"""Voice assistant for the widget: push-to-talk AND the "Hi Ultron" wake word.

Push-to-talk cycle:  beep -> record until you stop talking -> transcribe (Whisper via
onnxruntime) -> pick an intent (status / schedule / news) -> build the answer from real
data -> speak it.

Wake word:  a background thread listens to the mic. Short bursts of speech are transcribed
locally; if the first words contain "Ultron" it wakes up.
    "Hi Ultron, what's the status?"   -> answers straight away (one breath)
    "Hi Ultron"                       -> beep, then it listens for your question
Audio is never saved or sent anywhere; it only lives in memory while being transcribed.

Heavy imports (sounddevice, onnx_asr, pyttsx3) are lazy, so the backend still starts if one
of them is missing; the problem is then reported through get_state()["error"].

Terminal tests:   python voice.py          (one push-to-talk cycle)
                  python voice.py wake     (wake word; prints what it hears; Ctrl+C to stop)
"""
from __future__ import annotations

import os
import queue
import sys
import re
import threading
import time
import traceback
import json
import urllib.request
import urllib.error
from collections import deque
from datetime import datetime
from difflib import SequenceMatcher
from typing import Callable

import numpy as np
import psutil

# ---------- Tunables ----------
SR = 16000                 # Whisper wants 16 kHz mono
BLOCK_S = 0.05             # analysis block: 50 ms
CALIBRATE_S = 0.4          # measure room noise right after the beep
START_TIMEOUT_S = 6.0      # give up if you say nothing
END_SILENCE_S = 1.0        # stop after this much silence once you have spoken
MAX_UTTERANCE_S = 10.0
MIN_SPEECH_S = 0.3
PRE_ROLL_S = 0.3           # keep a little audio from before speech was detected
THRESH_MULT = 3.0          # speech threshold = noise floor * this
MIN_THRESHOLD = 0.0025     # RMS floor; lower it if your mic is very quiet
MODEL_NAME = os.getenv("ULTRON_STT_MODEL", "whisper-base")
VOICE_HINT = os.getenv("ULTRON_VOICE", "")        # e.g. "zira" to force a voice by name
SPEECH_RATE = int(os.getenv("ULTRON_RATE", "175"))

# ---------- Ollama conversational AI ----------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

SYSTEM_PROMPT = """
You are ULTRON, a sophisticated personal AI assistant inspired by JARVIS.

PERSONALITY:
- Calm, intelligent, professional and occasionally witty.
- Speak naturally.

RESPONSE RULES:
- Keep spoken answers to 1–3 short sentences.
- Aim for a maximum of 30 words.
- Give only the most important information.
- Never use Markdown, asterisks, bullet points or numbered lists.
- Do not repeat information unnecessarily.
- Give more detail only when the user asks.
- Be honest. Never invent live system information.
"""

_chat_history = deque(maxlen=12)
_chat_lock = threading.Lock()

# wake word
WAKE_WORD = "ultron"
WAKE_ALIASES = {a.strip() for a in os.getenv("ULTRON_WAKE_ALIASES", "").lower().split(",") if a.strip()}
WAKE_SIMILARITY = 0.8      # how close a heard word must be to "ultron" (1.0 = identical)
WAKE_END_SILENCE_S = 0.7   # a wake phrase is over after this much silence
WAKE_MAX_S = 8.0           # longer bursts are conversation/video, not a command: not transcribed
WAKE_MIN_SPEECH_S = 0.35   # ignore clicks and bumps
QUIET_AFTER_S = 0.8        # ignore the mic for a moment after Ultron has spoken (no self-trigger)

# ---------- Shared state ----------
_state: dict = {"state": "idle", "heard": "", "intent": "", "reply": "", "error": ""}
_state_lock = threading.Lock()
_cycle_lock = threading.Lock()          # held while a command is being listened to / answered
_providers: dict[str, Callable] = {}
_wake_on = False
_quiet_until = 0.0


def _set(**kw) -> None:
    with _state_lock:
        _state.update(kw)


def get_state() -> dict:
    with _state_lock:
        return {**_state, "wake": _wake_on}


def configure(
    *,
    metrics: Callable,
    events: Callable,
    news: Callable,
    ollama_url: str = OLLAMA_URL,
    ollama_model: str = OLLAMA_MODEL,
) -> None:
    """Receive data providers and local AI configuration from main.py."""
    global OLLAMA_URL, OLLAMA_MODEL

    _providers.update(metrics=metrics, events=events, news=news)
    OLLAMA_URL = ollama_url.rstrip("/")
    OLLAMA_MODEL = ollama_model


def missing_deps() -> str | None:
    for mod in ("sounddevice", "onnx_asr", "pyttsx3"):
        try:
            __import__(mod)
        except Exception as e:
            return f"{mod}: {e.__class__.__name__}: {e}"
    return None


# ---------- Speech-to-text model ----------
_model = None
_model_lock = threading.Lock()
_recognize_lock = threading.Lock()  # one transcription at a time


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            import onnx_asr

            _model = onnx_asr.load_model(MODEL_NAME)
        return _model


def preload() -> None:
    """Load the model in the background at startup so the first question isn't slow."""
    try:
        _get_model()
    except Exception as e:
        print(f"[voice] model preload failed: {e!r}")


# ---------- Microphone hub: one open stream, many listeners ----------
class _MicHub:
    """Keeps a single input stream open and hands every block to each subscriber's queue,
    so the wake-word listener and a push-to-talk recording can use the mic at the same time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._subs: list[queue.Queue] = []
        self._stream = None

    def subscribe(self) -> queue.Queue:
        import sounddevice as sd

        q: queue.Queue = queue.Queue(maxsize=400)  # ~20 s of audio
        with self._lock:
            if self._stream is None:
                stream = sd.InputStream(
                    samplerate=SR, channels=1, dtype="float32",
                    blocksize=int(SR * BLOCK_S), callback=self._callback,
                )
                stream.start()
                self._stream = stream
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)
            if not self._subs and self._stream is not None:  # last listener gone: release the mic
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def _callback(self, indata, frames, time_info, status):
        block = indata[:, 0].copy()
        for q in list(self._subs):
            try:
                q.put_nowait(block)
            except queue.Full:
                pass


_hub = _MicHub()


def _drain(q: queue.Queue) -> None:
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def _next_block(q: queue.Queue) -> np.ndarray:
    try:
        return q.get(timeout=2.0)
    except queue.Empty:
        raise RuntimeError("the microphone delivered no audio")


# ---------- Recording ----------
def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def beep() -> None:
    try:
        import winsound

        winsound.Beep(880, 120)  # "I'm listening"
    except Exception:
        pass


def record_utterance(debug: bool = True) -> np.ndarray | None:
    """Record from the mic until you stop talking. None = nothing heard."""
    q = _hub.subscribe()
    try:
        beep()
        _drain(q)  # drop the beep itself and anything older
        calib = [_rms(_next_block(q)) for _ in range(int(CALIBRATE_S / BLOCK_S))]
        floor = float(np.percentile(calib, 25))  # robust if you start talking during calibration
        thresh = max(floor * THRESH_MULT, MIN_THRESHOLD)
        if debug:
            print(f"[voice] noise floor {floor:.4f}, speech threshold {thresh:.4f}")

        pre: deque = deque(maxlen=max(1, int(PRE_ROLL_S / BLOCK_S)))
        chunks: list[np.ndarray] = []
        started = False
        hot = silent = speech_blocks = 0
        t_wait = time.monotonic()
        t_speech = 0.0
        while True:
            data = _next_block(q)
            loud = _rms(data) > thresh
            now = time.monotonic()
            if not started:
                pre.append(data)
                hot = hot + 1 if loud else 0
                if hot >= 2:  # two loud blocks in a row, so a single click doesn't trigger it
                    started, t_speech, chunks = True, now, list(pre)
                    speech_blocks = hot
                elif now - t_wait > START_TIMEOUT_S:
                    return None
            else:
                chunks.append(data)
                if loud:
                    silent = 0
                    speech_blocks += 1
                else:
                    silent += 1
                if silent * BLOCK_S >= END_SILENCE_S or now - t_speech > MAX_UTTERANCE_S:
                    break
    finally:
        _hub.unsubscribe(q)
    if speech_blocks * BLOCK_S < MIN_SPEECH_S:
        return None
    return np.concatenate(chunks)


# ---------- Transcription ----------
_NONSPEECH = re.compile(r"^\s*[\[\(].*[\]\)]\s*$")  # Whisper writes things like "[silence]"


def _normalize(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if 0 < peak < 0.5:  # quiet mic: bring the level up
        audio = audio * (0.9 / peak)
    return audio.astype(np.float32)


def transcribe(audio: np.ndarray) -> str:
    model = _get_model()
    with _recognize_lock:
        text = str(model.recognize(_normalize(audio), sample_rate=SR)).strip()
    return "" if _NONSPEECH.match(text) else text


# ---------- Intent routing ----------
INTENT_WORDS = {  # tie -> first listed wins
    "status": ("status", "system", "cpu", "processor", "ram", "memory", "gpu", "graphics",
               "performance", "laptop", "computer", "usage", "load"),
    "schedule": ("schedule", "calendar", "meeting", "meetings", "agenda", "appointment",
                 "appointments", "event", "events", "busy", "plan", "what do i have",
                 "what's on", "whats on"),
    "news": ("news", "headline", "headlines", "happening", "world"),
}
GREETINGS = ("hello", "hi", "hey", "good morning", "good evening")


def route(text: str) -> str:
    t = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", text.lower())) + " "
    scores = {k: sum(f" {w} " in t for w in ws) for k, ws in INTENT_WORDS.items()}
    best = max(scores, key=lambda k: scores[k])
    if scores[best] > 0:
        return best
    if any(f" {g} " in t for g in GREETINGS):
        return "greet"
    return "unknown"


# ---------- Answers (built from real data, not generated by the LLM) ----------
def _parse(iso: str) -> datetime:
    dt = datetime.fromisoformat(iso)
    return dt.astimezone().replace(tzinfo=None) if dt.tzinfo else dt  # Google Calendar gives offsets


def _clock(iso: str) -> str:
    dt = _parse(iso)
    return f"{dt.hour}:{dt.minute:02d}"


def answer_status() -> str:
    m = _providers["metrics"]()
    cpu = psutil.cpu_percent(interval=0.5)  # fresh sample; the shared reading can be stale
    ram, gpu = m["ram"], m.get("gpu")
    parts = [
        f"CPU is at {round(cpu)} percent",
        f"memory is at {round(ram['percent'])} percent, {ram['used_gb']} of {ram['total_gb']} gigabytes",
    ]
    if gpu:
        parts.append(
            f"GPU is at {gpu['percent']} percent, using {gpu['vram_used_mb']} of "
            f"{gpu['vram_total_mb']} megabytes of video memory"
        )
    else:
        parts.append("I can't read the GPU right now")
    msg = ", ".join(parts[:-1]) + f", and {parts[-1]}."
    if ram["percent"] >= 85:
        msg += " Memory is getting tight."
    elif cpu >= 85:
        msg += " The processor is under heavy load."
    return msg


def answer_schedule() -> str:
    events = _providers["events"]()
    if not events:
        return "You have nothing scheduled today."
    now = datetime.now()
    upcoming = [e for e in events if _parse(e["end"]) > now]
    n = len(events)
    head = f"You have {n} event{'s' if n != 1 else ''} today."
    if not upcoming:
        return head + " All of them are done."
    e = upcoming[0]
    if _parse(e["start"]) <= now:
        tail = f" Right now: {e['title']}, until {_clock(e['end'])}."
    else:
        tail = f" Next up is {e['title']} at {_clock(e['start'])}."
    if len(upcoming) > 1:
        tail += f" After that, {upcoming[1]['title']} at {_clock(upcoming[1]['start'])}."
    return head + tail


def answer_news() -> str:
    snap = _providers["news"]()
    items = snap.get("items", [])
    if not items:
        if snap.get("status") != "ready":
            return "I'm still fetching the headlines. Ask me again in a moment."
        return "I couldn't fetch any headlines right now."
    lines = []
    for it in items[:3]:
        text = it["summary"] if it.get("summarized") else it["title"]  # summaries are in English
        lines.append(f"{it['source']}: {text.rstrip('.')}.")
    return "Here are the top headlines. " + " ".join(lines)


ANSWERS = {
    "status": answer_status,
    "schedule": answer_schedule,
    "news": answer_news,
    "greet": lambda: "Hi, I'm Ultron. Ask me about your system status, your schedule, or the news.",
    "unknown": lambda: "I didn't catch that. Try asking about status, schedule, or news.",
}


# ---------- Speaking ----------
def _pick_voice(engine) -> str | None:
    voices = engine.getProperty("voices")
    hint = VOICE_HINT.lower()
    if hint:
        for v in voices:
            if hint in (v.name + " " + v.id).lower():
                return v.id
    for v in voices:  # otherwise prefer any English voice
        if "en-" in v.id.lower() or "english" in v.name.lower():
            return v.id
    return None

def clean_for_speech(text: str) -> str:
    """Remove Markdown formatting before sending text to speech."""

    text = re.sub(r"[*_#`~]", "", text)
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def speak(text: str) -> None:
    global _quiet_until
    import pyttsx3

    com = None
    try:  # worker threads need COM initialised before SAPI can be used
        import comtypes

        comtypes.CoInitialize()
        com = comtypes
    except Exception:
        pass
    try:
        engine = pyttsx3.init()
        vid = _pick_voice(engine)
        if vid:
            engine.setProperty("voice", vid)
        engine.setProperty("rate", SPEECH_RATE)
        engine.say(clean_for_speech(text))
        engine.runAndWait()
        engine.stop()
    finally:
        _quiet_until = time.monotonic() + QUIET_AFTER_S  # the wake listener must not hear Ultron itself
        if com is not None:
            try:
                com.CoUninitialize()
            except Exception:
                pass



# ---------- Safe Python tools available to Ollama ----------
OLLAMA_TOOLS = [
    {"type": "function", "function": {
        "name": "get_system_status",
        "description": "Get current CPU, RAM and GPU usage from this computer.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_today_schedule",
        "description": "Get today's calendar events and their start/end times.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_latest_news",
        "description": "Get the latest available news headlines and summaries.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
]


def _run_ollama_tool(name: str) -> str:
    """Run only explicitly allowlisted read-only tools."""
    if name == "get_system_status":
        data = _providers["metrics"]()
        data["cpu"]["percent"] = psutil.cpu_percent(interval=0.3)
        return json.dumps(data)

    if name == "get_today_schedule":
        return json.dumps({
            "events": _providers["events"](),
            "data_note": (
                "These are demonstration events from a placeholder calendar, "
                "not a connected personal calendar."
            ),
        })

    if name == "get_latest_news":
        snap = _providers["news"]()
        items = [
            {
                "source": item.get("source"),
                "title": item.get("title"),
                "summary": item.get("summary"),
            }
            for item in snap.get("items", [])[:3]
        ]
        return json.dumps({
            "status": snap.get("status"),
            "updated": snap.get("updated"),
            "items": items,
        })

    raise ValueError(f"Tool is not allowed: {name}")


def _ollama_chat(messages: list[dict], use_tools: bool = True) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "num_predict": 180,
            "num_ctx": 4096,
        },
    }
    if use_tools:
        payload["tools"] = OLLAMA_TOOLS

    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))



def ask_ollama(user_text: str) -> str:
    """Chat with Ollama and execute approved read-only tools when requested."""
    with _chat_lock:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *list(_chat_history),
            {"role": "user", "content": user_text},
        ]

        try:
            result = _ollama_chat(messages, use_tools=True)
            message = result.get("message", {})
            tool_calls = message.get("tool_calls", []) or []

            if tool_calls:
                messages.append(message)
                for call in tool_calls[:3]:
                    function = call.get("function", {})
                    name = function.get("name", "")
                    tool_result = _run_ollama_tool(name)
                    messages.append({
                        "role": "tool",
                        "name": name,
                        "content": tool_result,
                    })

                result = _ollama_chat(messages, use_tools=False)
                message = result.get("message", {})

            reply = message.get("content", "").strip()
            if not reply:
                raise RuntimeError("Ollama returned an empty response")

            _chat_history.append({"role": "user", "content": user_text})
            _chat_history.append({"role": "assistant", "content": reply})
            return reply

        except urllib.error.URLError as e:
            raise RuntimeError(
                "I cannot connect to Ollama. Please check that Ollama is running."
            ) from e


# ---------- Answering ----------
def _answer_and_speak(text: str) -> None:
    if not text.strip():
        reply = "I didn't catch that. Could you repeat it?"
        intent = "chat"
    else:
        intent = "chat"
        try:
            reply = ask_ollama(text)
        except Exception as e:
            reply = f"I'm having trouble processing that request. {e}"
            _set(error=f"{e.__class__.__name__}: {e}")

    _set(
        state="speaking",
        heard=text,
        intent=intent,
        reply=reply,
    )

    print(f"[voice] heard={text!r} intent={intent} reply={reply!r}")
    speak(reply)


def _listen_and_answer() -> None:
    """Record one question from the mic and answer it. The caller holds _cycle_lock."""
    _set(state="listening", heard="", intent="", reply="", error="")
    audio = record_utterance()
    if audio is None:
        _set(state="speaking", reply="I didn't hear anything.")
        speak("I didn't hear anything.")
        return
    _set(state="thinking")
    _answer_and_speak(transcribe(audio))


def run_cycle() -> None:
    """One push-to-talk cycle."""
    try:
        err = missing_deps()
        if err:
            _set(state="idle", error=f"Voice dependencies missing ({err})")
            return
        _listen_and_answer()
    except Exception as e:
        traceback.print_exc()
        _set(error=f"{e.__class__.__name__}: {e}")
    finally:
        _set(state="idle")


def start_cycle() -> bool:
    """Start one push-to-talk cycle in the background. False if one is already running."""
    if not _cycle_lock.acquire(blocking=False):
        return False

    def _worker():
        try:
            run_cycle()
        finally:
            _cycle_lock.release()

    threading.Thread(target=_worker, daemon=True).start()
    return True


# ---------- Wake word ----------
def _is_wake(token: str) -> bool:
    return (
        token == WAKE_WORD
        or token in WAKE_ALIASES
        or SequenceMatcher(None, token, WAKE_WORD).ratio() >= WAKE_SIMILARITY
    )


def find_wake(text: str) -> tuple[bool, str]:
    """Is the wake word among the first words? Returns (found, the rest of the sentence)."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    for i in range(min(4, len(words))):  # "hi ultron ..." / "hey, ultron ..." -- it has to come first
        for width in (1, 2):  # also catches a split like "ultra on"
            token = "".join(words[i : i + width])
            if len(token) >= 4 and _is_wake(token):
                return True, " ".join(words[i + width :])
    return False, ""


def _on_wake_segment(seg: np.ndarray) -> None:
    """Called by the wake thread with one burst of speech."""
    text = transcribe(seg)
    found, rest = find_wake(text)
    print(f"[wake] heard {text!r} -> {'WAKE' if found else 'ignored'}")
    if not found:
        return
    if not _cycle_lock.acquire(blocking=False):  # a push-to-talk cycle is already running
        return
    try:
        _set(state="thinking", heard=text, intent="", reply="", error="")
        if rest.strip():
            _answer_and_speak(rest)       # "Hi Ultron, what's the status?"
        else:
            _listen_and_answer()          # just "Hi Ultron": beep, then listen
    except Exception as e:
        traceback.print_exc()
        _set(error=f"{e.__class__.__name__}: {e}")
    finally:
        _set(state="idle")
        _cycle_lock.release()


_wake_thread: threading.Thread | None = None
_wake_stop = threading.Event()


def _wake_loop() -> None:
    global _wake_on, _quiet_until
    try:
        q = _hub.subscribe()
    except Exception as e:
        _set(error=f"Wake word: microphone unavailable ({e.__class__.__name__}: {e})")
        _wake_on = False
        return
    print("[wake] listening for 'Hi Ultron'...")
    pre: deque = deque(maxlen=max(1, int(PRE_ROLL_S / BLOCK_S)))
    chunks: list[np.ndarray] = []
    levels: list[float] = []
    floor: float | None = None
    started = False
    hot = silent = loud_blocks = 0
    try:
        while not _wake_stop.is_set():
            try:
                data = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if _cycle_lock.locked() or time.monotonic() < _quiet_until:
                # someone is talking to Ultron, or Ultron is talking: don't listen to that
                pre.clear(); chunks = []; levels = []
                started = False
                hot = silent = loud_blocks = 0
                continue
            level = _rms(data)
            floor = level if floor is None else floor
            thresh = max(floor * THRESH_MULT, MIN_THRESHOLD)
            loud = level > thresh
            if not started:
                if not loud:
                    floor = 0.98 * floor + 0.02 * level  # track the room noise while it's quiet
                pre.append(data)
                hot = hot + 1 if loud else 0
                if hot >= 2:
                    started, chunks, levels = True, list(pre), []
                    silent, loud_blocks = 0, hot
                continue
            chunks.append(data)
            levels.append(level)
            if loud:
                silent = 0
                loud_blocks += 1
            else:
                silent += 1
            duration = len(chunks) * BLOCK_S
            if silent * BLOCK_S >= WAKE_END_SILENCE_S or duration > WAKE_MAX_S:
                seg, spoke = np.concatenate(chunks), loud_blocks * BLOCK_S
                if duration > WAKE_MAX_S and levels:
                    floor = max(floor, float(np.percentile(levels, 30)))  # constant noise: raise the floor
                started, chunks, hot, silent, loud_blocks = False, [], 0, 0, 0
                pre.clear()
                if duration <= WAKE_MAX_S and spoke >= WAKE_MIN_SPEECH_S:
                    try:
                        _on_wake_segment(seg)
                    except Exception:
                        traceback.print_exc()
                    _drain(q)  # whatever piled up while we were busy is stale
                    _quiet_until = max(_quiet_until, time.monotonic() + 0.2)
    finally:
        _hub.unsubscribe(q)
        _wake_on = False
        print("[wake] stopped")


def start_wake() -> bool:
    """Start listening for the wake word. False if it can't (missing packages)."""
    global _wake_on, _wake_thread
    if _wake_thread is not None and _wake_thread.is_alive() and not _wake_stop.is_set():
        return True
    err = missing_deps()
    if err:
        _set(error=f"Wake word unavailable: voice dependencies missing ({err})")
        return False
    if _wake_thread is not None and _wake_thread.is_alive():
        _wake_thread.join(timeout=2)  # previous loop is still shutting down
    _wake_stop.clear()
    _wake_on = True
    _wake_thread = threading.Thread(target=_wake_loop, daemon=True)
    _wake_thread.start()
    return True


def stop_wake() -> None:
    global _wake_on
    _wake_stop.set()
    _wake_on = False


if __name__ == "__main__":
    import main  # noqa: F401  (registers the data providers and starts the model preload)
    import voice as v  # the copy of this module that main.py configured

    problem = v.missing_deps()
    if problem:
        raise SystemExit(f"Missing dependency: {problem}")
    if len(sys.argv) > 1 and sys.argv[1] == "wake":
        print("Say 'Hi Ultron, what's the status?' (Ctrl+C to stop). Everything it hears is printed.")
        v.start_wake()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            v.stop_wake()
        raise SystemExit(0)
    import pyttsx3

    print("Installed voices:")
    for voice in pyttsx3.init().getProperty("voices"):
        print("  -", voice.name)
    print("\nSpeak after the beep, e.g. 'what's the status of my system'...")
    v.run_cycle()
    print("Final state:", v.get_state())
