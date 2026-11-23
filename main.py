"""Widget backend: live system metrics + today's calendar events.

Run:  uvicorn main:app --host 127.0.0.1 --port 8765
"""
import asyncio
from datetime import datetime, timedelta

import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import voice  # push-to-talk assistant (see voice.py)

# ---------- GPU (NVIDIA) ----------
# Primary: NVML via pynvml. Fallback: parse `nvidia-smi` (ships with the driver).
import subprocess
import time
import traceback

try:
    import pynvml

    pynvml.nvmlInit()
    _gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # no NVIDIA driver / no GPU
    pynvml = None
    _gpu = None

_gpu_err_logged = False
_smi_cache: tuple[float, dict | None] = (0.0, None)


def _read_gpu_smi() -> dict | None:
    """Fallback reader. Cached for 2s so we don't spawn a process every tick."""
    global _smi_cache
    ts, val = _smi_cache
    if time.monotonic() - ts < 2:
        return val
    val = None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no console flash on Windows
        ).stdout.strip().splitlines()[0]
        util, used, total = (int(x) for x in out.split(","))
        val = {"percent": util, "vram_used_mb": used, "vram_total_mb": total}
    except Exception:
        pass
    _smi_cache = (time.monotonic(), val)
    return val


def read_gpu() -> dict | None:
    global _gpu_err_logged
    if _gpu is not None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_gpu)
            mem = pynvml.nvmlDeviceGetMemoryInfo(_gpu)
            return {
                "percent": util.gpu,
                "vram_used_mb": round(mem.used / 1024**2),
                "vram_total_mb": round(mem.total / 1024**2),
            }
        except Exception:
            if not _gpu_err_logged:  # print once, not every second
                traceback.print_exc()
                _gpu_err_logged = True
    return _read_gpu_smi()


# ---------- System metrics ----------
def read_metrics() -> dict:
    ram = psutil.virtual_memory()
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "cpu": {"percent": psutil.cpu_percent(interval=None)},
        "ram": {
            "percent": ram.percent,
            "used_gb": round(ram.used / 1024**3, 1),
            "total_gb": round(ram.total / 1024**3, 1),
        },
        "gpu": read_gpu(),
    }


# ---------- Calendar (swap this function for Google Calendar) ----------
def get_today_events() -> list[dict]:
    """Return today's events as [{title, start, end}], ISO-8601 strings.

    STUB: replace the body with the iCal URL or Google Calendar API call.
    Keep the return shape and nothing else in the app needs to change.
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    sample = [
        ("Stand-up", 9, 30, 9, 45),
        ("Deep work: widget project", 10, 0, 12, 0),
        ("Lunch", 12, 30, 13, 15),
    ]
    return [
        {
            "title": t,
            "start": (today + timedelta(hours=sh, minutes=sm)).isoformat(timespec="minutes"),
            "end": (today + timedelta(hours=eh, minutes=em)).isoformat(timespec="minutes"),
        }
        for t, sh, sm, eh, em in sample
    ]


# ---------- News (headlines + local LLM summaries) ----------
import html
import json
import os
import re
import threading
import urllib.request
import xml.etree.ElementTree as ET

# Ollama: set OLLAMA_MODEL to the exact name shown by `ollama list`.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

NEWS_FEEDS = [  # (label, url) -- edit freely
    ("BBC", "https://feeds.bbci.co.uk/news/rss.xml"),
    ("Tagesschau", "https://www.tagesschau.de/index~rss2.xml"),
]
ITEMS_PER_FEED = 3
NEWS_TTL_S = 20 * 60  # refresh headlines + summaries at most every 20 min

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub(" ", text or ""))).strip()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]  # strip XML namespace


def fetch_feed(source: str, url: str, limit: int) -> list[dict]:
    """Parse RSS 2.0 (<item>) or Atom (<entry>) with the stdlib only."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (system-widget)"})
    with urllib.request.urlopen(req, timeout=10) as r:
        root = ET.fromstring(r.read())
    items: list[dict] = []
    for el in root.iter():
        if _local(el.tag) not in ("item", "entry"):
            continue
        f: dict = {}
        for child in el:
            name = _local(child.tag)
            if name == "link":
                f.setdefault("link", child.get("href") or (child.text or "").strip())
            elif name in ("title", "description", "summary"):
                f.setdefault(name, child.text)
        title = _clean(f.get("title"))
        link = f.get("link", "")
        if not title or not link.startswith(("http://", "https://")):
            continue
        desc = _clean(f.get("description") or f.get("summary"))
        items.append(
            {
                "source": source,
                "title": title,
                "link": link,
                "desc": desc,
                "summary": desc[:140] + ("…" if len(desc) > 140 else ""),  # fallback until the LLM answers
                "summarized": False,
            }
        )
        if len(items) >= limit:
            break
    return items


def summarize(title: str, desc: str) -> str | None:
    prompt = (
        "Summarize this news item in one plain sentence of at most 25 words, in English. "
        "Reply with the sentence only.\n\n"
        f"Title: {title}\nDetails: {desc[:600]}"
    )
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 60},
        }
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=90) as r:  # first call may include model load time
        out = json.loads(r.read())["response"]
    return _clean(out).strip("\"") or None


_news: dict = {"status": "loading", "updated": None, "items": [], "errors": []}
_news_ts: float | None = None
_news_running = False
_news_lock = threading.Lock()


def _refresh_news() -> None:
    global _news, _news_ts, _news_running
    try:
        items: list[dict] = []
        errors: list[str] = []
        for source, url in NEWS_FEEDS:
            try:
                items += fetch_feed(source, url, ITEMS_PER_FEED)
            except Exception as e:
                errors.append(f"{source}: {e.__class__.__name__}")
        _news = {
            "status": "summarizing" if items else "ready",
            "updated": datetime.now().isoformat(timespec="seconds"),
            "items": items,
            "errors": errors,
        }
        for it in items:  # one at a time: friendly to a lower spec GPU
            try:
                s = summarize(it["title"], it["desc"])
            except Exception as e:
                errors.append(f"LLM: {e.__class__.__name__}")  # Ollama down/unreachable -> keep fallbacks
                break
            if s:
                it["summary"], it["summarized"] = s, True
        _news["status"] = "ready"
        # retry sooner if nothing could be fetched
        _news_ts = time.monotonic() - (NEWS_TTL_S - 60 if not items else 0)
    finally:
        _news_running = False


def _ensure_news_fresh() -> None:
    """Start a background refresh if the cached news is stale."""
    global _news_running
    stale = _news_ts is None or time.monotonic() - _news_ts > NEWS_TTL_S
    if stale:
        with _news_lock:
            if not _news_running:
                _news_running = True
                threading.Thread(target=_refresh_news, daemon=True).start()


def voice_news_snapshot() -> dict:
    _ensure_news_fresh()
    return _news


# ---------- Voice wiring ----------
voice.configure(
    metrics=read_metrics,
    events=get_today_events,
    news=voice_news_snapshot,
    ollama_url=OLLAMA_URL,
    ollama_model=OLLAMA_MODEL,
)
threading.Thread(target=voice.preload, daemon=True).start()  # load the speech model in the background


# ---------- App ----------
app = FastAPI(title="Widget backend")
psutil.cpu_percent(interval=None)  # prime: first call always returns 0.0


@app.get("/health")
def health():
    return {"ok": True, "gpu": read_gpu() is not None}


@app.get("/metrics")
def metrics():
    return read_metrics()


@app.get("/events/today")
def events_today():
    return get_today_events()


@app.get("/news")
def news():
    """Returns cached headlines immediately; refreshes in the background when stale."""
    _ensure_news_fresh()
    return {**_news, "items": [{k: v for k, v in i.items() if k != "desc"} for i in _news["items"]]}


@app.post("/voice/listen")
def voice_listen():
    """Push-to-talk: start one listen -> answer cycle in the background."""
    return {"started": voice.start_cycle()}


@app.get("/voice/state")
def voice_state():
    return {**voice.get_state(), "deps_error": voice.missing_deps()}


@app.post("/voice/wake")
def voice_wake(enabled: bool = True):
    """Turn the 'Hi Ultron' listener on or off."""
    if enabled:
        return {"enabled": voice.start_wake()}
    voice.stop_wake()
    return {"enabled": False}


@app.websocket("/ws/metrics")
async def ws_metrics(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json({**read_metrics(), "voice": voice.get_state()})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
