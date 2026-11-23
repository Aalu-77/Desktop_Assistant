"""Ultron: native always-on-top system widget (PySide6).

One Python process. It imports main.py and voice.py directly (no uvicorn, no browser).

    python widget.py

  - Drag the pill to move it. Click the rings to open/close the panel.
  - Click the mic, wait for the beep, then speak (status / schedule / news).
  - Say "Hi Ultron" (on by default; ULTRON_WAKE=0 starts with it off), e.g.
    "Hi Ultron, what's the status?"  Toggle it with a right-click on the widget.
  - Right-click for Quit (or use the X in the panel).
"""
from __future__ import annotations

import html
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import main  # data providers + voice wiring (this does NOT start a web server)
import voice

COLLAPSED = (290, 68)
EXPANDED = (330, 380)
PILL_H = 68
OK, WARM, HOT, BLUE = "#4ade80", "#fbbf24", "#f87171", "#60a5fa"
TEXT, MUTED = "#e8eaf0", "#8b90a0"
STATE_FILE = Path(__file__).with_name("widget_state.json")

QSS = f"""
QFrame#card {{ background: rgba(18, 20, 28, 220); border: 1px solid rgba(255, 255, 255, 23); border-radius: 18px; }}
QLabel {{ background: transparent; }}
QPushButton#tab {{ color: {MUTED}; background: transparent; border: none; padding: 5px 10px; border-radius: 8px; font-size: 12px; }}
QPushButton#tab:checked {{ background: rgba(255, 255, 255, 26); color: {TEXT}; }}
QPushButton#quit {{ color: {MUTED}; background: transparent; border: none; padding: 5px 9px; border-radius: 8px; font-size: 13px; }}
QPushButton#quit:hover {{ background: rgba(248, 113, 113, 50); color: {HOT}; }}
QFrame#item {{ background: rgba(255, 255, 255, 10); border-radius: 10px; }}
QFrame#next {{ background: rgba(74, 222, 128, 31); border-left: 3px solid {OK}; border-radius: 10px; }}
QFrame#you {{ background: rgba(255, 255, 255, 10); border-radius: 10px; }}
QFrame#bot {{ background: rgba(96, 165, 250, 31); border-left: 3px solid {BLUE}; border-radius: 10px; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ width: 6px; background: transparent; margin: 0; }}
QScrollBar::handle:vertical {{ background: rgba(255, 255, 255, 50); border-radius: 3px; min-height: 20px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QToolTip {{ color: {TEXT}; background: #1b1e29; border: 1px solid #333846; padding: 4px; }}
QMenu {{ background: #1b1e29; color: {TEXT}; border: 1px solid #333846; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background: rgba(255, 255, 255, 26); }}
"""


# ---------- Background data threads -> GUI (thread-safe via Qt signals) ----------
class Bus(QObject):
    metrics = Signal(object)
    voice = Signal(object)
    news = Signal(object)
    events = Signal(object)


def _fast_loop(bus: Bus, stop: threading.Event) -> None:
    tick = 0
    while not stop.is_set():
        try:
            if tick % 4 == 0:  # metrics once a second
                bus.metrics.emit(main.read_metrics())
            bus.voice.emit(voice.get_state())  # voice state four times a second
        except Exception:
            traceback.print_exc()
        tick += 1
        stop.wait(0.25)


def _slow_loop(bus: Bus, stop: threading.Event) -> None:
    n = 0
    while not stop.is_set():
        try:
            snap = main.voice_news_snapshot()  # also kicks off a refresh when the cache is stale
            bus.news.emit({**snap, "items": [dict(i) for i in snap.get("items", [])]})
            if n % 12 == 0:  # events once a minute
                bus.events.emit(main.get_today_events())
        except Exception:
            traceback.print_exc()
        n += 1
        stop.wait(5)


# ---------- Small custom-painted widgets ----------
class Grip(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(20, 36)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 90))
        for row in range(4):
            for col in range(2):
                p.drawEllipse(QPointF(5 + col * 7, 8 + row * 7), 1.4, 1.4)


class Dot(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(8, 8)
        self.on = True

    def set_on(self, on: bool):
        if on != self.on:
            self.on = on
            self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(OK if self.on else HOT))
        p.drawEllipse(QRectF(1, 1, 6, 6))


class Ring(QWidget):
    def __init__(self, label: str):
        super().__init__()
        self.label = label
        self.value: float | None = None
        self.setFixedSize(44, 54)

    def set_value(self, value: float | None, tip: str = ""):
        self.value = value
        self.setToolTip(tip)
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(5, 3, 34, 34)
        p.setPen(QPen(QColor(255, 255, 255, 26), 4))
        p.drawEllipse(rect)
        pct = self.value
        if pct is not None and pct > 0:
            color = QColor(HOT if pct > 85 else WARM if pct > 60 else OK)
            p.setPen(QPen(color, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(rect, 90 * 16, -int(max(pct, 2) / 100 * 360 * 16))
        f = QFont(self.font())
        f.setPixelSize(12)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.setPen(QColor(TEXT))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "-" if pct is None else str(round(pct)))
        f2 = QFont(self.font())
        f2.setPixelSize(9)
        f2.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        p.setFont(f2)
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(0, 39, 44, 14), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self.label)


class MicButton(QPushButton):
    STYLE = {  # state -> (background, icon colour)
        "idle": (QColor(255, 255, 255, 24), QColor(TEXT)),
        "listening": (QColor(OK), QColor("#0b1a10")),
        "thinking": (QColor(WARM), QColor("#1f1600")),
        "speaking": (QColor(BLUE), QColor("#06121f")),
    }

    def __init__(self):
        super().__init__()
        self.setFixedSize(44, 44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Ask Ultron")
        self.state = "idle"
        self.wake = False  # True while the "Hi Ultron" listener is on
        self.phase = 0.0
        self.hover = False
        t = QTimer(self)
        t.timeout.connect(self._pulse)
        t.start(40)

    def set_state(self, state: str):
        if state != self.state:
            self.state = state
            self.update()

    def set_wake(self, on: bool):
        if on != self.wake:
            self.wake = on
            self.setToolTip("Say 'Hi Ultron', or click to ask" if on else "Ask Ultron")
            self.update()

    def _pulse(self):
        if self.state in ("listening", "speaking"):
            self.phase = (self.phase + 0.04) % 1.0
            self.update()

    def enterEvent(self, e):
        self.hover = True
        self.update()

    def leaveEvent(self, e):
        self.hover = False
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg, fg = self.STYLE.get(self.state, self.STYLE["idle"])
        c = QPointF(self.width() / 2, self.height() / 2)
        if self.state in ("listening", "speaking"):
            ring = QColor(bg)
            ring.setAlpha(int(150 * (1 - self.phase)))
            p.setPen(QPen(ring, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            r = 16 + 6 * self.phase
            p.drawEllipse(c, r, r)
        if self.wake and self.state == "idle":  # thin green ring = I'm listening for "Hi Ultron"
            p.setPen(QPen(QColor(74, 222, 128, 110), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, 19, 19)
        fill = QColor(bg)
        if self.state == "idle" and self.hover:
            fill.setAlpha(46)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawEllipse(c, 16, 16)
        p.setPen(QPen(fg, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(c.x() - 3.2, c.y() - 9, 6.4, 11.5), 3.2, 3.2)
        p.drawArc(QRectF(c.x() - 6.5, c.y() - 7.5, 13, 13), 180 * 16, 180 * 16)
        p.drawLine(QPointF(c.x(), c.y() + 5.5), QPointF(c.x(), c.y() + 9))
        p.drawLine(QPointF(c.x() - 3, c.y() + 9), QPointF(c.x() + 3, c.y() + 9))


# ---------- Panel pages ----------
def label(text: str, size: int = 12, color: str = TEXT, bold: bool = False, rich: bool = False) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setStyleSheet(f"color: {color}; font-size: {size}px; font-weight: {600 if bold else 400};")
    if rich:
        lb.setTextFormat(Qt.TextFormat.RichText)
        lb.setOpenExternalLinks(True)  # opens in your default browser
    return lb


def card(kind: str, *children: QWidget) -> QFrame:
    f = QFrame()
    f.setObjectName(kind)
    v = QVBoxLayout(f)
    v.setContentsMargins(10, 8, 10, 8)
    v.setSpacing(2)
    for c in children:
        v.addWidget(c)
    return f


class ListPage(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        self.lay = QVBoxLayout(inner)
        self.lay.setContentsMargins(14, 10, 14, 14)
        self.lay.setSpacing(8)
        self.lay.addStretch(1)
        self.setWidget(inner)

    def set_widgets(self, widgets: list[QWidget]):
        while self.lay.count() > 1:
            w = self.lay.takeAt(0).widget()
            if w:
                w.hide()
                w.setParent(None)  # disappears at once; deleteLater alone waits for the event loop
                w.deleteLater()
        for i, w in enumerate(widgets):
            self.lay.insertWidget(i, w)


def _hhmm(iso: str) -> str:
    return voice._parse(iso).strftime("%H:%M")


def build_events(events: list[dict]) -> list[QWidget]:
    if not events:
        return [label("Nothing scheduled today.", 13, MUTED)]
    now = datetime.now()
    nxt = next((i for i, e in enumerate(events) if voice._parse(e["end"]) > now), -1)
    out = []
    for i, e in enumerate(events):
        past = nxt == -1 or i < nxt
        out.append(
            card(
                "next" if i == nxt else "item",
                label(f"{_hhmm(e['start'])}-{_hhmm(e['end'])}", 11, MUTED),
                label(e["title"], 13, MUTED if past else TEXT),
            )
        )
    return out


def build_news(snap: dict) -> list[QWidget]:
    items, status = snap.get("items", []), snap.get("status")
    if not items:
        if status != "ready":
            return [label("Fetching headlines...", 13, MUTED)]
        errs = snap.get("errors") or []
        return [label("No headlines. " + (errs[0] if errs else ""), 13, MUTED)]
    out: list[QWidget] = []
    for it in items:
        link = html.escape(it["link"], quote=True)
        head = f'<a href="{link}" style="color:{TEXT}; text-decoration:none; font-weight:600">{html.escape(it["title"])}</a>'
        out.append(
            card(
                "item",
                label(it["source"].upper(), 10, OK),
                label(head, 13, TEXT, rich=True),
                label(it["summary"], 12, "#c4c8d4" if it.get("summarized") else MUTED),
            )
        )
    if status == "summarizing":
        out.append(label("Summarizing with the local model...", 11, MUTED))
    elif snap.get("errors"):
        out.append(label("Issues: " + ", ".join(snap["errors"]), 11, MUTED))
    return out


def build_voice(v: dict) -> list[QWidget]:
    st = v.get("state", "idle")
    if st == "listening":
        return [label("Listening... speak now.", 13, MUTED)]
    if st == "thinking":
        return [label("Thinking...", 13, MUTED)]
    if not (v.get("heard") or v.get("reply") or v.get("error")):
        hint = "Say 'Hi Ultron' or click the mic" if v.get("wake") else "Click the mic"
        return [label(f"{hint} and ask about status, schedule or news.", 13, MUTED)]
    out: list[QWidget] = []
    if v.get("heard"):
        out.append(card("you", label("YOU", 10, MUTED), label(html.escape(v["heard"]), 13)))
    if v.get("reply"):
        who = "ULTRON" + (f" - {v['intent']}" if v.get("intent") else "")
        out.append(card("bot", label(who, 10, MUTED), label(html.escape(v["reply"]), 13)))
    if v.get("error"):
        out.append(label(v["error"], 11, HOT))
    return out


# ---------- The widget ----------
class Widget(QWidget):
    def __init__(self, start_threads: bool = True):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(QSS)
        self.is_open = False
        self._starting = False
        self._voice: dict = {}
        self._sig: dict = {}
        self._press = None
        self._moved = False
        self._press_y = 0.0
        self._last_metrics = time.monotonic()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        frame = QFrame()
        frame.setObjectName("card")
        outer.addWidget(frame)
        col = QVBoxLayout(frame)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        # pill
        pill = QWidget()
        pill.setFixedHeight(PILL_H)
        row = QHBoxLayout(pill)
        row.setContentsMargins(6, 0, 12, 0)
        row.setSpacing(4)
        row.addWidget(Grip())
        self.cpu, self.ram, self.gpu = Ring("CPU"), Ring("RAM"), Ring("GPU")
        row.addStretch(1)
        for r in (self.cpu, self.ram, self.gpu):
            row.addWidget(r)
            row.addStretch(1)
        self.mic = MicButton()
        self.mic.clicked.connect(self.start_voice)
        row.addWidget(self.mic)
        self.dot = Dot()
        row.addWidget(self.dot)
        col.addWidget(pill)

        # panel
        self.panel = QWidget()
        pv = QVBoxLayout(self.panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(0)
        tabs = QHBoxLayout()
        tabs.setContentsMargins(12, 10, 12, 0)
        tabs.setSpacing(6)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.stack = QStackedWidget()
        self.pages = [ListPage(), ListPage(), ListPage()]
        for i, name in enumerate(("Schedule", "News", "Voice")):
            b = QPushButton(name)
            b.setObjectName("tab")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, i=i: self.stack.setCurrentIndex(i))
            self.group.addButton(b, i)
            tabs.addWidget(b)
            self.stack.addWidget(self.pages[i])
        self.group.button(0).setChecked(True)
        tabs.addStretch(1)
        quit_btn = QPushButton("\u2715")
        quit_btn.setObjectName("quit")
        quit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        quit_btn.setToolTip("Quit widget")
        quit_btn.clicked.connect(QApplication.quit)
        tabs.addWidget(quit_btn)
        pv.addLayout(tabs)
        pv.addWidget(self.stack, 1)
        col.addWidget(self.panel, 1)
        self.panel.hide()
        self.setFixedSize(*COLLAPSED)

        self.pages[0].set_widgets([label("Loading...", 13, MUTED)])
        self.pages[1].set_widgets([label("Fetching headlines...", 13, MUTED)])
        self.pages[2].set_widgets(build_voice({}))

        self.bus = Bus()
        self.bus.metrics.connect(self.on_metrics)
        self.bus.voice.connect(self.on_voice)
        self.bus.news.connect(self.on_news)
        self.bus.events.connect(self.on_events)
        self._stop = threading.Event()
        if start_threads:
            for fn in (_fast_loop, _slow_loop):
                threading.Thread(target=fn, args=(self.bus, self._stop), daemon=True).start()
        QApplication.instance().aboutToQuit.connect(self._stop.set)
        stale = QTimer(self)
        stale.timeout.connect(lambda: self.dot.set_on(time.monotonic() - self._last_metrics < 4))
        stale.start(2000)
        self._place()
        if start_threads and os.getenv("ULTRON_WAKE", "1") != "0":  # set ULTRON_WAKE=0 to start with it off
            QTimer.singleShot(800, voice.start_wake)

    # ----- data -> UI -----
    def on_metrics(self, m: dict):
        self._last_metrics = time.monotonic()
        self.dot.set_on(True)
        self.cpu.set_value(m["cpu"]["percent"])
        r = m["ram"]
        self.ram.set_value(r["percent"], f"{r['used_gb']} / {r['total_gb']} GB")
        g = m.get("gpu")
        self.gpu.set_value(
            g["percent"] if g else None,
            f"VRAM {g['vram_used_mb']} / {g['vram_total_mb']} MB" if g else "No NVIDIA GPU detected",
        )

    def _refresh_mic(self):
        self.mic.set_state("listening" if self._starting else self._voice.get("state", "idle"))

    def on_voice(self, v: dict):
        self._voice = v
        self._refresh_mic()
        self.mic.set_wake(bool(v.get("wake")))
        sig = (v.get("state"), v.get("heard"), v.get("reply"), v.get("intent"), v.get("error"), v.get("wake"))
        if sig != self._sig.get("voice"):
            self._sig["voice"] = sig
            self.pages[2].set_widgets(build_voice(v))

    def on_news(self, snap: dict):
        sig = json.dumps(snap, default=str, sort_keys=True)
        if sig != self._sig.get("news"):
            self._sig["news"] = sig
            self.pages[1].set_widgets(build_news(snap))

    def on_events(self, events: list):
        sig = json.dumps(events, default=str) + datetime.now().strftime("%H%M")  # re-evaluate "next" each minute
        if sig != self._sig.get("events"):
            self._sig["events"] = sig
            self.pages[0].set_widgets(build_events(events))

    # ----- actions -----
    def start_voice(self):
        if self._starting or self._voice.get("state", "idle") != "idle":
            return
        self._starting = True
        self._refresh_mic()
        QTimer.singleShot(1500, self._end_starting)
        if self.is_open:
            self.group.button(2).setChecked(True)
            self.stack.setCurrentIndex(2)
        voice.start_cycle()

    def _end_starting(self):
        self._starting = False
        self._refresh_mic()

    def toggle(self):
        self.is_open = not self.is_open
        self.panel.setVisible(self.is_open)
        self.setFixedSize(*(EXPANDED if self.is_open else COLLAPSED))
        self._clamp()

    # ----- placement -----
    def _clamp(self):
        scr = self.screen() or QGuiApplication.primaryScreen()
        geo = scr.availableGeometry()
        x = max(geo.left(), min(self.x(), geo.right() - self.width() + 1))
        y = max(geo.top(), min(self.y(), geo.bottom() - self.height() + 1))
        if (x, y) != (self.x(), self.y()):
            self.move(x, y)

    def _place(self):
        try:
            s = json.loads(STATE_FILE.read_text())
            from PySide6.QtCore import QPoint

            if QGuiApplication.screenAt(QPoint(int(s["x"]), int(s["y"]))) is not None:
                self.move(int(s["x"]), int(s["y"]))
                return
        except Exception:
            pass
        geo = (QGuiApplication.primaryScreen()).availableGeometry()
        self.move(geo.right() - COLLAPSED[0] - 24, geo.top() + 24)

    def _save_pos(self):
        try:
            STATE_FILE.write_text(json.dumps({"x": self.x(), "y": self.y()}))
        except Exception:
            pass

    # ----- mouse: drag anywhere on the card, click the pill to open/close -----
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._press = e.globalPosition().toPoint()
            self._origin = self.pos()
            self._moved = False
            self._press_y = e.position().y()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._press is not None and e.buttons() & Qt.MouseButton.LeftButton:
            d = e.globalPosition().toPoint() - self._press
            if self._moved or d.manhattanLength() > 5:
                self._moved = True
                self.move(self._origin + d)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._press is not None:
            if self._moved:
                self._clamp()
                self._save_pos()
            elif self._press_y < PILL_H:
                self.toggle()
            self._press = None

    def _toggle_wake(self, on: bool):
        if on:
            voice.start_wake()
        else:
            voice.stop_wake()

    def contextMenuEvent(self, e):
        menu = QMenu(self)
        wake = menu.addAction("Listen for 'Hi Ultron'")
        wake.setCheckable(True)
        wake.setChecked(bool(self._voice.get("wake")))
        wake.triggered.connect(self._toggle_wake)
        menu.addAction("Quit Ultron", QApplication.quit)
        menu.exec(e.globalPos())


def run() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Ultron")
    app.setFont(QFont("Segoe UI", 9))
    w = Widget()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
