"""
Tiresias - audio_engine.py
============================
Phase 5: The Voice (Text-to-Speech Audio Alerts)

Provides a non-blocking audio alert system using pyttsx3.
A dedicated daemon thread owns the TTS engine and processes
speech requests from a queue, preventing the main video loop
from freezing during speech synthesis.
"""

import queue
import threading
import time

import pyttsx3


class AudioEngine:
    """Non-blocking text-to-speech engine using a dedicated thread + queue."""

    def __init__(self, cooldown_center: float = 3.0, cooldown_sides: float = 5.0):
        """
        Args:
            cooldown_center: Seconds before re-alerting the same object in Center zone.
            cooldown_sides:  Seconds before re-alerting the same object in Left/Right zones.
        """
        self.cooldown_center = cooldown_center
        self.cooldown_sides = cooldown_sides
        self._last_spoken: dict[str, float] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()

        # Start dedicated TTS thread (daemon = dies when main thread exits)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("[Tiresias] Audio engine started.")

    def _worker(self):
        """Runs on a dedicated thread. Creates a fresh pyttsx3 engine per speech
        to avoid the Windows COM bug where runAndWait() corrupts engine state."""
        while True:
            text = self._queue.get()
            if text is None:  # Shutdown signal
                break
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 175)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception:
                pass
            finally:
                # pyttsx3.init() caches engines internally — clear the cache
                # so the next call creates a truly fresh engine instance.
                pyttsx3._activeEngines.clear()

    def alert(self, label: str, zone: str, proximity: str):
        """
        Queue a spoken alert if cooldown has elapsed.

        Only alerts for NEAR objects. Center zone gets shorter cooldown
        (more urgent). Left/Right get longer cooldown.

        Args: 
            label:     Object class name (e.g., "person", "stairs").
            zone:      "Left", "Center", or "Right".
            proximity: "NEAR", "MID", or "FAR".
        """
        if proximity != "NEAR":
            return

        cooldown = self.cooldown_center if zone == "Center" else self.cooldown_sides
        key = f"{label}_{zone}"
        now = time.monotonic()

        if now - self._last_spoken.get(key, 0) < cooldown:
            return

        self._last_spoken[key] = now

        # Build natural spoken text
        if zone == "Center":
            text = f"{label} ahead, close"
        elif zone == "Left":
            text = f"{label} on your left, close"
        else:
            text = f"{label} on your right, close"

        self._queue.put_nowait(text)

    def shutdown(self):
        """Signal the worker thread to stop."""
        self._queue.put(None)
