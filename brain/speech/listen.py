"""Microphone capture with energy-based endpointing.

No wake word: the lamp only listens while someone is engaged with it, so
"looking at the lamp" is the push-to-talk. While the lamp itself is speaking
the listener is paused (half-duplex), so it never transcribes its own voice
from the laptop speaker.

Endpointing: a speech segment starts when short-term energy rises well above
an adaptive noise floor, and ends after END_SILENCE of quiet. A pre-roll
buffer keeps the first syllable. Audio stays in memory and is only handed to
the local speech recogniser.
"""
from __future__ import annotations

import collections
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

SR = 16000            # Whisper's native rate
BLOCK = 480           # 30 ms
START_BLOCKS = 4      # 120 ms of loud audio starts a segment
END_SILENCE = 0.8     # s of quiet ends it
PRE_ROLL = 0.35       # s kept from before the start trigger
MAX_SPEECH = 12.0     # s hard cap per utterance
MIN_SPEECH = 0.35     # s; shorter segments are clicks/coughs


@dataclass
class Utterance:
    audio: np.ndarray        # float32 mono 16 kHz
    t_start: float           # wall time speech started
    t_end: float             # wall time endpoint was detected

    @property
    def seconds(self) -> float:
        return len(self.audio) / SR


class Endpointer:
    """Pure logic (no audio device), so it can be unit-tested."""

    def __init__(self) -> None:
        self.floor = 0.005
        self.pre = collections.deque(maxlen=int(PRE_ROLL * SR / BLOCK))
        self.buf: list[np.ndarray] = []
        self.loud = 0
        self.quiet = 0
        self.active = False
        self.t_start = 0.0

    def reset(self) -> None:
        self.pre.clear()
        self.buf, self.loud, self.quiet, self.active = [], 0, 0, False

    def push(self, block: np.ndarray, now: float) -> Utterance | None:
        rms = float(np.sqrt(np.mean(block ** 2)) + 1e-9)
        on = max(0.012, self.floor * 3.5)
        off = max(0.008, self.floor * 2.0)
        if not self.active:
            # Track the noise floor only while nobody is talking.
            self.floor += (0.02 if rms < self.floor * 3 else 0.002) * (rms - self.floor)
            self.pre.append(block)
            self.loud = self.loud + 1 if rms > on else 0
            if self.loud >= START_BLOCKS:
                self.active, self.quiet = True, 0
                self.t_start = now - len(self.pre) * BLOCK / SR
                self.buf = list(self.pre)
            return None
        self.buf.append(block)
        self.quiet = self.quiet + 1 if rms < off else 0
        dur = len(self.buf) * BLOCK / SR
        if self.quiet * BLOCK / SR >= END_SILENCE or dur >= MAX_SPEECH:
            audio = np.concatenate(self.buf)
            trim = max(0, self.quiet - 6) * BLOCK          # drop trailing silence
            audio = audio[: len(audio) - trim] if trim else audio
            self.reset()
            if len(audio) / SR - 0.18 < MIN_SPEECH:
                return None
            return Utterance(audio.astype(np.float32), self.t_start, now)
        return None


class Listener:
    def __init__(self, device=None):
        import sounddevice as sd
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self.utterances: queue.Queue[Utterance] = queue.Queue()
        self.enabled = threading.Event()
        self.ep = Endpointer()
        self.level = 0.0                  # for the debug overlay
        self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                      blocksize=BLOCK, device=device,
                                      callback=lambda d, *_: self._q.put(d[:, 0].copy()))
        self._stream.start()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            block = self._q.get()
            self.level = float(np.sqrt(np.mean(block ** 2)))
            if not self.enabled.is_set():
                if self.ep.active:
                    self.ep.reset()
                continue
            utt = self.ep.push(block, time.time())
            if utt is not None:
                self.utterances.put(utt)

    def pause(self) -> None:
        self.enabled.clear()
        while not self._q.empty():           # drop audio captured meanwhile
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def resume(self) -> None:
        self.ep.reset()
        self.enabled.set()

    @property
    def hearing_speech(self) -> bool:
        return self.ep.active

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
