"""Local speech-to-text with faster-whisper (CTranslate2, int8 on CPU).

Model: base.en (~140 MB, downloaded once from HuggingFace into the user's
cache). On a 4-core laptop CPU it transcribes a 3 s utterance in a fraction of
a second, and no audio leaves the machine. tiny.en is faster but noticeably
worse on short commands; small.en is more accurate but ~3x slower.
"""
from __future__ import annotations

import os
import time

import numpy as np


class SpeechToText:
    def __init__(self, model: str | None = None, threads: int | None = None):
        from faster_whisper import WhisperModel
        name = model or os.environ.get("LAMP_STT_MODEL", "base.en")
        self.model = WhisperModel(name, device="cpu", compute_type="int8",
                                  cpu_threads=threads or min(4, os.cpu_count() or 4))
        self.transcribe(np.zeros(16000, dtype=np.float32))   # warm-up

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Returns (text, seconds taken)."""
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=1, vad_filter=False,
            condition_on_previous_text=False, without_timestamps=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, time.perf_counter() - t0
