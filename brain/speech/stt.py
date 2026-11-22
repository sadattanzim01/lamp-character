"""Turns your recorded speech into text, on the laptop (no cloud).

What this file does, in plain words:
    It runs Whisper, an open-source speech recognition model, through a fast
    library called faster-whisper. You give it the audio of one utterance and
    it gives back the words, typically in about 0.7 s.

Choices and trade-offs:
    * Runs locally on the CPU, so your voice never leaves the computer, and
      it works on the target laptop with no graphics card.
    * Model size "base.en" (English only, ~140 MB, downloaded once):
        tiny.en   faster, but mishears short commands noticeably more often
        base.en   the balance we picked
        small.en  more accurate, but about 3x slower
      You can switch with LAMP_STT_MODEL in .env.
    * "int8" stores the model's numbers as small whole numbers instead of
      decimals ("quantization"): much faster on a CPU, with almost no loss
      in accuracy.
"""
from __future__ import annotations

import os
import time

import numpy as np


class SpeechToText:
    """The speech recogniser. Create it once (loading takes a second), then call transcribe()."""

    def __init__(self, model: str | None = None, threads: int | None = None):
        from faster_whisper import WhisperModel   # imported here so it's only loaded when needed
        name = model or os.environ.get("LAMP_STT_MODEL", "base.en")
        # Use up to 4 CPU cores (the target laptop has exactly 4).
        self.model = WhisperModel(name, device="cpu", compute_type="int8",
                                  cpu_threads=threads or min(4, os.cpu_count() or 4))
        # "Warm-up": transcribe 1 s of silence now, so the first REAL sentence
        # isn't slowed down by one-time setup work.
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Turn audio (16 kHz) into text. Returns (text, seconds it took)."""
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(
            audio,
            language="en",                     # skip auto-detecting the language (saves time)
            beam_size=1,                       # take the single best guess (fastest search)
            vad_filter=False,                  # listen.py already trimmed the silence
            condition_on_previous_text=False,  # treat each utterance on its own
            without_timestamps=True)           # we only need the words, not their timing
        # Whisper may split the speech into several pieces; join them into one sentence.
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, time.perf_counter() - t0
