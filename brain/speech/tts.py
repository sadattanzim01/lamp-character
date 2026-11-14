"""Local text-to-speech.

Primary: Piper (neural, ONNX on CPU, ~60 MB voice downloaded once). It speaks
a short sentence in well under a second on a laptop CPU and sounds natural.
Fallback: the OS voice (macOS `say`, Linux `espeak-ng`), so the lamp can
always talk even before the voice model is downloaded.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

VOICE = os.environ.get("LAMP_VOICE", "en_US-lessac-medium")
VOICE_DIR = Path(__file__).resolve().parents[2] / "models" / "piper"
_HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/{locale}/{name}/{quality}/{voice}"


def download_voice(voice: str = VOICE) -> Path:
    lang, name, quality = voice.split("-")
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".onnx", ".onnx.json"):
        dst = VOICE_DIR / f"{voice}{suffix}"
        if not dst.exists():
            url = _HF.format(locale=lang, name=name, quality=quality, voice=voice + suffix)
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dst)
    return VOICE_DIR / f"{voice}.onnx"


class TextToSpeech:
    def __init__(self) -> None:
        self.backend = "system"
        self._voice = None
        model = VOICE_DIR / f"{VOICE}.onnx"
        try:
            from piper import PiperVoice
            if not model.exists():
                model = download_voice()
            self._voice = PiperVoice.load(str(model))
            self.backend = "piper"
        except Exception as exc:                       # noqa: BLE001
            print(f"[tts] Piper unavailable ({exc}); using the system voice")
        self.synthesize("Hi.")                         # warm-up

    def synthesize(self, text: str) -> tuple[np.ndarray, int, float]:
        """Returns (float32 mono audio, sample rate, seconds taken)."""
        t0 = time.perf_counter()
        if self.backend == "piper":
            audio, sr = self._piper(text)
        else:
            audio, sr = self._system(text)
        return audio, sr, time.perf_counter() - t0

    def _piper(self, text: str) -> tuple[np.ndarray, int]:
        v = self._voice
        if hasattr(v, "synthesize_stream_raw"):        # piper-tts <= 1.2
            raw = b"".join(v.synthesize_stream_raw(text))
            return np.frombuffer(raw, np.int16).astype(np.float32) / 32768, v.config.sample_rate
        chunks = list(v.synthesize(text))              # piper-tts >= 1.3
        audio = np.concatenate([c.audio_float_array for c in chunks]) if chunks else np.zeros(1)
        return audio.astype(np.float32), chunks[0].sample_rate if chunks else 22050

    def _system(self, text: str) -> tuple[np.ndarray, int]:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "say.wav"
            if platform.system() == "Darwin":
                subprocess.run(["say", "-o", str(out), "--data-format=LEI16@22050", text],
                               check=True)
            elif shutil.which("espeak-ng"):
                subprocess.run(["espeak-ng", "-w", str(out), text], check=True)
            else:
                raise RuntimeError("No TTS available: install piper-tts or espeak-ng")
            with wave.open(str(out)) as w:
                sr = w.getframerate()
                pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        return pcm.astype(np.float32) / 32768, sr


if __name__ == "__main__":       # python -m brain.speech.tts "Hello there"
    import sys
    import sounddevice as sd
    tts = TextToSpeech()
    audio, sr, dt = tts.synthesize(" ".join(sys.argv[1:]) or "Hello! I'm Lumi, your desk lamp.")
    print(f"{tts.backend}: {len(audio) / sr:.2f}s of audio in {dt * 1000:.0f} ms")
    sd.play(audio, sr)
    sd.wait()
