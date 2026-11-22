"""Gives Lumi its voice: turns text into speech, on the laptop (no cloud).

What this file does, in plain words:
    You give it a sentence; it gives back the audio of Lumi saying it, which
    the mixer (audio.py) then plays.

Two ways to speak:
    1. Piper (the main one): a small neural voice model that runs on the CPU.
       It sounds natural and makes a short sentence in about 0.1-0.3 s. The
       voice file (~60 MB) is downloaded once into models/piper/.
    2. The computer's built-in voice (the backup): `say` on macOS, `espeak-ng`
       on Linux. It sounds more robotic, but means Lumi can always talk, even
       if Piper isn't installed or the voice hasn't downloaded yet.

Try it: `python -m brain.speech.tts "Hello there"`
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

# Which Piper voice to use (change with LAMP_VOICE in .env).
# Names look like "language_REGION-speaker-quality".
VOICE = os.environ.get("LAMP_VOICE", "en_US-lessac-medium")
VOICE_DIR = Path(__file__).resolve().parents[2] / "models" / "piper"   # lamp-character/models/piper
# Where Piper's official voices are hosted.
_HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/{locale}/{name}/{quality}/{voice}"


def download_voice(voice: str = VOICE) -> Path:
    """Download a Piper voice (the model file + its settings file) if not already present."""
    lang, name, quality = voice.split("-")      # "en_US-lessac-medium" -> en_US, lessac, medium
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".onnx", ".onnx.json"):
        dst = VOICE_DIR / f"{voice}{suffix}"
        if not dst.exists():
            url = _HF.format(locale=lang, name=name, quality=quality, voice=voice + suffix)
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dst)
    return VOICE_DIR / f"{voice}.onnx"


class TextToSpeech:
    """Lumi's voice. Create it once, then call synthesize(text)."""

    def __init__(self) -> None:
        self.backend = "system"                  # assume the backup until Piper loads
        self._voice = None
        model = VOICE_DIR / f"{VOICE}.onnx"
        try:
            from piper import PiperVoice
            if not model.exists():
                model = download_voice()
            self._voice = PiperVoice.load(str(model))
            self.backend = "piper"
        except Exception as exc:                       # noqa: BLE001
            # Anything goes wrong (not installed, no internet...): use the backup voice.
            print(f"[tts] Piper unavailable ({exc}); using the system voice")
        # Warm-up: say one word now, so the first real sentence is quick.
        self.synthesize("Hi.")

    def synthesize(self, text: str) -> tuple[np.ndarray, int, float]:
        """Make the audio for `text`.
        Returns (audio as numbers between -1 and 1, sample rate, seconds it took)."""
        t0 = time.perf_counter()
        if self.backend == "piper":
            audio, sr = self._piper(text)
        else:
            audio, sr = self._system(text)
        return audio, sr, time.perf_counter() - t0

    def _piper(self, text: str) -> tuple[np.ndarray, int]:
        """Speak with Piper. Handles both the old and new versions of the Piper library,
        because its way of returning audio changed between versions."""
        v = self._voice
        if hasattr(v, "synthesize_stream_raw"):        # older Piper (1.2 and below): raw 16-bit bytes
            raw = b"".join(v.synthesize_stream_raw(text))
            return np.frombuffer(raw, np.int16).astype(np.float32) / 32768, v.config.sample_rate
        # Newer Piper (1.3+): gives the speech in chunks (roughly one per sentence); join them.
        chunks = list(v.synthesize(text))
        audio = np.concatenate([c.audio_float_array for c in chunks]) if chunks else np.zeros(1)
        return audio.astype(np.float32), chunks[0].sample_rate if chunks else 22050

    def _system(self, text: str) -> tuple[np.ndarray, int]:
        """Speak with the computer's built-in voice: record it to a temporary .wav file, then load it."""
        with tempfile.TemporaryDirectory() as d:        # the temp folder is deleted afterwards
            out = Path(d) / "say.wav"
            if platform.system() == "Darwin":           # macOS
                subprocess.run(["say", "-o", str(out), "--data-format=LEI16@22050", text],
                               check=True)
            elif shutil.which("espeak-ng"):             # Linux, if espeak-ng is installed
                subprocess.run(["espeak-ng", "-w", str(out), text], check=True)
            else:
                raise RuntimeError("No TTS available: install piper-tts or espeak-ng")
            with wave.open(str(out)) as w:
                sr = w.getframerate()
                pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        # 16-bit whole numbers (-32768..32767) -> decimals between -1 and 1.
        return pcm.astype(np.float32) / 32768, sr


# Run `python -m brain.speech.tts "your sentence"` to hear Lumi say it.
if __name__ == "__main__":
    import sys
    import sounddevice as sd
    tts = TextToSpeech()
    audio, sr, dt = tts.synthesize(" ".join(sys.argv[1:]) or "Hello! I'm Lumi, your desk lamp.")
    print(f"{tts.backend}: {len(audio) / sr:.2f}s of audio in {dt * 1000:.0f} ms")
    sd.play(audio, sr)
    sd.wait()
