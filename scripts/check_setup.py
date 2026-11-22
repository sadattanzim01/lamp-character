"""Setup check: run this once on a new computer before using Lumi.

    python scripts/check_setup.py

What this file does, in plain words:
    It tests each outside piece Lumi depends on, one at a time, and tells you
    clearly what's wrong if something fails:
      1. Claude API: is your API key working? (Lumi introduces itself in text.)
      2. Voice: downloads the Piper voice (~60 MB, first time only) and plays a test line.
      3. Speech recognition: downloads Whisper (~140 MB, first time only).
      4. Microphone: beeps, records 3 seconds, and shows what it heard.
    It ends with "ALL GOOD" or points to what failed. The install script for
    Ubuntu runs this automatically.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Let this script import the brain/ code (it lives one folder up).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def step(name: str):
    """Print a section heading."""
    print(f"\n== {name}")


def main() -> None:
    ok = True                    # becomes False if any check fails

    # ---- 1. Claude API: a tiny real request to prove the key and model work.
    step("Claude API")
    try:
        from brain.mind import Mind
        mind = Mind()
        r = mind.reply("Hi Lumi! Introduce yourself in one sentence.")
        print(f"   model {mind.model}: {r.say!r} [{r.gesture}, {r.mood}] in {r.seconds:.2f}s")
    except Exception as exc:                        # noqa: BLE001
        ok = False
        print(f"   FAILED: {exc}")
        print("   Check lamp-character/.env has ANTHROPIC_API_KEY=sk-ant-... and credit in the Console.")

    # ---- 2. Voice: make a sentence and play it out loud.
    step("Text to speech (first run downloads the Piper voice, ~60 MB)")
    import sounddevice as sd
    from brain.speech.tts import TextToSpeech
    tts = TextToSpeech()
    audio, sr, dt = tts.synthesize("Hello! I'm Lumi. Can you hear me?")
    print(f"   backend {tts.backend}: {len(audio) / sr:.1f}s of speech in {dt * 1000:.0f} ms")
    sd.play(audio, sr)
    sd.wait()

    # ---- 3. Speech recognition: loading it also downloads it the first time.
    step("Speech to text (first run downloads Whisper base.en, ~140 MB)")
    t0 = time.perf_counter()
    from brain.speech.stt import SpeechToText
    stt = SpeechToText()
    print(f"   loaded in {time.perf_counter() - t0:.1f}s")

    # ---- 4. Microphone: beep, record 3 s, check it's loud enough, and transcribe it.
    step("Microphone: say something like 'what a nice lamp' after the beep (3 s)")
    import numpy as np
    from brain.speech.listen import SR
    beep = (0.3 * np.sin(2 * np.pi * 880 * np.arange(int(0.15 * 22050)) / 22050)).astype(np.float32)
    sd.play(beep, 22050)
    sd.wait()
    rec = sd.rec(int(3 * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    # Average loudness. Near-silent usually means the terminal isn't allowed to use the mic.
    level = float(np.sqrt(np.mean(rec ** 2)))
    text, dt = stt.transcribe(rec[:, 0])
    print(f"   mic level {level:.4f} ({'OK' if level > 0.003 else 'very quiet: check mic permission'})")
    print(f"   heard: {text!r} in {dt * 1000:.0f} ms")
    if level <= 0.003:
        ok = False
        print("   macOS: System Settings > Privacy & Security > Microphone > enable your terminal app")

    print("\nALL GOOD" if ok else "\nSome checks failed (see above)")


if __name__ == "__main__":
    main()
