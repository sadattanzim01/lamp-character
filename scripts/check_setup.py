"""Pre-flight check: run once on a new machine before the demo.

    python scripts/check_setup.py

Checks the API key + model, downloads and times the speech models, records
3 s from the microphone and transcribes it, and speaks a test line.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def step(name: str):
    print(f"\n== {name}")


def main() -> None:
    ok = True

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

    step("Text to speech (first run downloads the Piper voice, ~60 MB)")
    import sounddevice as sd
    from brain.speech.tts import TextToSpeech
    tts = TextToSpeech()
    audio, sr, dt = tts.synthesize("Hello! I'm Lumi. Can you hear me?")
    print(f"   backend {tts.backend}: {len(audio) / sr:.1f}s of speech in {dt * 1000:.0f} ms")
    sd.play(audio, sr)
    sd.wait()

    step("Speech to text (first run downloads Whisper base.en, ~140 MB)")
    t0 = time.perf_counter()
    from brain.speech.stt import SpeechToText
    stt = SpeechToText()
    print(f"   loaded in {time.perf_counter() - t0:.1f}s")

    step("Microphone: say something like 'what a nice lamp' after the beep (3 s)")
    import numpy as np
    from brain.speech.listen import SR
    beep = (0.3 * np.sin(2 * np.pi * 880 * np.arange(int(0.15 * 22050)) / 22050)).astype(np.float32)
    sd.play(beep, 22050)
    sd.wait()
    rec = sd.rec(int(3 * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
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
