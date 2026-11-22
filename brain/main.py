"""The brain program: starts everything and runs the main loop.

How to start it (the body must already be running: `mjpython -m body.server`,
or just use ./run.sh, which starts both):
    python -m brain.main --show      # camera + debug window (press q to quit)
    python -m brain.main             # camera, no window
    python -m brain.main --fake      # a scripted pretend person, no camera (for testing)
    python -m brain.main --no-voice  # looks and reacts, but no microphone / Claude
    python -m brain.main --forget    # start with an empty object memory

What this file does, in plain words:
    1. Connects to the body and creates every part: audio, character,
       engagement, memory, camera, face detector, and (unless --no-voice)
       the microphone, speech recognition, Claude, voice, scene watcher and goals.
    2. Runs the main loop 15 times a second:
         camera frame -> find face -> update engagement -> character reacts
       (conversation and Claude run on their own threads, so this loop never waits).
    3. When you quit, prints the measurements (face detection speed, body
       command speed, how many scene looks) and shuts everything down cleanly.

How information flows:
    camera -> attention -> engagement -> character -> body + audio
    microphone -> conversation (STT -> Claude -> TTS) -> character
"""
from __future__ import annotations

import argparse
import json
import os
import logging
import time
from pathlib import Path

from brain.audio import Mixer
from brain.body_client import BodyClient
from brain.character import Character
from brain.engagement import Engagement
from brain.perception.attention import Observation

LOOP_HZ = 15                                              # main loop runs 15 times a second
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
log = logging.getLogger("brain")


def fake_person(t: float) -> Observation:
    """A pretend person for testing without a camera (--fake).

    At time t (seconds since start), returns what the camera "would" see. The
    script tests the tricky cases: a short glance away must NOT end the
    conversation, and someone walking past must NOT wake the lamp.
    """
    timeline = [  # (start, end, attention, azimuth)
        (2.0, 10.0, 1.0, 0.15),     # looks at the lamp
        (6.0, 6.4, 0.0, 0.0),       # blinks / looks at keyboard: must NOT disengage
        (13.0, 13.4, 0.82, -0.3),   # walks past glancing: must NOT engage
        (16.0, 21.0, 1.0, -0.2),    # comes back
    ]
    obs = Observation(t=time.time())
    for start, end, att, az in reversed(timeline):
        if start <= t < end:
            if att > 0:
                obs.face = obs.frontal = True
                obs.attention, obs.azimuth, obs.eyes = att, az, 2
            break
    return obs


def main() -> None:
    # ---- Command-line options (see the top of this file).
    p = argparse.ArgumentParser(description="Lamp brain")
    p.add_argument("--body", default="ws://127.0.0.1:8765")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--show", action="store_true", help="show camera debug window")
    p.add_argument("--fake", action="store_true", help="scripted person instead of camera")
    p.add_argument("--mute", action="store_true", help="no audio output device")
    p.add_argument("--no-voice", action="store_true", help="disable listening/talking")
    p.add_argument("--forget", action="store_true", help="clear scene memory at start")
    p.add_argument("--duration", type=float, default=0, help="stop after N seconds")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    # ---- 1. Create every part.
    LOG_DIR.mkdir(exist_ok=True)
    try:
        body = BodyClient(args.body)
    except OSError:
        # The most common mistake: forgetting to start the body first.
        raise SystemExit(f"Can't reach the lamp body at {args.body}.\n"
                         "Start it first in another terminal:  mjpython -m body.server  "
                         "(Linux: python -m body.server)")
    audio = Mixer(output=not args.mute)
    character = Character(body, audio)
    engagement = Engagement()

    # Object memory, loaded from logs/memory.json if it exists.
    from brain.memory import SceneMemory
    memory = SceneMemory(LOG_DIR / "memory.json")
    if args.forget:
        memory.clear()

    # Camera + face detector (skipped in --fake mode).
    cam = det = None
    if not args.fake:
        from brain.perception.attention import HaarAttention, draw
        from brain.perception.camera import Camera
        cam, det = Camera(args.camera), HaarAttention()
        import cv2

    # Voice, Claude, scene memory and goals (skipped with --no-voice or --fake).
    # These are imported only when needed, so --no-voice works without an API key.
    conversation = listener = watcher = None
    if not args.no_voice and not args.fake:
        from brain.conversation import Conversation
        from brain.mind import Mind
        from brain.speech.listen import Listener
        from brain.speech.stt import SpeechToText
        from brain.speech.tts import TextToSpeech
        log.info("loading speech models (first run downloads them)...")
        mind = Mind()
        stt, tts = SpeechToText(), TextToSpeech()
        listener = Listener()
        conversation = Conversation(character, audio, listener, stt, mind, tts, engagement,
                                    memory=memory, log_dir=LOG_DIR,
                                    get_frame=lambda: cam.latest()[1])
        from brain.perception.scene import SceneWatcher
        watcher = SceneWatcher(mind.client, mind.model, memory,
                               on_new=conversation.on_new_objects)
        from brain.goals import GoalRunner
        # Goals need precise localisation, so they default to a stronger vision model
        # (falls back to the conversation model if it is unavailable).
        conversation.goals = GoalRunner(mind.client,
                                        os.environ.get("LAMP_VISION_MODEL", "claude-sonnet-5-5"),
                                        character, lambda: cam.latest()[1], memory, LOG_DIR,
                                        fallback_model=mind.model)
        log.info("voice ready: stt=whisper %s, tts=%s, mind=%s",
                 getattr(stt, "name", "base.en"), tts.backend, mind.model)

    # ---- 2. The main loop.
    events = (LOG_DIR / "engagement.jsonl").open("a")    # every engagement change is logged here
    t0, last_seq = time.monotonic(), -1
    det_ms: list[float] = []
    log.info("brain running (%s)", "fake person" if args.fake else f"camera {args.camera}")
    try:
        while not args.duration or time.monotonic() - t0 < args.duration:
            tick_start = time.monotonic()
            obs = None                                     # what the camera saw this tick
            if args.fake:
                obs = fake_person(tick_start - t0)
            else:
                # Only process a frame we haven't seen before.
                seq, frame = cam.latest()
                if frame is not None and seq != last_seq:
                    last_seq = seq
                    obs = det.observe(frame)
                    det_ms.append(obs.ms)
                    # Let the scene watcher decide whether it's time to look at objects.
                    if watcher is not None:
                        watcher.offer(frame, obs.box, engagement.state.value == "engaged")
                    # The debug window: status text, where the lamp points
                    # (yellow circle) and any goal target (red box).
                    if args.show:
                        label = engagement.state.value
                        if conversation is not None:
                            label += (" | thinking/speaking" if conversation.busy else
                                      " | hearing you" if listener.hearing_speech else
                                      " | listening" if listener.enabled.is_set() else "")
                            label += f" | remembers {len(memory.items)}"
                        aim = target = None
                        g = body.state.get("gaze")
                        if g:
                            from brain.character import ATTENTIVE_EL
                            from brain.perception.attention import camera_angles_to_image
                            aim = camera_angles_to_image(g["azimuth"], g["elevation"] - ATTENTIVE_EL)
                        if conversation is not None and conversation.goals and conversation.goals.overlay:
                            ov = conversation.goals.overlay
                            target = (ov["name"], ov["target"], ov.get("box"))
                        cv2.imshow("lamp eyes", draw(frame, obs, label, aim, target))
                        if cv2.waitKey(1) & 0xFF == ord("q"):    # q = quit
                            break

            # Update engagement; if it changed, tell the character and conversation.
            if obs is not None:
                old = engagement.state
                change = engagement.update(time.monotonic(), obs.attention)
                if change is not None:
                    character.on_state(change, obs)
                    if conversation is not None:
                        conversation.on_state(change, old)
                    events.write(json.dumps({"t": round(time.time(), 3), "state": change.value,
                                             "score": round(engagement.score, 3)}) + "\n")
                    events.flush()
            character.tick(obs)                          # motion scripts, face-following, talking light

            # Sleep for whatever is left of this 1/15 s tick.
            time.sleep(max(0.0, 1 / LOOP_HZ - (time.monotonic() - tick_start)))
    except KeyboardInterrupt:
        pass                                             # Ctrl+C: fall through to the clean shutdown
    finally:
        # ---- 3. Print the measurements, then shut everything down.
        if det_ms:
            det_ms.sort()
            log.info("face detection: median %.1f ms, p95 %.1f ms over %d frames",
                     det_ms[len(det_ms) // 2], det_ms[int(len(det_ms) * 0.95)], len(det_ms))
        if watcher is not None and watcher.looks:
            log.info("scene looks: %d (last %.0f ms); objects in memory: %d",
                     watcher.looks, watcher.last_ms, len(memory.items))
        log.info("body command round trip: median %.1f ms",
                 sorted(body.rtt_ms)[len(body.rtt_ms) // 2] if body.rtt_ms else -1)
        if listener is not None:
            listener.close()
        audio.close()
        body.close()
        if cam:
            cam.close()


if __name__ == "__main__":
    main()
