"""Spoken interaction: listen -> transcribe -> think -> speak.

Runs on its own worker thread because each stage blocks (STT, the API call,
TTS). It never touches the body directly; it posts expressions to the
Character, which the main loop executes, so there is exactly one owner of
body commands.

Timeline of one turn (all measured and logged to logs/latency.jsonl):
  user stops talking
    +0.8 s   endpoint detected (silence threshold)
    + STT    local Whisper
    + LLM    Claude reply (words + gesture + mood)
    + TTS    local Piper
    -> first audio out of the speaker
"""
from __future__ import annotations

import json
import logging
import queue
import random
import re
import threading
import time
from pathlib import Path

from brain.engagement import State
from brain.perception.scene import image_block

log = logging.getLogger("conversation")

GREETINGS = ["Oh! Hi there.", "Hey! Hello.", "Oh, hello!"]
WELCOME_BACK = ["Oh, you're back.", "There you are."]
# Whisper hallucinates these on noise / breaths; ignore them.
# Utterances that point at the live scene get a camera frame attached.
DEICTIC = re.compile(r"\b(this|that|these|those|holding|hold|showing|show|see|look|looking|"
                     r"colou?r|in my hand|here|what is it|what's it)\b", re.IGNORECASE)
NOISE = re.compile(r"^\W*(you|thank you\.?|thanks for watching!?|bye\.?|uh+|um+|hmm+|\.+)?\W*$",
                   re.IGNORECASE)


class Conversation:
    def __init__(self, character, audio, listener, stt, mind, tts, engagement, memory=None,
                 log_dir: Path | None = None, get_frame=None):
        self.character, self.audio, self.listener = character, audio, listener
        self.stt, self.mind, self.tts = stt, mind, tts
        self.engagement, self.memory = engagement, memory
        self.get_frame = get_frame
        self.goals = None               # GoalRunner, attached by main
        self._last_comment = 0.0
        self.busy = False
        self._requests: queue.Queue = queue.Queue()
        self._latency = (log_dir / "latency.jsonl").open("a") if log_dir else None
        # Pre-synthesise greetings so the hello is instant.
        self._canned = {t: self.tts.synthesize(t)[:2] for t in GREETINGS + WELCOME_BACK}
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._pump_utterances, daemon=True).start()

    # ------------------------------------------------- engagement hooks
    def on_state(self, new: State, old: State) -> None:
        if new == State.ENGAGED:
            text = random.choice(WELCOME_BACK if old == State.LEAVING else GREETINGS)
            self._requests.put(("canned", text))
        elif new in (State.LEAVING, State.IDLE):
            self.listener.pause()
            if new == State.IDLE:
                self.mind.forget_conversation()   # next visitor starts fresh

    # ------------------------------------------------- worker
    def _pump_utterances(self) -> None:
        while True:
            utt = self.listener.utterances.get()
            if self.engagement.state == State.ENGAGED and not self.busy:
                self._requests.put(("utterance", utt))

    def _worker(self) -> None:
        while True:
            kind, payload = self._requests.get()
            self.busy = self.engagement.busy = True
            self.listener.pause()
            try:
                if kind == "canned":
                    pcm, sr = self._canned[payload]
                    self._speak(pcm, sr, gesture="none", mood="happy", text=payload)
                elif kind == "utterance":
                    self._turn(payload)
                elif kind == "say":                  # used by later steps
                    text, gesture, mood = payload
                    pcm, sr, _ = self.tts.synthesize(text)
                    self._speak(pcm, sr, gesture, mood, text)
            except Exception:                         # noqa: BLE001
                log.exception("conversation turn failed")
                self.character.post(lambda: self.character.express_error())
                time.sleep(1.0)
            finally:
                self.busy = self.engagement.busy = False
                if self.engagement.state == State.ENGAGED:
                    time.sleep(0.25)                  # let the room echo die down
                    self.listener.resume()

    def _turn(self, utt) -> None:
        self.character.post(self.character.express_thinking)
        text, t_stt = self.stt.transcribe(utt.audio)
        log.info("heard (%.1fs audio, stt %.0f ms): %r", utt.seconds, t_stt * 1000, text)
        if not text or NOISE.match(text):
            self.character.post(self.character.express_listening)
            return
        context = self.memory.describe() if self.memory else ""
        image = None
        if self.get_frame is not None and DEICTIC.search(text):
            frame = self.get_frame()
            if frame is not None:
                image = image_block(frame)
        reply = self.mind.reply(text, context=context, image=image)
        log.info("reply (llm %.0f ms%s): %r [%s, %s] light=%s remember=%s", reply.seconds * 1000,
                 ", +image" if image else "", reply.say, reply.gesture, reply.mood,
                 reply.light, [o.get("name") for o in reply.remember])
        if reply.light and isinstance(reply.light.get("rgb"), list) and len(reply.light["rgb"]) == 3:
            rgb, bright = reply.light["rgb"], reply.light.get("brightness", 0.9)
            self.character.post(lambda: self.character.set_user_light(rgb, bright))
        elif reply.light and "brightness" in reply.light:
            rgb = self.character.engaged_light()[0]
            bright = reply.light["brightness"]
            self.character.post(lambda: self.character.set_user_light(rgb, bright))
        if reply.remember and self.memory is not None:
            self.memory.update([{**o, "shown": bool(o.get("shown", False))} for o in reply.remember])
        if reply.notes:
            log.info("notes: %s", reply.notes)
        pcm, sr, t_tts = self.tts.synthesize(reply.say)
        t_audio = self._speak(pcm, sr, "none" if reply.goal else reply.gesture,
                              reply.mood, reply.say)
        if not reply.goal:
            # A normal conversational turn ends any goal hold: the lamp turns back to you.
            self.character.post(self.character.release_hold)
        if reply.goal and self.goals is not None:
            log.info("goal: %r", reply.goal)
            self.goals.run(reply.goal, say=self._say_now)
        if self._latency:
            rec = {"t": round(time.time(), 3), "speech_s": round(utt.seconds, 2),
                   "endpoint_s": round(utt.t_end - (utt.t_start + utt.seconds), 2),
                   "stt_s": round(t_stt, 3), "llm_s": round(reply.seconds, 3),
                   "tts_s": round(t_tts, 3),
                   "endpoint_to_audio_s": round(t_audio - utt.t_end, 3),
                   "image": image is not None,
                   "tokens_in": reply.input_tokens, "tokens_out": reply.output_tokens,
                   "heard": text, "said": reply.say}
            self._latency.write(json.dumps(rec) + "\n")
            self._latency.flush()
            log.info("latency: endpoint->audio %.2f s (stt %.2f, llm %.2f, tts %.2f)",
                     rec["endpoint_to_audio_s"], t_stt, reply.seconds, t_tts)

    def _speak(self, pcm, sr, gesture: str, mood: str, text: str) -> float:
        """Play speech with matching body language; blocks until finished."""
        started = threading.Event()
        done_box: list = []

        def go():
            done_box.append(self.character.speak(pcm, sr, gesture, mood))
            started.set()
        self.character.post(go)
        started.wait(2.0)
        t_audio = time.time()
        if done_box:
            done_box[0].wait(timeout=len(pcm) / sr + 5)
        return t_audio

    def _say_now(self, text: str, mood: str = "neutral") -> None:
        """Speak immediately from the worker thread (used during goals)."""
        pcm, sr, _ = self.tts.synthesize(text)
        self._speak(pcm, sr, "none", mood, text)

    def on_new_objects(self, new) -> None:
        """Scene watcher noticed something new: glance at it and remark (throttled)."""
        if (self.engagement.state != State.ENGAGED or self.busy
                or time.monotonic() - self._last_comment < 8.0):
            return
        self._last_comment = time.monotonic()
        obj = new[0]
        az, el = self.character.gaze_for_image(obj.x, obj.y)
        self.character.post(lambda: self.character.glance_at(az, el))
        article = "an" if obj.name[:1].lower() in "aeiou" else "a"
        self.say(f"Ooh, {article} {obj.name}.", gesture="none", mood="curious")

    def say(self, text: str, gesture: str = "none", mood: str = "neutral") -> None:
        """Queue a line for the lamp to say (used by memory / goal steps)."""
        self._requests.put(("say", (text, gesture, mood)))
