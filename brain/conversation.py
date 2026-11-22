"""The talking loop: listen -> turn speech into text -> think -> speak.

What this file does, in plain words:
    This is where a conversation turn happens, start to finish:
      1. the microphone catches what you said (listen.py)
      2. Whisper turns it into text (stt.py)
      3. Claude decides the reply, gesture, mood, light, notes, goal (mind.py)
      4. Piper turns the reply into speech (tts.py)
      5. Lumi says it, with matching body language (character.py)
    It also says hello when you arrive, comments on new objects you show it,
    and hands physical requests ("shine on the book") to goals.py.

Why it runs on its own thread:
    Steps 2-4 each take a moment (about 0.3 to 1.5 s). If they ran on the main
    loop, the camera and face-following would freeze while Lumi thinks. So
    they run on a separate "worker" thread. That thread never moves the lamp
    itself: it posts requests to the Character, which runs them on the main
    loop, so only one part of the program ever controls the body.

How long one turn takes (every turn is measured and saved to logs/latency.jsonl):
    you stop talking
      + 0.8 s  waiting for silence to be sure you've finished
      + ~0.7 s speech-to-text (Whisper, on the laptop)
      + ~1.2 s Claude's reply
      + ~0.3 s text-to-speech (Piper, on the laptop)
      -> Lumi starts speaking (about 3.3 s in total, median)
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

# Short ready-made lines, so the hello is instant (no waiting on Claude).
GREETINGS = ["Oh! Hi there.", "Hey! Hello.", "Oh, hello!"]
WELCOME_BACK = ["Oh, you're back.", "There you are."]
# Words that mean you're pointing at something in the room ("what's THIS?",
# "what am I HOLDING?"). When one appears, a camera photo is sent with that question.
DEICTIC = re.compile(r"\b(this|that|these|those|holding|hold|showing|show|see|look|looking|"
                     r"colou?r|in my hand|here|what is it|what's it)\b", re.IGNORECASE)
# Whisper sometimes "hears" these from background noise or breathing; treat them
# as nothing being said.
NOISE = re.compile(r"^\W*(you|thank you\.?|thanks for watching!?|bye\.?|uh+|um+|hmm+|\.+)?\W*$",
                   re.IGNORECASE)


class Conversation:
    """Runs spoken conversation on a background thread."""

    def __init__(self, character, audio, listener, stt, mind, tts, engagement, memory=None,
                 log_dir: Path | None = None, get_frame=None):
        self.character, self.audio, self.listener = character, audio, listener
        self.stt, self.mind, self.tts = stt, mind, tts
        self.engagement, self.memory = engagement, memory
        self.get_frame = get_frame            # a function that returns the newest camera frame
        self.goals = None                     # the GoalRunner (goals.py), connected later by main.py
        self._last_comment = 0.0
        self.busy = False                     # True while Lumi is thinking or talking
        # Jobs for the worker thread: ("canned", text), ("utterance", audio) or ("say", ...).
        self._requests: queue.Queue = queue.Queue()
        self._latency = (log_dir / "latency.jsonl").open("a") if log_dir else None
        # Make the greeting audio once now, so saying hello later is instant.
        self._canned = {t: self.tts.synthesize(t)[:2] for t in GREETINGS + WELCOME_BACK}
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._pump_utterances, daemon=True).start()

    # ------------------------------------------------- reacting to engagement
    def on_state(self, new: State, old: State) -> None:
        """Called when engagement changes (see engagement.py)."""
        if new == State.ENGAGED:
            # Say hello (or "you're back" if you only stepped away briefly).
            text = random.choice(WELCOME_BACK if old == State.LEAVING else GREETINGS)
            self._requests.put(("canned", text))
        elif new in (State.LEAVING, State.IDLE):
            self.listener.pause()                 # stop listening once you look away
            if new == State.IDLE:
                # Clear the word-for-word chat (notes about you are kept; see mind.py).
                self.mind.forget_conversation()

    # ------------------------------------------------- background threads
    def _pump_utterances(self) -> None:
        """Pass finished speech from the microphone to the worker, but only if
        you're engaged and Lumi isn't already busy."""
        while True:
            utt = self.listener.utterances.get()
            if self.engagement.state == State.ENGAGED and not self.busy:
                self._requests.put(("utterance", utt))

    def _worker(self) -> None:
        """The worker thread: handle one job at a time."""
        while True:
            kind, payload = self._requests.get()
            # Mark busy (so engagement won't end mid-conversation) and stop
            # listening, so Lumi never hears its own voice.
            self.busy = self.engagement.busy = True
            self.listener.pause()
            try:
                if kind == "canned":                 # a ready-made greeting
                    pcm, sr = self._canned[payload]
                    self._speak(pcm, sr, gesture="none", mood="happy", text=payload)
                elif kind == "utterance":            # you said something: a full turn
                    self._turn(payload)
                elif kind == "say":                  # a line from elsewhere (e.g. "Ooh, a book.")
                    text, gesture, mood = payload
                    pcm, sr, _ = self.tts.synthesize(text)
                    self._speak(pcm, sr, gesture, mood, text)
            except Exception:                         # noqa: BLE001
                # Anything fails (e.g. no internet): show a "can't do that" reaction
                # instead of crashing, then carry on.
                log.exception("conversation turn failed")
                self.character.post(lambda: self.character.express_error())
                time.sleep(1.0)
            finally:
                self.busy = self.engagement.busy = False
                if self.engagement.state == State.ENGAGED:
                    time.sleep(0.25)                  # let the room's echo of Lumi's voice fade
                    self.listener.resume()

    def _turn(self, utt) -> None:
        """One full turn: you said `utt` -> Lumi replies (and maybe acts)."""
        # Show it's thinking (blip, lavender light, head tilt) while it works.
        self.character.post(self.character.express_thinking)
        # 1. Speech -> text.
        text, t_stt = self.stt.transcribe(utt.audio)
        log.info("heard (%.1fs audio, stt %.0f ms): %r", utt.seconds, t_stt * 1000, text)
        if not text or NOISE.match(text):
            self.character.post(self.character.express_listening)
            return                                   # nothing real was said
        # 2. Gather context: what Lumi remembers, plus a photo if you pointed at something.
        context = self.memory.describe() if self.memory else ""
        image = None
        if self.get_frame is not None and DEICTIC.search(text):
            frame = self.get_frame()
            if frame is not None:
                image = image_block(frame)
        # 3. Ask Claude.
        reply = self.mind.reply(text, context=context, image=image)
        log.info("reply (llm %.0f ms%s): %r [%s, %s] light=%s remember=%s", reply.seconds * 1000,
                 ", +image" if image else "", reply.say, reply.gesture, reply.mood,
                 reply.light, [o.get("name") for o in reply.remember])
        # 4. Apply what Claude asked for, besides the words:
        #    a light change (a full colour, or just a brightness change),
        if reply.light and isinstance(reply.light.get("rgb"), list) and len(reply.light["rgb"]) == 3:
            rgb, bright = reply.light["rgb"], reply.light.get("brightness", 0.9)
            self.character.post(lambda: self.character.set_user_light(rgb, bright))
        elif reply.light and "brightness" in reply.light:
            rgb = self.character.engaged_light()[0]            # keep the current colour
            bright = reply.light["brightness"]
            self.character.post(lambda: self.character.set_user_light(rgb, bright))
        #    objects to remember from the photo,
        if reply.remember and self.memory is not None:
            self.memory.update([{**o, "shown": bool(o.get("shown", False))} for o in reply.remember])
        #    and facts about you (mind.py already saved them; this just logs them).
        if reply.notes:
            log.info("notes: %s", reply.notes)
        # 5. Text -> speech, and say it. (No gesture if a goal is coming: the
        #    goal will move the lamp instead.)
        pcm, sr, t_tts = self.tts.synthesize(reply.say)
        t_audio = self._speak(pcm, sr, "none" if reply.goal else reply.gesture,
                              reply.mood, reply.say)
        if not reply.goal:
            # A normal chat turn ends any goal "hold": the lamp turns back to you.
            self.character.post(self.character.release_hold)
        # 6. A physical request? Hand it to the goal runner (plan, aim, check).
        if reply.goal and self.goals is not None:
            log.info("goal: %r", reply.goal)
            self.goals.run(reply.goal, say=self._say_now)
        # 7. Log how long each part took (used for the technical note's measurements).
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
        """Say something with matching body language, and wait until it's finished.

        Asks the Character (on the main loop) to start speaking, waits for the
        speech to end, and returns the time the audio started (for the latency log).
        """
        started = threading.Event()
        done_box: list = []

        def go():
            done_box.append(self.character.speak(pcm, sr, gesture, mood))
            started.set()
        self.character.post(go)
        started.wait(2.0)
        t_audio = time.time()
        if done_box:
            done_box[0].wait(timeout=len(pcm) / sr + 5)   # wait for the speech to finish
        return t_audio

    def _say_now(self, text: str, mood: str = "neutral") -> None:
        """Speak right away from this thread (used by goals: "There, the book is lit up.")."""
        pcm, sr, _ = self.tts.synthesize(text)
        self._speak(pcm, sr, "none", mood, text)

    def on_new_objects(self, new) -> None:
        """The scene watcher spotted a new object you're showing: glance at it and say
        "Ooh, a ...". At most once every 8 s, and never while busy, so it doesn't chatter."""
        if (self.engagement.state != State.ENGAGED or self.busy
                or time.monotonic() - self._last_comment < 8.0):
            return
        self._last_comment = time.monotonic()
        obj = new[0]
        az, el = self.character.gaze_for_image(obj.x, obj.y)
        self.character.post(lambda: self.character.glance_at(az, el))
        article = "an" if obj.name[:1].lower() in "aeiou" else "a"   # "an apple", "a book"
        self.say(f"Ooh, {article} {obj.name}.", gesture="none", mood="curious")

    def say(self, text: str, gesture: str = "none", mood: str = "neutral") -> None:
        """Queue a line for Lumi to say when it's free."""
        self._requests.put(("say", (text, gesture, mood)))
