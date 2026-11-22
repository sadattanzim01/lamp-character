"""Lumi's personality: how it EXPRESSES itself with motion, light, sound and music.

What this file does, in plain words:
    Whenever something happens (you look at it, you leave, it starts talking),
    this file decides how Lumi reacts, using ALL its channels at once: how it
    moves, what colour it glows, what sound it makes, and whether music plays.
    For example, when you start paying attention it doesn't just say hi: the
    music fades, it chirps, glows orange, perks up, wiggles, then settles into
    a steady reading light while facing you.

    Because every reaction is decided in this one place, Lumi feels like one
    creature instead of separate features glued together. To change its
    personality, you'd only edit this file.

Two techniques worth knowing:
    * Motion scripts: a reaction can be a list of moves (look at you -> perk
      -> wiggle). Each move starts only after the body reports the previous
      one is done, so gestures never get cut off halfway.
    * One owner: other threads (speech, goals) never move the body directly.
      They "post" a request, and this file runs it on the main loop. That way
      two parts of the program can never send conflicting moves at once.
"""
from __future__ import annotations

import collections
import logging
import math
import threading
import time
from concurrent.futures import Future
from typing import Callable

import numpy as np

from brain.audio import Mixer
from brain.body_client import BodyClient, BodyError
from brain.engagement import State
from brain.perception.attention import Observation, image_to_camera_angles

log = logging.getLogger("character")

# Light colours as (red, green, blue), each 0-1. Each one carries a feeling.
SLEEPY = (1.0, 0.78, 0.50)     # warm and dim: asleep
CURIOUS = (1.0, 0.85, 0.62)    # a bit brighter: "hm?"
HAPPY = (1.0, 0.62, 0.28)      # orange glow: greeting you
FOCUS = (1.0, 0.93, 0.80)      # near-white while you're engaged: a good reading light
UNSURE = (0.62, 0.74, 1.0)     # cool blue: when you drift away
THINKING = (0.78, 0.66, 1.0)   # lavender: while it waits for speech recognition / Claude
# Each mood Claude can pick -> (colour, brightness).
MOOD_LIGHT = {
    "neutral": (FOCUS, 0.85), "happy": (HAPPY, 1.0), "curious": (CURIOUS, 0.9),
    "thinking": (THINKING, 0.75), "sad": (UNSURE, 0.45), "playful": ((1.0, 0.55, 0.75), 1.0),
}

ATTENTIVE_EL = -0.10           # how much to look down (rad) at a face in the middle of the picture
FOLLOW_DEADBAND = math.radians(4)   # ignore face movements smaller than 4 degrees (no jitter)
FOLLOW_PERIOD = 0.25           # update face-following at most 4 times a second


class Character:
    """Decides and performs Lumi's reactions. Call tick() every frame from the main loop."""

    def __init__(self, body: BodyClient, audio: Mixer):
        self.body, self.audio = body, audio
        self._script: list[Callable[[], Future | None]] = []   # moves still to do
        self._current: Future | None = None                     # the move happening now
        self._last_gaze = (0.0, ATTENTIVE_EL)                   # where it last looked
        self._last_follow = 0.0
        self._breath_t = 0.0                                    # for the sleepy "breathing" light
        self._breath_up = False
        self.state = State.IDLE
        # Requests from other threads, waiting to run on the main loop.
        self._posted: collections.deque[Callable[[], None]] = collections.deque()
        # While talking: (voice loudness over time, start time, colour, brightness).
        self._speech: tuple[np.ndarray, float, tuple, float] | None = None
        self._speech_tick = 0.0
        self.user_light: tuple[tuple, float] | None = None   # set when you ask for a light colour
        self._hold_until = 0.0     # until this time, don't follow your face (it's aiming at a goal)
        self.fall_asleep(initial=True)

    def engaged_light(self) -> tuple[tuple, float]:
        """The light to use while engaged: your chosen colour if you asked for one, else FOCUS."""
        return self.user_light or (FOCUS, 0.9)

    def set_user_light(self, rgb, brightness: float) -> None:
        """You asked for a colour/brightness: remember it and switch to it."""
        rgb = tuple(float(min(1, max(0, c))) for c in rgb)              # keep each value in 0-1
        self.user_light = (rgb, float(min(1.0, max(0.05, brightness))))   # never fully dark
        self.body.light(*self.user_light, fade=0.6)

    @staticmethod
    def gaze_for_image(x: float, y: float) -> tuple[float, float]:
        """Which way to point the lamp to look at a spot in the camera picture.
        Assumes the webcam sits where the lamp is and faces the same way (see the
        technical note's limitations)."""
        az, el = image_to_camera_angles(x, y)
        return az, ATTENTIVE_EL + el

    def glance_at(self, az: float, el: float) -> None:
        """Glance at something new (like an object you hold up): blip, look, tilt.
        Face-following resumes afterwards."""
        self.audio.sfx("notice", 0.5)
        self._run(self._motion(self.body.gaze, az, el, speed=0.8),
                  self._motion(self.body.gesture, "tilt"))

    def on_main(self, fn: Callable, timeout: float = 5.0):
        """For other threads: run fn on the main loop, wait for it, and return its result."""
        done, box = threading.Event(), []
        self.post(lambda: (box.append(fn()), done.set()))
        done.wait(timeout)
        return box[0] if box else None

    def aim(self, az: float, el: float, hold_s: float = 20.0):
        """Point the lamp (and its light) in a direction and KEEP it there for 20 s
        instead of turning back to your face. Used by goals ("shine on the book").
        Returns a placeholder that completes when the move is done."""
        self._hold_until = time.monotonic() + hold_s
        self._script = []                       # cancel any other planned moves
        try:
            _, done = self.body.gaze(az, el, speed=0.7)
            self._current = done
            return done
        except BodyError as exc:
            log.warning("aim rejected: %s", exc)
            return None

    def release_hold(self) -> None:
        """Stop holding the aim; go back to following your face."""
        self._hold_until = 0.0

    def post(self, fn: Callable[[], None]) -> None:
        """For other threads: ask the main loop to run fn on its next tick."""
        self._posted.append(fn)

    # ----------------------------------------------------------- motion scripts
    def _run(self, *steps: Callable[[], Future | None]) -> None:
        """Start a new list of moves, replacing whatever was planned."""
        self._script = list(steps)
        self._current = None

    def _motion(self, fn, *args, **kw) -> Callable[[], Future | None]:
        """Wrap a body command as a script step. The step returns a placeholder
        that completes when the move finishes (or None if the body refused it)."""
        def step():
            try:
                _, done = fn(*args, **kw)
                return done
            except BodyError as exc:
                log.warning("body rejected %s%s: %s", fn.__name__, args, exc)
                return None
        return step

    def _do(self, fn: Callable[[], None]) -> Callable[[], None]:
        """Wrap any instant action (e.g. change the light) as a script step."""
        def step():
            fn()
            return None
        return step

    @property
    def animating(self) -> bool:
        """True while a move or a script is still running."""
        return bool(self._script) or (self._current is not None and not self._current.done())

    # ----------------------------------------------------------- reactions
    def fall_asleep(self, initial: bool = False) -> None:
        """Dim warm light, lullaby, slump into the sleep pose (quicker at startup)."""
        self.body.light(SLEEPY, 0.12, fade=0.5 if initial else 2.5)
        self.audio.music("idle", 0.35, fade=1.0 if initial else 3.0)
        self._run(self._motion(self.body.pose, "sleep", speed=0.3))

    def on_state(self, new: State, obs: Observation) -> None:
        """React to an engagement change (see engagement.py) with motion + light + sound + music."""
        old, self.state = self.state, new
        self._hold_until = 0.0
        log.info("engagement %s -> %s (att %.2f)", old.value, new.value, obs.attention)
        az, el = self._target(obs)              # where your face is

        if new == State.NOTICING:
            # "Hm?": a curious blip, a little brighter, and lift the head part of the
            # way toward you (not all the way: it isn't sure yet).
            self.audio.sfx("notice", 0.6)
            self.body.light(CURIOUS, 0.35, fade=0.5)
            self._run(self._motion(self.body.gaze, az * 0.6, (el + -0.6) / 2, speed=0.35))

        elif new == State.ENGAGED and old == State.LEAVING:
            # You came back before it gave up: a quick "oh, hi again!" wiggle.
            self.audio.sfx("hello", 0.8)
            self.body.light(*self.engaged_light(), fade=0.3)
            self._run(self._motion(self.body.gaze, az, el, speed=0.8),
                      self._motion(self.body.gesture, "wiggle"))

        elif new == State.ENGAGED:
            # The full greeting: the music steps aside, it chirps, glows orange,
            # turns to you, perks up, wiggles, then settles into a steady reading light.
            self.audio.music(None, fade=1.0)
            self.audio.sfx("hello")
            self.body.light(HAPPY, 1.0, fade=0.25)
            self._run(self._motion(self.body.gaze, az, el, speed=1.0),
                      self._motion(self.body.gesture, "perk"),
                      self._motion(self.body.gesture, "wiggle"),
                      self._do(lambda: self.body.light(*self.engaged_light(), fade=1.2)))

        elif new == State.LEAVING:
            # You looked away: unsure. Cool blue light and a quizzical head tilt.
            self.body.light(UNSURE, 0.55, fade=0.8)
            self._run(self._motion(self.body.gesture, "tilt"))

        elif new == State.IDLE and old == State.NOTICING:
            # False alarm (it never fully engaged): quietly settle back down.
            self.fall_asleep()

        elif new == State.IDLE:
            # Goodbye: forget your light choice, sigh, droop, dim, and doze off
            # as the lullaby comes back.
            self.user_light = None
            self.audio.sfx("bye", 0.8)
            self.body.light(SLEEPY, 0.3, fade=1.0)
            self._run(self._motion(self.body.gesture, "droop"),
                      self._do(self.fall_asleep))

    # ----------------------------------------------------------- conversation
    def speak(self, pcm: np.ndarray, sr: int, gesture: str, mood: str) -> threading.Event:
        """Say something, with the gesture and mood colour Claude chose.

        While it talks, the light brightness follows the loudness of its voice
        (updated 10 times a second), so the light visibly "talks" along.
        Returns a flag that's set when the speech finishes.
        """
        rgb, base = MOOD_LIGHT.get(mood, MOOD_LIGHT["neutral"])
        if self.user_light is not None:          # a colour you asked for beats the mood colour
            rgb, base = self.user_light
        # Measure the voice's loudness in 0.1 s slices (the "envelope"), scaled to 0-1.
        hop = max(1, sr // 10)
        n = len(pcm) // hop
        env = np.sqrt(np.mean(pcm[: n * hop].reshape(n, hop) ** 2, axis=1)) if n else np.zeros(1)
        env = env / (env.max() + 1e-6)
        self._speech = (env, time.monotonic(), rgb, base)
        self.body.light(rgb, base, fade=0.15)
        if gesture and gesture != "none":
            self._run(self._motion(self.body.gesture, gesture))
        return self.audio.voice(pcm, sr)

    def express_thinking(self) -> None:
        """While waiting on speech recognition and Claude: bubbly blip, lavender light, head tilt."""
        self.audio.sfx("think", 0.5)
        self.body.light(THINKING, 0.7, fade=0.3)
        self._run(self._motion(self.body.gesture, "tilt"))

    def express_listening(self) -> None:
        """Back to the normal engaged light (e.g. after mishearing a cough)."""
        self.body.light(*self.engaged_light(), fade=0.4)

    def express_error(self) -> None:
        """Something went wrong: low wobble sound, blue light, head shake."""
        self.audio.sfx("error", 0.7)
        self.body.light(UNSURE, 0.6, fade=0.3)
        self._run(self._motion(self.body.gesture, "shake"))

    # ----------------------------------------------------------- every frame
    def tick(self, obs: Observation | None) -> None:
        """Called about 15 times a second by the main loop."""
        now = time.monotonic()
        # 1. Run requests posted by other threads.
        while self._posted:
            self._posted.popleft()()
        # 2. Make the light "talk" while speaking.
        if self._speech is not None:
            self._speech_light(now)
        # 3. Motion script: when the current move is done, start the next one.
        if self._current is None or self._current.done():
            self._current = None
            if self._script:
                self._current = self._script.pop(0)()
                return
        # 4. With nothing scripted: follow your face while engaged (unless holding
        #    an aim for a goal), or slowly "breathe" the light while asleep.
        holding = now < self._hold_until
        if (self.state == State.ENGAGED and obs is not None and obs.face
                and not self.animating and not holding):
            self._follow(obs, now)
        elif self.state == State.IDLE and not self.animating:
            self._breathe(now)

    def _speech_light(self, now: float) -> None:
        """Set the brightness from the voice's loudness right now (55-100% of normal)."""
        env, t0, rgb, base = self._speech
        i = int((now - t0) * 10)                  # which 0.1 s slice we're in
        if i >= len(env):
            # Finished talking: back to the normal engaged light.
            self._speech = None
            self.body.light(*self.engaged_light(), fade=0.6)
            return
        if now - self._speech_tick >= 0.1:
            self._speech_tick = now
            self.body.light(rgb, base * (0.55 + 0.45 * float(env[i])), fade=0.1)

    def _target(self, obs: Observation) -> tuple[float, float]:
        """Which way to look to face you. Up/down movement is softened (x0.8) so
        it doesn't bob around. If no face is visible, keep looking where it was."""
        if not obs.face:
            return self._last_gaze
        return obs.azimuth, ATTENTIVE_EL + 0.8 * obs.elevation

    def _follow(self, obs: Observation, now: float) -> None:
        """Turn to keep facing you: at most 4 updates a second, and only if you
        moved more than 4 degrees, so it follows smoothly instead of jittering."""
        if now - self._last_follow < FOLLOW_PERIOD:
            return
        az, el = self._target(obs)
        if math.hypot(az - self._last_gaze[0], el - self._last_gaze[1]) < FOLLOW_DEADBAND:
            return
        self._last_follow, self._last_gaze = now, (az, el)
        try:
            self.body.gaze(az, el, speed=0.5)
        except BodyError as exc:
            log.debug("gaze rejected: %s", exc)

    def _breathe(self, now: float) -> None:
        """While asleep, slowly pulse the light up and down every 2.2 s, like
        breathing, so the lamp still feels alive."""
        if now - self._breath_t < 2.2:
            return
        self._breath_t, self._breath_up = now, not self._breath_up
        self.body.light(SLEEPY, 0.2 if self._breath_up else 0.08, fade=2.2)
