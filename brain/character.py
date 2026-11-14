"""The character: one place that decides how the lamp *expresses* itself.

Each engagement transition maps to a coordinated reaction across all
channels (motion, light, sound effect, music), so the lamp behaves as one
creature rather than separate demos. Changing the personality means editing
this file only.

Motion sequences are scripts of steps; a step runs only after the previous
motion reports "done", so gestures finish before gaze tracking resumes.
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

# Light palette (rgb, 0..1)
SLEEPY = (1.0, 0.78, 0.50)     # warm, low
CURIOUS = (1.0, 0.85, 0.62)
HAPPY = (1.0, 0.62, 0.28)      # orange glow on greeting
FOCUS = (1.0, 0.93, 0.80)      # near-white while engaged: a good reading light
UNSURE = (0.62, 0.74, 1.0)     # cool when the person drifts away
THINKING = (0.78, 0.66, 1.0)   # lavender while waiting on speech / the mind
MOOD_LIGHT = {
    "neutral": (FOCUS, 0.85), "happy": (HAPPY, 1.0), "curious": (CURIOUS, 0.9),
    "thinking": (THINKING, 0.75), "sad": (UNSURE, 0.45), "playful": ((1.0, 0.55, 0.75), 1.0),
}

ATTENTIVE_EL = -0.10           # rad, gaze elevation for a face at image centre
FOLLOW_DEADBAND = math.radians(4)
FOLLOW_PERIOD = 0.25           # s between gaze updates while tracking


class Character:
    def __init__(self, body: BodyClient, audio: Mixer):
        self.body, self.audio = body, audio
        self._script: list[Callable[[], Future | None]] = []
        self._current: Future | None = None
        self._last_gaze = (0.0, ATTENTIVE_EL)
        self._last_follow = 0.0
        self._breath_t = 0.0
        self._breath_up = False
        self.state = State.IDLE
        self._posted: collections.deque[Callable[[], None]] = collections.deque()
        self._speech: tuple[np.ndarray, float, tuple, float] | None = None  # env, t0, rgb, base
        self._speech_tick = 0.0
        self.user_light: tuple[tuple, float] | None = None   # set when the person asks
        self._hold_until = 0.0     # while > now, don't track the face (lamp is aimed at a goal)
        self.fall_asleep(initial=True)

    def engaged_light(self) -> tuple[tuple, float]:
        """Light while engaged: the person's choice if they made one, else FOCUS."""
        return self.user_light or (FOCUS, 0.9)

    def set_user_light(self, rgb, brightness: float) -> None:
        rgb = tuple(float(min(1, max(0, c))) for c in rgb)
        self.user_light = (rgb, float(min(1.0, max(0.05, brightness))))
        self.body.light(*self.user_light, fade=0.6)

    @staticmethod
    def gaze_for_image(x: float, y: float) -> tuple[float, float]:
        """Lamp gaze toward a point in the camera image. Assumes the camera sits at the
        lamp and faces the same way (see README: physical notes)."""
        az, el = image_to_camera_angles(x, y)
        return az, ATTENTIVE_EL + el

    def glance_at(self, az: float, el: float) -> None:
        """Look at something (e.g. a newly noticed object); tracking resumes after."""
        self.audio.sfx("notice", 0.5)
        self._run(self._motion(self.body.gaze, az, el, speed=0.8),
                  self._motion(self.body.gesture, "tilt"))

    def on_main(self, fn: Callable, timeout: float = 5.0):
        """Run fn on the main loop from another thread and return its result."""
        done, box = threading.Event(), []
        self.post(lambda: (box.append(fn()), done.set()))
        done.wait(timeout)
        return box[0] if box else None

    def aim(self, az: float, el: float, hold_s: float = 20.0):
        """Point the shade (and light) at a direction and hold it there instead of
        returning to face tracking. Returns the motion's done-future."""
        self._hold_until = time.monotonic() + hold_s
        self._script = []
        try:
            _, done = self.body.gaze(az, el, speed=0.7)
            self._current = done
            return done
        except BodyError as exc:
            log.warning("aim rejected: %s", exc)
            return None

    def release_hold(self) -> None:
        self._hold_until = 0.0

    def post(self, fn: Callable[[], None]) -> None:
        """Thread-safe: run fn on the main loop at the next tick."""
        self._posted.append(fn)

    # ----------------------------------------------------------- scripting
    def _run(self, *steps: Callable[[], Future | None]) -> None:
        """Replace whatever is queued with a new motion script."""
        self._script = list(steps)
        self._current = None

    def _motion(self, fn, *args, **kw) -> Callable[[], Future | None]:
        def step():
            try:
                _, done = fn(*args, **kw)
                return done
            except BodyError as exc:
                log.warning("body rejected %s%s: %s", fn.__name__, args, exc)
                return None
        return step

    def _do(self, fn: Callable[[], None]) -> Callable[[], None]:
        def step():
            fn()
            return None
        return step

    @property
    def animating(self) -> bool:
        return bool(self._script) or (self._current is not None and not self._current.done())

    # ----------------------------------------------------------- reactions
    def fall_asleep(self, initial: bool = False) -> None:
        self.body.light(SLEEPY, 0.12, fade=0.5 if initial else 2.5)
        self.audio.music("idle", 0.35, fade=1.0 if initial else 3.0)
        self._run(self._motion(self.body.pose, "sleep", speed=0.3))

    def on_state(self, new: State, obs: Observation) -> None:
        old, self.state = self.state, new
        self._hold_until = 0.0
        log.info("engagement %s -> %s (att %.2f)", old.value, new.value, obs.attention)
        az, el = self._target(obs)

        if new == State.NOTICING:
            # "Hm?" Lift the head part-way towards the person and brighten a little.
            self.audio.sfx("notice", 0.6)
            self.body.light(CURIOUS, 0.35, fade=0.5)
            self._run(self._motion(self.body.gaze, az * 0.6, (el + -0.6) / 2, speed=0.35))

        elif new == State.ENGAGED and old == State.LEAVING:
            # Welcome back: quick wiggle, straight back to tracking.
            self.audio.sfx("hello", 0.8)
            self.body.light(*self.engaged_light(), fade=0.3)
            self._run(self._motion(self.body.gaze, az, el, speed=0.8),
                      self._motion(self.body.gesture, "wiggle"))

        elif new == State.ENGAGED:
            # Greeting: music steps aside, chirp, orange glow, perk + wiggle,
            # then settle into a steady focus light and face the person.
            self.audio.music(None, fade=1.0)
            self.audio.sfx("hello")
            self.body.light(HAPPY, 1.0, fade=0.25)
            self._run(self._motion(self.body.gaze, az, el, speed=1.0),
                      self._motion(self.body.gesture, "perk"),
                      self._motion(self.body.gesture, "wiggle"),
                      self._do(lambda: self.body.light(*self.engaged_light(), fade=1.2)))

        elif new == State.LEAVING:
            # Unsure: cool light, quizzical head tilt towards where they were.
            self.body.light(UNSURE, 0.55, fade=0.8)
            self._run(self._motion(self.body.gesture, "tilt"))

        elif new == State.IDLE and old == State.NOTICING:
            # False alarm: quietly settle back down.
            self.fall_asleep()

        elif new == State.IDLE:
            # Goodbye: sigh, droop, dim, music returns as it dozes off.
            self.user_light = None
            self.audio.sfx("bye", 0.8)
            self.body.light(SLEEPY, 0.3, fade=1.0)
            self._run(self._motion(self.body.gesture, "droop"),
                      self._do(self.fall_asleep))

    # ----------------------------------------------------------- conversation
    def speak(self, pcm: np.ndarray, sr: int, gesture: str, mood: str) -> threading.Event:
        """Start speech audio with a matching gesture and mood light.

        While the voice plays, the light intensity follows the speech envelope
        (10 Hz), so the light visibly 'talks'."""
        rgb, base = MOOD_LIGHT.get(mood, MOOD_LIGHT["neutral"])
        if self.user_light is not None:          # an explicit request beats mood tint
            rgb, base = self.user_light
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
        self.audio.sfx("think", 0.5)
        self.body.light(THINKING, 0.7, fade=0.3)
        self._run(self._motion(self.body.gesture, "tilt"))

    def express_listening(self) -> None:
        self.body.light(*self.engaged_light(), fade=0.4)

    def express_error(self) -> None:
        self.audio.sfx("error", 0.7)
        self.body.light(UNSURE, 0.6, fade=0.3)
        self._run(self._motion(self.body.gesture, "shake"))

    # ----------------------------------------------------------- per tick
    def tick(self, obs: Observation | None) -> None:
        now = time.monotonic()
        while self._posted:
            self._posted.popleft()()
        if self._speech is not None:
            self._speech_light(now)
        # Advance the motion script.
        if self._current is None or self._current.done():
            self._current = None
            if self._script:
                self._current = self._script.pop(0)()
                return

        holding = now < self._hold_until
        if (self.state == State.ENGAGED and obs is not None and obs.face
                and not self.animating and not holding):
            self._follow(obs, now)
        elif self.state == State.IDLE and not self.animating:
            self._breathe(now)

    def _speech_light(self, now: float) -> None:
        env, t0, rgb, base = self._speech
        i = int((now - t0) * 10)
        if i >= len(env):
            self._speech = None
            self.body.light(*self.engaged_light(), fade=0.6)
            return
        if now - self._speech_tick >= 0.1:
            self._speech_tick = now
            self.body.light(rgb, base * (0.55 + 0.45 * float(env[i])), fade=0.1)

    def _target(self, obs: Observation) -> tuple[float, float]:
        if not obs.face:
            return self._last_gaze
        return obs.azimuth, ATTENTIVE_EL + 0.8 * obs.elevation

    def _follow(self, obs: Observation, now: float) -> None:
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
        """Slow light 'breathing' while asleep: the lamp still feels alive."""
        if now - self._breath_t < 2.2:
            return
        self._breath_t, self._breath_up = now, not self._breath_up
        self.body.light(SLEEPY, 0.2 if self._breath_up else 0.08, fade=2.2)
