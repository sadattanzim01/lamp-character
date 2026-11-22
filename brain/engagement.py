"""Decides when Lumi should notice you, engage with you, and let you go.

What this file does, in plain words:
    Every camera frame, the face detector gives an "attention" number: about
    1.0 if someone is facing the camera, 0 if not. Those numbers are noisy:
    one frame can miss your face, or you blink, or glance at the keyboard.
    If the lamp reacted to every single frame it would flicker on and off.

    So this file is a "state machine": the lamp is always in exactly one of
    four moods, and it only switches mood when the evidence is clear enough
    for long enough.

    IDLE --(attention rising)--> NOTICING --(stays up)--> ENGAGED
      ^                             |                         |
      |<------(drops away)----------+                (attention lost)
      |                                                       v
      +<-----------------(stays away)------------------- LEAVING

    IDLE      asleep, nobody paying attention
    NOTICING  "hm?", someone might be looking; not committed yet
    ENGAGED   talking with someone
    LEAVING   they looked away; the lamp hesitates before giving up

The key idea, "hysteresis":
    It's harder to START engaging than to STAY engaged. Becoming engaged needs
    a strong signal for 0.6 s; ending it needs a weak signal for 1.2 s. So a
    blink or a quick glance away never ends a conversation, and someone
    walking past doesn't wake the lamp. All the thresholds live in
    EngagementConfig so they're easy to tune.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    """The four engagement states (see the diagram above)."""
    IDLE = "idle"
    NOTICING = "noticing"
    ENGAGED = "engaged"
    LEAVING = "leaving"


@dataclass
class EngagementConfig:
    """All the tuning numbers in one place. Scores go from 0 (no one) to 1 (clearly looking)."""
    ema_tau: float = 0.35        # smoothing time in seconds: higher = calmer but slower to react
    notice_at: float = 0.45      # smoothed score that makes the lamp go "hm?" (NOTICING)
    engage_at: float = 0.60      # score needed to engage...
    engage_hold: float = 0.6     # ...held for this many seconds
    release_at: float = 0.30     # below this, the person counts as looking away
    leave_hold: float = 1.2      # looking away this long -> LEAVING (the lamp hesitates)
    idle_hold: float = 1.5       # still away this much longer -> IDLE (goodbye)
    hold_while_busy: bool = True  # never walk away while the lamp is listening or talking


@dataclass
class Engagement:
    """Tracks the current state. Call update() once per camera frame."""
    cfg: EngagementConfig = field(default_factory=EngagementConfig)
    state: State = State.IDLE
    score: float = 0.0            # the smoothed attention score
    _t: float | None = None       # time of the previous update
    _since: float = 0.0           # when the current condition started (for the "held for X s" rules)
    busy: bool = False            # set by the conversation code while the lamp listens or speaks

    def update(self, t: float, attention: float) -> State | None:
        """Feed in one frame's attention (0-1) at time t (seconds).

        Returns the new state if it just changed, otherwise None.
        """
        if self._t is None:
            self._t, self._since = t, t
        dt = max(0.0, t - self._t)
        self._t = t
        # Smoothing (an "exponential moving average"): move the score part of
        # the way toward this frame's value. The longer since the last frame,
        # the bigger the step. One odd frame barely moves it; a steady signal
        # moves it all the way within about a second.
        a = min(1.0, dt / self.cfg.ema_tau) if self.cfg.ema_tau > 0 else 1.0
        self.score += a * (attention - self.score)
        s, c = self.score, self.cfg

        new = self.state
        if self.state == State.IDLE:
            # Someone seems to be looking: start paying attention.
            if s >= c.notice_at:
                new = State.NOTICING
        elif self.state == State.NOTICING:
            if s < c.release_at:
                new = State.IDLE                     # false alarm, go back to sleep
            elif s >= c.engage_at and t - self._since >= c.engage_hold:
                new = State.ENGAGED                  # they really are looking: engage
        elif self.state == State.ENGAGED:
            if s >= c.release_at or (self.busy and c.hold_while_busy):
                # Still looking (or the lamp is mid-conversation): restart the
                # "how long have they been away" timer.
                self._since = t
            elif t - self._since >= c.leave_hold:
                new = State.LEAVING                  # away long enough: start hesitating
        elif self.state == State.LEAVING:
            if s >= c.engage_at:
                new = State.ENGAGED                  # they came back: welcome back
            elif t - self._since >= c.idle_hold:
                new = State.IDLE                     # really gone: goodbye

        if new != self.state:
            self.state, self._since = new, t         # remember when this state began
            return new
        return None
