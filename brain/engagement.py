"""Engagement state machine: turns noisy per-frame attention into decisions.

    IDLE --(attention rising)--> NOTICING --(sustained)--> ENGAGED
      ^                             |                         |
      |<------(drops away)----------+                (attention lost)
      |                                                       v
      +<-----------------(stays away)------------------- LEAVING

Hysteresis: it is easy to *stay* engaged and harder to *become* engaged,
so a blink, a glance at the keyboard or one missed detection never ends a
conversation, and a person walking past does not wake the lamp. All
thresholds are here so they can be tuned against the engagement log.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    IDLE = "idle"
    NOTICING = "noticing"
    ENGAGED = "engaged"
    LEAVING = "leaving"


@dataclass
class EngagementConfig:
    ema_tau: float = 0.35        # s, smoothing of the per-frame attention score
    notice_at: float = 0.45      # smoothed score that makes the lamp look up
    engage_at: float = 0.60      # ...and that must hold for engage_hold seconds
    engage_hold: float = 0.6
    release_at: float = 0.30     # below this the person counts as "away"
    leave_hold: float = 1.2      # away this long -> LEAVING (the lamp hesitates)
    idle_hold: float = 1.5       # still away after LEAVING -> IDLE
    hold_while_busy: bool = True  # never disengage mid-conversation (set by brain)


@dataclass
class Engagement:
    cfg: EngagementConfig = field(default_factory=EngagementConfig)
    state: State = State.IDLE
    score: float = 0.0
    _t: float | None = None
    _since: float = 0.0           # time the current condition started
    busy: bool = False            # brain sets this while listening/speaking

    def update(self, t: float, attention: float) -> State | None:
        """Feed one observation. Returns the new state if it changed."""
        if self._t is None:
            self._t, self._since = t, t
        dt = max(0.0, t - self._t)
        self._t = t
        a = min(1.0, dt / self.cfg.ema_tau) if self.cfg.ema_tau > 0 else 1.0
        self.score += a * (attention - self.score)
        s, c = self.score, self.cfg

        new = self.state
        if self.state == State.IDLE:
            if s >= c.notice_at:
                new = State.NOTICING
        elif self.state == State.NOTICING:
            if s < c.release_at:
                new = State.IDLE
            elif s >= c.engage_at and t - self._since >= c.engage_hold:
                new = State.ENGAGED
        elif self.state == State.ENGAGED:
            if s >= c.release_at or (self.busy and c.hold_while_busy):
                self._since = t                      # reset the "away" timer
            elif t - self._since >= c.leave_hold:
                new = State.LEAVING
        elif self.state == State.LEAVING:
            if s >= c.engage_at:
                new = State.ENGAGED                  # came back: welcome back
            elif t - self._since >= c.idle_hold:
                new = State.IDLE

        if new != self.state:
            self.state, self._since = new, t
            return new
        return None
