"""Action executor: turns protocol commands into safe body motion.

Layering (bottom up):
  LampBody   - "hardware": servos, limits, light, physics. On a real robot this
               is the only layer that would be replaced.
  Executor   - action primitives: pose, gaze, joints, gesture, light, stop.
               Owns command ids, sequencing and completion reporting.
  server.py  - protocol transport (WebSocket + JSON).

Motion commands share one channel: a new motion command preempts the running
one (reported as "preempted"). Light has its own channel and never interrupts
motion, so the character can glow while it moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from body.kinematics import EL_RANGE, solve_gaze
from body.lamp_body import POSES, LampBody
from body.model import JOINTS

B, S, E, N, H = range(5)

# Gestures are keyframes of joint *offsets* from the pose the lamp is in when
# the gesture starts, so a nod works whether the lamp is looking up or aside.
# Each keyframe: (offsets {joint index: rad}, speed 0..1, hold seconds).
GESTURES: dict[str, list[tuple[dict[int, float], float, float]]] = {
    "nod":    [({H: 0.22}, 1.0, 0.0), ({H: -0.05}, 1.0, 0.0),
               ({H: 0.18}, 1.0, 0.0), ({}, 0.8, 0.0)],
    "shake":  [({N: 0.30}, 1.0, 0.0), ({N: -0.30}, 1.0, 0.0),
               ({N: 0.22}, 1.0, 0.0), ({}, 0.8, 0.0)],
    "wiggle": [({B: 0.18, N: -0.15}, 1.0, 0.0), ({B: -0.18, N: 0.15}, 1.0, 0.0),
               ({B: 0.12, N: -0.10}, 1.0, 0.0), ({}, 0.8, 0.0)],
    "perk":   [({S: 0.20, E: -0.12, H: -0.20}, 1.0, 0.25), ({}, 0.5, 0.0)],
    "droop":  [({S: -0.35, E: 0.10, H: 0.20}, 0.35, 0.8)],  # stays drooped
    "tilt":   [({N: 0.30, H: -0.10}, 0.6, 0.6), ({}, 0.5, 0.0)],
}


@dataclass
class _Motion:
    cmd_id: str
    steps: list[tuple[np.ndarray, float, float]] = field(default_factory=list)
    hold: float = 0.0                 # seconds to hold the current keyframe
    reached_at: float | None = None   # sim time the keyframe was reached
    adjusted: list[str] = field(default_factory=list)


class Executor:
    def __init__(self, body: LampBody):
        self.body = body
        self._scratch = mujoco.MjData(body.model)
        self._emitter = body.model.body("light_emitter_link").id
        self._motion: _Motion | None = None
        self.events: list[dict] = []        # outgoing: done / preempted

    # ------------------------------------------------------------ dispatch
    def handle(self, msg: dict) -> dict:
        """Execute one command. Returns the ack message."""
        kind = msg.get("type")
        cid = str(msg.get("id", ""))
        try:
            if kind == "pose":
                return self._start(cid, [(np.array(POSES[msg["name"]]), msg.get("speed", 0.7), 0.0)])
            if kind == "joints":
                q = self.body.goal.copy()
                for name, val in msg["q"].items():
                    q[JOINTS.index(name)] = float(val)
                return self._start(cid, [(q, msg.get("speed", 0.7), 0.0)])
            if kind == "gaze":
                az, el = float(msg["azimuth"]), float(msg["elevation"])
                q = solve_gaze(self.body.model, self._scratch, self.body.joint_qadr,
                               self.body.soft_lo, self.body.soft_hi, self._emitter, az, el)
                ack = self._start(cid, [(q, msg.get("speed", 0.7), 0.0)])
                if not EL_RANGE[0] <= el <= EL_RANGE[1]:
                    ack["adjusted"].append(f"elevation clipped to {EL_RANGE}")
                return ack
            if kind == "gesture":
                base = self.body.goal.copy()
                steps = []
                for offs, speed, hold in GESTURES[msg["name"]]:
                    q = base.copy()
                    for i, dv in offs.items():
                        q[i] += dv
                    steps.append((q, speed * msg.get("intensity", 1.0) ** 0.5, hold))
                return self._start(cid, steps)
            if kind == "light":
                self.body.set_light(msg["rgb"], msg["intensity"], msg.get("fade", 0.4))
                return {"type": "ack", "id": cid, "ok": True, "adjusted": []}
            if kind == "stop":
                self._preempt()
                self.body.set_goal(self.body.data.qpos[self.body.joint_qadr], 1.0)
                return {"type": "ack", "id": cid, "ok": True, "adjusted": []}
            return {"type": "ack", "id": cid, "ok": False, "error": f"unknown type {kind!r}"}
        except (KeyError, ValueError, TypeError) as exc:
            return {"type": "ack", "id": cid, "ok": False, "error": f"bad command: {exc}"}

    def _start(self, cid: str, steps) -> dict:
        self._preempt()
        self._motion = _Motion(cid, list(steps))
        ack = {"type": "ack", "id": cid, "ok": True, "adjusted": []}
        if not self._next_step():
            ack.update(ok=False, error="pose rejected: no collision-free variant")
            self._motion = None
        else:
            ack["adjusted"] = list(self._motion.adjusted)
        return ack

    def _preempt(self) -> None:
        if self._motion is not None:
            self.events.append({"type": "done", "id": self._motion.cmd_id, "status": "preempted"})
            self._motion = None

    def _next_step(self) -> bool:
        m = self._motion
        q, speed, hold = m.steps.pop(0)
        rep = self.body.set_goal(q, speed)
        m.adjusted += rep["adjusted"]
        m.hold, m.reached_at = hold, None
        return rep["accepted"]

    # ------------------------------------------------------------ loop
    def step(self) -> None:
        self.body.step()
        m = self._motion
        if m is None or not self.body.at_goal:
            return
        now = self.body.data.time
        if m.reached_at is None:
            m.reached_at = now
        if now - m.reached_at < m.hold:
            return
        if m.steps:
            if not self._next_step():
                self.events.append({"type": "done", "id": m.cmd_id, "status": "rejected"})
                self._motion = None
            return
        self.events.append({"type": "done", "id": m.cmd_id, "status": "completed"})
        self._motion = None

    def capabilities(self) -> dict:
        return {
            "joints": JOINTS,
            "soft_limits": {j: [round(float(lo), 3), round(float(hi), 3)]
                            for j, lo, hi in zip(JOINTS, self.body.soft_lo, self.body.soft_hi)},
            "velocity_limits": {j: float(v) for j, v in zip(JOINTS, self.body.vmax)},
            "poses": sorted(POSES),
            "gestures": sorted(GESTURES),
            "gaze_elevation_range": list(EL_RANGE),
            "gaze_frame": "azimuth 0 = lamp forward, +left; elevation 0 = horizontal, +up (rad)",
        }
