"""Turns commands from the brain into safe lamp movements.

What this file does, in plain words:
    The brain sends simple commands like "nod", "look over there" or "glow
    blue". This file is the lamp's to-do manager: it turns each command into
    the actual joint targets, runs multi-step moves (like a nod: down, up,
    down, back) one step at a time, and reports back when each one finishes.

How the body is layered (bottom to top):
    LampBody  (lamp_body.py)  the "hardware": motors, limits, light, physics.
                              On a real robot, this is the only layer you'd swap.
    Executor  (this file)     the actions: pose, gaze, joints, gesture, light, stop.
                              Tracks each command by its id and reports when it's done.
    server.py                 the network connection that carries the commands.

Two rules about overlapping commands:
    * Movement has ONE lane. A new movement command cancels the one in progress,
      and the cancelled one is reported as "preempted" (interrupted).
      Example: if the lamp is shaking its head and you ask it to look at you,
      it stops shaking and turns.
    * Light has its OWN lane. Changing the light never interrupts a movement,
      so the lamp can glow while it moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from body.kinematics import EL_RANGE, solve_gaze
from body.lamp_body import POSES, LampBody
from body.model import JOINTS

# Short names for the joint positions in a pose list:
# B = base, S = shoulder, E = elbow, N = neck, H = head.
B, S, E, N, H = range(5)

# Gestures, written as small changes ("offsets") from wherever the lamp is when
# the gesture starts. That's why a nod works whether the lamp is looking up, down
# or to the side: it always nods relative to its current pose.
#
# Each gesture is a list of steps. Each step is:
#     ({joint: change in radians}, speed 0-1, seconds to hold before the next step)
# An empty {} step means "go back to where you started".
GESTURES: dict[str, list[tuple[dict[int, float], float, float]]] = {
    # head down, up, down, back: "yes"
    "nod":    [({H: 0.22}, 1.0, 0.0), ({H: -0.05}, 1.0, 0.0),
               ({H: 0.18}, 1.0, 0.0), ({}, 0.8, 0.0)],
    # neck left, right, left, back: "no"
    "shake":  [({N: 0.30}, 1.0, 0.0), ({N: -0.30}, 1.0, 0.0),
               ({N: 0.22}, 1.0, 0.0), ({}, 0.8, 0.0)],
    # base and neck twist in opposite directions: a happy little dance
    "wiggle": [({B: 0.18, N: -0.15}, 1.0, 0.0), ({B: -0.18, N: 0.15}, 1.0, 0.0),
               ({B: 0.12, N: -0.10}, 1.0, 0.0), ({}, 0.8, 0.0)],
    # stand taller and look up for a quarter second: excited
    "perk":   [({S: 0.20, E: -0.12, H: -0.20}, 1.0, 0.25), ({}, 0.5, 0.0)],
    # slump forward slowly and stay there: sad / sleepy (no return step)
    "droop":  [({S: -0.35, E: 0.10, H: 0.20}, 0.35, 0.8)],
    # head tilted to one side for a moment: curious / unsure
    "tilt":   [({N: 0.30, H: -0.10}, 0.6, 0.6), ({}, 0.5, 0.0)],
}


@dataclass
class _Motion:
    """The movement currently running, and the steps it still has left."""
    cmd_id: str                                     # the id the brain gave this command
    steps: list[tuple[np.ndarray, float, float]] = field(default_factory=list)  # (angles, speed, hold)
    hold: float = 0.0                 # how long to pause at the current step
    reached_at: float | None = None   # when the current step was reached (simulation time)
    adjusted: list[str] = field(default_factory=list)   # anything the body changed for safety


class Executor:
    """Runs one movement at a time on the lamp and reports how each one ended."""

    def __init__(self, body: LampBody):
        self.body = body
        self._scratch = mujoco.MjData(body.model)   # throwaway copy for the gaze math
        self._emitter = body.model.body("light_emitter_link").id
        self._motion: _Motion | None = None          # the movement in progress, if any
        # Messages waiting to be sent to the brain, like
        # {"type": "done", "id": "c7", "status": "completed"}
        self.events: list[dict] = []

    # ------------------------------------------------------------ dispatch
    def handle(self, msg: dict) -> dict:
        """Carry out one command from the brain and return the immediate reply ("ack").

        The ack says whether the command was accepted, and lists anything the
        body had to change to keep it safe. Movements report again later
        (through self.events) when they finish or get interrupted.
        """
        kind = msg.get("type")
        cid = str(msg.get("id", ""))
        try:
            if kind == "pose":        # e.g. {"type": "pose", "name": "sleep"}
                return self._start(cid, [(np.array(POSES[msg["name"]]), msg.get("speed", 0.7), 0.0)])
            if kind == "joints":      # set specific joints; the others keep their current goal
                q = self.body.goal.copy()
                for name, val in msg["q"].items():
                    q[JOINTS.index(name)] = float(val)
                return self._start(cid, [(q, msg.get("speed", 0.7), 0.0)])
            if kind == "gaze":        # "look in this direction": kinematics.py works out the joints
                az, el = float(msg["azimuth"]), float(msg["elevation"])
                q = solve_gaze(self.body.model, self._scratch, self.body.joint_qadr,
                               self.body.soft_lo, self.body.soft_hi, self._emitter, az, el)
                ack = self._start(cid, [(q, msg.get("speed", 0.7), 0.0)])
                # Tell the brain if it asked to look higher/lower than the lamp can.
                if not EL_RANGE[0] <= el <= EL_RANGE[1]:
                    ack["adjusted"].append(f"elevation clipped to {EL_RANGE}")
                return ack
            if kind == "gesture":     # e.g. "nod": turn the offsets into actual angles
                base = self.body.goal.copy()
                steps = []
                for offs, speed, hold in GESTURES[msg["name"]]:
                    q = base.copy()
                    for i, dv in offs.items():
                        q[i] += dv
                    # A lower "intensity" makes the gesture gentler (slower).
                    steps.append((q, speed * msg.get("intensity", 1.0) ** 0.5, hold))
                return self._start(cid, steps)
            if kind == "light":       # separate lane: never interrupts movement
                self.body.set_light(msg["rgb"], msg["intensity"], msg.get("fade", 0.4))
                return {"type": "ack", "id": cid, "ok": True, "adjusted": []}
            if kind == "stop":        # cancel the movement and freeze where the lamp is now
                self._preempt()
                self.body.set_goal(self.body.data.qpos[self.body.joint_qadr], 1.0)
                return {"type": "ack", "id": cid, "ok": True, "adjusted": []}
            return {"type": "ack", "id": cid, "ok": False, "error": f"unknown type {kind!r}"}
        except (KeyError, ValueError, TypeError) as exc:
            # A bad command (unknown pose name, missing field...) gets a clear
            # error back instead of crashing the body.
            return {"type": "ack", "id": cid, "ok": False, "error": f"bad command: {exc}"}

    def _start(self, cid: str, steps) -> dict:
        """Begin a new movement, interrupting any movement already running."""
        self._preempt()
        self._motion = _Motion(cid, list(steps))
        ack = {"type": "ack", "id": cid, "ok": True, "adjusted": []}
        if not self._next_step():
            # The very first step was unsafe even after adjustment: refuse.
            ack.update(ok=False, error="pose rejected: no collision-free variant")
            self._motion = None
        else:
            ack["adjusted"] = list(self._motion.adjusted)
        return ack

    def _preempt(self) -> None:
        """Cancel the current movement and tell the brain it was interrupted."""
        if self._motion is not None:
            self.events.append({"type": "done", "id": self._motion.cmd_id, "status": "preempted"})
            self._motion = None

    def _next_step(self) -> bool:
        """Send the movement's next step to the body. False if the body refused it."""
        m = self._motion
        q, speed, hold = m.steps.pop(0)
        rep = self.body.set_goal(q, speed)     # lamp_body makes it safe first
        m.adjusted += rep["adjusted"]
        m.hold, m.reached_at = hold, None
        return rep["accepted"]

    # ------------------------------------------------------------ loop
    def step(self) -> None:
        """Called every 10 ms: run physics, then move on to the next step when ready."""
        self.body.step()
        m = self._motion
        if m is None or not self.body.at_goal:
            return                               # nothing running, or still moving
        # The current step has been reached. Pause there if it asks for a hold.
        now = self.body.data.time
        if m.reached_at is None:
            m.reached_at = now
        if now - m.reached_at < m.hold:
            return
        # More steps? Start the next one.
        if m.steps:
            if not self._next_step():
                self.events.append({"type": "done", "id": m.cmd_id, "status": "rejected"})
                self._motion = None
            return
        # No steps left: the whole movement is finished.
        self.events.append({"type": "done", "id": m.cmd_id, "status": "completed"})
        self._motion = None

    def capabilities(self) -> dict:
        """Everything the lamp can do, sent to the brain when it first connects.

        The brain uses this to know the joint names and limits, the available
        poses and gestures, and how far up/down the lamp can look, without any
        of it being hard-coded on the brain side.
        """
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
