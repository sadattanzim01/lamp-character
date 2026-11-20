"""The lamp's "hardware": its joints, how they move, and its light.

What this file does, in plain words:
    This is the lowest layer of the lamp. It knows nothing about people,
    speech or AI. It only understands two kinds of request:

      * "move the joints to these angles" (a joint goal)
          The goal is first made safe: kept inside the joints' safe limits and
          checked so the lamp won't hit itself. Then each joint glides there
          smoothly: it speeds up, cruises, and slows down, never going faster
          than the URDF allows.

      * "change the light to this colour and brightness" (a light goal)
          The light fades to the new setting over a short time instead of
          jumping.

    Every physical thing the character does (nodding, looking at you, glowing
    orange) is built out of these two requests.

Why it's separate: on a real lamp, this is the only file you'd swap out (for
code that talks to real motors and LEDs). Everything above it would stay the same.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from body.model import JOINTS, build_model, urdf_limits

# Ready-made poses. Each list gives the 5 joint angles (in radians) in this order:
#   [base_yaw, shoulder_pitch, elbow_pitch, neck_yaw, head_pitch]
#
# How the signs work: a positive shoulder leans the arm back, a negative elbow
# folds the upper arm forward, and a positive head_pitch tips the shade down.
# The lamp's front faces the world's -x direction.
#
# A real limit we discovered: the shade's collision shape (a cylinder about
# 11 cm wide) bumps into the upper arm as soon as head_pitch goes above ~0,
# even though the URDF's safe limit says +0.6 is allowed. So to look DOWN, the
# lamp leans its whole arm forward; the head joint is mainly used to look UP.
# "gaze" in the comments below = how far the shade points above (+) or below (-)
# horizontal.
POSES: dict[str, list[float]] = {
    "sleep":     [0.0, -0.30, -0.60, 0.00, -0.05],  # slumped over, gaze -49 deg
    "rest":      [0.0,  0.20, -0.90, 0.00, -0.05],  # relaxed, gaze -37 deg
    "attentive": [0.0,  0.50, -1.20, 0.00, -0.55],  # facing someone sitting at the desk
    "perk_up":   [0.0,  0.75, -1.35, 0.00, -0.75],  # tall and eager, looking up
    "look_down": [0.0, -0.20, -0.90, 0.00, -0.05],  # inspecting the desk, gaze -60 deg
    "curious":   [0.0,  0.45, -1.15, 0.35, -0.45],  # head turned to the side, quizzical
}

# How quickly each joint is allowed to speed up or slow down (rad/s^2).
# Joints that swing the whole arm are kept gentle so a real 3 kg base wouldn't
# rock on the desk. The two head joints only carry ~0.8 kg, so they can move
# snappily, and that's where most of the lamp's expressiveness comes from.
#               base  shoulder  elbow  neck  head
ACCEL_LIMIT = np.array([3.0, 4.0, 5.0, 12.0, 12.0])


@dataclass
class LightState:
    """One light setting: a colour (red, green, blue, each 0-1) and a brightness (0-1)."""
    rgb: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.85, 0.6]))
    intensity: float = 0.0  # 0 = off, 1 = full brightness


class LampBody:
    """The simulated lamp. Call step() 100 times a second to keep it running."""

    def __init__(self, control_hz: float = 100.0):
        self.model = build_model()            # the lamp built in model.py
        self.data = mujoco.MjData(self.model)  # its live state (angles, speeds, forces...)
        self.control_dt = 1.0 / control_hz    # time per control tick (0.01 s)
        # Physics runs at 500 Hz, so each 100 Hz control tick = 5 physics steps.
        self.substeps = max(1, round(self.control_dt / self.model.opt.timestep))

        # Read each joint's safe range and top speed from the URDF.
        lim = urdf_limits()
        self.soft_lo = np.array([lim[j]["soft_lower"] for j in JOINTS])
        self.soft_hi = np.array([lim[j]["soft_upper"] for j in JOINTS])
        self.vmax = np.array([lim[j]["velocity"] for j in JOINTS])
        # Where each joint's angle and speed are stored inside MuJoCo's arrays.
        self.joint_qadr = np.array([self.model.joint(j).qposadr[0] for j in JOINTS])
        self.joint_dof = np.array([self.model.joint(j).dofadr[0] for j in JOINTS])

        # A second, throwaway copy of the state used to test poses ("would this
        # pose collide?") without disturbing the real simulation.
        self._check = mujoco.MjData(self.model)
        # IDs of parts we refer to often.
        self._base = self.model.body("base_link").id
        self._spot = self.model.light("lamp_spot").id
        self._emitter = self.model.body("light_emitter_link").id
        self._bulb = self.model.material("bulb").id

        # Start asleep.
        start = np.array(POSES["sleep"])
        self.data.qpos[self.joint_qadr] = start
        # setpoint = where we're currently telling the motors to be (it glides
        #            toward the goal a little each tick)
        # setvel   = how fast that setpoint is moving right now
        # goal     = where the joints should end up
        self.setpoint = start.copy()
        self.setvel = np.zeros(len(JOINTS))
        self.goal = start.copy()
        self.speed = 1.0                      # fraction of max speed (0.05-1)

        # Light starts off. The _light_* fields track an ongoing fade from one
        # setting to another (_light_t goes from 0 to 1 over _light_dur seconds).
        self.light = LightState()
        self._light_from = LightState()
        self._light_to = LightState()
        self._light_t = 1.0
        self._light_dur = 0.0
        self._apply_light(self.light)
        mujoco.mj_forward(self.model, self.data)   # compute positions for the start pose

    # ----------------------------------------------------------- commands
    def set_goal(self, q, speed: float = 1.0) -> dict:
        """Ask the joints to move to angles `q`, after making the request safe.

        1. Clamp every angle to the joint's safe ("soft") range.
        2. If that pose would make the lamp hit itself (or the table), tilt the
           head up a little (0.05 rad at a time) until it's clear.
        3. If no tilt fixes it, refuse and keep the old goal.

        Returns a report like {"accepted": True, "adjusted": [...]} listing
        anything the body changed, so whoever asked can see what really happened.
        """
        q = np.asarray(q, dtype=float)
        report = {"accepted": True, "adjusted": []}
        # Step 1: keep inside the safe range, and note which joints were clipped.
        goal = np.clip(q, self.soft_lo, self.soft_hi)
        report["adjusted"] += [f"{JOINTS[i]}: soft limit"
                               for i in np.flatnonzero(np.abs(goal - q) > 1e-6)]
        # Step 2: raise the head until the pose no longer collides.
        h = JOINTS.index("head_pitch_joint")
        start_h = goal[h]
        while self.collides(goal) and goal[h] > self.soft_lo[h]:
            goal[h] = max(self.soft_lo[h], goal[h] - 0.05)
        # Step 3: still colliding? Refuse.
        if self.collides(goal):
            report["accepted"] = False
            return report
        if goal[h] != start_h:
            report["adjusted"].append(
                f"head_pitch_joint: {start_h:+.2f} -> {goal[h]:+.2f} to clear upper arm")
        self.goal = goal
        self.speed = float(np.clip(speed, 0.05, 1.0))
        return report

    def set_pose(self, name: str, speed: float = 1.0) -> dict:
        """Move to one of the named POSES above, e.g. set_pose("sleep")."""
        return self.set_goal(POSES[name], speed)

    def collides(self, q) -> bool:
        """Would pose `q` make the lamp touch itself or the table?

        The base resting on the table is normal, so that contact is ignored.
        This uses the scratch copy, so the real lamp doesn't move.
        """
        d = self._check
        d.qpos[self.joint_qadr] = q
        mujoco.mj_kinematics(self.model, d)   # where would every part be?
        mujoco.mj_collision(self.model, d)    # which parts would be touching?
        for c in d.contact[: d.ncon]:
            pair = {self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]}
            if pair <= {0, self._base}:       # 0 = the world/table, so base-on-table
                continue
            if c.dist < -1e-4:                # parts overlap (negative distance)
                return True
        return False

    def set_light(self, rgb, intensity: float, fade_s: float = 0.4) -> None:
        """Fade the light to a new colour and brightness over `fade_s` seconds."""
        self._light_from = LightState(self.light.rgb.copy(), self.light.intensity)
        self._light_to = LightState(np.clip(np.asarray(rgb, float), 0, 1),
                                    float(np.clip(intensity, 0, 1)))
        self._light_dur = max(fade_s, 1e-6)   # avoid dividing by zero for instant changes
        self._light_t = 0.0                   # restart the fade from the beginning

    # ----------------------------------------------------------- loop
    def step(self) -> None:
        """Run one 10 ms tick: move the setpoints, update the light fade, simulate physics."""
        dt = self.control_dt
        vmax = self.vmax * self.speed
        # Smooth motion, joint by joint:
        #   * err     = how far each joint still has to go
        #   * v_brake = the fastest speed from which the joint can still stop in
        #               time (physics: v = sqrt(2 * a * distance))
        #   * v_des   = desired speed: never above the top speed or the braking speed
        #   * dv      = change the speed by at most ACCEL_LIMIT per tick
        # Result: each joint accelerates, cruises, then decelerates into the
        # target, like a real servo with a motion profile.
        err = self.goal - self.setpoint
        v_brake = np.sqrt(2 * ACCEL_LIMIT * np.abs(err))
        v_des = np.sign(err) * np.minimum(vmax, v_brake)
        dv = np.clip(v_des - self.setvel, -ACCEL_LIMIT * dt, ACCEL_LIMIT * dt)
        self.setvel += dv
        self.setpoint += self.setvel * dt
        # Once a joint is basically there and almost still, snap it exactly to
        # the goal so it doesn't jitter around forever.
        done = (np.abs(err) < 1e-3) & (np.abs(self.setvel) < 0.05)
        self.setpoint[done] = self.goal[done]
        self.setvel[done] = 0.0

        # Hand the setpoints to the motors.
        self.data.ctrl[:] = self.setpoint

        # Light fade. "smoothstep" eases in and out so the fade starts and ends
        # gently instead of changing at a constant rate.
        if self._light_t < 1.0:
            self._light_t = min(1.0, self._light_t + dt / self._light_dur)
            s = self._light_t * self._light_t * (3 - 2 * self._light_t)  # smoothstep
            a, b = self._light_from, self._light_to
            self.light = LightState(a.rgb + (b.rgb - a.rgb) * s,
                                    a.intensity + (b.intensity - a.intensity) * s)
            self._apply_light(self.light)

        # Advance the physics (5 small steps per tick).
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def _apply_light(self, light: LightState) -> None:
        """Push a light setting into the simulation: the spotlight and the shade's glow."""
        on = light.intensity
        # The spotlight that lights up the table.
        self.model.light_diffuse[self._spot] = light.rgb * on * 1.4
        self.model.light_specular[self._spot] = light.rgb * on * 0.2
        self.model.light_active[self._spot] = on > 0.01   # fully off when nearly dark
        # The glow inside the shade: dull grey when off, coloured and glowing when on.
        off = np.array([0.55, 0.55, 0.55])
        self.model.mat_rgba[self._bulb, :3] = off + (light.rgb - off) * min(1.0, on * 1.5)
        self.model.mat_emission[self._bulb] = 0.05 + 0.45 * on

    # ----------------------------------------------------------- state
    @property
    def at_goal(self) -> bool:
        """True once every joint's setpoint has reached its goal."""
        return bool(np.all(self.setpoint == self.goal))

    def state(self) -> dict:
        """A snapshot of the lamp, sent to the brain 20 times a second.

        Includes joint angles (q) and speeds (qd), how hard each motor is
        working (torque_frac: 0.5 = half of its torque limit), whether it has
        reached its goal, where the shade is actually pointing (gaze), and the
        current light.
        """
        q = self.data.qpos[self.joint_qadr]
        qd = self.data.qvel[self.joint_dof]
        # Total torque each motor produces, including the part spent holding the
        # arm up against gravity, compared with its URDF limit.
        torque = self.data.qfrc_actuator[self.joint_dof]
        effort = np.abs(self.model.actuator_forcerange[:, 1])
        # Where the shade points, computed from the ACTUAL joint angles (like
        # reading the motors' encoders on a real robot), not from what we asked
        # for. The goal code uses this to check whether its aim was right.
        #   azimuth   = left/right angle (0 = straight ahead, + = lamp's left)
        #   elevation = up/down angle (0 = horizontal, + = up)
        x = self.data.xmat[self._emitter].reshape(3, 3)[:, 0]   # direction the shade faces
        gaze = [float(np.arctan2(-x[1], -x[0])), float(np.arcsin(np.clip(x[2], -1, 1)))]
        return {
            "t": round(float(self.data.time), 3),
            "q": {j: round(float(v), 4) for j, v in zip(JOINTS, q)},
            "qd": {j: round(float(v), 4) for j, v in zip(JOINTS, qd)},
            "torque_frac": {j: round(float(abs(t) / e), 3) for j, t, e in zip(JOINTS, torque, effort)},
            "at_goal": self.at_goal,
            "gaze": {"azimuth": round(gaze[0], 4), "elevation": round(gaze[1], 4)},
            "light": {"rgb": [round(float(c), 3) for c in self.light.rgb],
                      "intensity": round(self.light.intensity, 3)},
        }
