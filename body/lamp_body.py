"""The simulated lamp body: physics, servo motion profiles and the light.

The body knows nothing about people, speech or AI. It accepts two kinds of
command and executes them safely:

  * joint goals   -> clamped to the URDF soft limits, then approached with a
                     velocity- and acceleration-limited profile so no joint
                     ever exceeds its URDF velocity limit
  * light goals   -> colour + brightness, faded over a given duration

Everything the character "does" physically is built from these two.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from body.model import JOINTS, build_model, urdf_limits

# Named poses: [base_yaw, shoulder_pitch, elbow_pitch, neck_yaw, head_pitch].
# Positive shoulder leans the arm back; negative elbow folds the upper arm
# forward; positive head_pitch tips the shade down. The lamp faces world -x.
#
# Physical constraint found in the supplied model: the shade's collision
# cylinder (r = 0.11 m) meets the upper arm once head_pitch > ~0, although the
# URDF soft limit allows +0.6. So the lamp looks *down* by leaning its arm,
# and uses head_pitch mainly to look *up*. Gaze elevation in the comments is
# the shade axis angle above horizontal.
POSES: dict[str, list[float]] = {
    "sleep":     [0.0, -0.30, -0.60, 0.00, -0.05],  # slumped, gaze -49 deg
    "rest":      [0.0,  0.20, -0.90, 0.00, -0.05],  # relaxed, gaze -37 deg
    "attentive": [0.0,  0.50, -1.20, 0.00, -0.55],  # facing a seated person
    "perk_up":   [0.0,  0.75, -1.35, 0.00, -0.75],  # tall and eager, gaze up
    "look_down": [0.0, -0.20, -0.90, 0.00, -0.05],  # inspect the desk, -60 deg
    "curious":   [0.0,  0.45, -1.15, 0.35, -0.45],  # head turned, quizzical
}

# Per-joint acceleration limits (rad/s^2). Joints that swing the whole arm are
# kept gentle so the 3 kg base would not rock on a real desk; the light head
# joints (~0.8 kg load) can snap quickly, which is where expression lives.
ACCEL_LIMIT = np.array([3.0, 4.0, 5.0, 12.0, 12.0])


@dataclass
class LightState:
    rgb: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.85, 0.6]))
    intensity: float = 0.0  # 0..1


class LampBody:
    def __init__(self, control_hz: float = 100.0):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.control_dt = 1.0 / control_hz
        self.substeps = max(1, round(self.control_dt / self.model.opt.timestep))

        lim = urdf_limits()
        self.soft_lo = np.array([lim[j]["soft_lower"] for j in JOINTS])
        self.soft_hi = np.array([lim[j]["soft_upper"] for j in JOINTS])
        self.vmax = np.array([lim[j]["velocity"] for j in JOINTS])
        self.joint_qadr = np.array([self.model.joint(j).qposadr[0] for j in JOINTS])
        self.joint_dof = np.array([self.model.joint(j).dofadr[0] for j in JOINTS])

        self._check = mujoco.MjData(self.model)  # scratch data for pose checks
        self._base = self.model.body("base_link").id
        self._spot = self.model.light("lamp_spot").id
        self._emitter = self.model.body("light_emitter_link").id
        self._bulb = self.model.material("bulb").id

        start = np.array(POSES["sleep"])
        self.data.qpos[self.joint_qadr] = start
        self.setpoint = start.copy()
        self.setvel = np.zeros(len(JOINTS))
        self.goal = start.copy()
        self.speed = 1.0

        self.light = LightState()
        self._light_from = LightState()
        self._light_to = LightState()
        self._light_t = 1.0
        self._light_dur = 0.0
        self._apply_light(self.light)
        mujoco.mj_forward(self.model, self.data)

    # ----------------------------------------------------------- commands
    def set_goal(self, q, speed: float = 1.0) -> dict:
        """Validate and set joint goals.

        1. Clamp to URDF soft limits.
        2. If the goal pose self-collides (or hits the table), lift the head
           (lower head_pitch) in 0.05 rad steps until it is free.
        3. If no free pose is found, reject and keep the previous goal.
        Returns a report so callers (and logs) can see what the body changed.
        """
        q = np.asarray(q, dtype=float)
        report = {"accepted": True, "adjusted": []}
        goal = np.clip(q, self.soft_lo, self.soft_hi)
        report["adjusted"] += [f"{JOINTS[i]}: soft limit"
                               for i in np.flatnonzero(np.abs(goal - q) > 1e-6)]
        h = JOINTS.index("head_pitch_joint")
        start_h = goal[h]
        while self.collides(goal) and goal[h] > self.soft_lo[h]:
            goal[h] = max(self.soft_lo[h], goal[h] - 0.05)
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
        return self.set_goal(POSES[name], speed)

    def collides(self, q) -> bool:
        """True if pose q has any lamp contact other than base-on-table."""
        d = self._check
        d.qpos[self.joint_qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_collision(self.model, d)
        for c in d.contact[: d.ncon]:
            pair = {self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]}
            if pair <= {0, self._base}:
                continue
            if c.dist < -1e-4:
                return True
        return False

    def set_light(self, rgb, intensity: float, fade_s: float = 0.4) -> None:
        self._light_from = LightState(self.light.rgb.copy(), self.light.intensity)
        self._light_to = LightState(np.clip(np.asarray(rgb, float), 0, 1),
                                    float(np.clip(intensity, 0, 1)))
        self._light_dur = max(fade_s, 1e-6)
        self._light_t = 0.0

    # ----------------------------------------------------------- loop
    def step(self) -> None:
        """Advance one control tick (motion profile, light fade, physics)."""
        dt = self.control_dt
        vmax = self.vmax * self.speed
        # Velocity we would like: proportional approach, capped by vmax.
        # The sqrt term lets the joint brake in time with ACCEL_LIMIT.
        err = self.goal - self.setpoint
        v_brake = np.sqrt(2 * ACCEL_LIMIT * np.abs(err))
        v_des = np.sign(err) * np.minimum(vmax, v_brake)
        dv = np.clip(v_des - self.setvel, -ACCEL_LIMIT * dt, ACCEL_LIMIT * dt)
        self.setvel += dv
        self.setpoint += self.setvel * dt
        # Snap when close so we do not dither forever.
        done = (np.abs(err) < 1e-3) & (np.abs(self.setvel) < 0.05)
        self.setpoint[done] = self.goal[done]
        self.setvel[done] = 0.0

        self.data.ctrl[:] = self.setpoint

        if self._light_t < 1.0:
            self._light_t = min(1.0, self._light_t + dt / self._light_dur)
            s = self._light_t * self._light_t * (3 - 2 * self._light_t)  # smoothstep
            a, b = self._light_from, self._light_to
            self.light = LightState(a.rgb + (b.rgb - a.rgb) * s,
                                    a.intensity + (b.intensity - a.intensity) * s)
            self._apply_light(self.light)

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def _apply_light(self, light: LightState) -> None:
        on = light.intensity
        self.model.light_diffuse[self._spot] = light.rgb * on * 1.4
        self.model.light_specular[self._spot] = light.rgb * on * 0.2
        self.model.light_active[self._spot] = on > 0.01
        # Shade glow: dim grey when off, tinted and emissive when on.
        off = np.array([0.55, 0.55, 0.55])
        self.model.mat_rgba[self._bulb, :3] = off + (light.rgb - off) * min(1.0, on * 1.5)
        self.model.mat_emission[self._bulb] = 0.05 + 0.45 * on

    # ----------------------------------------------------------- state
    @property
    def at_goal(self) -> bool:
        return bool(np.all(self.setpoint == self.goal))

    def state(self) -> dict:
        q = self.data.qpos[self.joint_qadr]
        qd = self.data.qvel[self.joint_dof]
        # qfrc_actuator includes the gravity-compensation share, so this is
        # the total torque each servo is producing against its effort limit.
        torque = self.data.qfrc_actuator[self.joint_dof]
        effort = np.abs(self.model.actuator_forcerange[:, 1])
        # Measured gaze: shade axis from the *actual* joint angles (as encoders
        # would report on hardware), not from the commanded goal.
        x = self.data.xmat[self._emitter].reshape(3, 3)[:, 0]
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
