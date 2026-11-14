"""Gaze inverse kinematics: "point the shade in this direction" -> joint goals.

Gaze is expressed in the lamp's own frame:
  azimuth   0 = straight ahead (world -x), positive = towards the lamp's left
  elevation 0 = horizontal, positive = up

The brain only ever says *where* to look. This module decides *how*: the arm
posture is chosen from the target elevation (lean forward to look down, stand
tall to look up), the neck takes a share of small turns so the motion reads as
a glance, and a damped least-squares solve on the real MuJoCo kinematics fixes
base_yaw, shoulder and head so the shade axis actually hits the target.
"""
from __future__ import annotations

import mujoco
import numpy as np

# Arm posture by gaze elevation (rad): (elevation, shoulder, elbow).
# Derived from the hand-tuned character poses in lamp_body.POSES.
_POSTURE = np.array([
    [-1.05, -0.20, -0.90],
    [-0.65,  0.20, -0.90],
    [-0.15,  0.50, -1.20],
    [ 0.25,  0.75, -1.35],
])
NECK_SHARE = 0.35    # fraction of azimuth taken by the neck
NECK_MAX = 0.50      # rad
HEAD_SAFE_MAX = -0.05  # head_pitch above this hits the upper arm (see lamp_body)
EL_RANGE = (-1.05, 0.22)  # reachable gaze elevation with the shade clear of the arm


def gaze_of(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    """(azimuth, elevation) of the shade axis for the current kinematics."""
    x = data.xmat[body_id].reshape(3, 3)[:, 0]
    return np.array([np.arctan2(-x[1], -x[0]), np.arcsin(np.clip(x[2], -1, 1))])


def _direction(az: float, el: float) -> np.ndarray:
    return np.array([-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)])


def solve_gaze(model, scratch, qadr, lo, hi, emitter_id, az, el, iters=30):
    """Return joint goals [base, shoulder, elbow, neck, head] for a gaze target."""
    el = float(np.clip(el, *EL_RANGE))
    hi = hi.copy()
    hi[4] = min(hi[4], HEAD_SAFE_MAX)
    s = np.interp(el, _POSTURE[:, 0], _POSTURE[:, 1])
    e = np.interp(el, _POSTURE[:, 0], _POSTURE[:, 2])
    neck = float(np.clip(NECK_SHARE * az, -NECK_MAX, NECK_MAX))
    q = np.array([az - neck, s, e, neck, s + e - el])
    q = np.clip(q, lo, hi)
    prior = q.copy()
    target = _direction(az, el)

    free = [0, 1, 2, 3, 4]           # all joints; priors keep the expressive posture
    w_prior = np.array([0.02, 0.05, 0.1, 0.05, 0.05])
    for _ in range(iters):
        scratch.qpos[qadr] = q
        mujoco.mj_kinematics(model, scratch)
        cur = scratch.xmat[emitter_id].reshape(3, 3)[:, 0].copy()  # copy: xmat is a live view
        err = target - cur
        if np.linalg.norm(err) < 1e-4:
            break
        J = np.zeros((3, len(free)))
        for k, i in enumerate(free):
            dq = q.copy()
            dq[i] += 1e-5
            scratch.qpos[qadr] = dq
            mujoco.mj_kinematics(model, scratch)
            J[:, k] = (scratch.xmat[emitter_id].reshape(3, 3)[:, 0] - cur) / 1e-5
        # Damped least squares with a pull back toward the expressive posture.
        A = J.T @ J + np.diag(w_prior + 1e-3)
        b = J.T @ err + w_prior * (prior[free] - q[free])
        q[free] += np.linalg.solve(A, b)
        q = np.clip(q, lo, hi)
    return q
