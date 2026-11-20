"""Works out which joint angles make the lamp look in a given direction.

What this file does, in plain words:
    The brain never says "set the shoulder to 0.5 radians". It says "look
    over there", as a direction. This file turns that direction into the 5
    joint angles that make the shade actually point there. In robotics this is
    called inverse kinematics (IK): going from "where I want to point" back
    to "how to bend my joints".

How a direction is described (in the lamp's own point of view):
    azimuth   = left/right. 0 = straight ahead, positive = toward the lamp's left
    elevation = up/down.    0 = level/horizontal, positive = up
    (Both are angles in radians. 1 radian is about 57 degrees.)

How it solves it, in three steps:
    1. Pick a natural-looking starting posture: lean forward to look down,
       stand tall to look up. Let the neck do part of a sideways turn so it
       reads like a glance, and turn the base for the rest.
    2. Measure where that posture really points, using the actual 3D model.
    3. Nudge the joints a little at a time until the shade points exactly at
       the target, while staying as close as possible to the natural posture
       from step 1 (so it still looks like a character, not a machine).
"""
from __future__ import annotations

import mujoco
import numpy as np

# Natural arm postures for different up/down gaze angles.
# Each row: (gaze elevation, shoulder angle, elbow angle), all in radians.
# For an in-between elevation we blend the two nearest rows.
# These come from the hand-tuned poses in lamp_body.POSES.
_POSTURE = np.array([
    [-1.05, -0.20, -0.90],   # looking far down at the desk: lean forward
    [-0.65,  0.20, -0.90],   # looking down a bit
    [-0.15,  0.50, -1.20],   # looking at a person
    [ 0.25,  0.75, -1.35],   # looking up: stand tall
])
NECK_SHARE = 0.35          # the neck takes 35% of a left/right turn, the base the rest
NECK_MAX = 0.50            # but the neck never turns more than 0.5 rad (~29 degrees)
HEAD_SAFE_MAX = -0.05      # tipping the head down past this hits the upper arm (see lamp_body.py)
EL_RANGE = (-1.05, 0.22)   # the up/down range the lamp can reach without the shade hitting the arm


def gaze_of(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    """Where the shade is pointing right now, as (azimuth, elevation)."""
    x = data.xmat[body_id].reshape(3, 3)[:, 0]   # the shade's "forward" direction in 3D
    return np.array([np.arctan2(-x[1], -x[0]), np.arcsin(np.clip(x[2], -1, 1))])


def _direction(az: float, el: float) -> np.ndarray:
    """Turn (azimuth, elevation) into a 3D arrow of length 1 in world coordinates.
    The minus signs are because the lamp's "straight ahead" is the world's -x."""
    return np.array([-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)])


def solve_gaze(model, scratch, qadr, lo, hi, emitter_id, az, el, iters=30):
    """Find the 5 joint angles [base, shoulder, elbow, neck, head] that point the
    shade at (az, el).

    model, scratch  the MuJoCo model and a throwaway state copy to experiment on
    qadr            where each joint's angle lives in MuJoCo's arrays
    lo, hi          each joint's safe range
    emitter_id      the light's frame (we aim this frame's forward direction)
    iters           max refinement steps (it usually finishes much sooner)
    """
    # Keep the target inside what the lamp can physically reach, and never let
    # the head tip down far enough to hit the arm.
    el = float(np.clip(el, *EL_RANGE))
    hi = hi.copy()
    hi[4] = min(hi[4], HEAD_SAFE_MAX)

    # Step 1: natural starting posture.
    s = np.interp(el, _POSTURE[:, 0], _POSTURE[:, 1])   # shoulder, blended from the table
    e = np.interp(el, _POSTURE[:, 0], _POSTURE[:, 2])   # elbow, blended from the table
    neck = float(np.clip(NECK_SHARE * az, -NECK_MAX, NECK_MAX))
    # Base does the rest of the turn; the head angle makes up the remaining tilt
    # (roughly: gaze elevation = shoulder + elbow - head).
    q = np.array([az - neck, s, e, neck, s + e - el])
    q = np.clip(q, lo, hi)
    prior = q.copy()            # remember this natural posture
    target = _direction(az, el)

    # Steps 2-3: refine with "damped least squares", a standard IK method.
    free = [0, 1, 2, 3, 4]      # all 5 joints may be adjusted
    # How strongly each joint is pulled back toward the natural posture.
    # Small numbers = that joint is free to change; bigger = it prefers to stay.
    w_prior = np.array([0.02, 0.05, 0.1, 0.05, 0.05])
    for _ in range(iters):
        # Where does the shade point with the current guess?
        scratch.qpos[qadr] = q
        mujoco.mj_kinematics(model, scratch)
        # .copy() matters: MuJoCo's array gets overwritten by the next calculation.
        cur = scratch.xmat[emitter_id].reshape(3, 3)[:, 0].copy()
        err = target - cur
        if np.linalg.norm(err) < 1e-4:
            break               # close enough: pointing at the target
        # Build the "Jacobian" J: how much the pointing direction changes when
        # each joint moves a tiny bit. We measure it by actually nudging each
        # joint by 0.00001 rad and seeing what happens.
        J = np.zeros((3, len(free)))
        for k, i in enumerate(free):
            dq = q.copy()
            dq[i] += 1e-5
            scratch.qpos[qadr] = dq
            mujoco.mj_kinematics(model, scratch)
            J[:, k] = (scratch.xmat[emitter_id].reshape(3, 3)[:, 0] - cur) / 1e-5
        # Solve for the joint change that best reduces the pointing error, while
        # the w_prior terms gently pull back toward the natural posture. The tiny
        # 1e-3 "damping" keeps the steps stable when the math gets touchy.
        A = J.T @ J + np.diag(w_prior + 1e-3)
        b = J.T @ err + w_prior * (prior[free] - q[free])
        q[free] += np.linalg.solve(A, b)
        q = np.clip(q, lo, hi)  # never leave the safe ranges
    return q
