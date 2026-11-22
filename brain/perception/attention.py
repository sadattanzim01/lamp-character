"""Finds your face in each camera frame and decides whether you're looking at the lamp.

What this file does, in plain words:
    For every frame it answers two questions:
      1. Is someone FACING the lamp right now? (an "attention" score, 0 to 1)
      2. If so, WHERE are they? (which direction, so the lamp can turn to them)

How it detects faces:
    It uses OpenCV's built-in "Haar cascade" detectors, one for faces seen from
    the front and one for eyes. The handy part: the front-face detector only
    finds faces turned roughly toward the camera (within about 25 degrees). So
    "a face was found" already means "someone is facing me", which is exactly
    the signal we need. Finding the eyes as well makes us more confident.

Why this detector (the trade-off):
    + It ships inside OpenCV: nothing to download.
    + It's fast: about 7 ms per frame on one CPU core, fine for the target laptop.
    - It can't tell exactly where your eyes are looking, only that your face is
      turned toward the camera. A face-landmark model (like MediaPipe) could
      estimate head angle, but needs a model download and its API recently
      changed. Because all the face logic lives in this one file, swapping
      detectors later would only touch this file.

Privacy: this is the only file that ever looks at the camera pixels. It passes
just a handful of numbers on to the rest of the brain.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

PROC_WIDTH = 320          # shrink frames to 320 px wide before detecting: much faster, still accurate enough
MIN_FACE_FRAC = 0.10      # ignore faces narrower than 10% of the frame (people far in the background)
HFOV_DEG = 65.0           # how wide a typical laptop webcam sees, left edge to right edge, in degrees


@dataclass
class Observation:
    """What the camera saw in one frame."""
    t: float                  # when (wall-clock seconds)
    face: bool = False        # was a face found?
    frontal: bool = False     # was it facing the camera? (same as face for this detector)
    eyes: int = 0             # how many eyes were found inside the face (0, 1 or 2)
    azimuth: float = 0.0      # left/right direction to the face in radians (+ = lamp's left)
    elevation: float = 0.0    # up/down direction to the face in radians (+ = up)
    size: float = 0.0         # face width as a fraction of the frame (bigger = closer)
    attention: float = 0.0    # 0 = no one looking, ~1 = clearly looking at the lamp
    box: tuple[int, int, int, int] | None = None   # face rectangle (x, y, width, height) in pixels
    ms: float = 0.0           # how long the detection took, in milliseconds


class HaarAttention:
    """Face + eye detector. Call observe(frame) for each new camera frame."""

    def __init__(self) -> None:
        # Load the two detectors that come bundled with opencv-python 4.x.
        base = cv2.data.haarcascades
        self.face = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self.eyes = cv2.CascadeClassifier(base + "haarcascade_eye_tree_eyeglasses.xml")  # works with glasses too
        if self.face.empty() or self.eyes.empty():
            # OpenCV 5 stopped bundling these files, which is why requirements pin 4.x.
            raise RuntimeError("OpenCV Haar cascades missing (needs opencv-python 4.x)")
        self._last_box: tuple[int, int, int, int] | None = None   # reserved for face tracking (not used yet)

    def observe(self, frame_bgr: np.ndarray) -> Observation:
        """Look for a face in one frame and return what was found."""
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        # Prepare the image: shrink it, make it greyscale, and even out the
        # brightness (equalizeHist) so detection works in dim or bright rooms.
        scale = PROC_WIDTH / w
        small = cv2.resize(frame_bgr, (PROC_WIDTH, int(h * scale)))
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        min_px = int(PROC_WIDTH * MIN_FACE_FRAC)
        # Find faces. minNeighbors=6 means "only trust a face that's detected
        # several times at slightly different sizes": fewer false alarms.
        faces = self.face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6,
                                           minSize=(min_px, min_px))
        obs = Observation(t=time.time())
        if len(faces):
            # If there are several people, use the largest face (the closest person).
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            # Look for eyes only in the top part of the face, where eyes are.
            roi = gray[y : y + fh // 2 + fh // 8, x : x + fw]
            eyes = self.eyes.detectMultiScale(roi, 1.1, 4, minSize=(fw // 8, fw // 8))
            # Centre of the face, in full-size pixels.
            cx, cy = (x + fw / 2) / scale, (y + fh / 2) / scale
            # Turn that pixel position into a direction (angles), using the
            # camera's field of view. f is the camera's "focal length" in pixels.
            f = (w / 2) / math.tan(math.radians(HFOV_DEG) / 2)
            obs.face = obs.frontal = True
            obs.eyes = min(2, len(eyes))
            # A face on the left of the picture is on the lamp's left, because
            # the lamp and the camera face the same way.
            obs.azimuth = math.atan2(w / 2 - cx, f)
            obs.elevation = math.atan2(h / 2 - cy, f)
            obs.size = fw / PROC_WIDTH
            obs.box = (int(x / scale), int(y / scale), int(fw / scale), int(fh / scale))
            # Attention score: a face facing us = 0.65; each visible eye adds 0.175
            # (so a face with both eyes = 1.0).
            obs.attention = 0.65 + 0.175 * obs.eyes
        obs.ms = (time.perf_counter() - t0) * 1000
        return obs


def image_to_camera_angles(x: float, y: float, aspect: float = 0.75) -> tuple[float, float]:
    """Turn a spot in the picture into a direction from the camera.

    x, y are positions as fractions of the picture: (0, 0) = top-left corner,
    (0.5, 0.5) = centre, (1, 1) = bottom-right. Returns (azimuth, elevation)
    in radians, with + azimuth = left side of the picture (the lamp's left).
    `aspect` = picture height / width (480 / 640 = 0.75).
    """
    t = math.tan(math.radians(HFOV_DEG) / 2)
    return math.atan((0.5 - x) * 2 * t), math.atan((0.5 - y) * 2 * t * aspect)


def camera_angles_to_image(az: float, el: float, aspect: float = 0.75) -> tuple[float, float]:
    """The reverse: which spot in the picture a direction points at.
    Used to draw where the lamp is aiming on the camera window."""
    t = math.tan(math.radians(HFOV_DEG) / 2)
    return 0.5 - math.tan(az) / (2 * t), 0.5 - math.tan(el) / (2 * t * aspect)


def draw(frame: np.ndarray, obs: Observation, state: str, aim=None, target=None) -> np.ndarray:
    """Draw the debug overlay for the --show camera window.

    Draws, on a copy of the frame:
      * a red box around a goal's target object (sized to the object), with its name
      * a yellow circle where the lamp is actually pointing (`aim`)
      * a box around your face: green if clearly looking, blue if less sure
      * a status line: engagement state, attention score, eyes found, detection time
    aim and target positions are fractions of the picture (0-1), like above.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    if target is not None:
        name, (cx, cy) = target[0], target[1]
        bw, bh = target[2] if len(target) > 2 and target[2] else (0.0, 0.0)
        tx, ty = int(cx * w), int(cy * h)
        if bw > 0.02 and bh > 0.02:
            # We know the object's size: draw a box that fits it.
            x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(out, name, (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # No size known: just mark the centre with a small square.
            cv2.drawMarker(out, (tx, ty), (0, 0, 255), cv2.MARKER_SQUARE, 40, 2)
            cv2.putText(out, name, (tx + 24, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if aim is not None and 0 <= aim[0] <= 1 and 0 <= aim[1] <= 1:
        # Where the lamp is pointing (only drawn if it's inside the picture).
        ax, ay = int(aim[0] * w), int(aim[1] * h)
        cv2.circle(out, (ax, ay), 26, (0, 220, 255), 2)
        cv2.drawMarker(out, (ax, ay), (0, 220, 255), cv2.MARKER_CROSS, 18, 2)
    if obs.box:
        x, y, w, h = obs.box
        color = (80, 200, 80) if obs.attention >= 0.8 else (60, 180, 230)   # (colours are Blue, Green, Red)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    cv2.putText(out, f"{state}  att={obs.attention:.2f}  eyes={obs.eyes}  {obs.ms:.1f}ms",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
