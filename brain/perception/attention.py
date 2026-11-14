"""Per-frame attention estimate from the laptop camera.

Signal: is someone *facing* the lamp, and where are they?

Detector: OpenCV Haar cascades (frontal face + eyes). The frontal-face model
only fires when a face is turned roughly towards the camera (about +/-25 deg
yaw), which is the "looking toward the character" cue we need, and it ships
inside opencv-python: no model download, ~3-8 ms per frame at 320 px on one
CPU core. Tradeoff: it gives no precise head pose or eye gaze, so attention is
"facing the camera" rather than "eyes on the lamp". A landmark model
(MediaPipe Face Landmarker) would add head pose at the cost of a model
download and a dependency whose API changed recently; the Detector interface
keeps that swap local to this file.

Only this module ever sees pixels. It outputs a few numbers per frame.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

PROC_WIDTH = 320          # frames are downscaled to this width for detection
MIN_FACE_FRAC = 0.10      # ignore faces narrower than 10% of the frame (far away)
HFOV_DEG = 65.0           # typical laptop webcam horizontal field of view


@dataclass
class Observation:
    t: float
    face: bool = False
    frontal: bool = False     # frontal face found
    eyes: int = 0             # eyes found inside the face (0-2)
    azimuth: float = 0.0      # rad, + = lamp's left (camera image left)
    elevation: float = 0.0    # rad, + = up, relative to camera axis
    size: float = 0.0         # face width / frame width
    attention: float = 0.0    # 0..1 instantaneous score
    box: tuple[int, int, int, int] | None = None   # in full-frame pixels
    ms: float = 0.0           # detection time


class HaarAttention:
    def __init__(self) -> None:
        base = cv2.data.haarcascades
        self.face = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self.eyes = cv2.CascadeClassifier(base + "haarcascade_eye_tree_eyeglasses.xml")
        if self.face.empty() or self.eyes.empty():
            raise RuntimeError("OpenCV Haar cascades missing (needs opencv-python 4.x)")
        self._last_box: tuple[int, int, int, int] | None = None

    def observe(self, frame_bgr: np.ndarray) -> Observation:
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        scale = PROC_WIDTH / w
        small = cv2.resize(frame_bgr, (PROC_WIDTH, int(h * scale)))
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        min_px = int(PROC_WIDTH * MIN_FACE_FRAC)
        faces = self.face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6,
                                           minSize=(min_px, min_px))
        obs = Observation(t=time.time())
        if len(faces):
            # Track the largest face (closest person); prefer the one near last.
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            roi = gray[y : y + fh // 2 + fh // 8, x : x + fw]   # upper half: eyes
            eyes = self.eyes.detectMultiScale(roi, 1.1, 4, minSize=(fw // 8, fw // 8))
            cx, cy = (x + fw / 2) / scale, (y + fh / 2) / scale
            f = (w / 2) / math.tan(math.radians(HFOV_DEG) / 2)
            obs.face = obs.frontal = True
            obs.eyes = min(2, len(eyes))
            obs.azimuth = math.atan2(w / 2 - cx, f)     # image left -> lamp's left
            obs.elevation = math.atan2(h / 2 - cy, f)
            obs.size = fw / PROC_WIDTH
            obs.box = (int(x / scale), int(y / scale), int(fw / scale), int(fh / scale))
            # Frontal face is the main cue; visible eyes raise confidence.
            obs.attention = 0.65 + 0.175 * obs.eyes
        obs.ms = (time.perf_counter() - t0) * 1000
        return obs


def image_to_camera_angles(x: float, y: float, aspect: float = 0.75) -> tuple[float, float]:
    """Normalised image position (0..1) -> (azimuth, elevation) from the camera axis.
    Azimuth + = image left (the lamp's left, since lamp and camera face the same way)."""
    t = math.tan(math.radians(HFOV_DEG) / 2)
    return math.atan((0.5 - x) * 2 * t), math.atan((0.5 - y) * 2 * t * aspect)


def camera_angles_to_image(az: float, el: float, aspect: float = 0.75) -> tuple[float, float]:
    """Inverse of image_to_camera_angles."""
    t = math.tan(math.radians(HFOV_DEG) / 2)
    return 0.5 - math.tan(az) / (2 * t), 0.5 - math.tan(el) / (2 * t * aspect)


def draw(frame: np.ndarray, obs: Observation, state: str, aim=None, target=None) -> np.ndarray:
    """Debug overlay for the --show window. aim/target: normalised image points."""
    out = frame.copy()
    h, w = out.shape[:2]
    if target is not None:
        name, (cx, cy) = target[0], target[1]
        bw, bh = target[2] if len(target) > 2 and target[2] else (0.0, 0.0)
        tx, ty = int(cx * w), int(cy * h)
        if bw > 0.02 and bh > 0.02:                       # box sized to the object
            x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(out, name, (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.drawMarker(out, (tx, ty), (0, 0, 255), cv2.MARKER_SQUARE, 40, 2)
            cv2.putText(out, name, (tx + 24, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if aim is not None and 0 <= aim[0] <= 1 and 0 <= aim[1] <= 1:
        ax, ay = int(aim[0] * w), int(aim[1] * h)
        cv2.circle(out, (ax, ay), 26, (0, 220, 255), 2)       # where the lamp points
        cv2.drawMarker(out, (ax, ay), (0, 220, 255), cv2.MARKER_CROSS, 18, 2)
    if obs.box:
        x, y, w, h = obs.box
        color = (80, 200, 80) if obs.attention >= 0.8 else (60, 180, 230)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    cv2.putText(out, f"{state}  att={obs.attention:.2f}  eyes={obs.eyes}  {obs.ms:.1f}ms",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
