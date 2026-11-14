"""Camera capture on a background thread; the brain always reads the newest frame.

Dropping stale frames (instead of queueing them) keeps reaction latency bounded
when detection is slower than the camera.
"""
from __future__ import annotations

import platform
import threading
import time

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        backend = {"Darwin": cv2.CAP_AVFOUNDATION, "Linux": cv2.CAP_V4L2}.get(
            platform.system(), cv2.CAP_ANY)
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {index}. macOS: allow camera access for your "
                "terminal in System Settings > Privacy & Security > Camera. "
                "Linux: check /dev/video* and that your user is in the 'video' group.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame, self._seq = frame, self._seq + 1

    def latest(self) -> tuple[int, np.ndarray | None]:
        """(sequence number, newest frame). Same seq means no new frame yet."""
        with self._lock:
            return self._seq, self._frame

    def close(self) -> None:
        self._running = False
        self.cap.release()
