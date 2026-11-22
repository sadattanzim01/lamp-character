"""Reads video from the webcam, always keeping just the newest picture.

What this file does, in plain words:
    A background thread grabs frames from the camera as fast as the camera
    sends them (about 30 per second) and keeps only the most recent one. The
    rest of the brain asks for "the latest frame" whenever it's ready.

Why only the newest frame:
    If the face detector is ever slower than the camera, a queue of frames
    would pile up and the lamp would react to things that happened seconds
    ago. Throwing old frames away keeps the lamp reacting to *now*.
"""
from __future__ import annotations

import platform
import threading
import time

import cv2
import numpy as np


class Camera:
    """The webcam. Use latest() to get the newest frame."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        # Pick the right camera system for the operating system:
        # AVFoundation on macOS ("Darwin"), Video4Linux on Linux, otherwise let OpenCV choose.
        backend = {"Darwin": cv2.CAP_AVFOUNDATION, "Linux": cv2.CAP_V4L2}.get(
            platform.system(), cv2.CAP_ANY)
        self.cap = cv2.VideoCapture(index, backend)   # index 0 = the built-in camera
        if not self.cap.isOpened():
            # The most common causes, spelled out so the fix is obvious.
            raise RuntimeError(
                f"Cannot open camera {index}. macOS: allow camera access for your "
                "terminal in System Settings > Privacy & Security > Camera. "
                "Linux: check /dev/video* and that your user is in the 'video' group.")
        # 640x480 is plenty for finding faces, and keeps processing fast.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame: np.ndarray | None = None   # the newest picture
        self._seq = 0                            # counts frames, so readers can tell when a new one arrived
        self._lock = threading.Lock()            # the frame is written by one thread and read by another
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        """Background thread: keep reading frames and replace the stored one."""
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)                 # camera hiccup: wait a moment and try again
                continue
            with self._lock:
                self._frame, self._seq = frame, self._seq + 1

    def latest(self) -> tuple[int, np.ndarray | None]:
        """Return (frame number, newest frame).

        If the frame number is the same as last time you asked, no new picture
        has arrived yet, so there's no need to process it again.
        """
        with self._lock:
            return self._seq, self._frame

    def close(self) -> None:
        """Stop reading and release the camera so other programs can use it."""
        self._running = False
        self.cap.release()
