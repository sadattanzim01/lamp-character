"""Scene watcher: decides *when* to look closely, then asks Claude what it sees.

Cloud vision is slow (~1-2 s) and costs tokens, so it is event-driven:
  * only while someone is engaged with the lamp (nothing is sent when idle),
  * when the image changed a lot and then held still for a moment
    (someone holds an object up), or every 30 s as a refresh,
  * at most one request in flight, and at least 4 s apart.

Change detection runs locally on a 64x48 greyscale thumbnail with the
person's face masked out, so talking and head movement do not trigger it.

Data sent: one JPEG (512 px wide, ~30-50 kB) per look. The prompt asks for
objects only and forbids describing or identifying people.
"""
from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger("scene")

THUMB = (64, 48)
PIXEL_DELTA = 25         # grey levels (0-255) for a thumbnail pixel to count as changed
CHANGE_FRAC = 0.02       # >2% of pixels changed = something new (a mug is ~3% of the frame)
SETTLE_S = 0.7            # change must hold still this long before we look
MIN_GAP_S = 4.0
# No periodic refresh: one baseline look when someone engages, then only on change.
# (A 30 s refresh produced 52 looks in one session and filled memory with background.)

SCENE_TOOL = {
    "name": "record_scene",
    "description": "Record the notable physical objects visible in the camera image.",
    "input_schema": {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "description": "Up to 6 distinct, notable objects. Skip walls, furniture "
                               "edges, clothing and anything part of a person.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "short common name, e.g. 'red mug'"},
                        "description": {"type": "string",
                                        "description": "colour, material, size, any readable text"},
                        "where": {"type": "string",
                                  "description": "e.g. 'held up by the person', 'on the desk, left'"},
                        "x": {"type": "number", "description": "object centre, 0=left 1=right"},
                        "y": {"type": "number", "description": "object centre, 0=top 1=bottom"},
                        "shown": {"type": "boolean",
                                  "description": "true if a person is holding it up or "
                                                 "presenting it; false for background items"},
                    },
                    "required": ["name", "description", "where", "x", "y", "shown"],
                },
            },
        },
        "required": ["objects"],
    },
}
SCENE_PROMPT = ("You are the eyes of a desk lamp robot. List the notable objects in this "
                "webcam image, especially anything a person is holding up or showing. "
                "Do not describe, identify or guess anything about people themselves, "
                "including their clothing or accessories.")


def _changed(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of thumbnail pixels that changed noticeably."""
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)) > PIXEL_DELTA))


def encode_jpeg(frame: np.ndarray, width: int = 512, quality: int = 80) -> str:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def image_block(frame: np.ndarray, width: int = 512) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": encode_jpeg(frame, width=width)}}


class SceneWatcher:
    def __init__(self, client, model: str, memory, on_new: Callable[[list], None] | None = None):
        self.client, self.model, self.memory, self.on_new = client, model, memory, on_new
        self._ref: np.ndarray | None = None      # thumbnail at the last look
        self._changed_at: float | None = None
        self._last_look = 0.0
        self._busy = False
        self.looks = 0
        self.last_ms = 0.0

    def offer(self, frame: np.ndarray, face_box, engaged: bool) -> None:
        """Called every frame from the main loop; cheap unless it decides to look."""
        if not engaged or self._busy or frame is None:
            self._changed_at = None
            return
        now = time.monotonic()
        thumb = self._thumb(frame, face_box)
        diff = 1.0 if self._ref is None else _changed(thumb, self._ref)
        if diff > CHANGE_FRAC:
            if self._changed_at is None:
                self._changed_at, self._settle_ref = now, thumb
            elif _changed(thumb, self._settle_ref) > CHANGE_FRAC / 2:
                self._changed_at, self._settle_ref = now, thumb   # still moving: wait
        settled = self._changed_at is not None and now - self._changed_at >= SETTLE_S
        baseline = self._ref is None
        if (settled or baseline) and now - self._last_look >= MIN_GAP_S:
            self.look(frame, reason="change" if settled else "baseline", thumb=thumb)

    def look(self, frame: np.ndarray, reason: str = "asked", thumb=None) -> None:
        self._busy, self._last_look, self._changed_at = True, time.monotonic(), None
        self._ref = thumb if thumb is not None else self._thumb(frame, None)
        threading.Thread(target=self._describe, args=(frame.copy(), reason), daemon=True).start()

    def _describe(self, frame: np.ndarray, reason: str) -> None:
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=600, tools=[SCENE_TOOL],
                tool_choice={"type": "tool", "name": "record_scene"},
                messages=[{"role": "user", "content": [image_block(frame),
                                                       {"type": "text", "text": SCENE_PROMPT}]}])
            objs = next((b.input.get("objects", []) for b in resp.content if b.type == "tool_use"), [])
            self.last_ms = (time.perf_counter() - t0) * 1000
            self.looks += 1
            new = self.memory.update(objs)
            log.info("look (%s) %.0f ms: %s | new: %s", reason, self.last_ms,
                     [o.get("name") for o in objs], [n.name for n in new])
            shown_new = [n for n in new if n.shown]
            if shown_new and self.on_new:
                self.on_new(shown_new)
        except Exception:                      # noqa: BLE001
            log.exception("scene look failed")
        finally:
            self._busy = False

    @staticmethod
    def _thumb(frame: np.ndarray, face_box) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if face_box:
            x, y, w, h = face_box
            gray = gray.copy()
            gray[max(0, y):y + h, max(0, x):x + w] = 0       # ignore the face itself
        return cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA)
