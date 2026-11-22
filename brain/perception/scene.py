"""Decides WHEN to take a closer look at the scene, then asks Claude what objects it sees.

What this file does, in plain words:
    Asking Claude about a photo takes 1-2 seconds and costs money, so the lamp
    can't do it for every frame. This file watches the camera cheaply on the
    laptop and only sends a photo when something worth looking at happens:

      * only while someone is engaged with the lamp (nothing is ever sent
        when the lamp is idle),
      * once when someone first engages (a "baseline" look at the room),
      * then only when the picture CHANGES a lot and then HOLDS STILL for a
        moment, which is what happens when someone holds up an object,
      * at most one request at a time, and at least 4 seconds apart.

    Claude replies with a list of objects (name, description, where it is,
    and whether the person is showing it), which goes into the lamp's memory.

How it notices a change without sending anything:
    It keeps a tiny 64x48 greyscale copy of the picture and counts how many of
    its pixels changed. Your face is blacked out first, so talking or moving
    your head doesn't count as a change.

What gets sent to the cloud: one small JPEG (512 px wide, ~30-50 KB) per look.
The instructions ask for objects only and forbid describing or identifying people.
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

THUMB = (64, 48)         # size of the tiny comparison picture
PIXEL_DELTA = 25         # a thumbnail pixel counts as "changed" if its brightness moved by more than this (0-255)
CHANGE_FRAC = 0.02       # more than 2% of pixels changed = something new appeared (a mug is ~3% of the picture)
SETTLE_S = 0.7           # the change must hold still this long before we look (so the photo isn't blurry mid-motion)
MIN_GAP_S = 4.0          # never look more often than once every 4 seconds
# There's no "look every N seconds" timer on purpose. An earlier version looked
# every 30 s and took 52 photos in one session, filling memory with background clutter.

# The "form" Claude fills in for each look. Making Claude answer through a tool
# with a fixed shape (a "schema") means we always get clean, structured data
# back instead of free text we'd have to pick apart.
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
# The instruction sent with each photo.
SCENE_PROMPT = ("You are the eyes of a desk lamp robot. List the notable objects in this "
                "webcam image, especially anything a person is holding up or showing. "
                "Do not describe, identify or guess anything about people themselves, "
                "including their clothing or accessories.")


def _changed(a: np.ndarray, b: np.ndarray) -> float:
    """What fraction of the two thumbnails' pixels are noticeably different (0-1)."""
    # int16 so subtracting two brightness values can go negative without wrapping around.
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)) > PIXEL_DELTA))


def encode_jpeg(frame: np.ndarray, width: int = 512, quality: int = 80) -> str:
    """Shrink a frame, compress it as a JPEG, and turn it into text (base64) for the API."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def image_block(frame: np.ndarray, width: int = 512) -> dict:
    """Wrap a frame in the format the Claude API expects for an image.
    Also used by conversation.py ("what's this?") and goals.py."""
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": encode_jpeg(frame, width=width)}}


class SceneWatcher:
    """Watches for meaningful changes and sends a photo to Claude when one happens."""

    def __init__(self, client, model: str, memory, on_new: Callable[[list], None] | None = None):
        # client = the Anthropic API client, memory = where objects are saved,
        # on_new = called when someone shows the lamp a new object (so it can react).
        self.client, self.model, self.memory, self.on_new = client, model, memory, on_new
        self._ref: np.ndarray | None = None      # thumbnail from the last look (None = haven't looked yet)
        self._changed_at: float | None = None    # when the current change started (None = no change)
        self._last_look = 0.0                    # when we last sent a photo
        self._busy = False                       # True while waiting for Claude's answer
        self.looks = 0                           # how many looks so far (printed on exit)
        self.last_ms = 0.0                       # how long the last look took

    def offer(self, frame: np.ndarray, face_box, engaged: bool) -> None:
        """Called for every camera frame. Cheap, unless it decides it's time to look."""
        if not engaged or self._busy or frame is None:
            self._changed_at = None
            return
        now = time.monotonic()
        thumb = self._thumb(frame, face_box)
        # How different is the scene from the last time we looked?
        diff = 1.0 if self._ref is None else _changed(thumb, self._ref)
        if diff > CHANGE_FRAC:
            if self._changed_at is None:
                # A change just started: note when, and what it looks like right now.
                self._changed_at, self._settle_ref = now, thumb
            elif _changed(thumb, self._settle_ref) > CHANGE_FRAC / 2:
                # Still moving (the object is being waved around): restart the wait.
                self._changed_at, self._settle_ref = now, thumb
        settled = self._changed_at is not None and now - self._changed_at >= SETTLE_S
        baseline = self._ref is None          # never looked yet this session
        if (settled or baseline) and now - self._last_look >= MIN_GAP_S:
            self.look(frame, reason="change" if settled else "baseline", thumb=thumb)

    def look(self, frame: np.ndarray, reason: str = "asked", thumb=None) -> None:
        """Send this frame to Claude on a background thread (so the camera loop never waits)."""
        self._busy, self._last_look, self._changed_at = True, time.monotonic(), None
        # Remember what the scene looked like now, to compare future frames against.
        self._ref = thumb if thumb is not None else self._thumb(frame, None)
        threading.Thread(target=self._describe, args=(frame.copy(), reason), daemon=True).start()

    def _describe(self, frame: np.ndarray, reason: str) -> None:
        """Background thread: ask Claude what's in the photo and save the answer to memory."""
        t0 = time.perf_counter()
        try:
            # tool_choice forces Claude to answer by filling in SCENE_TOOL.
            resp = self.client.messages.create(
                model=self.model, max_tokens=600, tools=[SCENE_TOOL],
                tool_choice={"type": "tool", "name": "record_scene"},
                messages=[{"role": "user", "content": [image_block(frame),
                                                       {"type": "text", "text": SCENE_PROMPT}]}])
            objs = next((b.input.get("objects", []) for b in resp.content if b.type == "tool_use"), [])
            self.last_ms = (time.perf_counter() - t0) * 1000
            self.looks += 1
            new = self.memory.update(objs)       # returns only objects it hadn't seen before
            log.info("look (%s) %.0f ms: %s | new: %s", reason, self.last_ms,
                     [o.get("name") for o in objs], [n.name for n in new])
            # React only to NEW objects the person is SHOWING (not background clutter).
            shown_new = [n for n in new if n.shown]
            if shown_new and self.on_new:
                self.on_new(shown_new)
        except Exception:                      # noqa: BLE001
            # A failed look (network hiccup, etc.) is logged and skipped; the lamp carries on.
            log.exception("scene look failed")
        finally:
            self._busy = False

    @staticmethod
    def _thumb(frame: np.ndarray, face_box) -> np.ndarray:
        """Make the tiny greyscale comparison picture, with the face blacked out."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if face_box:
            x, y, w, h = face_box
            gray = gray.copy()
            gray[max(0, y):y + h, max(0, x):x + w] = 0       # ignore the face itself
        return cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA)
