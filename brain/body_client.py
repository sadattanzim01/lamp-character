"""The brain's remote control for the lamp body.

What this file does, in plain words:
    The body runs as a separate program (body/server.py). This file is the
    brain's side of that connection: it connects over the WebSocket, sends
    commands, and listens for replies. Everywhere else in the brain, moving
    the lamp is as simple as:

        body.gesture("nod")
        body.gaze(0.3, -0.1)                 # look a bit left and slightly down
        body.light((1, 0.6, 0.3), 1.0)       # glow orange

How replies work:
    * Every command gets an immediate reply ("ack") saying whether the body
      accepted it and what, if anything, it had to change to keep it safe.
    * Movement commands also get a "done" message later, when they finish
      ("completed") or get cut off by a newer movement ("preempted"). You can
      either wait for that (wait=True) or carry on and check it later.
    * The body also sends its latest state 20 times a second; the newest copy
      is always in `client.state`.

The message format is described in docs/PROTOCOL.md.
"""
from __future__ import annotations

import itertools
import json
import threading
import time
from concurrent.futures import Future

from websockets.sync.client import connect


class BodyError(RuntimeError):
    """Raised when the body rejects a command (e.g. an unknown pose name)."""
    pass


class BodyClient:
    """Connection to the lamp body. Create one, then call its command methods."""

    def __init__(self, url: str = "ws://127.0.0.1:8765", timeout: float = 5.0):
        self.ws = connect(url, open_timeout=timeout)
        # The body always speaks first with a "hello" listing what it can do.
        hello = json.loads(self.ws.recv(timeout=timeout))
        if hello.get("type") != "hello":
            raise BodyError(f"expected hello, got {hello}")
        self.capabilities = hello["capabilities"]   # poses, gestures, joint limits, ...
        self.state: dict = {}                        # latest state from the body
        self.timeout = timeout
        self._ids = itertools.count(1)               # gives each command an id: c1, c2, c3...
        # "Futures" are placeholders for answers that haven't arrived yet.
        # We keep one per command id, and fill it in when the matching reply comes back.
        self._acks: dict[str, Future] = {}           # waiting for the immediate reply
        self._dones: dict[str, Future] = {}          # waiting for "movement finished"
        self._lock = threading.Lock()                # the two dicts are shared between threads
        self.rtt_ms: list[float] = []                # round-trip times, for the latency measurements
        # A background thread reads everything the body sends.
        threading.Thread(target=self._reader, daemon=True).start()

    # ---------------------------------------------------------------- io
    def _reader(self) -> None:
        """Background thread: sort each incoming message and wake up whoever is waiting for it."""
        for raw in self.ws:
            msg = json.loads(raw)
            kind, cid = msg.get("type"), msg.get("id")
            if kind == "state":                     # regular status update: just keep the newest
                self.state = msg
            elif kind == "ack":                     # immediate reply to command `cid`
                with self._lock:
                    fut = self._acks.pop(cid, None)
                if fut:
                    fut.set_result(msg)
            elif kind == "done":                    # movement `cid` finished or was interrupted
                with self._lock:
                    fut = self._dones.pop(cid, None)
                if fut:
                    fut.set_result(msg["status"])

    def send(self, msg: dict, motion: bool = False) -> tuple[dict, Future | None]:
        """Send one command and wait for its ack.

        Returns (ack, done_future). done_future is only set for movement
        commands; call done_future.result() to wait until the movement ends.
        Raises BodyError if the body rejects the command.
        """
        cid = f"c{next(self._ids)}"
        msg = {**msg, "id": cid}
        ack_f: Future = Future()
        done_f: Future | None = Future() if motion else None
        # Register the placeholders BEFORE sending, so a very fast reply can't be missed.
        with self._lock:
            self._acks[cid] = ack_f
            if done_f:
                self._dones[cid] = done_f
        t0 = time.perf_counter()
        self.ws.send(json.dumps(msg))
        ack = ack_f.result(self.timeout)            # wait (up to 5 s) for the reply
        self.rtt_ms.append((time.perf_counter() - t0) * 1000)
        if not ack.get("ok"):
            with self._lock:
                self._dones.pop(cid, None)          # rejected: no "done" will ever come
            raise BodyError(ack.get("error", "rejected"))
        return ack, done_f

    # ---------------------------------------------------------- commands
    # Movement commands. With wait=True they return only when the movement ends,
    # as (ack, "completed" / "preempted"). With wait=False they return at once
    # as (ack, done_future).

    def pose(self, name: str, speed: float = 0.7, wait: bool = False):
        """Go to a named pose, e.g. "sleep", "attentive", "perk_up"."""
        return self._motion({"type": "pose", "name": name, "speed": speed}, wait)

    def gaze(self, azimuth: float, elevation: float, speed: float = 0.7, wait: bool = False):
        """Point the shade in a direction (radians): azimuth + = lamp's left, elevation + = up."""
        return self._motion({"type": "gaze", "azimuth": azimuth,
                             "elevation": elevation, "speed": speed}, wait)

    def gesture(self, name: str, intensity: float = 1.0, wait: bool = False):
        """Play a gesture: "nod", "shake", "wiggle", "perk", "droop" or "tilt"."""
        return self._motion({"type": "gesture", "name": name, "intensity": intensity}, wait)

    def joints(self, q: dict[str, float], speed: float = 0.7, wait: bool = False):
        """Set specific joint angles directly, e.g. {"neck_yaw_joint": 0.3}."""
        return self._motion({"type": "joints", "q": q, "speed": speed}, wait)

    def light(self, rgb, intensity: float, fade: float = 0.4) -> dict:
        """Fade the light to a colour (r, g, b each 0-1) and brightness (0-1)."""
        ack, _ = self.send({"type": "light", "rgb": list(rgb), "intensity": intensity,
                            "fade": fade})
        return ack

    def stop(self) -> dict:
        """Stop moving and hold the current position."""
        ack, _ = self.send({"type": "stop"})
        return ack

    def _motion(self, msg: dict, wait: bool):
        """Shared helper for the movement commands above."""
        ack, done = self.send(msg, motion=True)
        if wait:
            return ack, done.result(timeout=30)
        return ack, done

    def close(self) -> None:
        """Disconnect from the body."""
        self.ws.close()
