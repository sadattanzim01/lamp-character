"""Brain-side client for the body protocol (see docs/PROTOCOL.md).

Every command returns its ack immediately (the body validated it and may have
adjusted it); motion commands can then be awaited until "completed" or
"preempted". The latest body state is always available as `client.state`.
"""
from __future__ import annotations

import itertools
import json
import threading
import time
from concurrent.futures import Future

from websockets.sync.client import connect


class BodyError(RuntimeError):
    pass


class BodyClient:
    def __init__(self, url: str = "ws://127.0.0.1:8765", timeout: float = 5.0):
        self.ws = connect(url, open_timeout=timeout)
        hello = json.loads(self.ws.recv(timeout=timeout))
        if hello.get("type") != "hello":
            raise BodyError(f"expected hello, got {hello}")
        self.capabilities = hello["capabilities"]
        self.state: dict = {}
        self.timeout = timeout
        self._ids = itertools.count(1)
        self._acks: dict[str, Future] = {}
        self._dones: dict[str, Future] = {}
        self._lock = threading.Lock()
        self.rtt_ms: list[float] = []
        threading.Thread(target=self._reader, daemon=True).start()

    # ---------------------------------------------------------------- io
    def _reader(self) -> None:
        for raw in self.ws:
            msg = json.loads(raw)
            kind, cid = msg.get("type"), msg.get("id")
            if kind == "state":
                self.state = msg
            elif kind == "ack":
                with self._lock:
                    fut = self._acks.pop(cid, None)
                if fut:
                    fut.set_result(msg)
            elif kind == "done":
                with self._lock:
                    fut = self._dones.pop(cid, None)
                if fut:
                    fut.set_result(msg["status"])

    def send(self, msg: dict, motion: bool = False) -> tuple[dict, Future | None]:
        cid = f"c{next(self._ids)}"
        msg = {**msg, "id": cid}
        ack_f: Future = Future()
        done_f: Future | None = Future() if motion else None
        with self._lock:
            self._acks[cid] = ack_f
            if done_f:
                self._dones[cid] = done_f
        t0 = time.perf_counter()
        self.ws.send(json.dumps(msg))
        ack = ack_f.result(self.timeout)
        self.rtt_ms.append((time.perf_counter() - t0) * 1000)
        if not ack.get("ok"):
            with self._lock:
                self._dones.pop(cid, None)
            raise BodyError(ack.get("error", "rejected"))
        return ack, done_f

    # ---------------------------------------------------------- commands
    def pose(self, name: str, speed: float = 0.7, wait: bool = False):
        return self._motion({"type": "pose", "name": name, "speed": speed}, wait)

    def gaze(self, azimuth: float, elevation: float, speed: float = 0.7, wait: bool = False):
        return self._motion({"type": "gaze", "azimuth": azimuth,
                             "elevation": elevation, "speed": speed}, wait)

    def gesture(self, name: str, intensity: float = 1.0, wait: bool = False):
        return self._motion({"type": "gesture", "name": name, "intensity": intensity}, wait)

    def joints(self, q: dict[str, float], speed: float = 0.7, wait: bool = False):
        return self._motion({"type": "joints", "q": q, "speed": speed}, wait)

    def light(self, rgb, intensity: float, fade: float = 0.4) -> dict:
        ack, _ = self.send({"type": "light", "rgb": list(rgb), "intensity": intensity,
                            "fade": fade})
        return ack

    def stop(self) -> dict:
        ack, _ = self.send({"type": "stop"})
        return ack

    def _motion(self, msg: dict, wait: bool):
        ack, done = self.send(msg, motion=True)
        if wait:
            return ack, done.result(timeout=30)
        return ack, done

    def close(self) -> None:
        self.ws.close()
