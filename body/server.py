"""Body process: simulation + WebSocket protocol server.

    mjpython -m body.server            # macOS, with 3D window
    python -m body.server              # Linux, with 3D window
    python -m body.server --headless   # no window (tests, CI, SSH)

Threading model: the main thread owns the simulation (MjData is never touched
by any other thread). Connection threads only parse JSON and enqueue; the main
loop drains the queue once per 10 ms control tick, executes, and replies.
See docs/PROTOCOL.md for the message format.
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from body.executor import Executor
from body.lamp_body import LampBody

PROTOCOL_VERSION = 1
STATE_HZ = 20
log = logging.getLogger("body")


class BodyServer:
    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self.inbox: queue.Queue[tuple[ServerConnection, dict, float]] = queue.Queue()
        self.clients: set[ServerConnection] = set()
        self.lock = threading.Lock()
        self.body = LampBody()
        self.exec = Executor(self.body)

    # --------------------------------------------------- network threads
    def _handler(self, conn: ServerConnection) -> None:
        with self.lock:
            self.clients.add(conn)
        log.info("brain connected from %s", conn.remote_address)
        self._send(conn, {"type": "hello", "v": PROTOCOL_VERSION,
                          "capabilities": self.exec.capabilities()})
        try:
            for raw in conn:
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        raise ValueError("message must be a JSON object")
                except ValueError as exc:
                    self._send(conn, {"type": "ack", "id": None, "ok": False, "error": str(exc)})
                    continue
                self.inbox.put((conn, msg, time.perf_counter()))
        except ConnectionClosed:
            pass
        finally:
            with self.lock:
                self.clients.discard(conn)
            log.info("brain disconnected")

    def _send(self, conn: ServerConnection, msg: dict) -> None:
        try:
            conn.send(json.dumps(msg))
        except ConnectionClosed:
            pass

    def _broadcast(self, msg: dict) -> None:
        with self.lock:
            clients = list(self.clients)
        for c in clients:
            self._send(c, msg)

    # --------------------------------------------------- main loop
    def _drain(self, block_until: float | None) -> None:
        """Execute queued commands; optionally keep waiting until a deadline."""
        while True:
            try:
                if block_until is None:
                    conn, msg, t_in = self.inbox.get_nowait()
                else:
                    wait = block_until - time.perf_counter()
                    if wait <= 0:
                        return
                    conn, msg, t_in = self.inbox.get(timeout=wait)
            except queue.Empty:
                return
            ack = self.exec.handle(msg)
            ack["body_ms"] = round((time.perf_counter() - t_in) * 1000, 2)
            self._send(conn, ack)
            if not ack.get("ok"):
                log.warning("rejected %s: %s", msg, ack.get("error"))

    def run(self, headless: bool) -> None:
        server = serve(self._handler, self.host, self.port)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        log.info("body listening on ws://%s:%d", self.host, self.port)

        viewer = None
        if not headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.body.model, self.body.data)
            viewer.cam.lookat[:] = [-0.05, 0, 0.35]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.5, -30, -12

        dt = self.body.control_dt
        state_every = max(1, round(1 / (STATE_HZ * dt)))
        tick, next_t = 0, time.perf_counter()
        try:
            while viewer is None or viewer.is_running():
                self._drain(block_until=None)
                self.exec.step()
                for ev in self.exec.events:
                    self._broadcast(ev)
                self.exec.events.clear()

                tick += 1
                if tick % state_every == 0:
                    self._broadcast({"type": "state", **self.body.state()})
                if viewer is not None:
                    viewer.sync()

                # Real-time pacing. Instead of sleeping, wait on the inbox so a
                # command arriving mid-tick is acknowledged immediately.
                next_t += dt
                self._drain(block_until=next_t)
                if time.perf_counter() - next_t > 0.25:  # fell far behind: resync
                    next_t = time.perf_counter()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            if viewer is not None:
                viewer.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Lamp body process")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    BodyServer(args.host, args.port).run(args.headless)


if __name__ == "__main__":
    main()
