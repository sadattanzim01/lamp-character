"""The body program: runs the lamp simulation and listens for commands from the brain.

How to start it:
    mjpython -m body.server            # macOS, with the 3D window
    python -m body.server              # Linux, with the 3D window
    python -m body.server --headless   # no window (tests, servers, SSH)

What this file does, in plain words:
    It opens a WebSocket (a two-way network connection) on this computer at
    ws://127.0.0.1:8765. The brain connects to it and sends commands as small
    JSON messages. This program:
      * runs the physics 100 times a second,
      * carries out each command (using executor.py),
      * replies to every command right away ("ack"),
      * tells the brain when movements finish ("done"),
      * sends the lamp's state 20 times a second ("state").
    The message format is written up in docs/PROTOCOL.md.

How the threads are organised (and why):
    The main thread is the only one allowed to touch the simulation. The
    network threads just receive messages, read the JSON, and drop them into a
    queue (the "inbox"). The main thread picks them up between physics ticks.
    That way two threads can never change the simulation at the same time, so
    there are no race conditions and no locks needed around the physics.
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

PROTOCOL_VERSION = 1   # bump this if the message format ever changes
STATE_HZ = 20          # how often the lamp's state is sent to the brain
log = logging.getLogger("body")


class BodyServer:
    """Owns the lamp simulation and the network connection to the brain."""

    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        # Commands waiting to run: (which connection sent it, the message, when it arrived).
        self.inbox: queue.Queue[tuple[ServerConnection, dict, float]] = queue.Queue()
        self.clients: set[ServerConnection] = set()   # everyone currently connected
        self.lock = threading.Lock()                  # protects the clients set only
        self.body = LampBody()
        self.exec = Executor(self.body)

    # --------------------------------------------------- network threads
    def _handler(self, conn: ServerConnection) -> None:
        """Runs on its own thread for each connected brain."""
        with self.lock:
            self.clients.add(conn)
        log.info("brain connected from %s", conn.remote_address)
        # First message: say hello and describe everything the lamp can do.
        self._send(conn, {"type": "hello", "v": PROTOCOL_VERSION,
                          "capabilities": self.exec.capabilities()})
        try:
            for raw in conn:                       # wait for each incoming message
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        raise ValueError("message must be a JSON object")
                except ValueError as exc:
                    # Not valid JSON: reply with an error and keep going.
                    self._send(conn, {"type": "ack", "id": None, "ok": False, "error": str(exc)})
                    continue
                # Hand it to the main thread; note the arrival time to measure delay.
                self.inbox.put((conn, msg, time.perf_counter()))
        except ConnectionClosed:
            pass
        finally:
            with self.lock:
                self.clients.discard(conn)
            log.info("brain disconnected")

    def _send(self, conn: ServerConnection, msg: dict) -> None:
        """Send one message, ignoring it quietly if that brain has disconnected."""
        try:
            conn.send(json.dumps(msg))
        except ConnectionClosed:
            pass

    def _broadcast(self, msg: dict) -> None:
        """Send a message to every connected brain."""
        with self.lock:
            clients = list(self.clients)   # copy, so we don't hold the lock while sending
        for c in clients:
            self._send(c, msg)

    # --------------------------------------------------- main loop
    def _drain(self, block_until: float | None) -> None:
        """Run every command waiting in the inbox.

        block_until=None  -> handle what's there now and return straight away.
        block_until=time  -> keep waiting for new commands until that time.
                             This is how the loop "sleeps" between ticks while
                             still answering a new command the instant it arrives.
        """
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
            # How long the command waited plus ran inside the body, in ms.
            ack["body_ms"] = round((time.perf_counter() - t_in) * 1000, 2)
            self._send(conn, ack)
            if not ack.get("ok"):
                log.warning("rejected %s: %s", msg, ack.get("error"))

    def run(self, headless: bool) -> None:
        """Start the network server, open the 3D window, and run the 100 Hz loop."""
        # The WebSocket server runs in a background thread.
        server = serve(self._handler, self.host, self.port)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        log.info("body listening on ws://%s:%d", self.host, self.port)

        # The 3D window (skipped with --headless), with a nice starting camera angle.
        viewer = None
        if not headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.body.model, self.body.data)
            viewer.cam.lookat[:] = [-0.05, 0, 0.35]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.5, -30, -12

        dt = self.body.control_dt                        # 0.01 s per tick
        state_every = max(1, round(1 / (STATE_HZ * dt)))  # send state every 5th tick (20 Hz)
        tick, next_t = 0, time.perf_counter()
        try:
            while viewer is None or viewer.is_running():
                # 1. Run any commands that arrived.
                self._drain(block_until=None)
                # 2. Advance the lamp by one tick.
                self.exec.step()
                # 3. Report finished/interrupted movements.
                for ev in self.exec.events:
                    self._broadcast(ev)
                self.exec.events.clear()

                # 4. Every 5th tick, send the lamp's full state.
                tick += 1
                if tick % state_every == 0:
                    self._broadcast({"type": "state", **self.body.state()})
                # 5. Redraw the 3D window.
                if viewer is not None:
                    viewer.sync()

                # 6. Wait until it's time for the next tick, so the simulation runs
                #    at real-world speed. Instead of a plain sleep, we wait on the
                #    inbox: a command that arrives mid-wait is handled immediately.
                #    (This cut the command reply time from ~10 ms to ~1 ms.)
                next_t += dt
                self._drain(block_until=next_t)
                if time.perf_counter() - next_t > 0.25:  # fell far behind (e.g. laptop slept): catch up
                    next_t = time.perf_counter()
        except KeyboardInterrupt:
            pass                                          # Ctrl+C: shut down cleanly
        finally:
            server.shutdown()
            if viewer is not None:
                viewer.close()


def main() -> None:
    """Read the command-line options and start the body."""
    p = argparse.ArgumentParser(description="Lamp body process")
    p.add_argument("--host", default="127.0.0.1")   # only this computer can connect
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    BodyServer(args.host, args.port).run(args.headless)


if __name__ == "__main__":
    main()
