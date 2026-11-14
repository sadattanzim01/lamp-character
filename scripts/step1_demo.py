"""Step 1 check: the lamp wakes up, looks around, and goes back to sleep.

Live window (macOS needs mjpython, which ships with the mujoco package):
    mjpython scripts/step1_demo.py
Headless (Linux / CI), saves a GIF and prints tracking + torque stats:
    MUJOCO_GL=egl python scripts/step1_demo.py --record out.gif
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from body.lamp_body import LampBody  # noqa: E402

WARM = [1.0, 0.82, 0.55]
COOL = [0.55, 0.75, 1.0]
HAPPY = [1.0, 0.6, 0.25]

# (time_s, action, args) — a tiny scripted "performance"
SCRIPT = [
    (0.5, "light", (WARM, 0.15, 1.0)),          # dim glow while asleep
    (2.0, "pose", ("rest", 0.4)),               # slowly stir
    (2.8, "light", (WARM, 0.6, 0.8)),
    (4.0, "pose", ("perk_up", 1.0)),            # notice someone: fast perk-up
    (4.0, "light", (HAPPY, 1.0, 0.3)),
    (5.5, "pose", ("attentive", 0.7)),
    (6.5, "goal", ([0.5, 0.5, -1.2, 0.3, -0.45], 0.6)),   # look left
    (8.0, "goal", ([-0.5, 0.5, -1.2, -0.3, -0.45], 0.6)), # look right
    (9.5, "pose", ("curious", 0.6)),
    (9.5, "light", (COOL, 0.8, 0.6)),
    (11.0, "pose", ("look_down", 0.5)),
    (12.5, "pose", ("sleep", 0.3)),              # lose interest, settle
    (12.5, "light", (WARM, 0.1, 2.0)),
]
DURATION = 16.0


def run(record: str | None) -> None:
    body = LampBody()
    events = list(SCRIPT)
    frames, peaks = [], {}
    viewer = renderer = cam = None

    if record:
        import mujoco
        renderer = mujoco.Renderer(body.model, 360, 480)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [-0.05, 0, 0.35]
        cam.distance, cam.azimuth, cam.elevation = 1.5, -30, -12
    else:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(body.model, body.data)
        viewer.cam.lookat[:] = [-0.05, 0, 0.35]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.5, -30, -12

    t0 = time.perf_counter()
    while body.data.time < DURATION:
        while events and events[0][0] <= body.data.time:
            _, kind, args = events.pop(0)
            if kind == "pose":
                body.set_pose(*args)
            elif kind == "goal":
                body.set_goal(*args)
            elif kind == "light":
                body.set_light(*args)
        body.step()

        st = body.state()
        for j, f in st["torque_frac"].items():
            peaks.setdefault(j, [0.0, 0.0])
            peaks[j][0] = max(peaks[j][0], f)
        for j, v in st["qd"].items():
            peaks[j][1] = max(peaks[j][1], abs(v))

        if record:
            if int(round(body.data.time / body.control_dt)) % 5 == 0:  # 20 fps
                renderer.update_scene(body.data, cam)
                frames.append(renderer.render())
        else:
            if not viewer.is_running():
                break
            viewer.sync()
            # real-time pacing
            lag = body.data.time - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)

    print("joint                 peak torque/effort   peak |vel| (rad/s)  vel limit")
    for i, (j, (tf, v)) in enumerate(peaks.items()):
        print(f"{j:22s} {tf:10.2f}            {v:8.2f}           {body.vmax[i]:.2f}")

    if record and frames:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(record, save_all=True, append_images=imgs[1:], duration=50, loop=0)
        print(f"saved {record} ({len(imgs)} frames)")
    if viewer:
        viewer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--record", help="render headless to this GIF instead of opening a window")
    run(p.parse_args().record)
