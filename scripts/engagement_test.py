"""Engagement reliability test with ground-truth cues.

Runs the same perception + engagement code as the lamp (no body, no API),
while you follow on-screen / beep cues:
    LOOK  - look at the screen/camera as if talking to the lamp
    AWAY  - turn your head away (look at your phone, the side, the window)

    python scripts/engagement_test.py            # 10 LOOK/AWAY trials, ~2.5 min
    python scripts/engagement_test.py --trials 5

Stop the brain first (the camera can only be used by one program).
Results: logs/engagement_test.json, summary printed at the end.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.engagement import Engagement, State          # noqa: E402
from brain.perception.attention import HaarAttention     # noqa: E402
from brain.perception.camera import Camera               # noqa: E402

LOOK_S, AWAY_S = 6.0, 7.0
REACTION_S = 0.4          # typical human reaction to the cue, subtracted from latencies


def beep(freq: float) -> None:
    try:
        import sounddevice as sd
        t = np.arange(int(0.15 * 22050)) / 22050
        sd.play((0.25 * np.sin(2 * np.pi * freq * t)).astype(np.float32), 22050)
    except Exception:                                    # noqa: BLE001
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    cam, det, eng = Camera(args.camera), HaarAttention(), Engagement()
    phases = [("AWAY", 4.0)] + [p for _ in range(args.trials) for p in (("LOOK", LOOK_S), ("AWAY", AWAY_S))]
    log: list[dict] = []          # per frame: t, phase, attention, state
    events: list[dict] = []       # state changes
    print(f"{args.trials} trials. Follow the cue in the window. Starting with AWAY for 4 s...")
    last_seq = -1
    t0 = time.monotonic()
    for k, (phase, dur) in enumerate(phases):
        beep(880 if phase == "LOOK" else 440)
        p_start = time.monotonic()
        while time.monotonic() - p_start < dur:
            seq, frame = cam.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.005)
                continue
            last_seq = seq
            obs = det.observe(frame)
            now = time.monotonic()
            change = eng.update(now, obs.attention)
            log.append({"t": round(now - t0, 3), "phase": phase, "k": k,
                        "att": obs.attention, "state": eng.state.value, "ms": round(obs.ms, 2)})
            if change is not None:
                events.append({"t": round(now - t0, 3), "phase": phase, "k": k,
                               "since_cue": round(now - p_start, 3), "state": change.value})
            view = cv2.flip(frame, 1)
            color = (60, 200, 60) if phase == "LOOK" else (60, 60, 230)
            cv2.rectangle(view, (0, 0), (view.shape[1], 70), color, -1)
            cv2.putText(view, f"{phase}  ({k // 2 + 1}/{args.trials})  {dur - (now - p_start):.0f}s",
                        (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            cv2.putText(view, f"lamp: {eng.state.value}", (15, view.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow("engagement test", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cam.close()
                raise SystemExit("aborted")
    cam.close()
    cv2.destroyAllWindows()

    # ---------------------------------------------------------------- scoring
    engage_lat, leave_lat, idle_lat = [], [], []
    hits = misses = drops = false_engage = 0
    for k, (phase, _) in enumerate(phases):
        ev = [e for e in events if e["k"] == k]
        if phase == "LOOK":
            before = [e for e in events if e["k"] < k]
            already = bool(before) and before[-1]["state"] == "engaged"
            first = next((e for e in ev if e["state"] == "engaged"), None)
            if already and not any(e["state"] in ("leaving", "idle") for e in ev):
                hits += 1                    # still engaged from before the cue: a hit, no latency
            elif first:
                hits += 1
                engage_lat.append(max(0.0, first["since_cue"] - REACTION_S))
                # dropped while still looking (after engaging)?
                drops += sum(1 for e in ev if e["t"] > first["t"] and e["state"] in ("leaving", "idle"))
            else:
                misses += 1
        else:
            leave = next((e for e in ev if e["state"] == "leaving"), None)
            idle = next((e for e in ev if e["state"] == "idle"), None)
            if leave:
                leave_lat.append(max(0.0, leave["since_cue"] - REACTION_S))
            if idle:
                idle_lat.append(max(0.0, idle["since_cue"] - REACTION_S))
            # re-engaged during AWAY after having let go = false engagement
            if idle:
                false_engage += sum(1 for e in ev if e["t"] > idle["t"] and e["state"] == "engaged")

    def frame_rate(phase: str, want_face: bool) -> float:
        # skip the first 1 s of each phase (person is still turning)
        fr = [r for r in log if r["phase"] == phase and
              r["t"] - next(e["t"] for e in log if e["k"] == r["k"]) > 1.0]
        ok = [r for r in fr if (r["att"] > 0) == want_face]
        return len(ok) / max(1, len(fr))

    looks = sum(1 for p, _ in phases if p == "LOOK")
    aways = sum(1 for p, _ in phases[1:] if p == "AWAY")
    med = lambda v: round(statistics.median(v), 2) if v else None   # noqa: E731
    res = {
        "trials": args.trials,
        "engaged_when_looking": f"{hits}/{looks}",
        "engage_latency_s_median": med(engage_lat),
        "engage_latency_s_max": round(max(engage_lat), 2) if engage_lat else None,
        "let_go_when_away": f"{len(idle_lat)}/{aways}",
        "to_leaving_s_median": med(leave_lat),
        "to_idle_s_median": med(idle_lat),
        "false_drops_while_looking": drops,
        "false_engagements_while_away": false_engage,
        "frame_face_detected_when_looking": round(frame_rate("LOOK", True), 3),
        "frame_no_face_when_away": round(frame_rate("AWAY", False), 3),
        "detection_ms_median": med([r["ms"] for r in log]),
        "frames": len(log),
    }
    out = Path(__file__).resolve().parent.parent / "logs" / "engagement_test.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": res, "events": events}, indent=1))
    print("\nRESULTS")
    for k, v in res.items():
        if v is not None:
            print(f"  {k:36s} {v}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
