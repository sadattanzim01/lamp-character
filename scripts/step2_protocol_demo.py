"""Step 2 check: drive the body entirely over the protocol.

Terminal 1:  mjpython -m body.server          (macOS; Linux: python -m body.server)
Terminal 2:  python scripts/step2_protocol_demo.py

Prints every ack (including body-side adjustments), checks preemption and
rejection behaviour, and reports command round-trip latency.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.body_client import BodyClient, BodyError  # noqa: E402

WARM, HAPPY, COOL = [1.0, 0.82, 0.55], [1.0, 0.55, 0.2], [0.5, 0.72, 1.0]


def main() -> None:
    body = BodyClient()
    caps = body.capabilities
    print("connected. poses:", caps["poses"], "gestures:", caps["gestures"])

    print("\n-- wake up")
    body.light(WARM, 0.2, fade=1.0)
    body.pose("rest", speed=0.4, wait=True)
    body.light(HAPPY, 1.0, fade=0.3)
    body.pose("perk_up", speed=1.0, wait=True)
    print("   wiggle ->", body.gesture("wiggle", wait=True)[1])

    print("\n-- look around with gaze (body solves the joints)")
    for az, el in [(0.6, -0.1), (-0.6, -0.1), (0.0, -0.8)]:
        ack, status = body.gaze(az, el, wait=True)
        print(f"   gaze az={az:+.1f} el={el:+.1f}: {status} {ack['adjusted'] or ''}")
    print("   nod ->", body.gesture("nod", wait=True)[1])

    print("\n-- body protects itself")
    ack, _ = body.joints({"head_pitch_joint": 0.6}, wait=True)
    print("   head_pitch +0.6 requested ->", ack["adjusted"])
    ack, _ = body.gaze(0.0, 0.8, wait=True)
    print("   gaze el=+0.8 requested   ->", ack["adjusted"])
    try:
        body.pose("backflip")
    except BodyError as e:
        print("   unknown pose            -> rejected:", e)

    print("\n-- a new motion preempts the running one")
    _, first = body.gesture("shake")
    time.sleep(0.3)
    _, second = body.pose("attentive")
    print("   shake ->", first.result(5), "| attentive ->", second.result(10))

    print("\n-- go back to sleep")
    body.light(COOL, 0.6, fade=0.5)
    body.gesture("droop", wait=True)
    body.light(WARM, 0.08, fade=2.0)
    body.pose("sleep", speed=0.3, wait=True)

    st = body.state
    print("\nfinal state:", {k: st[k] for k in ("q", "light")})
    rtt = sorted(body.rtt_ms)
    print(f"\ncommand round-trip over {len(rtt)} commands: "
          f"median {statistics.median(rtt):.1f} ms, "
          f"p95 {rtt[int(0.95 * (len(rtt) - 1))]:.1f} ms, max {rtt[-1]:.1f} ms")
    body.close()


if __name__ == "__main__":
    main()
