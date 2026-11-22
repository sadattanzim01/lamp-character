"""The second demo: controls the lamp body purely by sending it messages over the network.

How to run it (two terminals):
    Terminal 1:  mjpython -m body.server          (Linux: python -m body.server)
    Terminal 2:  python scripts/step2_protocol_demo.py

What this file does, in plain words:
    It acts like a tiny stand-in brain. It connects to the body program and
    tests the protocol (the message format) step by step, printing what happens:
      1. wake up: light up, stir, perk up, wiggle
      2. look around: it only says WHICH DIRECTION to look; the body works
         out the joint angles itself
      3. the body protects itself: asking for an unsafe head angle, an
         impossible look-up, or a pose that doesn't exist gets adjusted or
         rejected, and the body says what it changed
      4. interrupting: a new movement cancels the one in progress
         (the cancelled one reports "preempted")
      5. back to sleep
    At the end it prints how fast the body replied to commands (round trip).
    That's where the "about 1 ms" figure comes from.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

# Let this script import the brain/ code (it lives one folder up).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.body_client import BodyClient, BodyError  # noqa: E402

# Light colours (red, green, blue, each 0-1).
WARM, HAPPY, COOL = [1.0, 0.82, 0.55], [1.0, 0.55, 0.2], [0.5, 0.72, 1.0]


def main() -> None:
    body = BodyClient()                      # connect; the body says hello with its capabilities
    caps = body.capabilities
    print("connected. poses:", caps["poses"], "gestures:", caps["gestures"])

    print("\n-- wake up")
    body.light(WARM, 0.2, fade=1.0)
    body.pose("rest", speed=0.4, wait=True)
    body.light(HAPPY, 1.0, fade=0.3)
    body.pose("perk_up", speed=1.0, wait=True)
    print("   wiggle ->", body.gesture("wiggle", wait=True)[1])

    # Directions are (azimuth, elevation) in radians: left/right and up/down.
    print("\n-- look around with gaze (body solves the joints)")
    for az, el in [(0.6, -0.1), (-0.6, -0.1), (0.0, -0.8)]:
        ack, status = body.gaze(az, el, wait=True)
        print(f"   gaze az={az:+.1f} el={el:+.1f}: {status} {ack['adjusted'] or ''}")
    print("   nod ->", body.gesture("nod", wait=True)[1])

    # Ask for things the body shouldn't do, and print how it responds.
    print("\n-- body protects itself")
    ack, _ = body.joints({"head_pitch_joint": 0.6}, wait=True)
    print("   head_pitch +0.6 requested ->", ack["adjusted"])
    ack, _ = body.gaze(0.0, 0.8, wait=True)
    print("   gaze el=+0.8 requested   ->", ack["adjusted"])
    try:
        body.pose("backflip")
    except BodyError as e:
        print("   unknown pose            -> rejected:", e)

    # Start a head shake, then interrupt it after 0.3 s with a new pose.
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
    # How long each command took to get a reply (sent -> ack received).
    rtt = sorted(body.rtt_ms)
    print(f"\ncommand round-trip over {len(rtt)} commands: "
          f"median {statistics.median(rtt):.1f} ms, "
          f"p95 {rtt[int(0.95 * (len(rtt) - 1))]:.1f} ms, max {rtt[-1]:.1f} ms")
    body.close()


if __name__ == "__main__":
    main()
