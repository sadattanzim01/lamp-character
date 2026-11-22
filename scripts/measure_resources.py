"""Measures how much CPU and memory Lumi uses while it's running.

How to run it: start the lamp (./run.sh), then in a second terminal:
    python scripts/measure_resources.py --seconds 120
While it measures, actually USE the lamp (look at it, talk, show an object,
give a goal), so the numbers reflect real work and not just idling.

What this file does, in plain words:
    It finds the two running programs (the body and the brain), checks their
    CPU use, memory and thread count twice a second for the whole period,
    then prints a summary table and saves it to logs/resources.json.

How to read the numbers:
    * CPU % is per core: 100% = one core fully busy, so on a 10-core Mac the
      maximum is 1000%. The summary also converts it to a share of a 4-core
      machine, because that's the Ubuntu laptop the challenge targets.
    * RSS = how much RAM the program is actually using right now.
    * p95 = 95% of the samples were at or below this value (ignores rare spikes).
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
# How to recognise each program: by the module name in its command line.
ROLES = {"body": "body.server", "brain": "brain.main"}


def find() -> dict[str, psutil.Process]:
    """Find the body and brain programs among everything running on the computer.

    If several processes mention the name (e.g. a shell or `timeout` wrapper
    that launched it), pick the one using the most memory: that's the real program.
    """
    found: dict[str, psutil.Process] = {}
    for p in psutil.process_iter(["pid", "cmdline", "memory_info"]):
        cmd = " ".join(p.info.get("cmdline") or [])
        for role, marker in ROLES.items():
            if marker in cmd and "measure_resources" not in cmd:
                cur = found.get(role)
                if cur is None or p.info["memory_info"].rss > cur.memory_info().rss:
                    found[role] = p
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    procs = find()
    missing = [r for r in ROLES if r not in procs]
    if missing:
        raise SystemExit(f"Not running: {', '.join(missing)}. Start the lamp first.")
    # The first CPU reading is always 0 (it measures "since last time"), so take
    # a throwaway reading first.
    for p in procs.values():
        p.cpu_percent(None)
    samples = {r: {"cpu": [], "rss_mb": [], "threads": []} for r in procs}
    print(f"sampling {', '.join(f'{r}={p.pid}' for r, p in procs.items())} for {args.seconds:.0f}s "
          "- interact with the lamp now...")
    # ---- Sample every 0.5 s until time is up.
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        time.sleep(args.interval)
        for role, p in procs.items():
            try:
                with p.oneshot():
                    samples[role]["cpu"].append(p.cpu_percent(None))
                    samples[role]["rss_mb"].append(p.memory_info().rss / 2**20)
                    samples[role]["threads"].append(p.num_threads())
            except psutil.NoSuchProcess:
                pass                             # that program quit mid-measurement

    # ---- Summarise: average, p95 and maximum CPU; average and peak memory.
    summary = {"machine": {"system": platform.system(), "machine": platform.machine(),
                           "cpus": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)},
               "seconds": args.seconds, "processes": {}}
    print(f"\n{'process':8s} {'CPU mean':>9s} {'CPU p95':>8s} {'CPU max':>8s} {'RSS mean':>9s} {'RSS max':>8s}")
    for role, s in samples.items():
        cpu = sorted(s["cpu"])
        row = {
            "cpu_mean_pct": round(statistics.mean(cpu), 1),
            "cpu_p95_pct": round(cpu[int(0.95 * (len(cpu) - 1))], 1),
            "cpu_max_pct": round(cpu[-1], 1),
            "rss_mean_mb": round(statistics.mean(s["rss_mb"]), 0),
            "rss_max_mb": round(max(s["rss_mb"]), 0),
            "threads_max": max(s["threads"]),
        }
        summary["processes"][role] = row
        print(f"{role:8s} {row['cpu_mean_pct']:8.1f}% {row['cpu_p95_pct']:7.1f}% {row['cpu_max_pct']:7.1f}% "
              f"{row['rss_mean_mb']:7.0f}MB {row['rss_max_mb']:6.0f}MB")
    # Totals for body + brain together.
    total_mean = sum(r["cpu_mean_pct"] for r in summary["processes"].values())
    total_rss = sum(r["rss_max_mb"] for r in summary["processes"].values())
    summary["total"] = {"cpu_mean_pct": round(total_mean, 1),
                        "share_of_4_cores_pct": round(total_mean / 4, 1),
                        "rss_max_mb": round(total_rss, 0)}
    print(f"\ntotal: {total_mean:.0f}% of one core on average (= {total_mean / 4:.0f}% of a 4-core "
          f"machine), peak memory {total_rss:.0f} MB")
    out = ROOT / "logs" / "resources.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
