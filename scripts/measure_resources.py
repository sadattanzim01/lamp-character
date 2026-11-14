"""Measure CPU and memory of the running body and brain processes.

Start the lamp normally (body + brain), then in a third terminal:
    python scripts/measure_resources.py --seconds 120

Interact with the lamp while it samples (engage, talk, show an object, give a
goal) so the numbers cover real work, not just idle. Writes
logs/resources.json. CPU % is per core (100% = one full core); the summary
also shows the share of a 4-core machine (the Ubuntu target).
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
ROLES = {"body": "body.server", "brain": "brain.main"}


def find() -> dict[str, psutil.Process]:
    """Largest process per role (skips wrappers like `timeout` or shells that
    merely mention the module name in their command line)."""
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
    for p in procs.values():
        p.cpu_percent(None)                      # prime the counters
    samples = {r: {"cpu": [], "rss_mb": [], "threads": []} for r in procs}
    print(f"sampling {', '.join(f'{r}={p.pid}' for r, p in procs.items())} for {args.seconds:.0f}s "
          "- interact with the lamp now...")
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
                pass

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
