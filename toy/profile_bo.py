"""Profile the BO script with cProfile, serial vs parallel."""
from __future__ import annotations

import cProfile
import pstats
import io
import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def profile_run(n_processes: int, tag: str) -> str:
    """Run cProfile on run_bo.py with given n_processes, return stats string."""
    profile_path = f"/tmp/bo_profile_{tag}.prof"
    cmd = (
        f"PYTHONPATH=src python -m cProfile -o {profile_path} "
        f"scripts/run_bo.py "
        f"--n-evaluations 5 --n-initial 3 --n-outcomes 3 "
        f"--winner-outcomes 3 --n-processes {n_processes} --seed 0"
    )
    print(f"\n{'='*70}")
    print(f"PROFILING: {tag} (n_processes={n_processes})")
    print(f"Command: {cmd}")
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    rc = os.system(cmd)
    wall = time.time() - t0

    # Parse cProfile stats
    pr = pstats.Stats(profile_path)
    pr.sort_stats("cumulative")

    buf = io.StringIO()
    pr.stream = buf
    pr.print_stats(25)

    result = buf.getvalue()
    result += f"\n\n--- WALL TIME: {wall:.1f}s ---\n"
    return result


if __name__ == "__main__":
    print("BO PROFILER", flush=True)

    serial = profile_run(1, "serial")
    parallel = profile_run(8, "parallel")

    # Write combined report
    report_path = "toy/profile_bo_results.txt"
    with open(report_path, "w") as f:
        f.write("BO PROFILING RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write("SERIAL (n_processes=1)\n")
        f.write("=" * 70 + "\n")
        f.write(serial)
        f.write("\n\n")
        f.write("PARALLEL (n_processes=8)\n")
        f.write("=" * 70 + "\n")
        f.write(parallel)

    print(f"\nReport saved to {report_path}")
    print(serial)
