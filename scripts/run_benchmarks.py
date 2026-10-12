#!/usr/bin/env python3
"""GeneWeaver Benchmark Suite Execution Script.

Entry point to execute benchmarks (CPU, CUDA, Dask) from scripts/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
_BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GeneWeaver Benchmark Runner."
    )
    parser.add_argument(
        "suite",
        choices=["cpu", "cuda", "dask", "all"],
        default="cpu",
        nargs="?",
        help="Benchmark suite to run (cpu, cuda, dask, all). Default: cpu",
    )

    args = parser.parse_args()
    suites = ["cpu", "cuda", "dask"] if args.suite == "all" else [args.suite]

    for suite_name in suites:
        bench_script = _BENCHMARKS_DIR / f"{suite_name}_benchmark.py"
        if not bench_script.exists():
            print(f"Benchmark script not found: {bench_script}", file=sys.stderr)
            return 1

        print(f"\n{'='*70}")
        print(f"  Running {suite_name.upper()} Benchmark: {bench_script.name}")
        print(f"{'='*70}\n")
        
        proc = subprocess.run([sys.executable, str(bench_script)])
        if proc.returncode != 0:
            print(f"{suite_name.upper()} benchmark failed with return code {proc.returncode}", file=sys.stderr)
            return proc.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
