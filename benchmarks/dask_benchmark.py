"""GeneWeaver -- Dask Distributed Sequence Alignment Benchmark.

Week 3 Day 1 & Day 2: Dask Benchmark & CPU vs Dask Comparison.

Measures and compares sequential CPU chunk alignment against Dask-distributed
chunk alignment across multiple workload sizes and multi-round repetitions.

Adheres strictly to project benchmarking standards:
- Real wall-clock measurements (time.perf_counter()).
- Fixed RNG seed for deterministic DNA generation.
- Cold-start isolation (warm-up phase reported separately).
- 100% result equivalence validation prior to timing collection.
- Zero fabricated or simulated numbers.
"""

from __future__ import annotations

import os
import platform
import random
import sys
import time
from typing import Any

# Ensure src/ is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alignment.cpu_alignment import (
    align_chunks,
    align_sequence,
    AlignmentResult,
)
from distributed.dask_processor import (
    DaskGenomeProcessor,
    is_dask_available,
    get_dask_info,
)

DNA_BASES = "ACGT"
RANDOM_SEED = 42
NUM_ROUNDS = 5
MUTATION_RATE = 0.10

# Benchmark configurations: (label, chunk_size, num_chunks)
CONFIGURATIONS = [
    ("Small",  100,   500),    # 50,000 bases
    ("Medium", 1_000, 500),    # 500,000 bases
    ("Large",  5_000, 200),    # 1,000,000 bases
]


def generate_random_dna(length: int, rng: random.Random) -> str:
    """Generate a random DNA sequence of specified length."""
    return "".join(rng.choices(DNA_BASES, k=length))


def generate_mutated_sequence(
    reference: str, mutation_rate: float, rng: random.Random
) -> str:
    """Generate a sequence with point mutations at given mutation rate."""
    chars = list(reference)
    for i in range(len(chars)):
        if rng.random() < mutation_rate:
            alts = [b for b in DNA_BASES if b != chars[i]]
            chars[i] = rng.choice(alts)
    return "".join(chars)


def generate_benchmark_workload(
    chunk_size: int,
    num_chunks: int,
    mutation_rate: float,
    rng: random.Random,
) -> tuple[str, list[str]]:
    """Generate target query and candidate chunks for alignment benchmark."""
    target = generate_random_dna(chunk_size, rng)
    chunks = [
        generate_mutated_sequence(target, mutation_rate, rng)
        for _ in range(num_chunks)
    ]
    return target, chunks


def format_duration(seconds: float) -> str:
    """Format duration in human-readable units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def print_separator(char: str = "-", width: int = 92) -> None:
    print(char * width)


def run_benchmark() -> None:
    print()
    print("============================================================================================")
    print("       GeneWeaver -- Dask Distributed vs CPU Baseline Alignment Benchmark")
    print("============================================================================================")
    print()

    # 1. Environment & Diagnostics
    dask_info = get_dask_info()
    print("Environment:")
    print(f"  Platform         : {platform.platform()}")
    print(f"  Processor        : {platform.processor()}")
    print(f"  Python           : {sys.version.split()[0]}")
    print(f"  Dask Available   : {dask_info['available']}")
    print(f"  Dask Version     : {dask_info['dask_version']}")
    print(f"  Distributed Ver  : {dask_info['distributed_version']}")
    print(f"  RNG Seed         : {RANDOM_SEED}")
    print(f"  Mutation Rate    : {MUTATION_RATE * 100:.0f}%")
    print(f"  Rounds per Config: {NUM_ROUNDS}")
    print()

    if not dask_info["available"]:
        print("ERROR: Dask is not available. Aborting benchmark.")
        return

    # 2. Processor Initialization & Warm-Up
    print("Initializing Dask local cluster (threads=4)...")
    t_init_start = time.perf_counter()
    processor = DaskGenomeProcessor(processes=False, threads_per_worker=4)
    t_init = time.perf_counter() - t_init_start
    print(f"  Dask cluster startup time: {format_duration(t_init)}")

    print("Running warm-up alignment to isolate cold-start / JIT overhead...")
    rng_warmup = random.Random(RANDOM_SEED)
    warmup_target, warmup_chunks = generate_benchmark_workload(
        50, 10, MUTATION_RATE, rng_warmup
    )
    t_warmup_start = time.perf_counter()
    warmup_res = processor.align_chunks(warmup_chunks, warmup_target, batch_size=1)
    t_warmup = time.perf_counter() - t_warmup_start
    print(f"  Warm-up alignment ({len(warmup_res)} chunks): {format_duration(t_warmup)}")
    print()

    try:
        # 3. Workload Benchmarks
        print_separator("=")
        print(
            f"{'Config':<10s} {'Chunk':>6s} {'Chunks':>7s} {'Bases':>10s} | "
            f"{'CPU Min':>9s} {'CPU Avg':>9s} | "
            f"{'Dask Min':>9s} {'Dask Avg':>9s} | "
            f"{'Speedup':>8s} {'Equiv':>6s}"
        )
        print_separator("=")

        benchmark_summary = []

        for label, chunk_sz, num_chk in CONFIGURATIONS:
            total_bases = chunk_sz * num_chk
            rng = random.Random(RANDOM_SEED)
            target, chunks = generate_benchmark_workload(
                chunk_sz, num_chk, MUTATION_RATE, rng
            )

            # Equivalence check
            cpu_ref = align_chunks(chunks, target)
            dask_check = processor.align_chunks(chunks, target, batch_size=50)
            is_equiv = cpu_ref == dask_check
            assert is_equiv, f"Result mismatch in configuration {label}!"

            # Measure CPU baseline across NUM_ROUNDS
            cpu_times = []
            for _ in range(NUM_ROUNDS):
                t0 = time.perf_counter()
                res = align_chunks(chunks, target)
                t1 = time.perf_counter()
                cpu_times.append(t1 - t0)

            # Measure Dask execution across NUM_ROUNDS (batched to reduce task serialization overhead)
            # Choose batch_size so we have 4-8 tasks across workers
            batch_sz = max(1, num_chk // 4)
            dask_times = []
            for _ in range(NUM_ROUNDS):
                t0 = time.perf_counter()
                res = processor.align_chunks(chunks, target, batch_size=batch_sz)
                t1 = time.perf_counter()
                dask_times.append(t1 - t0)

            cpu_min = min(cpu_times)
            cpu_avg = sum(cpu_times) / len(cpu_times)
            dask_min = min(dask_times)
            dask_avg = sum(dask_times) / len(dask_times)

            speedup = cpu_min / dask_min if dask_min > 0 else 0.0

            benchmark_summary.append({
                "label": label,
                "chunk_size": chunk_sz,
                "num_chunks": num_chk,
                "total_bases": total_bases,
                "cpu_min": cpu_min,
                "cpu_avg": cpu_avg,
                "dask_min": dask_min,
                "dask_avg": dask_avg,
                "speedup": speedup,
                "batch_size": batch_sz,
            })

            equiv_str = "OK" if is_equiv else "FAIL"
            print(
                f"{label:<10s} {chunk_sz:>6d} {num_chk:>7d} {total_bases:>10,d} | "
                f"{format_duration(cpu_min):>9s} {format_duration(cpu_avg):>9s} | "
                f"{format_duration(dask_min):>9s} {format_duration(dask_avg):>9s} | "
                f"{speedup:>7.2f}x {equiv_str:>6s}"
            )

        print_separator()
        print()

        # Detailed Analysis
        print("Analysis & Findings:")
        print("  1. Result Equivalence: 100% identical AlignmentResult outputs between CPU and Dask.")
        for item in benchmark_summary:
            sp = item["speedup"]
            lbl = item["label"]
            if sp < 1.0:
                print(
                    f"  2. [{lbl}] Task scheduling overhead ({format_duration(item['dask_min'])}) "
                    f"exceeds compute time ({format_duration(item['cpu_min'])}). Speedup: {sp:.2f}x (expected for small tasks)."
                )
            else:
                print(
                    f"  2. [{lbl}] Dask parallel execution achieves {sp:.2f}x speedup over single-core CPU."
                )

        print()
        print("Benchmark completed successfully.")
        print()

    finally:
        processor.close()


if __name__ == "__main__":
    run_benchmark()
