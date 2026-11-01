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


def print_separator(char: str = "-", width: int = 112) -> None:
    print(char * width)


def run_benchmark() -> None:
    print()
    print("=" * 112)
    print("       GeneWeaver -- Dask Distributed & Local-Scheduler Performance Benchmark")
    print("=" * 112)
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

    # 2. Scheduler Initialization & Warm-Up Latency (Isolated Measurement)
    print("Scheduler Initialization & Startup Metrics:")
    t_dist_init_start = time.perf_counter()
    dist_processor = DaskGenomeProcessor(processes=False, threads_per_worker=4)
    t_dist_init = time.perf_counter() - t_dist_init_start
    print(f"  1. Dask LocalCluster startup time (threads=4) : {format_duration(t_dist_init)}")

    t_local_init_start = time.perf_counter()
    local_processor = DaskGenomeProcessor(scheduler="threads")
    t_local_init = time.perf_counter() - t_local_init_start
    print(f"  2. Dask local-scheduler startup time (in-proc): {format_duration(t_local_init)}")

    # Warm-up phase
    print("\nRunning warm-up alignments to isolate cold-start overhead...")
    rng_warmup = random.Random(RANDOM_SEED)
    warmup_target, warmup_chunks = generate_benchmark_workload(
        50, 10, MUTATION_RATE, rng_warmup
    )
    t_warmup_dist_start = time.perf_counter()
    warmup_dist_res = dist_processor.align_chunks(warmup_chunks, warmup_target, batch_size=1)
    t_warmup_dist = time.perf_counter() - t_warmup_dist_start

    t_warmup_local_start = time.perf_counter()
    warmup_local_res = local_processor.align_chunks(warmup_chunks, warmup_target, batch_size=1)
    t_warmup_local = time.perf_counter() - t_warmup_local_start
    print(f"  Warm-up LocalCluster (10 chunks)    : {format_duration(t_warmup_dist)}")
    print(f"  Warm-up local-scheduler (10 chunks) : {format_duration(t_warmup_local)}")
    print()

    try:
        # 3. Workload Benchmarks across Schedulers
        print_separator("=")
        print(
            f"{'Config':<8s} {'Chunk':>5s} {'Chunks':>6s} {'Bases':>9s} | "
            f"{'CPU Min':>8s} {'CPU Avg':>8s} | "
            f"{'Local Min':>9s} {'Local Avg':>9s} | "
            f"{'Dist Min':>8s} {'Dist Avg':>8s} | "
            f"{'Loc Spd':>7s} {'Dist Spd':>8s} {'Equiv':>5s}"
        )
        print_separator("=")

        benchmark_summary = []

        for label, chunk_sz, num_chk in CONFIGURATIONS:
            total_bases = chunk_sz * num_chk
            rng = random.Random(RANDOM_SEED)
            target, chunks = generate_benchmark_workload(
                chunk_sz, num_chk, MUTATION_RATE, rng
            )

            # Equivalence verification
            cpu_ref = align_chunks(chunks, target)
            local_check = local_processor.align_chunks(chunks, target, batch_size=50)
            dist_check = dist_processor.align_chunks(chunks, target, batch_size=50)
            assert cpu_ref == local_check, f"Result mismatch in local scheduler for {label}!"
            assert cpu_ref == dist_check, f"Result mismatch in distributed scheduler for {label}!"

            # Measure CPU baseline across NUM_ROUNDS
            cpu_times = []
            for _ in range(NUM_ROUNDS):
                t0 = time.perf_counter()
                _ = align_chunks(chunks, target)
                t1 = time.perf_counter()
                cpu_times.append(t1 - t0)

            # Measure Dask local-scheduler across NUM_ROUNDS
            batch_sz = max(1, num_chk // 4)
            local_times = []
            for _ in range(NUM_ROUNDS):
                t0 = time.perf_counter()
                _ = local_processor.align_chunks(chunks, target, batch_size=batch_sz)
                t1 = time.perf_counter()
                local_times.append(t1 - t0)

            # Measure Dask distributed LocalCluster across NUM_ROUNDS
            dist_times = []
            for _ in range(NUM_ROUNDS):
                t0 = time.perf_counter()
                _ = dist_processor.align_chunks(chunks, target, batch_size=batch_sz)
                t1 = time.perf_counter()
                dist_times.append(t1 - t0)

            cpu_min = min(cpu_times)
            cpu_avg = sum(cpu_times) / len(cpu_times)
            local_min = min(local_times)
            local_avg = sum(local_times) / len(local_times)
            dist_min = min(dist_times)
            dist_avg = sum(dist_times) / len(dist_times)

            local_speedup = cpu_min / local_min if local_min > 0 else 0.0
            dist_speedup = cpu_min / dist_min if dist_min > 0 else 0.0

            benchmark_summary.append({
                "label": label,
                "chunk_size": chunk_sz,
                "num_chunks": num_chk,
                "total_bases": total_bases,
                "cpu_min": cpu_min,
                "cpu_avg": cpu_avg,
                "local_min": local_min,
                "local_avg": local_avg,
                "dist_min": dist_min,
                "dist_avg": dist_avg,
                "local_speedup": local_speedup,
                "dist_speedup": dist_speedup,
                "batch_size": batch_sz,
            })

            print(
                f"{label:<8s} {chunk_sz:>5d} {num_chk:>6d} {total_bases:>9,d} | "
                f"{format_duration(cpu_min):>8s} {format_duration(cpu_avg):>8s} | "
                f"{format_duration(local_min):>9s} {format_duration(local_avg):>9s} | "
                f"{format_duration(dist_min):>8s} {format_duration(dist_avg):>8s} | "
                f"{local_speedup:>6.2f}x {dist_speedup:>7.2f}x {'OK':>5s}"
            )

        print_separator()
        print()

        # 4. Task Granularity / Batch Sizing Analysis
        print("Task Granularity & Sizing Analysis (Medium Workload: 1,000 bp x 500 chunks = 500,000 bases):")
        print_separator("-", width=72)
        print(f"{'Batch Size':<12s} {'Tasks Created':>14s} | {'Local Exec Time':>16s} | {'Overhead Impact':>18s}")
        print_separator("-", width=72)

        rng = random.Random(RANDOM_SEED)
        med_target, med_chunks = generate_benchmark_workload(1000, 500, MUTATION_RATE, rng)
        t_batch_1 = None
        for b_sz in [1, 10, 50, 125, 250]:
            n_tasks = (500 + b_sz - 1) // b_sz
            t0 = time.perf_counter()
            _ = local_processor.align_chunks(med_chunks, med_target, batch_size=b_sz)
            t_exec = time.perf_counter() - t0
            if b_sz == 1:
                t_batch_1 = t_exec
                overhead_desc = "Baseline (1 task/chunk)"
            else:
                pct = ((t_batch_1 - t_exec) / t_batch_1) * 100 if t_batch_1 else 0
                overhead_desc = f"{pct:+.1f}% faster"
            print(f"{b_sz:<12d} {n_tasks:>14d} | {format_duration(t_exec):>16s} | {overhead_desc:>18s}")
        print_separator("-", width=72)
        print()

        # 5. Bottleneck Analysis & Findings
        print("Analysis & Findings:")
        print("  1. Result Equivalence: 100% deterministic output match across CPU, local-scheduler, and distributed.")
        print(f"  2. Scheduler Setup Latency: LocalCluster initialization ({format_duration(t_dist_init)}) represents")
        print(f"     a fixed one-time cost. Dask local-scheduler requires near-zero setup ({format_duration(t_local_init)}).")
        print("  3. Task Granularity (Batch Sizing): Batching 125 chunks/task vs micro-tasks (batch_size=1)")
        print("     eliminates excessive task graph construction and Dask queue scheduling overhead.")
        for item in benchmark_summary:
            lbl = item["label"]
            lsp = item["local_speedup"]
            dsp = item["dist_speedup"]
            print(
                f"  4. [{lbl:<6s}] Local Speedup: {lsp:.2f}x | Dist Speedup: {dsp:.2f}x "
                f"(CPU: {format_duration(item['cpu_min'])}, Local: {format_duration(item['local_min'])}, Dist: {format_duration(item['dist_min'])})"
            )

        print()
        print("Benchmark completed successfully.")
        print()

    finally:
        dist_processor.close()
        local_processor.close()


if __name__ == "__main__":
    run_benchmark()
