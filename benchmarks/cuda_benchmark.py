"""Validation and Benchmark Entry Point for CUDA DNA Alignment.

Week 2 Day 2: CUDA Benchmarking, Host/Device Transfer Analysis, and CPU vs GPU Validation.

This script benchmarks the CUDA implementation using actual timing measurements
and compares it directly against the Week 1 CPU reference implementation across
multiple input sizes (Small, Medium, Large).

Key Methodology:
- Warm-up execution to trigger CUDA context initialization and JIT compilation.
- Granular component breakdown:
    * Host preparation
    * Host-to-Device transfer (cuda.to_device)
    * CUDA kernel execution (dna_align_kernel with explicit cuda.synchronize)
    * Device-to-Host transfer (copy_to_host with explicit cuda.synchronize)
    * Result reconstruction
    * GPU Total = H2D + Kernel + D2H
- CPU vs GPU comparison:
    * Speedup = CPU Time / GPU Total Time
    * Honest accounting for transfer overhead on smaller batches
- Truthful reporting: If CUDA runtime is unavailable (e.g., blocked by OS Code
  Integrity policy), the script detects and reports the exact limitation without
  faking numbers.
"""

from __future__ import annotations

import math
import os
import platform
import random
import sys
import time
from typing import List, Tuple

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alignment.cpu_alignment import (
    AlignmentResult,
    align_sequence,
    align_sequences,
)
from alignment.cuda_alignment import (
    cuda_align_sequence,
    cuda_align_sequences,
    cuda_align_sequences_timed,
    is_cuda_available,
    get_cuda_device_info,
    CUDANotAvailableError,
    CUDATimingBreakdown,
)

DNA_BASES = "ACGT"
RANDOM_SEED = 42
MUTATION_RATE = 0.10
NUM_ROUNDS = 5

# Input size configurations for Day 2 CPU vs GPU benchmark
BENCHMARK_CONFIGS: List[Tuple[str, int, int]] = [
    ("Small",  100,    1_000),   # 100 bp x 1,000 pairs = 100,000 bases
    ("Medium", 1_000,  1_000),   # 1,000 bp x 1,000 pairs = 1,000,000 bases
    ("Large",  10_000,   100),   # 10,000 bp x 100 pairs = 1,000,000 bases
]


def generate_random_dna(length: int, rng: random.Random) -> str:
    return "".join(rng.choices(DNA_BASES, k=length))


def generate_pair_with_mutations(
    length: int,
    mutation_rate: float,
    rng: random.Random,
) -> Tuple[str, str]:
    ref = generate_random_dna(length, rng)
    query_chars = list(ref)
    for i in range(length):
        if rng.random() < mutation_rate:
            alternatives = [b for b in DNA_BASES if b != ref[i]]
            query_chars[i] = rng.choice(alternatives)
    return ref, "".join(query_chars)


def format_time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def run_cuda_benchmark() -> None:
    print("=" * 86)
    print("   GeneWeaver -- Week 2 Day 2: CPU vs CUDA Alignment Benchmark & Analysis")
    print("=" * 86)

    # 1. Environment & Hardware Diagnostics
    print("\n--- Environment & Hardware Diagnostics ---")
    print(f"  Platform      : {platform.platform()}")
    print(f"  Python        : {sys.version.split()[0]} ({platform.architecture()[0]})")
    print(f"  Executable    : {sys.executable}")

    cuda_info = get_cuda_device_info()
    available = is_cuda_available()

    print(f"  CUDA Available: {available}")
    if available:
        print(f"  Device Name   : {cuda_info.get('device_name')}")
        print(f"  Compute Cap   : {cuda_info.get('compute_capability')}")
    else:
        print(f"  Limitation    : {cuda_info.get('reason')}")

    # 2. Early exit if CUDA is unavailable
    if not available:
        print("\n--- Status: Environment Restricted ---")
        print("  CUDA execution is unavailable in this environment.")
        print(f"  Reason: {cuda_info.get('reason')}")
        print("  Rule adherence: NO simulated or fake GPU numbers will be reported.")
        print("\n  Testing fallback error handling...")
        try:
            cuda_align_sequences([("ACGT", "ACGT")])
            print("  ERROR: Expected CUDANotAvailableError but execution succeeded!")
        except CUDANotAvailableError as err:
            print(f"  PASS : CUDANotAvailableError properly raised as expected:\n         {err}")

        print("\n  Running CPU Baseline Timings across configurations for reference:")
        print("-" * 86)
        print(f"{'Config':<10} {'SeqLen':>8} {'Pairs':>8} {'Total Bases':>12} {'CPU Time (min)':>16} {'Throughput':>18}")
        print("-" * 86)
        for label, seq_len, num_pairs in BENCHMARK_CONFIGS:
            rng = random.Random(RANDOM_SEED)
            pairs = [generate_pair_with_mutations(seq_len, MUTATION_RATE, rng) for _ in range(num_pairs)]
            total_bases = seq_len * num_pairs
            times = []
            for _ in range(NUM_ROUNDS):
                t0 = time.perf_counter()
                _ = align_sequences(pairs)
                times.append(time.perf_counter() - t0)
            min_t = min(times)
            throughput = total_bases / min_t
            print(f"{label:<10} {seq_len:>8,d} {num_pairs:>8,d} {total_bases:>12,d} {format_time(min_t):>16} {throughput:>15,.0f} bp/s")
        print("-" * 86)
        print("Benchmark completed. CUDA GPU benchmarks cleanly deferred until OS policy permits.")
        return

    # 3. CUDA Warm-Up Execution
    print("\n--- CUDA Warm-up Phase ---")
    print("  Performing warm-up execution to trigger JIT compilation and device context initialization...")
    warmup_pairs = [("ACGT" * 25, "ACGT" * 25) for _ in range(10)]
    _ = cuda_align_sequences(warmup_pairs)
    print("  Warm-up completed successfully.")

    # 4. Correctness & Parity Verification
    print("\n--- CPU vs GPU Correctness Verification ---")
    parity_rng = random.Random(RANDOM_SEED)
    test_pairs = [generate_pair_with_mutations(50, MUTATION_RATE, parity_rng) for _ in range(10)]
    cpu_ref = align_sequences(test_pairs)
    gpu_out, _ = cuda_align_sequences_timed(test_pairs)

    all_matched = True
    for i, (c_res, g_res) in enumerate(zip(cpu_ref, gpu_out)):
        if c_res != g_res:
            all_matched = False
            print(f"  FAIL on pair {i}: CPU={c_res} != GPU={g_res}")

    if all_matched:
        print("  PASS: 100% Correctness Parity Verified (CPU result == GPU result).")
    else:
        print("  FAIL: Correctness mismatch detected!")
        return

    # 5. Multi-round Benchmarking
    print("\n--- Multi-Configuration Benchmark (5 rounds each) ---")
    print("-" * 110)
    print(
        f"{'Size':<10} {'Pairs':>6} {'SeqLen':>7} "
        f"{'CPU (min)':>12} {'H2D (min)':>11} {'Kernel (min)':>13} {'D2H (min)':>11} {'GPU Tot':>11} {'Speedup':>9}"
    )
    print("-" * 110)

    for label, seq_len, num_pairs in BENCHMARK_CONFIGS:
        rng = random.Random(RANDOM_SEED)
        pairs = [generate_pair_with_mutations(seq_len, MUTATION_RATE, rng) for _ in range(num_pairs)]

        # Time CPU
        cpu_times = []
        for _ in range(NUM_ROUNDS):
            t0 = time.perf_counter()
            _ = align_sequences(pairs)
            cpu_times.append(time.perf_counter() - t0)
        cpu_min = min(cpu_times)

        # Time GPU with breakdown
        h2d_times = []
        kernel_times = []
        d2h_times = []
        gpu_totals = []
        for _ in range(NUM_ROUNDS):
            _, timing = cuda_align_sequences_timed(pairs)
            h2d_times.append(timing.h2d_transfer_time)
            kernel_times.append(timing.kernel_execution_time)
            d2h_times.append(timing.d2h_transfer_time)
            gpu_totals.append(timing.total_gpu_time)

        min_h2d = min(h2d_times)
        min_kernel = min(kernel_times)
        min_d2h = min(d2h_times)
        min_gpu_tot = min(gpu_totals)
        speedup = cpu_min / min_gpu_tot if min_gpu_tot > 0 else 0.0

        speedup_str = f"{speedup:.2f}x" if speedup >= 1.0 else f"{speedup:.2f}x (slower)"
        print(
            f"{label:<10} {num_pairs:>6,d} {seq_len:>7,d} "
            f"{format_time(cpu_min):>12} {format_time(min_h2d):>11} {format_time(min_kernel):>13} {format_time(min_d2h):>11} "
            f"{format_time(min_gpu_tot):>11} {speedup_str:>9}"
        )
    print("-" * 110)


if __name__ == "__main__":
    run_cuda_benchmark()
