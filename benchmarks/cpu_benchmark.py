"""CPU DNA Alignment Benchmark -- Week 1 Baseline.

Measures wall-clock execution time of the CPU alignment implementation
across multiple input sizes using realistic random DNA sequences.

Usage:
    python benchmarks/cpu_benchmark.py

All timings are measured with time.perf_counter() which provides the
highest available resolution on each platform.  Setup (sequence
generation) is explicitly excluded from the measured alignment time.

This script intentionally avoids GPU/CUDA/Dask imports.
"""

from __future__ import annotations

import os
import platform
import random
import sys
import time
from typing import List, Tuple

# ── Make src/ importable ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alignment.cpu_alignment import align_sequence, align_sequences, AlignmentResult


# ── Constants ─────────────────────────────────────────────────────────────
DNA_BASES = "ACGT"
RANDOM_SEED = 42  # fixed seed for reproducibility

# Benchmark configurations: (sequence_length, number_of_pairs)
BENCHMARK_CONFIGS: List[Tuple[int, int]] = [
    (100,    1_000),    # short sequences, many pairs
    (1_000,  1_000),    # medium sequences
    (10_000, 100),      # long sequences
    (100_000, 10),      # very long sequences, few pairs
]


# ── Helpers ───────────────────────────────────────────────────────────────
def generate_random_dna(length: int, rng: random.Random) -> str:
    """Generate a random DNA sequence of the given length."""
    return "".join(rng.choices(DNA_BASES, k=length))


def generate_pair_with_mutations(
    length: int,
    mutation_rate: float,
    rng: random.Random,
) -> Tuple[str, str]:
    """Generate a (reference, query) pair with a given mutation rate.

    Parameters
    ----------
    length : int
        Length of each sequence.
    mutation_rate : float
        Fraction of positions to mutate (0.0 – 1.0).
    rng : random.Random
        Seeded RNG for reproducibility.

    Returns
    -------
    tuple[str, str]
        (reference, query) where query has ~mutation_rate fraction of
        bases changed relative to reference.
    """
    ref = generate_random_dna(length, rng)
    query_chars = list(ref)
    for i in range(length):
        if rng.random() < mutation_rate:
            # Pick a different base
            alternatives = [b for b in DNA_BASES if b != ref[i]]
            query_chars[i] = rng.choice(alternatives)
    return ref, "".join(query_chars)


def format_time(seconds: float) -> str:
    """Format a duration in human-readable form."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def print_separator(char: str = "-", width: int = 78) -> None:
    print(char * width)


# ── Single-pair benchmark ────────────────────────────────────────────────
def benchmark_single(
    seq_len: int,
    num_pairs: int,
    mutation_rate: float,
    rng: random.Random,
) -> dict:
    """Benchmark align_sequence() called individually per pair.

    Returns a dict with timing and result summary.
    """
    # ── Setup (excluded from timing) ─────────────────────────────────
    pairs = [
        generate_pair_with_mutations(seq_len, mutation_rate, rng)
        for _ in range(num_pairs)
    ]

    # ── Timed alignment ──────────────────────────────────────────────
    results: List[AlignmentResult] = []
    t_start = time.perf_counter()
    for ref, qry in pairs:
        results.append(align_sequence(ref, qry))
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    total_bases = seq_len * num_pairs

    avg_score = sum(r.score for r in results) / len(results)
    avg_mismatches = sum(r.mismatches for r in results) / len(results)

    return {
        "seq_len": seq_len,
        "num_pairs": num_pairs,
        "total_bases": total_bases,
        "elapsed_s": elapsed,
        "per_pair_s": elapsed / num_pairs,
        "bases_per_sec": total_bases / elapsed if elapsed > 0 else float("inf"),
        "avg_score": avg_score,
        "avg_mismatches": avg_mismatches,
    }


# ── Batch benchmark ──────────────────────────────────────────────────────
def benchmark_batch(
    seq_len: int,
    num_pairs: int,
    mutation_rate: float,
    rng: random.Random,
) -> dict:
    """Benchmark align_sequences() (batch API).

    Returns a dict with timing and result summary.
    """
    pairs = [
        generate_pair_with_mutations(seq_len, mutation_rate, rng)
        for _ in range(num_pairs)
    ]

    t_start = time.perf_counter()
    results = align_sequences(pairs)
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    total_bases = seq_len * num_pairs

    avg_score = sum(r.score for r in results) / len(results)

    return {
        "seq_len": seq_len,
        "num_pairs": num_pairs,
        "total_bases": total_bases,
        "elapsed_s": elapsed,
        "per_pair_s": elapsed / num_pairs,
        "bases_per_sec": total_bases / elapsed if elapsed > 0 else float("inf"),
        "avg_score": avg_score,
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    rng = random.Random(RANDOM_SEED)
    mutation_rate = 0.10  # 10% mismatch rate -- realistic for variant analysis

    print()
    print("+==================================================================+")
    print("|       GeneWeaver -- CPU DNA Alignment Benchmark (Week 1)        |")
    print("+==================================================================+")
    print()

    # ── Environment info ─────────────────────────────────────────────
    print("Environment:")
    print(f"  Python     : {sys.version}")
    print(f"  Platform   : {platform.platform()}")
    print(f"  Processor  : {platform.processor()}")
    print(f"  Mutation   : {mutation_rate * 100:.0f}%")
    print(f"  RNG Seed   : {RANDOM_SEED}")
    print()

    # ── Per-pair benchmarks ──────────────────────────────────────────
    print_separator("=")
    print("  Per-pair benchmark  (align_sequence called individually)")
    print_separator("=")
    print(
        f"{'Seq Len':>10s}  {'Pairs':>8s}  {'Total Bases':>12s}  "
        f"{'Wall Time':>12s}  {'Per Pair':>12s}  {'Bases/sec':>14s}  "
        f"{'Avg Score':>10s}"
    )
    print_separator()

    single_results = []
    for seq_len, num_pairs in BENCHMARK_CONFIGS:
        r = benchmark_single(seq_len, num_pairs, mutation_rate, rng)
        single_results.append(r)
        print(
            f"{r['seq_len']:>10,d}  {r['num_pairs']:>8,d}  "
            f"{r['total_bases']:>12,d}  "
            f"{format_time(r['elapsed_s']):>12s}  "
            f"{format_time(r['per_pair_s']):>12s}  "
            f"{r['bases_per_sec']:>14,.0f}  "
            f"{r['avg_score']:>10.4f}"
        )

    print_separator()
    print()

    # ── Batch benchmarks ─────────────────────────────────────────────
    # Reset RNG so batch gets same sequences for fair comparison
    rng = random.Random(RANDOM_SEED)

    print_separator("=")
    print("  Batch benchmark  (align_sequences called once per config)")
    print_separator("=")
    print(
        f"{'Seq Len':>10s}  {'Pairs':>8s}  {'Total Bases':>12s}  "
        f"{'Wall Time':>12s}  {'Per Pair':>12s}  {'Bases/sec':>14s}  "
        f"{'Avg Score':>10s}"
    )
    print_separator()

    batch_results = []
    for seq_len, num_pairs in BENCHMARK_CONFIGS:
        r = benchmark_batch(seq_len, num_pairs, mutation_rate, rng)
        batch_results.append(r)
        print(
            f"{r['seq_len']:>10,d}  {r['num_pairs']:>8,d}  "
            f"{r['total_bases']:>12,d}  "
            f"{format_time(r['elapsed_s']):>12s}  "
            f"{format_time(r['per_pair_s']):>12s}  "
            f"{r['bases_per_sec']:>14,.0f}  "
            f"{r['avg_score']:>10.4f}"
        )

    print_separator()
    print()

    # ── Summary ──────────────────────────────────────────────────────
    total_bases_all = sum(r["total_bases"] for r in single_results)
    total_time_all = sum(r["elapsed_s"] for r in single_results)
    print(f"  Total bases aligned (per-pair) : {total_bases_all:>14,d}")
    print(f"  Total wall-clock time          : {format_time(total_time_all):>14s}")
    if total_time_all > 0:
        print(
            f"  Overall throughput             : "
            f"{total_bases_all / total_time_all:>14,.0f} bases/sec"
        )
    print()
    print("Benchmark complete.")
    print()


if __name__ == "__main__":
    main()
