from __future__ import annotations

import os
import platform
import random
import sys
import time
from typing import List, Tuple

# ── Make src/ importable ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alignment.cpu_alignment import (
    align_sequence,
    align_sequences,
    align_chunks,
    AlignmentResult,
)


# ── Constants ─────────────────────────────────────────────────────────────
DNA_BASES = "ACGT"
RANDOM_SEED = 42  # fixed seed for reproducibility
NUM_ROUNDS = 5    # number of repetitions per benchmark config

# Benchmark configurations: (label, sequence_length, number_of_pairs)
BENCHMARK_CONFIGS: List[Tuple[str, int, int]] = [
    ("Small",       100,     1_000),   # short sequences, many pairs (100 kbp)
    ("Medium",      1_000,   1_000),   # medium sequences (1 Mbp)
    ("Large",       10_000,  100),     # long sequences (1 Mbp)
    ("Extra Large", 100_000, 10),      # very long sequences, few pairs (1 Mbp)
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
    """Generate a (reference, query) pair where the query has random mutations.

    Parameters
    ----------
    length : int
        Sequence length in base pairs.
    mutation_rate : float
        Probability of mutation at each position.
    rng : random.Random
        Seeded random number generator for reproducibility.

    Returns
    -------
    Tuple[str, str]
        Tuple of (reference_sequence, mutated_query_sequence).
    """
    ref = generate_random_dna(length, rng)
    query_chars = list(ref)
    for i in range(length):
        if rng.random() < mutation_rate:
            alternatives = [b for b in DNA_BASES if b != ref[i]]
            query_chars[i] = rng.choice(alternatives)
    return ref, "".join(query_chars)


def format_time(seconds: float) -> str:
    """Format a duration in human-readable form."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def print_separator(char: str = "-", width: int = 86) -> None:
    print(char * width)


# ── Single-pair benchmark ────────────────────────────────────────────────
def benchmark_single(
    label: str,
    seq_len: int,
    num_pairs: int,
    mutation_rate: float,
    rng: random.Random,
    num_rounds: int = NUM_ROUNDS,
) -> dict:
    """Benchmark align_sequence() called individually per pair over multiple rounds.

    Returns a dict with timing metrics (min, avg) and result summary.
    """
    # ── Setup (excluded from timing) ─────────────────────────────────
    pairs = [
        generate_pair_with_mutations(seq_len, mutation_rate, rng)
        for _ in range(num_pairs)
    ]

    # ── Timed alignment across multiple rounds ───────────────────────
    run_times: List[float] = []
    last_results: List[AlignmentResult] = []

    for _ in range(num_rounds):
        results: List[AlignmentResult] = []
        t_start = time.perf_counter()
        for ref, qry in pairs:
            results.append(align_sequence(ref, qry))
        t_end = time.perf_counter()
        run_times.append(t_end - t_start)
        last_results = results

    min_elapsed = min(run_times)
    avg_elapsed = sum(run_times) / len(run_times)
    total_bases = seq_len * num_pairs

    avg_score = sum(r.score for r in last_results) / len(last_results)
    avg_mismatches = sum(r.mismatches for r in last_results) / len(last_results)

    return {
        "label": label,
        "seq_len": seq_len,
        "num_pairs": num_pairs,
        "total_bases": total_bases,
        "rounds": num_rounds,
        "min_elapsed_s": min_elapsed,
        "avg_elapsed_s": avg_elapsed,
        "per_pair_s": min_elapsed / num_pairs,
        "bases_per_sec": total_bases / min_elapsed if min_elapsed > 0 else float("inf"),
        "avg_score": avg_score,
        "avg_mismatches": avg_mismatches,
    }


# ── Batch benchmark ──────────────────────────────────────────────────────
def benchmark_batch(
    label: str,
    seq_len: int,
    num_pairs: int,
    mutation_rate: float,
    rng: random.Random,
    num_rounds: int = NUM_ROUNDS,
) -> dict:
    """Benchmark align_sequences() (batch API) over multiple rounds.

    Returns a dict with timing metrics (min, avg) and result summary.
    """
    pairs = [
        generate_pair_with_mutations(seq_len, mutation_rate, rng)
        for _ in range(num_pairs)
    ]

    run_times: List[float] = []
    last_results: List[AlignmentResult] = []

    for _ in range(num_rounds):
        t_start = time.perf_counter()
        results = align_sequences(pairs)
        t_end = time.perf_counter()
        run_times.append(t_end - t_start)
        last_results = results

    min_elapsed = min(run_times)
    avg_elapsed = sum(run_times) / len(run_times)
    total_bases = seq_len * num_pairs

    avg_score = sum(r.score for r in last_results) / len(last_results)

    return {
        "label": label,
        "seq_len": seq_len,
        "num_pairs": num_pairs,
        "total_bases": total_bases,
        "rounds": num_rounds,
        "min_elapsed_s": min_elapsed,
        "avg_elapsed_s": avg_elapsed,
        "per_pair_s": min_elapsed / num_pairs,
        "bases_per_sec": total_bases / min_elapsed if min_elapsed > 0 else float("inf"),
        "avg_score": avg_score,
    }


# ── Chunk alignment benchmark ─────────────────────────────────────────────
def benchmark_chunks(
    label: str,
    chunk_size: int,
    num_chunks: int,
    mutation_rate: float,
    rng: random.Random,
    num_rounds: int = NUM_ROUNDS,
) -> dict:
    """Benchmark align_chunks() (Member 2 chunk format) over multiple rounds.

    Simulates aligning genomic chunks against a target sequence.
    """
    target = generate_random_dna(chunk_size, rng)
    chunks = []
    for _ in range(num_chunks):
        _, query = generate_pair_with_mutations(chunk_size, mutation_rate, rng)
        chunks.append(query)

    run_times: List[float] = []
    last_results: List[AlignmentResult] = []

    for _ in range(num_rounds):
        t_start = time.perf_counter()
        results = align_chunks(chunks, target)
        t_end = time.perf_counter()
        run_times.append(t_end - t_start)
        last_results = results

    min_elapsed = min(run_times)
    avg_elapsed = sum(run_times) / len(run_times)
    total_bases = chunk_size * num_chunks

    avg_score = sum(r.score for r in last_results) / len(last_results)

    return {
        "label": label,
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "total_bases": total_bases,
        "rounds": num_rounds,
        "min_elapsed_s": min_elapsed,
        "avg_elapsed_s": avg_elapsed,
        "per_chunk_s": min_elapsed / num_chunks,
        "bases_per_sec": total_bases / min_elapsed if min_elapsed > 0 else float("inf"),
        "avg_score": avg_score,
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    rng = random.Random(RANDOM_SEED)
    mutation_rate = 0.10  # 10% mismatch rate -- realistic for variant analysis

    print()
    print("+=============================================================================+")
    print("|          GeneWeaver -- CPU DNA Alignment Benchmark (Week 1 Baseline)        |")
    print("+=============================================================================+")
    print()

    # ── Environment info ─────────────────────────────────────────────
    print("Environment:")
    print(f"  Python     : {sys.version}")
    print(f"  Platform   : {platform.platform()}")
    print(f"  Processor  : {platform.processor()}")
    print(f"  Mutation   : {mutation_rate * 100:.0f}%")
    print(f"  RNG Seed   : {RANDOM_SEED}")
    print(f"  Repetitions: {NUM_ROUNDS} rounds per configuration")
    print()

    # ── Per-pair benchmarks ──────────────────────────────────────────
    print_separator("=")
    print("  Per-pair benchmark  (align_sequence called individually)")
    print_separator("=")
    print(
        f"{'Size':<12s} {'Seq Len':>8s} {'Pairs':>7s} {'Total Bases':>12s} "
        f"{'Avg Time':>11s} {'Min Time':>11s} {'Per Pair':>10s} "
        f"{'Bases/sec':>12s} {'Score':>7s}"
    )
    print_separator()

    single_results = []
    for label, seq_len, num_pairs in BENCHMARK_CONFIGS:
        r = benchmark_single(label, seq_len, num_pairs, mutation_rate, rng)
        single_results.append(r)
        print(
            f"{r['label']:<12s} {r['seq_len']:>8,d} {r['num_pairs']:>7,d} "
            f"{r['total_bases']:>12,d} {format_time(r['avg_elapsed_s']):>11s} "
            f"{format_time(r['min_elapsed_s']):>11s} {format_time(r['per_pair_s']):>10s} "
            f"{r['bases_per_sec']:>12,.0f} {r['avg_score']:>7.4f}"
        )

    print_separator()
    print()

    # ── Batch benchmarks ─────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)

    print_separator("=")
    print("  Batch benchmark  (align_sequences called once per config)")
    print_separator("=")
    print(
        f"{'Size':<12s} {'Seq Len':>8s} {'Pairs':>7s} {'Total Bases':>12s} "
        f"{'Avg Time':>11s} {'Min Time':>11s} {'Per Pair':>10s} "
        f"{'Bases/sec':>12s} {'Score':>7s}"
    )
    print_separator()

    batch_results = []
    for label, seq_len, num_pairs in BENCHMARK_CONFIGS:
        r = benchmark_batch(label, seq_len, num_pairs, mutation_rate, rng)
        batch_results.append(r)
        print(
            f"{r['label']:<12s} {r['seq_len']:>8,d} {r['num_pairs']:>7,d} "
            f"{r['total_bases']:>12,d} {format_time(r['avg_elapsed_s']):>11s} "
            f"{format_time(r['min_elapsed_s']):>11s} {format_time(r['per_pair_s']):>10s} "
            f"{r['bases_per_sec']:>12,.0f} {r['avg_score']:>7.4f}"
        )

    print_separator()
    print()

    # ── Chunk alignment benchmark ─────────────────────────────────────
    rng = random.Random(RANDOM_SEED)

    print_separator("=")
    print("  Chunk alignment benchmark  (align_chunks - Member 2 format)")
    print_separator("=")
    print(
        f"{'Size':<12s} {'Chunk Sz':>8s} {'Chunks':>7s} {'Total Bases':>12s} "
        f"{'Avg Time':>11s} {'Min Time':>11s} {'Per Chunk':>10s} "
        f"{'Bases/sec':>12s} {'Score':>7s}"
    )
    print_separator()

    chunk_results = []
    chunk_configs = [
        ("Small",       100,   1_000),
        ("Medium",      1_000, 1_000),
        ("Large",       5_000, 200),
    ]
    for label, chunk_sz, num_chk in chunk_configs:
        r = benchmark_chunks(label, chunk_sz, num_chk, mutation_rate, rng)
        chunk_results.append(r)
        print(
            f"{r['label']:<12s} {r['chunk_size']:>8,d} {r['num_chunks']:>7,d} "
            f"{r['total_bases']:>12,d} {format_time(r['avg_elapsed_s']):>11s} "
            f"{format_time(r['min_elapsed_s']):>11s} {format_time(r['per_chunk_s']):>10s} "
            f"{r['bases_per_sec']:>12,.0f} {r['avg_score']:>7.4f}"
        )

    print_separator()
    print()

    # ── Summary ──────────────────────────────────────────────────────
    total_bases_all = sum(r["total_bases"] for r in single_results)
    total_min_time_all = sum(r["min_elapsed_s"] for r in single_results)
    total_avg_time_all = sum(r["avg_elapsed_s"] for r in single_results)
    print(f"  Total bases per configuration set  : {total_bases_all:>14,d}")
    print(f"  Total min wall-clock time (single) : {format_time(total_min_time_all):>14s}")
    print(f"  Total avg wall-clock time (single) : {format_time(total_avg_time_all):>14s}")
    if total_min_time_all > 0:
        print(
            f"  Peak throughput (best-of-{NUM_ROUNDS})       : "
            f"{total_bases_all / total_min_time_all:>14,.0f} bases/sec"
        )
    print()
    print("Benchmark complete.")
    print()


if __name__ == "__main__":
    main()
