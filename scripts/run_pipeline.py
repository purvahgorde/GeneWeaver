#!/usr/bin/env python3
"""GeneWeaver Pipeline Execution Script.

Entry point to execute end-to-end genome sequence alignment across
available backends (CPU, CUDA, Dask).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Set up paths
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alignment.integration import run_pipeline, run_pipeline_timed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GeneWeaver CRISPR sequence alignment pipeline."
    )
    parser.add_argument(
        "--fasta",
        type=str,
        default=str(_REPO_ROOT / "data" / "mock_genome.fasta"),
        help="Path to input FASTA file (default: data/mock_genome.fasta)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="ATGCGTACGT",
        help="Target DNA sequence to align against (default: ATGCGTACGT)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Length of each genome chunk (default: 10)",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Execution backend (default: cpu)",
    )
    parser.add_argument(
        "--shared-mem",
        action="store_true",
        help="Enable CUDA on-chip shared-memory target caching optimization",
    )
    parser.add_argument(
        "--fallback-to-cpu",
        action="store_true",
        help="Fallback cleanly to CPU if CUDA backend is unavailable or fails",
    )
    parser.add_argument(
        "--timed",
        action="store_true",
        help="Display microsecond execution timings",
    )

    args = parser.parse_args()

    print("--- GeneWeaver Pipeline Runner ---")
    print(f"FASTA File : {args.fasta}")
    print(f"Target     : {args.target} (length: {len(args.target)})")
    print(f"Chunk Size : {args.chunk_size}")
    print(f"Backend    : {args.backend}")
    print(f"Shared Mem : {args.shared_mem}")
    print(f"Fallback   : {args.fallback_to_cpu}")

    try:
        if args.timed:
            results, timings = run_pipeline_timed(
                fasta_path=args.fasta,
                target=args.target,
                chunk_size=args.chunk_size,
                backend=args.backend,
                use_shared_memory=args.shared_mem,
                fallback_to_cpu=args.fallback_to_cpu,
            )
            print(f"\nCompleted in {timings.total_pipeline_time * 1000:.3f} ms (Backend used: {timings.backend}):")
            print(f"  Parse Time    : {timings.parse_time * 1000:.3f} ms")
            print(f"  Chunk Time    : {timings.chunk_time * 1000:.3f} ms")
            print(f"  Alignment Time: {timings.alignment_time * 1000:.3f} ms")
            if timings.cuda_timing:
                ct = timings.cuda_timing
                print(f"  H2D Transfer  : {ct.h2d_transfer_time * 1000:.3f} ms")
                print(f"  Kernel Exec   : {ct.kernel_execution_time * 1000:.3f} ms")
                print(f"  D2H Transfer  : {ct.d2h_transfer_time * 1000:.3f} ms")
            print(f"  Aligned Chunks: {len(results)}")
        else:
            results = run_pipeline(
                fasta_path=args.fasta,
                target=args.target,
                chunk_size=args.chunk_size,
                backend=args.backend,
                use_shared_memory=args.shared_mem,
                fallback_to_cpu=args.fallback_to_cpu,
            )
            print(f"\nAligned {len(results)} chunks successfully.")

        return 0
    except Exception as exc:
        print(f"Error executing pipeline: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
