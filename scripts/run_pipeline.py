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
from parser.fasta_parser import parse_fasta
from chunking.genome_chunker import chunk_sequence
from scoring.biological_scoring import score_alignment_results, calculate_gc_content
from pam.pam_detector import find_pam_sites
from visualization.mismatch_visualizer import format_dna_alignment


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
        choices=["cpu", "cuda", "dask", "auto"],
        default="cpu",
        help="Execution backend (cpu, cuda, dask, auto; default: cpu)",
    )
    parser.add_argument(
        "--shared-mem",
        action="store_true",
        help="Enable CUDA on-chip shared-memory target caching optimization",
    )
    parser.add_argument(
        "--fallback-to-cpu",
        action="store_true",
        help="Fallback cleanly to CPU if accelerated backend is unavailable or fails",
    )
    parser.add_argument(
        "--timed",
        action="store_true",
        help="Display execution timings breakdown",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Compute and display biological scoring summary and PAM analysis",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display formatted color-coded DNA mismatch alignment in terminal",
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
        if args.backend == "dask":
            from distributed.dask_processor import (
                process_fasta_dask,
                process_fasta_dask_timed,
                is_dask_available,
                DaskNotAvailableError,
            )

            if not is_dask_available():
                if args.fallback_to_cpu:
                    print("\n[Notice] Dask is unavailable; falling back cleanly to CPU.")
                    if args.timed:
                        results, timings = run_pipeline_timed(
                            fasta_path=args.fasta,
                            target=args.target,
                            chunk_size=args.chunk_size,
                            backend="cpu",
                        )
                        print(f"\nCompleted in {timings.total_pipeline_time * 1000:.3f} ms (Backend used: {timings.backend}):")
                        print(f"  Parse Time    : {timings.parse_time * 1000:.3f} ms")
                        print(f"  Chunk Time    : {timings.chunk_time * 1000:.3f} ms")
                        print(f"  Alignment Time: {timings.alignment_time * 1000:.3f} ms")
                        print(f"  Aligned Chunks: {len(results)}")
                    else:
                        results = run_pipeline(
                            fasta_path=args.fasta,
                            target=args.target,
                            chunk_size=args.chunk_size,
                            backend="cpu",
                        )
                        print(f"\nAligned {len(results)} chunks successfully.")
                else:
                    raise DaskNotAvailableError("Dask is requested but not available in current environment.")
            else:
                try:
                    if args.timed:
                        results, dask_timing = process_fasta_dask_timed(
                            fasta_path=args.fasta,
                            target=args.target,
                            chunk_size=args.chunk_size,
                            skip_unequal=True,
                        )
                        print(f"\nCompleted in {dask_timing.total_pipeline_time * 1000:.3f} ms (Backend used: dask):")
                        print(f"  Parse Time    : {dask_timing.parse_time * 1000:.3f} ms")
                        print(f"  Chunk Time    : {dask_timing.chunk_time * 1000:.3f} ms")
                        print(f"  Alignment Time: {dask_timing.dask_align_time * 1000:.3f} ms")
                        print(f"  Dask Tasks    : {dask_timing.num_tasks}")
                        print(f"  Aligned Chunks: {len(results)}")
                    else:
                        results = process_fasta_dask(
                            fasta_path=args.fasta,
                            target=args.target,
                            chunk_size=args.chunk_size,
                            skip_unequal=True,
                        )
                        print(f"\nAligned {len(results)} chunks successfully.")
                except Exception:
                    if not args.fallback_to_cpu:
                        raise
                    print("\n[Notice] Dask execution encountered an error; falling back cleanly to CPU.")
                    if args.timed:
                        results, timings = run_pipeline_timed(
                            fasta_path=args.fasta,
                            target=args.target,
                            chunk_size=args.chunk_size,
                            backend="cpu",
                        )
                        print(f"\nCompleted in {timings.total_pipeline_time * 1000:.3f} ms (Backend used: {timings.backend}):")
                        print(f"  Parse Time    : {timings.parse_time * 1000:.3f} ms")
                        print(f"  Chunk Time    : {timings.chunk_time * 1000:.3f} ms")
                        print(f"  Alignment Time: {timings.alignment_time * 1000:.3f} ms")
                        print(f"  Aligned Chunks: {len(results)}")
                    else:
                        results = run_pipeline(
                            fasta_path=args.fasta,
                            target=args.target,
                            chunk_size=args.chunk_size,
                            backend="cpu",
                        )
                        print(f"\nAligned {len(results)} chunks successfully.")
        elif args.timed:
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

        if args.score:
            scoring = score_alignment_results(results)
            target_gc = calculate_gc_content(args.target)
            print("\n--- Biological Scoring Summary ---")
            print(f"Target GC Content : {target_gc:.2f}%")
            print(f"Total Chunks      : {int(scoring['total_chunks'])}")
            print(f"Total Matches     : {int(scoring['total_matches'])}")
            print(f"Total Mismatches  : {int(scoring['total_mismatches'])}")
            print(f"Avg Similarity    : {scoring['average_similarity']:.2f}%")
            print(f"Avg Score         : {scoring['average_score']:.4f}")
            print(f"Max Score         : {scoring['max_score']:.4f}")
            print(f"Min Score         : {scoring['min_score']:.4f}")

            # PAM sites in target
            pam_sites = find_pam_sites(args.target)
            if pam_sites:
                print(f"SpCas9 PAMs in Target: {pam_sites}")

        if args.visualize:
            sequences = parse_fasta(args.fasta)
            all_chunks = [
                chk
                for seq in sequences
                for chk in chunk_sequence(seq, args.chunk_size)
                if len(chk) == len(args.target)
            ]
            print("\n--- DNA Alignment Visualization ---")
            from rich.console import Console
            console = Console()
            formatted_text = format_dna_alignment(all_chunks, args.target)
            console.print(formatted_text)

        return 0
    except Exception as exc:
        print(f"Error executing pipeline: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
