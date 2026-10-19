"""Integration pipeline connecting Member 2's parser/chunker with Member 1's CPU/CUDA alignment.

Week 2 Day 3: CUDA Pipeline Integration.

Provides an end-to-end execution path:
FASTA file -> BioPython SeqIO parser -> Genome chunker -> DNA Chunks -> Alignment (CPU or CUDA).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parser.fasta_parser import parse_fasta
from chunking.genome_chunker import chunk_sequence
from alignment.cpu_alignment import align_chunks, AlignmentResult
from alignment.cuda_alignment import (
    cuda_align_chunks,
    cuda_align_chunks_timed,
    CUDANotAvailableError,
    CUDATimingBreakdown,
    is_cuda_available,
)
from alignment.dispatcher import align_chunks_dispatch, normalize_backend


@dataclass(frozen=True)
class PipelineTimingBreakdown:

    parse_time: float
    chunk_time: float
    alignment_time: float
    total_pipeline_time: float
    num_sequences: int
    num_chunks: int
    num_aligned: int
    backend: str
    cuda_timing: CUDATimingBreakdown | None = None


def run_pipeline(
    fasta_path: str | Path,
    target: str,
    chunk_size: int,
    *,
    backend: str = "cpu",
    skip_unequal: bool = True,
    threads_per_block: int = 128,
    use_shared_memory: bool = False,
) -> list[AlignmentResult]:
    
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError(f"chunk_size must be an int, got {type(chunk_size).__name__}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    norm_backend = normalize_backend(backend)

    sequences = parse_fasta(fasta_path)
    all_chunks: list[str] = []
    for seq in sequences:
        chunks = chunk_sequence(seq, chunk_size)
        all_chunks.extend(chunks)

    return align_chunks_dispatch(
        all_chunks,
        target,
        backend=norm_backend,
        skip_unequal=skip_unequal,
        threads_per_block=threads_per_block,
        use_shared_memory=use_shared_memory,
    )


def run_pipeline_timed(
    fasta_path: str | Path,
    target: str,
    chunk_size: int,
    *,
    backend: str = "cpu",
    skip_unequal: bool = True,
    threads_per_block: int = 128,
    use_shared_memory: bool = False,
) -> tuple[list[AlignmentResult], PipelineTimingBreakdown]:
    
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError(f"chunk_size must be an int, got {type(chunk_size).__name__}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    norm_backend = normalize_backend(backend)

    t_wall_start = time.perf_counter()

    # Stage 1: FASTA Parsing
    t_parse_start = time.perf_counter()
    sequences = parse_fasta(fasta_path)
    t_parse = time.perf_counter() - t_parse_start

    # Stage 2: Genome Chunking
    t_chunk_start = time.perf_counter()
    all_chunks: list[str] = []
    for seq in sequences:
        chunks = chunk_sequence(seq, chunk_size)
        all_chunks.extend(chunks)
    t_chunk = time.perf_counter() - t_chunk_start

    # Stage 3: Alignment (CPU or CUDA)
    cuda_timing: CUDATimingBreakdown | None = None
    if norm_backend == "cuda":
        results, cuda_timing = cuda_align_chunks_timed(
            all_chunks,
            target,
            skip_unequal=skip_unequal,
            threads_per_block=threads_per_block,
            use_shared_memory=use_shared_memory,
        )
        t_align = cuda_timing.total_wall_time
    else:
        t_align_start = time.perf_counter()
        results = align_chunks(all_chunks, target, skip_unequal=skip_unequal)
        t_align = time.perf_counter() - t_align_start

    total_time = time.perf_counter() - t_wall_start

    breakdown = PipelineTimingBreakdown(
        parse_time=t_parse,
        chunk_time=t_chunk,
        alignment_time=t_align,
        total_pipeline_time=total_time,
        num_sequences=len(sequences),
        num_chunks=len(all_chunks),
        num_aligned=len(results),
        backend=norm_backend,
        cuda_timing=cuda_timing,
    )

    return results, breakdown


if __name__ == "__main__":
    mock_path = REPO_ROOT / "data" / "mock_genome.fasta"
    target_query = "ATGCGTACGT"  # 10-mer guide target

    print("--- GeneWeaver End-to-End Pipeline Execution ---")
    print(f"Target sequence: {target_query} (length: {len(target_query)})")
    print(f"CUDA Available : {is_cuda_available()}")

    # CPU Execution
    cpu_res, cpu_time = run_pipeline_timed(
        mock_path, target_query, chunk_size=10, backend="cpu", skip_unequal=True
    )
    print(f"\n[CPU Backend] Processed {len(cpu_res)} chunks:")
    print(f"  Parse time    : {cpu_time.parse_time * 1e3:.2f} ms")
    print(f"  Chunk time    : {cpu_time.chunk_time * 1e3:.2f} ms")
    print(f"  Alignment time: {cpu_time.alignment_time * 1e3:.2f} ms")
    print(f"  Total time    : {cpu_time.total_pipeline_time * 1e3:.2f} ms")

    # CUDA Execution
    if is_cuda_available():
        gpu_res, gpu_time = run_pipeline_timed(
            mock_path, target_query, chunk_size=10, backend="cuda", skip_unequal=True
        )
        print(f"\n[CUDA Backend] Processed {len(gpu_res)} chunks:")
        print(f"  Parse time    : {gpu_time.parse_time * 1e3:.2f} ms")
        print(f"  Chunk time    : {gpu_time.chunk_time * 1e3:.2f} ms")
        if gpu_time.cuda_timing:
            ct = gpu_time.cuda_timing
            print(f"  H2D Transfer  : {ct.h2d_transfer_time * 1e3:.2f} ms")
            print(f"  Kernel Exec   : {ct.kernel_execution_time * 1e3:.2f} ms")
            print(f"  D2H Transfer  : {ct.d2h_transfer_time * 1e3:.2f} ms")
            print(f"  GPU Total     : {ct.total_gpu_time * 1e3:.2f} ms")
        print(f"  Total time    : {gpu_time.total_pipeline_time * 1e3:.2f} ms")
        assert cpu_res == gpu_res
        print("  Parity: CPU and CUDA results match 100%.")
    else:
        print("\n[CUDA Backend] Skipped kernel execution: CUDA unavailable in environment.")
