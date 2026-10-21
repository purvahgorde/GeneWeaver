from __future__ import annotations

from typing import Sequence, Tuple

from .cpu_alignment import (
    AlignmentResult,
    align_sequence,
    align_sequences,
    align_chunks,
    align_chunk_pairs,
)
from .cuda_alignment import (
    cuda_align_sequence,
    cuda_align_sequences,
    cuda_align_chunks,
    cuda_align_chunk_pairs,
    is_cuda_available,
    CUDANotAvailableError,
)

SUPPORTED_BACKENDS: tuple[str, ...] = ("cpu", "cuda")


def normalize_backend(backend: str) -> str:
    
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a string, got {type(backend).__name__}")

    b = backend.lower().strip()
    if b in ("cuda", "gpu"):
        return "cuda"
    if b == "cpu":
        return "cpu"

    raise ValueError(
        f"Unsupported backend '{backend}'. Supported backends: {SUPPORTED_BACKENDS}."
    )


def align_sequence_dispatch(
    reference: str,
    query: str,
    *,
    backend: str = "cpu",
) -> AlignmentResult:
    
    target_backend = normalize_backend(backend)
    if target_backend == "cuda":
        return cuda_align_sequence(reference, query)
    return align_sequence(reference, query)


def align_sequences_dispatch(
    pairs: list[tuple[str, str]],
    *,
    backend: str = "cpu",
    threads_per_block: int = 128,
) -> list[AlignmentResult]:
    
    target_backend = normalize_backend(backend)
    if target_backend == "cuda":
        return cuda_align_sequences(pairs, threads_per_block=threads_per_block)
    return align_sequences(pairs)


def align_chunks_dispatch(
    chunks: list[str],
    target: str,
    *,
    backend: str = "cpu",
    skip_unequal: bool = False,
    threads_per_block: int = 128,
    use_shared_memory: bool = False,
) -> list[AlignmentResult]:
    
    target_backend = normalize_backend(backend)
    if target_backend == "cuda":
        return cuda_align_chunks(
            chunks,
            target,
            skip_unequal=skip_unequal,
            threads_per_block=threads_per_block,
            use_shared_memory=use_shared_memory,
        )
    return align_chunks(chunks, target, skip_unequal=skip_unequal)


def align_chunk_pairs_dispatch(
    reference_chunks: list[str],
    query_chunks: list[str],
    *,
    backend: str = "cpu",
    threads_per_block: int = 128,
) -> list[AlignmentResult]:
    
    target_backend = normalize_backend(backend)
    if target_backend == "cuda":
        return cuda_align_chunk_pairs(
            reference_chunks,
            query_chunks,
            threads_per_block=threads_per_block,
        )
    return align_chunk_pairs(reference_chunks, query_chunks)
