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

SUPPORTED_BACKENDS: tuple[str, ...] = ("cpu", "cuda", "auto")


def normalize_backend(backend: str) -> str:
    """Validate and normalize backend string identifier.

    Supported backends: 'cpu', 'cuda' (alias 'gpu'), and 'auto'.
    """
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a string, got {type(backend).__name__}")

    b = backend.lower().strip()
    if b in ("cuda", "gpu"):
        return "cuda"
    if b == "cpu":
        return "cpu"
    if b == "auto":
        return "auto"

    raise ValueError(
        f"Unsupported backend '{backend}'. Supported backends: {SUPPORTED_BACKENDS}."
    )


def align_sequence_dispatch(
    reference: str,
    query: str,
    *,
    backend: str = "cpu",
    fallback_to_cpu: bool = False,
) -> AlignmentResult:
    """Dispatch single sequence pair alignment to requested backend with optional fallback."""
    target_backend = normalize_backend(backend)
    if target_backend == "cuda":
        try:
            return cuda_align_sequence(reference, query)
        except (CUDANotAvailableError, Exception):
            if fallback_to_cpu:
                return align_sequence(reference, query)
            raise
    elif target_backend == "auto":
        if is_cuda_available():
            try:
                return cuda_align_sequence(reference, query)
            except Exception:
                return align_sequence(reference, query)
        return align_sequence(reference, query)
    return align_sequence(reference, query)


def align_sequences_dispatch(
    pairs: list[tuple[str, str]],
    *,
    backend: str = "cpu",
    threads_per_block: int = 128,
    fallback_to_cpu: bool = False,
) -> list[AlignmentResult]:
    """Dispatch batch of sequence pairs to requested backend with optional fallback."""
    target_backend = normalize_backend(backend)
    if target_backend == "cuda":
        try:
            return cuda_align_sequences(pairs, threads_per_block=threads_per_block)
        except (CUDANotAvailableError, Exception):
            if fallback_to_cpu:
                return align_sequences(pairs)
            raise
    elif target_backend == "auto":
        if is_cuda_available():
            try:
                return cuda_align_sequences(pairs, threads_per_block=threads_per_block)
            except Exception:
                return align_sequences(pairs)
        return align_sequences(pairs)
    return align_sequences(pairs)


def align_chunks_dispatch(
    chunks: list[str],
    target: str,
    *,
    backend: str = "cpu",
    skip_unequal: bool = False,
    threads_per_block: int = 128,
    use_shared_memory: bool = False,
    fallback_to_cpu: bool = False,
) -> list[AlignmentResult]:
    """Dispatch candidate chunks alignment to requested backend with optional fallback."""
    target_backend = normalize_backend(backend)
    if target_backend == "cuda":
        try:
            return cuda_align_chunks(
                chunks,
                target,
                skip_unequal=skip_unequal,
                threads_per_block=threads_per_block,
                use_shared_memory=use_shared_memory,
            )
        except (CUDANotAvailableError, Exception):
            if fallback_to_cpu:
                return align_chunks(chunks, target, skip_unequal=skip_unequal)
            raise
    elif target_backend == "auto":
        if is_cuda_available():
            try:
                return cuda_align_chunks(
                    chunks,
                    target,
                    skip_unequal=skip_unequal,
                    threads_per_block=threads_per_block,
                    use_shared_memory=use_shared_memory,
                )
            except Exception:
                return align_chunks(chunks, target, skip_unequal=skip_unequal)
        return align_chunks(chunks, target, skip_unequal=skip_unequal)
    return align_chunks(chunks, target, skip_unequal=skip_unequal)


def align_chunk_pairs_dispatch(
    reference_chunks: list[str],
    query_chunks: list[str],
    *,
    backend: str = "cpu",
    threads_per_block: int = 128,
    fallback_to_cpu: bool = False,
) -> list[AlignmentResult]:
    """Dispatch chunk pair alignment to requested backend with optional fallback."""
    target_backend = normalize_backend(backend)
    if target_backend == "cuda":
        try:
            return cuda_align_chunk_pairs(
                reference_chunks,
                query_chunks,
                threads_per_block=threads_per_block,
            )
        except (CUDANotAvailableError, Exception):
            if fallback_to_cpu:
                return align_chunk_pairs(reference_chunks, query_chunks)
            raise
    elif target_backend == "auto":
        if is_cuda_available():
            try:
                return cuda_align_chunk_pairs(
                    reference_chunks,
                    query_chunks,
                    threads_per_block=threads_per_block,
                )
            except Exception:
                return align_chunk_pairs(reference_chunks, query_chunks)
        return align_chunk_pairs(reference_chunks, query_chunks)
    return align_chunk_pairs(reference_chunks, query_chunks)
