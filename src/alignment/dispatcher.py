"""Unified execution interface and dispatcher for DNA sequence alignment.

Week 2 Day 3: CUDA Pipeline Integration.

Provides a clean, uniform dispatch layer allowing callers to choose between
CPU and CUDA alignment engines using identical inputs and producing identical
AlignmentResult output objects.

Supported backends:
- 'cpu': Single-threaded reference alignment engine (Week 1).
- 'cuda': GPU-accelerated Numba CUDA alignment engine (Week 2).
  'gpu' is accepted as an alias for 'cuda'.
"""

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
    """Validate and normalize backend identifier string.

    Parameters
    ----------
    backend : str
        Target backend ('cpu', 'cuda', or 'gpu').

    Returns
    -------
    str
        Canonical backend name ('cpu' or 'cuda').

    Raises
    ------
    TypeError
        If backend is not a string.
    ValueError
        If backend is not one of 'cpu', 'cuda', or 'gpu'.
    """
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
    """Align two equal-length DNA sequences using the selected backend.

    Parameters
    ----------
    reference : str
        Reference DNA sequence.
    query : str
        Query DNA sequence.
    backend : str, default="cpu"
        Alignment engine to use ('cpu' or 'cuda').

    Returns
    -------
    AlignmentResult
        Result conforming to the standard AlignmentResult contract.

    Raises
    ------
    CUDANotAvailableError
        If backend='cuda' and CUDA is not available.
    ValueError
        If backend is invalid or inputs fail sequence validation.
    TypeError
        If input types are invalid.
    """
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
    """Align a batch of DNA sequence pairs using the selected backend.

    Parameters
    ----------
    pairs : list[tuple[str, str]]
        List of (reference, query) sequence tuples.
    backend : str, default="cpu"
        Alignment engine to use ('cpu' or 'cuda').
    threads_per_block : int, default=128
        CUDA block dimension (ignored for CPU backend).

    Returns
    -------
    list[AlignmentResult]
        Alignment result for each pair, preserving input order.

    Raises
    ------
    CUDANotAvailableError
        If backend='cuda' and CUDA is not available.
    ValueError
        If backend is invalid or inputs fail sequence validation.
    TypeError
        If input types are invalid.
    """
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
    """Align genomic chunks against a target sequence using the selected backend.

    Parameters
    ----------
    chunks : list[str]
        List of genomic chunk strings.
    target : str
        Target DNA sequence to align each chunk against.
    backend : str, default="cpu"
        Alignment engine to use ('cpu' or 'cuda').
    skip_unequal : bool, default=False
        If True, chunks whose length differs from len(target) are skipped.
        If False, length mismatch raises a ValueError.
    threads_per_block : int, default=128
        CUDA block dimension (ignored for CPU backend).
    use_shared_memory : bool, default=False
        If True, utilizes Numba CUDA shared-memory optimization (`dna_align_shared_kernel`)
        when backend='cuda'.

    Returns
    -------
    list[AlignmentResult]
        Alignment results for matching chunks in order.

    Raises
    ------
    CUDANotAvailableError
        If backend='cuda' and CUDA is not available.
    ValueError
        If backend is invalid, target has invalid characters, or chunk length
        mismatches target when skip_unequal=False.
    TypeError
        If input types are invalid.
    """
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
    """Align corresponding pairs of chunks using the selected backend.

    Parameters
    ----------
    reference_chunks : list[str]
        Reference sequence chunks.
    query_chunks : list[str]
        Query sequence chunks.
    backend : str, default="cpu"
        Alignment engine to use ('cpu' or 'cuda').
    threads_per_block : int, default=128
        CUDA block dimension (ignored for CPU backend).

    Returns
    -------
    list[AlignmentResult]
        One result per chunk pair, in order.

    Raises
    ------
    CUDANotAvailableError
        If backend='cuda' and CUDA is not available.
    ValueError
        If backend is invalid or chunk lists have different lengths.
    TypeError
        If input types are invalid.
    """
    target_backend = normalize_backend(backend)
    if target_backend == "cuda":
        return cuda_align_chunk_pairs(
            reference_chunks,
            query_chunks,
            threads_per_block=threads_per_block,
        )
    return align_chunk_pairs(reference_chunks, query_chunks)
