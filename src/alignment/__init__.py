"""Alignment package for the GeneWeaver sequence alignment engine.

Exports public CPU alignment routines and result types.
"""

from .cpu_alignment import (
    align_sequence,
    align_sequences,
    align_chunks,
    align_chunk_pairs,
    AlignmentResult,
)
from .cuda_alignment import (
    cuda_align_sequence,
    cuda_align_sequences,
    cuda_align_sequences_timed,
    cuda_align_chunks,
    is_cuda_available,
    get_cuda_device_info,
    CUDANotAvailableError,
    CUDATimingBreakdown,
)

__all__ = [
    "align_sequence",
    "align_sequences",
    "align_chunks",
    "align_chunk_pairs",
    "AlignmentResult",
    "cuda_align_sequence",
    "cuda_align_sequences",
    "cuda_align_sequences_timed",
    "cuda_align_chunks",
    "is_cuda_available",
    "get_cuda_device_info",
    "CUDANotAvailableError",
    "CUDATimingBreakdown",
]

