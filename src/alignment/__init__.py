"""Alignment package for the GeneWeaver sequence alignment engine.

Exports public CPU and CUDA alignment routines, dispatchers, pipelines, and result types.
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
    cuda_align_chunks_shared,
    cuda_align_chunks_timed,
    cuda_align_chunk_pairs,
    is_cuda_available,
    get_cuda_device_info,
    CUDANotAvailableError,
    CUDATimingBreakdown,
    MAX_SHARED_TARGET_LEN,
    dna_align_shared_kernel,
)
from .dispatcher import (
    align_sequence_dispatch,
    align_sequences_dispatch,
    align_chunks_dispatch,
    align_chunk_pairs_dispatch,
    normalize_backend,
    SUPPORTED_BACKENDS,
)
from .integration import (
    run_pipeline,
    run_pipeline_timed,
    PipelineTimingBreakdown,
)

__all__ = [
    # CPU Core
    "align_sequence",
    "align_sequences",
    "align_chunks",
    "align_chunk_pairs",
    "AlignmentResult",
    # CUDA Core
    "cuda_align_sequence",
    "cuda_align_sequences",
    "cuda_align_sequences_timed",
    "cuda_align_chunks",
    "cuda_align_chunks_shared",
    "cuda_align_chunks_timed",
    "cuda_align_chunk_pairs",
    "is_cuda_available",
    "get_cuda_device_info",
    "CUDANotAvailableError",
    "CUDATimingBreakdown",
    "MAX_SHARED_TARGET_LEN",
    "dna_align_shared_kernel",
    # Dispatcher
    "align_sequence_dispatch",
    "align_sequences_dispatch",
    "align_chunks_dispatch",
    "align_chunk_pairs_dispatch",
    "normalize_backend",
    "SUPPORTED_BACKENDS",
    # End-to-End Pipeline
    "run_pipeline",
    "run_pipeline_timed",
    "PipelineTimingBreakdown",
]
