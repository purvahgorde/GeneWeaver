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

__all__ = [
    "align_sequence",
    "align_sequences",
    "align_chunks",
    "align_chunk_pairs",
    "AlignmentResult",
]

