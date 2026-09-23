"""GeneWeaver alignment package.

Provides CPU (and future GPU) DNA sequence alignment implementations.
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

