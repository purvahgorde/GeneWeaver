"""GeneWeaver alignment package.

Provides CPU (and future GPU) DNA sequence alignment implementations.
"""

from .cpu_alignment import align_sequence, align_sequences

__all__ = ["align_sequence", "align_sequences"]
