"""Biological scoring module for GeneWeaver.

Provides DNA sequence scoring, PAM site motif scoring, and biological metrics
for sequence alignments.
"""

from __future__ import annotations

from typing import Sequence
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from src.scoring.dna_scoring import (
    score_bases,
    score_sequences,
    validate_dna,
    MATCH_SCORE,
    MISMATCH_SCORE,
    VALID_BASES,
)
from src.pam.pam_detector import find_pam_sites
from src.analysis.pam_scoring import score_pam_sites
from src.alignment.cpu_alignment import AlignmentResult


def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content percentage of a DNA sequence.

    Parameters
    ----------
    sequence : str
        DNA sequence.

    Returns
    -------
    float
        Percentage of G and C bases (0.0 to 100.0).
    """
    validate_dna(sequence)
    if not sequence:
        return 0.0

    seq_upper = sequence.upper()
    gc_count = seq_upper.count("G") + seq_upper.count("C")
    return (gc_count / len(seq_upper)) * 100.0


def score_alignment_result(
    result: AlignmentResult,
    *,
    match_weight: float = 1.0,
    mismatch_weight: float = -1.0,
) -> float:
    """Calculate a weighted biological score for an AlignmentResult.

    Parameters
    ----------
    result : AlignmentResult
        Individual alignment result.
    match_weight : float, default=1.0
        Score weight per matching base.
    mismatch_weight : float, default=-1.0
        Score penalty per mismatching base.

    Returns
    -------
    float
        Computed biological score.
    """
    return (result.matches * match_weight) + (result.mismatches * mismatch_weight)


def score_alignment_results(
    results: Sequence[AlignmentResult],
) -> dict[str, float]:
    """Compute aggregate biological summary metrics over a collection of alignment results.

    Parameters
    ----------
    results : Sequence[AlignmentResult]
        Collection of alignment results.

    Returns
    -------
    dict[str, float]
        Dictionary of aggregate statistics:
        - total_chunks: Total number of aligned chunks
        - total_matches: Sum of matching base pairs
        - total_mismatches: Sum of mismatching base pairs
        - average_similarity: Percent similarity across all aligned bases
        - average_score: Arithmetic mean alignment score
        - max_score: Highest single chunk score
        - min_score: Lowest single chunk score
    """
    if not results:
        return {
            "total_chunks": 0.0,
            "total_matches": 0.0,
            "total_mismatches": 0.0,
            "average_similarity": 0.0,
            "average_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
        }

    total_matches = sum(r.matches for r in results)
    total_mismatches = sum(r.mismatches for r in results)
    total_compared = total_matches + total_mismatches
    similarity = (total_matches / total_compared * 100.0) if total_compared > 0 else 0.0
    scores = [r.score for r in results]

    return {
        "total_chunks": float(len(results)),
        "total_matches": float(total_matches),
        "total_mismatches": float(total_mismatches),
        "average_similarity": round(similarity, 4),
        "average_score": round(sum(scores) / len(scores), 4),
        "max_score": round(max(scores), 4),
        "min_score": round(min(scores), 4),
    }


__all__ = [
    "score_bases",
    "score_sequences",
    "validate_dna",
    "MATCH_SCORE",
    "MISMATCH_SCORE",
    "VALID_BASES",
    "find_pam_sites",
    "score_pam_sites",
    "calculate_gc_content",
    "score_alignment_result",
    "score_alignment_results",
]
