from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ── Valid DNA bases (upper-case IUPAC standard bases) ──────────────────────
VALID_BASES: frozenset[str] = frozenset("ACGT")


# ── Result container ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class AlignmentResult:

    matches: int
    mismatches: int
    alignment_length: int
    score: float
    mismatch_positions: List[int] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AlignmentResult(matches={self.matches}, "
            f"mismatches={self.mismatches}, "
            f"length={self.alignment_length}, "
            f"score={self.score:.4f})"
        )


# ── Validation helpers ────────────────────────────────────────────────────
def _validate_sequence(seq: str, name: str) -> str:
    
    if not isinstance(seq, str):
        raise TypeError(
            f"{name} must be a string, got {type(seq).__name__}"
        )
    seq_upper = seq.upper()
    invalid_chars = set(seq_upper) - VALID_BASES
    if invalid_chars and seq_upper != "":
        raise ValueError(
            f"{name} contains invalid characters: "
            f"{', '.join(sorted(invalid_chars))}. "
            f"Only A, C, G, T are allowed."
        )
    return seq_upper


# ── Core alignment function ──────────────────────────────────────────────
def align_sequence(reference: str, query: str) -> AlignmentResult:
    
    # ── Validate inputs ──────────────────────────────────────────────
    reference = _validate_sequence(reference, "reference")
    query = _validate_sequence(query, "query")

    if len(reference) != len(query):
        raise ValueError(
            f"Sequences must have equal length: "
            f"reference has {len(reference)}, query has {len(query)}."
        )

    # ── Handle empty sequences ───────────────────────────────────────
    alignment_length = len(reference)
    if alignment_length == 0:
        return AlignmentResult(
            matches=0,
            mismatches=0,
            alignment_length=0,
            score=0.0,
            mismatch_positions=[],
        )

    # ── Character-by-character comparison ────────────────────────────
    matches = 0
    mismatches = 0
    mismatch_positions: List[int] = []

    for i in range(alignment_length):
        if reference[i] == query[i]:
            matches += 1
        else:
            mismatches += 1
            mismatch_positions.append(i)

    score = matches / alignment_length

    return AlignmentResult(
        matches=matches,
        mismatches=mismatches,
        alignment_length=alignment_length,
        score=score,
        mismatch_positions=mismatch_positions,
    )


# ── Batch helper ─────────────────────────────────────────────────────────
def align_sequences(
    pairs: list[tuple[str, str]],
) -> list[AlignmentResult]:
    
    if not isinstance(pairs, list):
        raise TypeError(
            f"pairs must be a list, got {type(pairs).__name__}"
        )
    return [align_sequence(ref, qry) for ref, qry in pairs]