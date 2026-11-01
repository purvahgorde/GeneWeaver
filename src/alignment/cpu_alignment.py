"""CPU DNA Sequence Alignment Reference Implementation.

Provides deterministic, single-threaded, character-by-character DNA sequence
alignment algorithms for baseline correctness and benchmarking. Designed as the
unaccelerated reference oracle for subsequent GPU (CUDA) and distributed (Dask)
implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# Valid uppercase DNA bases
VALID_BASES: frozenset[str] = frozenset("ACGT")


@dataclass(frozen=True, eq=False)
class AlignmentResult:
    """Immutable result container for DNA sequence alignment.

    Attributes:
        matches: Number of identical base positions.
        mismatches: Number of differing base positions.
        alignment_length: Total length of the aligned sequences.
        score: Fraction of matching positions, or 0.0 for empty sequences.
        mismatch_positions: Zero-based indices where mismatches occurred.
    """

    matches: int
    mismatches: int
    alignment_length: int
    score: float
    mismatch_positions: list[int] = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        if not hasattr(other, "matches"):
            return False
        return (
            self.matches == getattr(other, "matches", None)
            and self.mismatches == getattr(other, "mismatches", None)
            and self.alignment_length == getattr(other, "alignment_length", None)
            and math.isclose(
                self.score,
                getattr(other, "score", 0.0),
                abs_tol=1e-5,
            )
            and list(self.mismatch_positions)
            == list(getattr(other, "mismatch_positions", []))
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.matches,
                self.mismatches,
                self.alignment_length,
                round(self.score, 6),
                tuple(self.mismatch_positions),
            )
        )

    def __repr__(self) -> str:
        return (
            f"AlignmentResult(matches={self.matches}, "
            f"mismatches={self.mismatches}, "
            f"length={self.alignment_length}, "
            f"score={self.score:.4f})"
        )


def _validate_sequence(seq: str, name: str) -> str:
    """Validate and normalize a DNA sequence to uppercase."""
    if not isinstance(seq, str):
        raise TypeError(f"{name} must be a string, got {type(seq).__name__}")

    seq_upper = seq.upper()
    invalid_chars = set(seq_upper) - VALID_BASES

    if invalid_chars and seq_upper != "":
        raise ValueError(
            f"{name} contains invalid characters: "
            f"{', '.join(sorted(invalid_chars))}. "
            "Only A, C, G, T are allowed."
        )

    return seq_upper


def align_sequence(reference: str, query: str) -> AlignmentResult:
    """Align two equal-length DNA sequences on the CPU.

    Comparison is case-insensitive and restricted to A, C, G, and T.
    """
    reference = _validate_sequence(reference, "reference")
    query = _validate_sequence(query, "query")

    if len(reference) != len(query):
        raise ValueError(
            f"Sequences must have equal length: "
            f"reference has {len(reference)}, query has {len(query)}."
        )

    alignment_length = len(reference)

    if alignment_length == 0:
        return AlignmentResult(
            matches=0,
            mismatches=0,
            alignment_length=0,
            score=0.0,
            mismatch_positions=[],
        )

    matches = 0
    mismatches = 0
    mismatch_positions: list[int] = []

    for i in range(alignment_length):
        if reference[i] == query[i]:
            matches += 1
        else:
            mismatches += 1
            mismatch_positions.append(i)

    return AlignmentResult(
        matches=matches,
        mismatches=mismatches,
        alignment_length=alignment_length,
        score=matches / alignment_length,
        mismatch_positions=mismatch_positions,
    )


def align_sequences(
    pairs: list[tuple[str, str]],
) -> list[AlignmentResult]:
    """Align a batch of DNA sequence pairs sequentially on the CPU."""
    if not isinstance(pairs, list):
        raise TypeError(f"pairs must be a list, got {type(pairs).__name__}")

    return [align_sequence(ref, qry) for ref, qry in pairs]


def align_chunks(
    chunks: list[str],
    target: str,
    *,
    skip_unequal: bool = False,
) -> list[AlignmentResult]:
    """Align genomic chunks against a target sequence.

    If skip_unequal is True, chunks whose lengths differ from the target are
    skipped. Otherwise, a length mismatch raises ValueError.
    """
    if not isinstance(chunks, list):
        raise TypeError(f"chunks must be a list, got {type(chunks).__name__}")
    if not isinstance(target, str):
        raise TypeError(f"target must be a string, got {type(target).__name__}")

    target_norm = _validate_sequence(target, "target")
    target_len = len(target_norm)
    results: list[AlignmentResult] = []

    for chunk in chunks:
        if not isinstance(chunk, str):
            raise TypeError(
                f"Each chunk must be a string, got {type(chunk).__name__}"
            )
        if skip_unequal and len(chunk) != target_len:
            continue

        results.append(align_sequence(chunk, target_norm))

    return results


def align_chunk_pairs(
    reference_chunks: list[str],
    query_chunks: list[str],
) -> list[AlignmentResult]:
    """Align corresponding pairs of chunks on the CPU."""
    if not isinstance(reference_chunks, list):
        raise TypeError(
            "reference_chunks must be a list, "
            f"got {type(reference_chunks).__name__}"
        )
    if not isinstance(query_chunks, list):
        raise TypeError(
            "query_chunks must be a list, "
            f"got {type(query_chunks).__name__}"
        )
    if len(reference_chunks) != len(query_chunks):
        raise ValueError(
            f"Chunk lists must have equal length: "
            f"reference has {len(reference_chunks)}, "
            f"query has {len(query_chunks)}."
        )

    return [
        align_sequence(ref, qry)
        for ref, qry in zip(reference_chunks, query_chunks)
    ]
