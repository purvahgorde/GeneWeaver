"""CPU DNA Sequence Alignment — Reference Implementation.

This module provides a simple, deterministic, character-by-character DNA
sequence alignment on the CPU.  It is intentionally kept free of GPU (CUDA /
Numba) and distributed (Dask) dependencies so that it can serve as the
correctness baseline when those accelerated back-ends are introduced in
later weeks.

Input contract
--------------
Both ``reference`` and ``query`` are plain Python strings composed of
upper-case IUPAC DNA bases (A, C, G, T).  The two strings must be equal
in length (pre-aligned / pre-chunked by the upstream parser).

If a parser / chunker module is added later by Member 2, it should
produce string pairs that satisfy this contract.

Output contract
---------------
``align_sequence`` returns an ``AlignmentResult`` dataclass with:
    - ``matches``        (int)  : number of positions where bases agree
    - ``mismatches``     (int)  : number of positions where bases differ
    - ``alignment_length`` (int)  : total compared positions (== len(reference))
    - ``score``          (float): fraction of matches  (matches / alignment_length)
    - ``mismatch_positions`` (list[int]): 0-based indices of mismatch positions

All outputs are fully deterministic for the same input pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ── Valid DNA bases (upper-case IUPAC standard bases) ──────────────────────
VALID_BASES: frozenset[str] = frozenset("ACGT")


# ── Result container ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class AlignmentResult:
    """Immutable container for the result of a pairwise DNA alignment.

    Attributes
    ----------
    matches : int
        Number of positions where the reference and query bases are identical.
    mismatches : int
        Number of positions where the reference and query bases differ.
    alignment_length : int
        Total number of compared positions (equal to the length of the
        input sequences).
    score : float
        Fraction of matching positions (matches / alignment_length).
        0.0 when alignment_length is 0.
    mismatch_positions : list[int]
        Sorted, 0-based indices where mismatches occur.
    """

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
    """Validate and normalise a single DNA sequence string.

    Parameters
    ----------
    seq : str
        The raw input sequence.
    name : str
        Human-readable label used in error messages (e.g. ``"reference"``).

    Returns
    -------
    str
        The validated, upper-cased sequence.

    Raises
    ------
    TypeError
        If *seq* is not a string.
    ValueError
        If *seq* contains characters outside the valid DNA alphabet.
    """
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
    """Perform character-by-character DNA alignment on the CPU.

    This is the reference implementation against which future GPU-accelerated
    versions will be validated.

    Parameters
    ----------
    reference : str
        The reference DNA sequence (e.g. from a FASTA file).
    query : str
        The query DNA sequence to align against the reference.

    Returns
    -------
    AlignmentResult
        A dataclass with match/mismatch counts, score, and positions.

    Raises
    ------
    TypeError
        If either argument is not a string.
    ValueError
        If the sequences contain invalid characters or differ in length.

    Examples
    --------
    >>> result = align_sequence("ACGT", "ACGT")
    >>> result.matches
    4
    >>> result.score
    1.0

    >>> result = align_sequence("ACGT", "ACGA")
    >>> result.mismatches
    1
    >>> result.mismatch_positions
    [3]
    """
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
    """Align multiple (reference, query) pairs sequentially on the CPU.

    This is a convenience wrapper that applies :func:`align_sequence` to
    every pair in *pairs*.  It exists so that higher-level callers
    (and, later, the GPU / Dask backends) share a uniform batch API.

    Parameters
    ----------
    pairs : list[tuple[str, str]]
        Each element is a ``(reference, query)`` string pair.

    Returns
    -------
    list[AlignmentResult]
        One result per input pair, in the same order.

    Raises
    ------
    TypeError
        If *pairs* is not a list.
    """
    if not isinstance(pairs, list):
        raise TypeError(
            f"pairs must be a list, got {type(pairs).__name__}"
        )
    return [align_sequence(ref, qry) for ref, qry in pairs]


# ── Chunk alignment helpers (Member 2 integration) ───────────────────────
def align_chunks(
    chunks: list[str],
    target: str,
    *,
    skip_unequal: bool = False,
) -> list[AlignmentResult]:
    """Align a list of genomic chunks against a target sequence on the CPU.

    This function provides the integration path between Member 2's chunker
    (which yields a list of chunk strings) and Member 1's CPU alignment.
    In CRISPR search workflows, the ``target`` represents the guide RNA or
    reference query, and ``chunks`` represent candidate genomic windows.

    Parameters
    ----------
    chunks : list[str]
        A list of DNA sequence strings (e.g. from ``chunk_sequence()``).
    target : str
        Target reference/query sequence to compare each chunk against.
    skip_unequal : bool, default=False
        If True, chunks whose length differs from ``len(target)`` (e.g.,
        residual tail chunks at chromosome boundaries) are omitted.
        If False, any length mismatch raises a ValueError from
        :func:`align_sequence`.

    Returns
    -------
    list[AlignmentResult]
        Alignment results for each matching chunk, preserving the existing
        ``AlignmentResult`` contract and input order.

    Raises
    ------
    TypeError
        If *chunks* is not a list or *target* is not a string.
    ValueError
        If *target* contains invalid characters, or if a chunk differs in
        length from *target* when *skip_unequal* is False.
    """
    if not isinstance(chunks, list):
        raise TypeError(f"chunks must be a list, got {type(chunks).__name__}")
    if not isinstance(target, str):
        raise TypeError(f"target must be a string, got {type(target).__name__}")

    # Validate target once upfront
    target_norm = _validate_sequence(target, "target")

    results: list[AlignmentResult] = []
    target_len = len(target_norm)

    for chunk in chunks:
        if not isinstance(chunk, str):
            raise TypeError(f"Each chunk must be a string, got {type(chunk).__name__}")
        if skip_unequal and len(chunk) != target_len:
            continue
        results.append(align_sequence(chunk, target_norm))

    return results


def align_chunk_pairs(
    reference_chunks: list[str],
    query_chunks: list[str],
) -> list[AlignmentResult]:
    """Align corresponding pairs of chunks from two chunk lists on the CPU.

    Parameters
    ----------
    reference_chunks : list[str]
        Reference sequence chunks.
    query_chunks : list[str]
        Query sequence chunks to compare against reference chunks.

    Returns
    -------
    list[AlignmentResult]
        One result per chunk pair, in order.

    Raises
    ------
    TypeError
        If either argument is not a list.
    ValueError
        If the two lists have different lengths.
    """
    if not isinstance(reference_chunks, list):
        raise TypeError(
            f"reference_chunks must be a list, got {type(reference_chunks).__name__}"
        )
    if not isinstance(query_chunks, list):
        raise TypeError(
            f"query_chunks must be a list, got {type(query_chunks).__name__}"
        )
    if len(reference_chunks) != len(query_chunks):
        raise ValueError(
            f"Chunk lists must have equal length: "
            f"reference has {len(reference_chunks)}, query has {len(query_chunks)}."
        )

    return [
        align_sequence(ref, qry)
        for ref, qry in zip(reference_chunks, query_chunks)
    ]