"""Mismatch visualization utilities for GeneWeaver.

Provides rich visual representations of sequence alignments, highlighting
matches, mismatches, and mismatch positions for terminal, dashboard, and text outputs.
"""

from __future__ import annotations

from typing import Sequence
from rich.text import Text


def format_dna_alignment(chunks: Sequence[str], target: str) -> Text:
    """Format aligned DNA chunks with Rich styling, highlighting matches and mismatches.

    Parameters
    ----------
    chunks : Sequence[str]
        Genomic sequence chunks to compare against the target.
    target : str
        Target guide sequence.

    Returns
    -------
    Text
        Rich Text object suitable for rendering in Textual dashboards or Rich console.
    """
    output = Text()

    output.append("TARGET  ", style="bold cyan")
    output.append(target + "\n\n", style="bold cyan")
    output.append("Legend: ", style="bold")
    output.append("Match ", style="green")
    output.append("Mismatch\n\n", style="red")

    if not chunks:
        output.append("No chunks available for comparison.")
        return output

    for index, chunk in enumerate(chunks, start=1):
        output.append(f"Chunk {index:<3} ", style="bold")
        mismatch_positions: list[int] = []

        for position, (base, target_base) in enumerate(
            zip(chunk, target), start=1
        ):
            if base == target_base:
                output.append(base, style="green")
            else:
                output.append(base, style="red")
                mismatch_positions.append(position)

        output.append("\n")
        output.append("         ")

        if mismatch_positions:
            positions_text = ", ".join(str(pos) for pos in mismatch_positions)
            output.append(
                f"Mismatch positions: {positions_text}\n",
                style="red",
            )
        else:
            output.append("Perfect match\n", style="green")

        output.append("\n")

    return output


def render_mismatch_ascii(chunk: str, target: str) -> str:
    """Render a text/ASCII representation of alignment between a chunk and target.

    Matches are indicated by '|', mismatches by '.', and length differences by ' '.

    Parameters
    ----------
    chunk : str
        Candidate sequence chunk.
    target : str
        Target reference sequence.

    Returns
    -------
    str
        Multi-line string displaying chunk, alignment match line, and target.
    """
    max_len = max(len(chunk), len(target))
    chunk_padded = chunk.ljust(max_len)
    target_padded = target.ljust(max_len)

    match_chars = []
    for c_base, t_base in zip(chunk_padded, target_padded):
        if c_base == " " or t_base == " ":
            match_chars.append(" ")
        elif c_base.upper() == t_base.upper():
            match_chars.append("|")
        else:
            match_chars.append(".")

    match_line = "".join(match_chars)
    return f"Chunk : {chunk}\n        {match_line}\nTarget: {target}"


def get_mismatch_indices(chunk: str, target: str, one_based: bool = True) -> list[int]:
    """Return indices of mismatching positions between two equal-length sequences.

    Parameters
    ----------
    chunk : str
        Candidate DNA sequence.
    target : str
        Target DNA sequence.
    one_based : bool, default=True
        Whether to return 1-based indexing (standard biological notation)
        or 0-based indexing.

    Returns
    -------
    list[int]
        List of position indices where chunk and target differ.
    """
    start = 1 if one_based else 0
    return [
        pos
        for pos, (c, t) in enumerate(zip(chunk.upper(), target.upper()), start=start)
        if c != t
    ]


__all__ = [
    "format_dna_alignment",
    "render_mismatch_ascii",
    "get_mismatch_indices",
]
