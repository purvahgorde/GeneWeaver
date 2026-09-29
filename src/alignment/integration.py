"""Integration pipeline connecting Member 2's parser/chunker with Member 1's CPU alignment.

Provides an end-to-end execution path:
FASTA file -> BioPython SeqIO parser -> Genome chunker -> CPU chunk alignment.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parser.fasta_parser import parse_fasta
from chunking.genome_chunker import chunk_sequence
from alignment.cpu_alignment import align_chunks, AlignmentResult


def run_pipeline(
    fasta_path: str | Path,
    target: str,
    chunk_size: int,
    *,
    skip_unequal: bool = True,
) -> list[AlignmentResult]:
    """Execute the end-to-end Week 1 DNA alignment pipeline.

    Parses a FASTA file using Member 2's parser, partitions all sequences into
    windows of size `chunk_size` using Member 2's genome chunker, and aligns
    each resulting chunk against `target` using Member 1's CPU alignment engine.

    Parameters
    ----------
    fasta_path : str | Path
        Path to the input FASTA file.
    target : str
        Target DNA sequence to align each genomic chunk against (e.g. guide RNA).
    chunk_size : int
        Window size in bases for genome chunking.
    skip_unequal : bool, default=True
        If True, residual tail chunks whose length differs from len(target)
        are skipped. If False, any length mismatch raises a ValueError.

    Returns
    -------
    list[AlignmentResult]
        Alignment results for each chunk, preserving chunk order.
    """
    sequences = parse_fasta(fasta_path)
    all_chunks: list[str] = []
    for seq in sequences:
        chunks = chunk_sequence(seq, chunk_size)
        all_chunks.extend(chunks)

    return align_chunks(all_chunks, target, skip_unequal=skip_unequal)


if __name__ == "__main__":
    mock_path = REPO_ROOT / "data" / "mock_genome.fasta"
    target_query = "ATGCGTACGT"  # 10-mer guide target
    results = run_pipeline(mock_path, target_query, chunk_size=10, skip_unequal=True)
    print(f"Processed {len(results)} chunks against target '{target_query}':")
    for i, res in enumerate(results):
        print(
            f"  Chunk {i:02d}: matches={res.matches}, "
            f"mismatches={res.mismatches}, score={res.score:.2f}, "
            f"mismatch_positions={res.mismatch_positions}"
        )
