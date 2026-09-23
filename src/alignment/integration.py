"""Pipeline integration connecting Member 2's parser/chunker to Member 1's CPU alignment.

End-to-end workflow:
    FASTA file
        ↓ (Member 2: parse_fasta)
    DNA sequences
        ↓ (Member 2: chunk_sequence)
    DNA chunks
        ↓ (Member 1: align_chunks)
    AlignmentResult list
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

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
) -> List[AlignmentResult]:
    """Execute the end-to-end FASTA -> Chunker -> CPU Alignment pipeline.

    Parameters
    ----------
    fasta_path : str or Path
        Path to the input FASTA file.
    target : str
        Target DNA sequence to align each chunk against (e.g. CRISPR guide).
    chunk_size : int
        Size of chunks produced by the chunker.
    skip_unequal : bool, default=True
        Whether to omit partial tail chunks whose length differs from
        ``len(target)``.

    Returns
    -------
    list[AlignmentResult]
        Alignment results for all processed chunks.
    """
    sequences = parse_fasta(fasta_path)
    all_chunks: List[str] = []
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
