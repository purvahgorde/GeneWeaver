import sys
from pathlib import Path

# Support running directly or as a package
_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.parser.fasta_parser import parse_fasta
    from src.chunking.genome_chunker import chunk_sequence
except ModuleNotFoundError:
    from parser.fasta_parser import parse_fasta
    from chunking.genome_chunker import chunk_sequence


def process_genome(file_path, chunk_size):
    """Parse FASTA sequence file and split into contiguous chunks.

    Parameters
    ----------
    file_path : str | Path
        Path to FASTA input file.
    chunk_size : int
        Size in base pairs for each chunk.

    Returns
    -------
    list[str]
        List of genomic chunk strings.
    """
    sequences = parse_fasta(file_path)

    all_chunks = []

    for sequence in sequences:
        chunks = chunk_sequence(sequence, chunk_size)
        all_chunks.extend(chunks)

    return all_chunks


def run_alignment_pipeline(
    file_path,
    target,
    chunk_size,
    *,
    backend="cpu",
    skip_unequal=True,
    threads_per_block=128,
    use_shared_memory=False,
    fallback_to_cpu=False,
):
    """End-to-end genomic alignment pipeline: FASTA -> Parse -> Chunk -> Align -> Score.

    Parameters
    ----------
    file_path : str | Path
        Path to FASTA file.
    target : str
        Target DNA sequence to align against.
    chunk_size : int
        Size of candidate windows.
    backend : str, default="cpu"
        Alignment backend ('cpu', 'cuda', 'gpu', 'auto').
    skip_unequal : bool, default=True
        Whether to skip tail chunks differing in length from target.
    threads_per_block : int, default=128
        CUDA block dimension.
    use_shared_memory : bool, default=False
        Whether to use CUDA shared-memory target caching optimization.
    fallback_to_cpu : bool, default=False
        Whether to gracefully fallback to CPU if CUDA is unavailable.

    Returns
    -------
    list[AlignmentResult]
        Alignment results for each chunk.
    """
    try:
        from src.alignment.integration import run_pipeline
    except ModuleNotFoundError:
        from alignment.integration import run_pipeline

    return run_pipeline(
        file_path,
        target,
        chunk_size,
        backend=backend,
        skip_unequal=skip_unequal,
        threads_per_block=threads_per_block,
        use_shared_memory=use_shared_memory,
        fallback_to_cpu=fallback_to_cpu,
    )


if __name__ == "__main__":
    chunks = process_genome("data/mock_genome.fasta", 10)

    print("Total chunks:", len(chunks))

    for chunk in chunks:
        print(chunk)
