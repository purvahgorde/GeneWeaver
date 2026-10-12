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
    sequences = parse_fasta(file_path)

    all_chunks = []

    for sequence in sequences:
        chunks = chunk_sequence(sequence, chunk_size)
        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    chunks = process_genome("data/mock_genome.fasta", 10)

    print("Total chunks:", len(chunks))

    for chunk in chunks:
        print(chunk)
