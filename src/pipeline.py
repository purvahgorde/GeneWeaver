from src.parser.fasta_parser import parse_fasta
from src.chunking.genome_chunker import chunk_sequence


def process_genome(file_path, chunk_size):
    """Parse a FASTA file and split all sequences into chunks."""
    sequences = parse_fasta(file_path)
    all_chunks = []

    for sequence in sequences:
        chunks = chunk_sequence(sequence, chunk_size)
        all_chunks.extend(chunks)

    return sequences, all_chunks


if __name__ == "__main__":
    sequences, chunks = process_genome(
        "data/mock_genome.fasta",
        10
    )

    print("Total sequences:", len(sequences))
    print("Total chunks:", len(chunks))

    for chunk in chunks:
        print(chunk)