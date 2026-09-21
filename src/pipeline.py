from src.parser.fasta_parser import parse_fasta
from src.chunking.genome_chunker import chunk_sequence


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