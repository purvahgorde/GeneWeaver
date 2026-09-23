from pathlib import Path
from Bio import SeqIO


def parse_fasta(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    records = SeqIO.parse(file_path, "fasta")

    sequences = []

    for record in records:
        sequences.append(str(record.seq))

    return sequences


if __name__ == "__main__":
    sequences = parse_fasta("data/mock_genome.fasta")

    print("Number of sequences:", len(sequences))

    for sequence in sequences:
        print(sequence)
