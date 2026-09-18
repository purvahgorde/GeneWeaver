from Bio import SeqIO
def parse_fasta(file_path):
    records = SeqIO.parse(file_path, "fasta")

    sequences = []

    for record in records:
        sequences.append(str(record.seq))

    return sequences

if __name__ == "__main__":
    sequences = parse_fasta("data/mock_genome.fasta")
    print(sequences)