from pathlib import Path
from Bio import SeqIO


def parse_fasta(file_path):
    """Parse a FASTA file and return a list of sequence strings.

    Parameters
    ----------
    file_path : str | Path
        Path to FASTA formatted file.

    Returns
    -------
    list[str]
        Extracted DNA sequence strings.
    """
    if file_path is None:
        raise TypeError("file_path must be a valid path string or Path object, got None")

    if isinstance(file_path, str) and not file_path.strip():
        raise ValueError("file_path cannot be empty")

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    if not file_path.is_file():
        raise FileNotFoundError(f"FASTA path is not a file: {file_path}")

    sequences = []
    with open(file_path, "r", encoding="utf-8") as handle:
        records = SeqIO.parse(handle, "fasta")
        for record in records:
            sequences.append(str(record.seq))

    return sequences


if __name__ == "__main__":
    sequences = parse_fasta("data/mock_genome.fasta")

    print("Number of sequences:", len(sequences))

    for sequence in sequences:
        print(sequence)
