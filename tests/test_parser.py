from src.parser.fasta_parser import parse_fasta


def test_parse_fasta():
    sequences = parse_fasta("data/mock_genome.fasta")

    assert len(sequences) == 2
    assert len(sequences[0]) > 0
    assert len(sequences[1]) > 0