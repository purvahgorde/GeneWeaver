from src.chunking.genome_chunker import chunk_sequence


def test_chunk_sequence():
    sequence = "ATGCGTACGT"

    chunks = chunk_sequence(sequence, 5)

    assert chunks == ["ATGCG", "TACGT"]