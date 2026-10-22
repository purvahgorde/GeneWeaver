"""Tests for the GeneWeaver genome processing pipeline."""

from pathlib import Path

from src.pipeline import process_genome


REPO_ROOT = Path(__file__).resolve().parent.parent


class TestGenomePipeline:
    """Tests for process_genome function."""

    def test_process_mock_genome(self):
        """Pipeline should parse and chunk the mock genome."""
        fasta_path = REPO_ROOT / "data" / "mock_genome.fasta"

        sequences, chunks = process_genome(fasta_path, 10)

        assert len(sequences) == 2
        assert len(chunks) == 10

    def test_pipeline_first_chunk(self):
        """Pipeline should generate the expected first chunk."""
        fasta_path = REPO_ROOT / "data" / "mock_genome.fasta"

        sequences, chunks = process_genome(fasta_path, 10)

        assert chunks[0] == "ATGCGTACGT"

    def test_pipeline_reconstructs_sequences(self):
        """Chunks should reconstruct the original sequences."""
        fasta_path = REPO_ROOT / "data" / "mock_genome.fasta"

        sequences, chunks = process_genome(fasta_path, 10)

        reconstructed_length = sum(len(chunk) for chunk in chunks)
        original_length = sum(len(sequence) for sequence in sequences)

        assert reconstructed_length == original_length