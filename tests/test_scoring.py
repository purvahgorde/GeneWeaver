"""Tests for DNA scoring utilities."""
import pytest

from src.scoring.dna_scoring import score_bases, score_sequences

class TestDnaScoring:
    """Tests for DNA base scoring."""

    def test_matching_bases(self):
        """Matching bases should receive a positive score."""
        assert score_bases("A", "A") == 1
        assert score_bases("C", "C") == 1
        assert score_bases("G", "G") == 1
        assert score_bases("T", "T") == 1

    def test_mismatching_bases(self):
        """Different bases should receive a negative score."""
        assert score_bases("A", "G") == -1
        assert score_bases("C", "T") == -1
        assert score_bases("G", "T") == -1

    def test_lowercase_bases(self):
        """Lowercase bases currently behave as distinct values."""
        assert score_bases("a", "a") == 1
        assert score_bases("a", "A") == -1

    def test_matching_sequences(self):
        """Identical sequences should receive the maximum score."""
        assert score_sequences("ATGC", "ATGC") == 4

    def test_mixed_sequence_score(self):
        """Mixed matches and mismatches should produce the correct score."""
        assert score_sequences("ATGC", "ATAC") == 2

    def test_different_sequences(self):
        """Completely different sequences should receive a negative score."""
        assert score_sequences("AAAA", "TTTT") == -4

    def test_different_sequence_lengths(self):
        """Sequences with different lengths should raise an error."""
        with pytest.raises(ValueError, match="same length"):
            score_sequences("ATGC", "ATG")