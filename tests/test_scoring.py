"""Tests for DNA scoring utilities."""

from src.scoring.dna_scoring import score_bases


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