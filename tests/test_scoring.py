
"""Tests for DNA scoring utilities."""

import pytest

from src.scoring.dna_scoring import (
    score_bases,
    score_sequences,
    validate_dna,
)


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
        """Lowercase and uppercase bases are treated equally."""
        assert score_bases("a", "a") == 1
        assert score_bases("a", "A") == 1
        assert score_bases("c", "C") == 1
        assert score_bases("a", "T") == -1

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


def test_validate_valid_dna():
    """Accept uppercase and lowercase DNA sequences."""
    validate_dna("ATGC")
    validate_dna("atgc")


def test_validate_invalid_dna():
    """Reject characters that are not valid DNA bases."""
    with pytest.raises(ValueError, match="only A, C, G, and T"):
        validate_dna("ATGX")


def test_validate_ambiguous_base():
    """Reject N because this version accepts only A, C, G, and T."""
    with pytest.raises(ValueError, match="only A, C, G, and T"):
        validate_dna("ATGN")


def test_score_bases_rejects_invalid_base():
    """Reject an invalid base during scoring."""
    with pytest.raises(ValueError, match="only A, C, G, and T"):
        score_bases("X", "A")


def test_score_sequences_rejects_invalid_dna():
    """Reject invalid DNA during sequence scoring."""
    with pytest.raises(ValueError, match="only A, C, G, and T"):
        score_sequences("ATGX", "ATGC")


def test_score_sequences_case_insensitive():
    """Score uppercase and lowercase sequences consistently."""
    assert score_sequences("atgc", "ATGC") == 4


def test_empty_sequences():
    """Empty sequences should have a score of zero."""
    assert score_sequences("", "") == 0


def test_validate_empty_dna():
    """An empty string contains no invalid DNA bases."""
    validate_dna("")


def test_validate_rejects_none():
    """Reject None as a DNA sequence."""
    with pytest.raises(ValueError, match="must be a string"):
        validate_dna(None)


def test_validate_rejects_number():
    """Reject numbers as DNA sequences."""
    with pytest.raises(ValueError, match="must be a string"):
        validate_dna(123)


def test_score_bases_rejects_multiple_characters():
    """Each input to base scoring must contain exactly one base."""
    with pytest.raises(ValueError):
        score_bases("AT", "AT")


def test_score_sequences_rejects_none():
    """Reject None as a sequence input."""
    with pytest.raises(ValueError, match="must be a string"):
        score_sequences(None, "ATGC")
