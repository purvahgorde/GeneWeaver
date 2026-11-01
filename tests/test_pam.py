
"""Tests for PAM sequence detection."""

from src.pam.pam_detector import find_pam_sites


def test_find_multiple_pam_sites():
    """Find all NGG motifs, including overlapping motifs."""
    result = find_pam_sites("ATGAGGCTGGG")

    assert result == [
        {"position": 3, "pam": "AGG"},
        {"position": 7, "pam": "TGG"},
        {"position": 8, "pam": "GGG"},
    ]


def test_find_no_pam_sites():
    """Return an empty list when no NGG motif exists."""
    assert find_pam_sites("ATATCTA") == []


def test_lowercase_sequence():
    """Detect PAM motifs in lowercase DNA."""
    assert find_pam_sites("atgagg") == [
        {"position": 3, "pam": "AGG"}
    ]


def test_sequence_too_short():
    """Return no sites for sequences shorter than three bases."""
    assert find_pam_sites("AG") == []
    assert find_pam_sites("") == []


def test_pam_at_start():
    """Detect a PAM beginning at position zero."""
    assert find_pam_sites("AGGATC") == [
        {"position": 0, "pam": "AGG"}
    ]