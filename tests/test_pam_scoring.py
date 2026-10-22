"""Tests for PAM detection and DNA scoring integration."""

import pytest

from src.analysis.pam_scoring import score_pam_sites


def test_score_multiple_pam_sites():
    """Find PAM sites and calculate their scores."""
    result = score_pam_sites("ATGAGGCTTGG", "AGG")

    assert result == [
        {"position": 3, "pam": "AGG", "score": 3},
        {"position": 8, "pam": "TGG", "score": 1},
    ]


def test_no_pam_sites():
    """Return an empty list when no PAM sites exist."""
    assert score_pam_sites("ATATCTA", "AGG") == []


def test_lowercase_sequence():
    """Handle lowercase DNA input."""
    result = score_pam_sites("atgagg", "AGG")

    assert result == [
        {"position": 3, "pam": "AGG", "score": 3}
    ]


def test_invalid_target_length():
    """Reject targets that are not three bases long."""
    with pytest.raises(ValueError, match="exactly 3 bases"):
        score_pam_sites("ATGAGG", "AG")


def test_all_mismatches():
    """Give a negative score when all bases mismatch."""
    result = score_pam_sites("AAAGG", "CCC")

    assert result == [
        {"position": 2, "pam": "AGG", "score": -3}
    ]