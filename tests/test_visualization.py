"""Tests for mismatch visualizer module."""

from rich.text import Text
from src.visualization.mismatch_visualizer import (
    format_dna_alignment,
    render_mismatch_ascii,
    get_mismatch_indices,
)


class TestMismatchVisualizer:
    """Tests for mismatch visualizer functions."""

    def test_format_dna_alignment_empty(self):
        result = format_dna_alignment([], "ATGC")
        assert isinstance(result, Text)
        assert "No chunks available" in result.plain

    def test_format_dna_alignment_with_matches_and_mismatches(self):
        chunks = ["ATGC", "ATGA"]
        target = "ATGC"
        result = format_dna_alignment(chunks, target)
        assert isinstance(result, Text)
        assert "TARGET  ATGC" in result.plain
        assert "Perfect match" in result.plain
        assert "Mismatch positions: 4" in result.plain

    def test_render_mismatch_ascii(self):
        ascii_out = render_mismatch_ascii("ATGA", "ATGC")
        assert "Chunk : ATGA" in ascii_out
        assert "Target: ATGC" in ascii_out
        assert "|||." in ascii_out

    def test_get_mismatch_indices_1_based(self):
        indices = get_mismatch_indices("ATGA", "ATGC", one_based=True)
        assert indices == [4]

    def test_get_mismatch_indices_0_based(self):
        indices = get_mismatch_indices("ATGA", "ATGC", one_based=False)
        assert indices == [3]

    def test_get_mismatch_indices_perfect_match(self):
        indices = get_mismatch_indices("ACGT", "ACGT")
        assert indices == []
