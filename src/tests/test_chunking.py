"""Tests for Member 2 genome chunker."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chunking.genome_chunker import chunk_sequence


class TestGenomeChunker:
    """Tests for chunk_sequence function."""

    def test_even_chunking(self):
        seq = "ACGTACGT"
        chunks = chunk_sequence(seq, 4)
        assert chunks == ["ACGT", "ACGT"]

    def test_tail_chunking(self):
        seq = "ACGTACGTA"
        chunks = chunk_sequence(seq, 4)
        assert chunks == ["ACGT", "ACGT", "A"]

    def test_chunk_size_larger_than_sequence(self):
        seq = "ACGT"
        chunks = chunk_sequence(seq, 10)
        assert chunks == ["ACGT"]

    def test_empty_sequence(self):
        chunks = chunk_sequence("", 4)
        assert chunks == []

    def test_reconstruction(self):
        seq = "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
        chunks = chunk_sequence(seq, 7)
        assert "".join(chunks) == seq

    def test_chunk_boundary_and_ordering(self):
        seq = "ABCDEFGHIJ"
        chunks = chunk_sequence(seq, 3)
        assert chunks == ["ABC", "DEF", "GHI", "J"]
        assert chunks[0] == "ABC"
        assert chunks[1] == "DEF"
        assert chunks[2] == "GHI"
        assert chunks[3] == "J"

    def test_invalid_chunk_size_non_positive(self):
        with pytest.raises(ValueError, match="positive integer"):
            chunk_sequence("ACGT", 0)

        with pytest.raises(ValueError, match="positive integer"):
            chunk_sequence("ACGT", -5)

    def test_invalid_types_raise(self):
        with pytest.raises(TypeError, match="chunk_size must be an integer"):
            chunk_sequence("ACGT", "4")  # type: ignore

        with pytest.raises(TypeError, match="chunk_size must be an integer"):
            chunk_sequence("ACGT", True)  # type: ignore

        with pytest.raises(TypeError, match="sequence must be a string"):
            chunk_sequence(12345, 4)  # type: ignore
