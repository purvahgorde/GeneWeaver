"""Tests for Member 2 FASTA parser."""

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

from parser.fasta_parser import parse_fasta


class TestFastaParser:
    """Tests for parse_fasta function."""

    def test_parse_mock_genome(self):
        mock_path = REPO_ROOT / "data" / "mock_genome.fasta"
        assert mock_path.exists()
        sequences = parse_fasta(mock_path)
        assert len(sequences) == 2
        assert sequences[0] == "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
        assert sequences[1] == "CGTACGTAGCTAGCTAACGTACGTAGCTAGCTAGCTAGCTA"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="FASTA file not found"):
            parse_fasta("non_existent_file.fasta")

    def test_custom_fasta(self, tmp_path):
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(">seq1\nACGT\n>seq2\nTGCA\n")
        sequences = parse_fasta(fasta_file)
        assert sequences == ["ACGT", "TGCA"]
