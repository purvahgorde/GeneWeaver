from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignment.cpu_alignment import (
    AlignmentResult,
    align_sequence,
    align_chunks,
    align_chunk_pairs,
)
from alignment.integration import run_pipeline
from parser.fasta_parser import parse_fasta
from chunking.genome_chunker import chunk_sequence


# ═════════════════════════════════════════════════════════════════════════
#  align_chunks tests
# ═════════════════════════════════════════════════════════════════════════
class TestAlignChunks:
    """Tests for align_chunks helper."""

    def test_exact_match_all_chunks(self):
        chunks = ["ACGT", "ACGT", "ACGT"]
        results = align_chunks(chunks, "ACGT")
        assert len(results) == 3
        for res in results:
            assert isinstance(res, AlignmentResult)
            assert res.matches == 4
            assert res.mismatches == 0
            assert res.score == 1.0

    def test_varying_mismatches_across_chunks(self):
        chunks = ["ACGT", "ACGA", "AAAA", "TGCA"]
        target = "ACGT"
        results = align_chunks(chunks, target)
        assert len(results) == 4
        assert results[0].matches == 4  # ACGT vs ACGT
        assert results[0].score == 1.0
        assert results[1].matches == 3  # ACGA vs ACGT (mismatch at pos 3)
        assert results[1].mismatch_positions == [3]
        assert results[2].matches == 1  # AAAA vs ACGT (matches at pos 0)
        assert results[3].matches == 0  # TGCA vs ACGT (total mismatch)
        assert results[3].score == 0.0

    def test_empty_chunks_list(self):
        results = align_chunks([], "ACGT")
        assert results == []

    def test_empty_target_with_empty_chunk(self):
        results = align_chunks([""], "")
        assert len(results) == 1
        assert results[0].alignment_length == 0
        assert results[0].score == 0.0

    def test_case_insensitivity(self):
        chunks = ["acgt", "AcGt"]
        results = align_chunks(chunks, "ACGT")
        assert len(results) == 2
        assert results[0].matches == 4
        assert results[1].matches == 4

    def test_invalid_chunks_type(self):
        with pytest.raises(TypeError, match="chunks must be a list"):
            align_chunks("ACGT", "ACGT")  # type: ignore

    def test_invalid_target_type(self):
        with pytest.raises(TypeError, match="target must be a string"):
            align_chunks(["ACGT"], 123)  # type: ignore

    def test_invalid_chunk_element_type(self):
        with pytest.raises(TypeError, match="Each chunk must be a string"):
            align_chunks(["ACGT", 123], "ACGT")  # type: ignore

    def test_invalid_characters_in_target(self):
        with pytest.raises(ValueError, match="target contains invalid characters"):
            align_chunks(["ACGT"], "ACGN")

    def test_invalid_characters_in_chunk(self):
        with pytest.raises(ValueError, match="reference contains invalid characters"):
            align_chunks(["ACGN"], "ACGT")

    def test_unequal_length_without_skip_raises(self):
        chunks = ["ACGT", "ACG"]
        with pytest.raises(ValueError, match="Sequences must have equal length"):
            align_chunks(chunks, "ACGT", skip_unequal=False)

    def test_unequal_length_with_skip_omits(self):
        chunks = ["ACGT", "ACG", "ACGT", "A", "ACGTA"]
        results = align_chunks(chunks, "ACGT", skip_unequal=True)
        # Only the two 4-bp chunks should be processed
        assert len(results) == 2
        assert all(r.alignment_length == 4 for r in results)


# ═════════════════════════════════════════════════════════════════════════
#  align_chunk_pairs tests
# ═════════════════════════════════════════════════════════════════════════
class TestAlignChunkPairs:
    """Tests for align_chunk_pairs helper."""

    def test_paired_chunks_alignment(self):
        ref_chunks = ["ACGT", "GGCC", "AAAA"]
        qry_chunks = ["ACGT", "GGCA", "TTTT"]
        results = align_chunk_pairs(ref_chunks, qry_chunks)
        assert len(results) == 3
        assert results[0].matches == 4
        assert results[1].matches == 3
        assert results[1].mismatches == 1
        assert results[2].matches == 0
        assert results[2].mismatches == 4

    def test_empty_lists(self):
        results = align_chunk_pairs([], [])
        assert results == []

    def test_mismatched_list_lengths_raises(self):
        with pytest.raises(ValueError, match="Chunk lists must have equal length"):
            align_chunk_pairs(["ACGT"], ["ACGT", "GGCC"])

    def test_invalid_types_raise(self):
        with pytest.raises(TypeError, match="reference_chunks must be a list"):
            align_chunk_pairs("ACGT", ["ACGT"])  # type: ignore
        with pytest.raises(TypeError, match="query_chunks must be a list"):
            align_chunk_pairs(["ACGT"], "ACGT")  # type: ignore


# ═════════════════════════════════════════════════════════════════════════
#  End-to-End Pipeline Integration tests (Member 2 + Member 1)
# ═════════════════════════════════════════════════════════════════════════
class TestEndToEndPipeline:
    """Verifies data flows through the complete Member 2 -> Member 1 pipeline."""

    @pytest.fixture
    def mock_fasta_path(self):
        path = REPO_ROOT / "data" / "mock_genome.fasta"
        assert path.exists(), f"Mock genome missing at {path}"
        return path

    def test_parse_fasta_returns_expected_structure(self, mock_fasta_path):
        """Verify Member 2's parse_fasta output contract."""
        sequences = parse_fasta(mock_fasta_path)
        assert isinstance(sequences, list)
        assert len(sequences) == 2
        assert all(isinstance(s, str) for s in sequences)
        # chr1 is 45 bp, chr2 is 41 bp
        assert len(sequences[0]) == 45
        assert len(sequences[1]) == 41

    def test_chunk_sequence_returns_expected_chunks(self, mock_fasta_path):
        """Verify Member 2's chunk_sequence output contract."""
        sequences = parse_fasta(mock_fasta_path)
        seq1 = sequences[0]
        chunks = chunk_sequence(seq1, 10)
        assert isinstance(chunks, list)
        assert len(chunks) == 5
        # First 4 chunks are 10 bp, 5th chunk is 5 bp tail
        assert [len(c) for c in chunks] == [10, 10, 10, 10, 5]
        # Chunks reassembled equal original sequence
        assert "".join(chunks) == seq1

    def test_end_to_end_flow_with_run_pipeline(self, mock_fasta_path):
        """Verify full FASTA -> chunker -> CPU alignment pipeline."""
        target = "ATGCGTACGT"  # 10 bp query matching start of chr1
        results = run_pipeline(mock_fasta_path, target, chunk_size=10, skip_unequal=True)

        # Total chunks = 5 from chr1 + 5 from chr2 = 10
        # 8 chunks have length 10, 2 chunks are shorter tails (5 and 1)
        # With skip_unequal=True, exactly 8 full chunks are aligned
        assert len(results) == 8
        assert all(isinstance(r, AlignmentResult) for r in results)
        assert all(r.alignment_length == 10 for r in results)

        # First chunk of chr1 starts with ATGCGTACGT -> perfect match
        first_result = results[0]
        assert first_result.matches == 10
        assert first_result.mismatches == 0
        assert first_result.score == 1.0
        assert first_result.mismatch_positions == []

    def test_end_to_end_tail_chunk_validation(self, mock_fasta_path):
        """When skip_unequal=False, tail chunk triggers strict length validation."""
        target = "ATGCGTACGT"
        with pytest.raises(ValueError, match="Sequences must have equal length"):
            run_pipeline(mock_fasta_path, target, chunk_size=10, skip_unequal=False)

    def test_end_to_end_even_chunk_size(self, mock_fasta_path):
        """With chunk_size=5 on chr1 (45 bp) and chr2 sliced, all chunks match length."""
        sequences = parse_fasta(mock_fasta_path)
        chr1_chunks = chunk_sequence(sequences[0], 5)  # 45 / 5 = 9 chunks
        assert len(chr1_chunks) == 9
        assert all(len(c) == 5 for c in chr1_chunks)

        results = align_chunks(chr1_chunks, "ATGCG")
        assert len(results) == 9
        assert results[0].matches == 5
        assert results[0].score == 1.0

    def test_missing_fasta_file_raises_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_pipeline("nonexistent_file.fasta", "ACGT", 10)
