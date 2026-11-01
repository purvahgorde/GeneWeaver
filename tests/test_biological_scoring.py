"""Tests for biological scoring and alignment result scoring metrics."""

import pytest
from src.alignment.cpu_alignment import AlignmentResult
from src.scoring.biological_scoring import (
    calculate_gc_content,
    score_alignment_result,
    score_alignment_results,
    score_bases,
    score_sequences,
    validate_dna,
    find_pam_sites,
    score_pam_sites,
)


class TestBiologicalScoring:
    """Tests for biological scoring functions."""

    def test_gc_content_calculation(self):
        assert calculate_gc_content("GCGC") == 100.0
        assert calculate_gc_content("ATAT") == 0.0
        assert calculate_gc_content("ACGT") == 50.0
        assert calculate_gc_content("") == 0.0

    def test_gc_content_invalid_sequence(self):
        with pytest.raises(ValueError):
            calculate_gc_content("ACGTN")

    def test_score_alignment_result_default_weights(self):
        res = AlignmentResult(matches=8, mismatches=2, alignment_length=10, score=0.8)
        assert score_alignment_result(res) == 6.0  # 8*1 - 2*1

    def test_score_alignment_result_custom_weights(self):
        res = AlignmentResult(matches=10, mismatches=0, alignment_length=10, score=1.0)
        assert score_alignment_result(res, match_weight=2.0, mismatch_weight=-3.0) == 20.0

    def test_score_alignment_results_empty(self):
        metrics = score_alignment_results([])
        assert metrics["total_chunks"] == 0.0
        assert metrics["average_score"] == 0.0
        assert metrics["average_similarity"] == 0.0

    def test_score_alignment_results_summary(self):
        r1 = AlignmentResult(matches=10, mismatches=0, alignment_length=10, score=1.0)
        r2 = AlignmentResult(matches=5, mismatches=5, alignment_length=10, score=0.5)
        metrics = score_alignment_results([r1, r2])

        assert metrics["total_chunks"] == 2.0
        assert metrics["total_matches"] == 15.0
        assert metrics["total_mismatches"] == 5.0
        assert metrics["average_similarity"] == 75.0
        assert metrics["average_score"] == 0.75
        assert metrics["max_score"] == 1.0
        assert metrics["min_score"] == 0.5

    def test_reexported_scoring_utilities(self):
        assert score_bases("A", "A") == 1
        assert score_sequences("ATGC", "ATGC") == 4
        assert len(find_pam_sites("ATGAGG")) == 1
        assert score_pam_sites("ATGAGG", "AGG")[0]["score"] == 3
