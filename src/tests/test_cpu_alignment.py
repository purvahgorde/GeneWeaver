import sys
import os

import pytest

# ── Make sure src/ is importable regardless of working directory ──────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alignment.cpu_alignment import (
    AlignmentResult,
    align_sequence,
    align_sequences,
    align_chunks,
    align_chunk_pairs,
    VALID_BASES,
)


# ═════════════════════════════════════════════════════════════════════════
#  Exact-match tests
# ═════════════════════════════════════════════════════════════════════════
class TestExactMatch:
    """Sequences that are identical should yield a perfect score."""

    def test_exact_match_4bp(self):
        result = align_sequence("ACGT", "ACGT")
        assert result.matches == 4
        assert result.mismatches == 0
        assert result.alignment_length == 4
        assert result.score == 1.0
        assert result.mismatch_positions == []

    def test_exact_match_long_sequence(self):
        seq = "ACGT" * 250  # 1 000 bases
        result = align_sequence(seq, seq)
        assert result.matches == 1000
        assert result.mismatches == 0
        assert result.score == 1.0
        assert result.mismatch_positions == []

    def test_single_base_match_A(self):
        result = align_sequence("A", "A")
        assert result.matches == 1
        assert result.score == 1.0

    def test_single_base_match_all_bases(self):
        """Each of A, C, G, T should match itself."""
        for base in "ACGT":
            r = align_sequence(base, base)
            assert r.matches == 1
            assert r.mismatches == 0
            assert r.score == 1.0

    def test_exact_match_2bp(self):
        result = align_sequence("AT", "AT")
        assert result.matches == 2
        assert result.alignment_length == 2
        assert result.score == 1.0

    def test_exact_match_homopolymer(self):
        """All-A vs all-A should be a perfect match."""
        seq = "A" * 100
        result = align_sequence(seq, seq)
        assert result.matches == 100
        assert result.score == 1.0


# ═════════════════════════════════════════════════════════════════════════
#  Mismatch tests
# ═════════════════════════════════════════════════════════════════════════
class TestMismatch:
    """Sequences with differing bases should report mismatches correctly."""

    def test_single_mismatch_last_position(self):
        result = align_sequence("ACGT", "ACGA")
        assert result.matches == 3
        assert result.mismatches == 1
        assert result.alignment_length == 4
        assert result.score == 0.75
        assert result.mismatch_positions == [3]

    def test_single_mismatch_first_position(self):
        """Mismatch at position 0 — boundary."""
        result = align_sequence("ACGT", "TCGT")
        assert result.mismatches == 1
        assert result.mismatch_positions == [0]

    def test_single_mismatch_middle(self):
        result = align_sequence("ACGT", "AGGT")
        assert result.mismatches == 1
        assert result.mismatch_positions == [1]

    def test_two_mismatches(self):
        result = align_sequence("AAAA", "ACGA")
        assert result.matches == 2
        assert result.mismatches == 2
        assert result.score == 0.5
        assert result.mismatch_positions == [1, 2]

    def test_three_mismatches(self):
        result = align_sequence("AAAA", "TCGA")
        assert result.matches == 1
        assert result.mismatches == 3
        assert result.mismatch_positions == [0, 1, 2]

    def test_all_mismatch(self):
        result = align_sequence("AAAA", "TTTT")
        assert result.matches == 0
        assert result.mismatches == 4
        assert result.score == 0.0
        assert result.mismatch_positions == [0, 1, 2, 3]

    def test_mismatch_positions_sorted(self):
        result = align_sequence("ACGTACGT", "TCGAACGA")
        # Mismatches at 0, 3, 7
        assert result.mismatch_positions == [0, 3, 7]
        assert result.mismatch_positions == sorted(result.mismatch_positions)

    def test_alternating_match_mismatch(self):
        """ACAC vs ATAG — mismatches at positions 1 and 3."""
        result = align_sequence("ACAC", "ATAG")
        assert result.matches == 2
        assert result.mismatches == 2
        assert result.mismatch_positions == [1, 3]

    def test_single_base_mismatch(self):
        """Shortest possible mismatch: 1bp."""
        result = align_sequence("A", "T")
        assert result.matches == 0
        assert result.mismatches == 1
        assert result.score == 0.0
        assert result.mismatch_positions == [0]


# ═════════════════════════════════════════════════════════════════════════
#  Short-sequence tests
# ═════════════════════════════════════════════════════════════════════════
class TestShortSequences:
    """Verify correct behavior for very short inputs (1-3 bp)."""

    def test_2bp_one_mismatch(self):
        result = align_sequence("AC", "AT")
        assert result.matches == 1
        assert result.mismatches == 1
        assert result.score == 0.5
        assert result.mismatch_positions == [1]

    def test_3bp_exact(self):
        result = align_sequence("ACG", "ACG")
        assert result.matches == 3
        assert result.score == 1.0

    def test_3bp_all_different(self):
        result = align_sequence("ACG", "TGA")
        assert result.mismatches == 3
        assert result.score == 0.0


# ═════════════════════════════════════════════════════════════════════════
#  Empty-sequence tests
# ═════════════════════════════════════════════════════════════════════════
class TestEmptySequence:
    """Empty inputs should return a zeroed result without errors."""

    def test_empty_sequence(self):
        result = align_sequence("", "")
        assert result.matches == 0
        assert result.mismatches == 0
        assert result.alignment_length == 0
        assert result.score == 0.0
        assert result.mismatch_positions == []

    def test_empty_result_type(self):
        """Even empty results should be an AlignmentResult instance."""
        result = align_sequence("", "")
        assert isinstance(result, AlignmentResult)


# ═════════════════════════════════════════════════════════════════════════
#  Invalid-input tests
# ═════════════════════════════════════════════════════════════════════════
class TestInvalidInput:
    """Invalid inputs should raise clear, specific exceptions."""

    def test_non_string_reference_int(self):
        with pytest.raises(TypeError, match="reference must be a string"):
            align_sequence(123, "ACGT")

    def test_non_string_query_int(self):
        with pytest.raises(TypeError, match="query must be a string"):
            align_sequence("ACGT", 456)

    def test_none_reference(self):
        with pytest.raises(TypeError):
            align_sequence(None, "ACGT")

    def test_none_query(self):
        with pytest.raises(TypeError):
            align_sequence("ACGT", None)

    def test_list_reference(self):
        with pytest.raises(TypeError, match="reference must be a string"):
            align_sequence(["A", "C", "G", "T"], "ACGT")

    def test_invalid_character_in_reference(self):
        with pytest.raises(ValueError, match="invalid characters"):
            align_sequence("ACGX", "ACGT")

    def test_invalid_character_in_query(self):
        with pytest.raises(ValueError, match="invalid characters"):
            align_sequence("ACGT", "ACGX")

    def test_numeric_characters(self):
        with pytest.raises(ValueError, match="invalid characters"):
            align_sequence("A1GT", "ACGT")

    def test_space_in_sequence(self):
        with pytest.raises(ValueError, match="invalid characters"):
            align_sequence("AC GT", "ACGT")

    def test_newline_in_sequence(self):
        with pytest.raises(ValueError, match="invalid characters"):
            align_sequence("AC\nGT", "ACGT")

    def test_unequal_length_ref_longer(self):
        with pytest.raises(ValueError, match="equal length"):
            align_sequence("ACGT", "ACG")

    def test_unequal_length_query_longer(self):
        with pytest.raises(ValueError, match="equal length"):
            align_sequence("ACG", "ACGT")

    def test_unequal_length_empty_vs_nonempty(self):
        with pytest.raises(ValueError, match="equal length"):
            align_sequence("", "A")

    def test_invalid_characters_iupac_ambiguity_n(self):
        with pytest.raises(ValueError, match="invalid characters"):
            align_sequence("ACGN", "ACGT")

    def test_invalid_characters_iupac_ambiguity_r_y(self):
        with pytest.raises(ValueError, match="invalid characters"):
            align_sequence("ACGR", "ACGY")

    def test_invalid_characters_query_iupac(self):
        with pytest.raises(ValueError, match="invalid characters"):
            align_sequence("ACGT", "ACGS")


# ═════════════════════════════════════════════════════════════════════════
#  Case-insensitivity tests
# ═════════════════════════════════════════════════════════════════════════
class TestCaseInsensitive:
    """Inputs should be normalised to upper-case before comparison."""

    def test_lowercase_exact_match(self):
        result = align_sequence("acgt", "acgt")
        assert result.matches == 4
        assert result.score == 1.0

    def test_mixed_case(self):
        result = align_sequence("AcGt", "aCgT")
        assert result.matches == 4
        assert result.score == 1.0

    def test_lowercase_mismatch(self):
        """Case normalisation should not mask a real mismatch."""
        result = align_sequence("acgt", "acga")
        assert result.mismatches == 1
        assert result.mismatch_positions == [3]


# ═════════════════════════════════════════════════════════════════════════
#  Boundary and edge-case tests
# ═════════════════════════════════════════════════════════════════════════
class TestBoundaryCases:
    """Edge cases that stress boundary conditions."""

    def test_all_same_base_mismatch(self):
        """Homopolymer reference vs different homopolymer query."""
        result = align_sequence("CCCC", "GGGG")
        assert result.mismatches == 4
        assert result.score == 0.0

    def test_score_precision_not_rounded(self):
        """Score should be exact rational division, not rounded."""
        result = align_sequence("AAA", "AAT")
        # 2/3 = 0.666...
        assert result.score == pytest.approx(2.0 / 3.0)

    def test_score_one_third(self):
        result = align_sequence("AAA", "ATT")
        assert result.score == pytest.approx(1.0 / 3.0)

    def test_large_sequence_correctness(self):
        """10 000 bases with one mismatch at the midpoint."""
        n = 10_000
        ref = "A" * n
        qry_list = list(ref)
        qry_list[n // 2] = "T"  # single mismatch at position 5000
        qry = "".join(qry_list)
        result = align_sequence(ref, qry)
        assert result.matches == n - 1
        assert result.mismatches == 1
        assert result.mismatch_positions == [n // 2]

    def test_mismatch_count_equals_len_of_mismatch_positions(self):
        """mismatches field must equal len(mismatch_positions) always."""
        result = align_sequence("ACGTACGT", "TCGATCGA")
        assert result.mismatches == len(result.mismatch_positions)

    def test_alternating_long_sequence(self):
        """1 000 bases with alternating mismatch pattern."""
        ref = "AC" * 500
        qry = "AG" * 500  # mismatches at odd positions 1, 3, 5, ..., 999
        result = align_sequence(ref, qry)
        assert result.matches == 500
        assert result.mismatches == 500
        assert result.score == 0.5
        assert result.mismatch_positions == list(range(1, 1000, 2))

    def test_extreme_boundary_first_and_last_mismatch(self):
        """100 bases with mismatches only at position 0 and 99."""
        ref_list = list("A" * 100)
        qry_list = list("A" * 100)
        qry_list[0] = "C"
        qry_list[99] = "G"
        result = align_sequence("".join(ref_list), "".join(qry_list))
        assert result.matches == 98
        assert result.mismatches == 2
        assert result.score == 0.98
        assert result.mismatch_positions == [0, 99]


# ═════════════════════════════════════════════════════════════════════════
#  Batch alignment tests
# ═════════════════════════════════════════════════════════════════════════
class TestAlignSequences:
    """The batch helper should process multiple pairs correctly."""

    def test_batch_multiple_pairs(self):
        pairs = [
            ("ACGT", "ACGT"),
            ("AAAA", "TTTT"),
            ("ACGT", "ACGA"),
        ]
        results = align_sequences(pairs)
        assert len(results) == 3
        assert results[0].score == 1.0
        assert results[1].score == 0.0
        assert results[2].mismatches == 1

    def test_batch_preserves_order(self):
        """Results must be in the same order as input pairs."""
        pairs = [
            ("A", "A"),
            ("A", "T"),
            ("G", "G"),
        ]
        results = align_sequences(pairs)
        assert results[0].score == 1.0
        assert results[1].score == 0.0
        assert results[2].score == 1.0

    def test_batch_single_pair(self):
        results = align_sequences([("ACGT", "ACGT")])
        assert len(results) == 1
        assert results[0].matches == 4

    def test_batch_empty_list(self):
        results = align_sequences([])
        assert results == []

    def test_batch_invalid_type(self):
        with pytest.raises(TypeError, match="pairs must be a list"):
            align_sequences("not a list")  # type: ignore[arg-type]

    def test_batch_propagates_validation_error(self):
        """If one pair is invalid, the error should propagate."""
        with pytest.raises(ValueError):
            align_sequences([("ACGT", "ACGT"), ("ACGX", "ACGT")])


# ═════════════════════════════════════════════════════════════════════════
#  AlignmentResult contract tests
# ═════════════════════════════════════════════════════════════════════════
class TestAlignmentResultContract:
    """The AlignmentResult should always satisfy its invariants."""

    def test_matches_plus_mismatches_equals_length(self):
        result = align_sequence("ACGTACGT", "ACGAACGA")
        assert result.matches + result.mismatches == result.alignment_length

    def test_score_bounds(self):
        # Perfect
        assert align_sequence("A", "A").score == 1.0
        # Zero
        assert align_sequence("A", "T").score == 0.0
        # Partial
        score = align_sequence("ACGT", "ACGA").score
        assert 0.0 <= score <= 1.0

    def test_score_consistency(self):
        """score == matches / alignment_length for non-empty."""
        result = align_sequence("ACGTACGT", "ACGAACGA")
        expected_score = result.matches / result.alignment_length
        assert result.score == pytest.approx(expected_score)

    def test_result_is_immutable(self):
        result = align_sequence("ACGT", "ACGT")
        with pytest.raises(AttributeError):
            result.matches = 99  # type: ignore[misc]

    def test_deterministic_multiple_runs(self):
        """Same inputs must always produce the exact same output."""
        for _ in range(10):
            r = align_sequence("ACGTACGT", "ACGAACGA")
            assert r.matches == 6
            assert r.mismatches == 2
            assert r.mismatch_positions == [3, 7]

    def test_equality(self):
        """Two results from the same input should be equal."""
        r1 = align_sequence("ACGTACGT", "ACGAACGA")
        r2 = align_sequence("ACGTACGT", "ACGAACGA")
        assert r1 == r2


# ═════════════════════════════════════════════════════════════════════════
#  Chunk alignment core tests
# ═════════════════════════════════════════════════════════════════════════
class TestAlignChunksCore:
    """Unit tests for align_chunks helper in cpu_alignment."""

    def test_align_chunks_exact_match(self):
        chunks = ["ACGT", "ACGT"]
        results = align_chunks(chunks, "ACGT")
        assert len(results) == 2
        assert all(r.score == 1.0 for r in results)

    def test_align_chunks_varying_scores(self):
        chunks = ["ACGT", "ACGA", "TGCA"]
        results = align_chunks(chunks, "ACGT")
        assert len(results) == 3
        assert results[0].score == 1.0
        assert results[1].score == 0.75
        assert results[2].score == 0.0

    def test_align_chunks_skip_unequal_true(self):
        chunks = ["ACGT", "ACG", "ACGT"]
        results = align_chunks(chunks, "ACGT", skip_unequal=True)
        assert len(results) == 2
        assert all(r.alignment_length == 4 for r in results)

    def test_align_chunks_skip_unequal_false_raises(self):
        chunks = ["ACGT", "ACG"]
        with pytest.raises(ValueError, match="equal length"):
            align_chunks(chunks, "ACGT", skip_unequal=False)

    def test_align_chunks_empty_list(self):
        results = align_chunks([], "ACGT")
        assert results == []

    def test_align_chunks_single_chunk(self):
        results = align_chunks(["ACGT"], "ACGT")
        assert len(results) == 1
        assert results[0].matches == 4

    def test_align_chunks_invalid_types(self):
        with pytest.raises(TypeError):
            align_chunks("not a list", "ACGT")
        with pytest.raises(TypeError):
            align_chunks(["ACGT"], 123)


# ═════════════════════════════════════════════════════════════════════════
#  Chunk pairs core tests
# ═════════════════════════════════════════════════════════════════════════
class TestAlignChunkPairsCore:
    """Unit tests for align_chunk_pairs helper in cpu_alignment."""

    def test_align_chunk_pairs_matching(self):
        ref_chunks = ["ACGT", "GGCC"]
        qry_chunks = ["ACGT", "GGCA"]
        results = align_chunk_pairs(ref_chunks, qry_chunks)
        assert len(results) == 2
        assert results[0].score == 1.0
        assert results[1].mismatches == 1

    def test_align_chunk_pairs_empty(self):
        results = align_chunk_pairs([], [])
        assert results == []

    def test_align_chunk_pairs_length_mismatch(self):
        with pytest.raises(ValueError, match="equal length"):
            align_chunk_pairs(["ACGT"], ["ACGT", "GGCC"])

    def test_align_chunk_pairs_type_validation(self):
        with pytest.raises(TypeError):
            align_chunk_pairs("not a list", ["ACGT"])
        with pytest.raises(TypeError):
            align_chunk_pairs(["ACGT"], "not a list")


# ═════════════════════════════════════════════════════════════════════════
#  Import verification
# ═════════════════════════════════════════════════════════════════════════
class TestImport:
    """Verify that the module and its public symbols are importable."""

    def test_module_importable(self):
        from alignment import cpu_alignment  # noqa: F811
        assert hasattr(cpu_alignment, "align_sequence")
        assert hasattr(cpu_alignment, "align_sequences")
        assert hasattr(cpu_alignment, "align_chunks")
        assert hasattr(cpu_alignment, "align_chunk_pairs")
        assert hasattr(cpu_alignment, "AlignmentResult")

    def test_package_level_imports(self):
        from alignment import align_sequence as fn  # noqa: F811
        assert callable(fn)

    def test_valid_bases_constant(self):
        assert VALID_BASES == frozenset("ACGT")