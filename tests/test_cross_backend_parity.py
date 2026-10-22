"""Cross-backend parity tests validating CPU, CUDA, and Dask equivalence."""

from __future__ import annotations

import pytest
from src.alignment.cpu_alignment import align_chunks, align_sequence, align_sequences, AlignmentResult
from src.alignment.cuda_alignment import (
    cuda_align_chunks,
    cuda_align_chunks_shared,
    cuda_align_sequence,
    cuda_align_sequences,
    is_cuda_available,
)
from src.distributed.dask_processor import (
    dask_align_chunks,
    dask_align_sequences,
    is_dask_available,
)
from src.scoring.biological_scoring import score_alignment_results, score_alignment_result
from src.visualization.mismatch_visualizer import format_dna_alignment, render_mismatch_ascii


class TestCrossBackendParity:
    """Verify identical alignment outcomes between CPU, CUDA, and Dask."""

    @pytest.fixture
    def sample_data(self):
        chunks = [
            "ATGCGTACGT",  # 100% match
            "ATGCCTACGT",  # 1 mismatch at pos 5
            "TTGCGTACGT",  # 1 mismatch at pos 1
            "ATGCGTACGA",  # 1 mismatch at pos 10
            "CGTACGTAGC",  # high mismatch
            "CCCCCCCCCC",  # all mismatch except C's
        ]
        target = "ATGCGTACGT"
        return chunks, target

    def test_cpu_dask_parity(self, sample_data):
        chunks, target = sample_data
        cpu_res = align_chunks(chunks, target)
        dask_res = dask_align_chunks(chunks, target, scheduler="sync")

        assert len(cpu_res) == len(dask_res)
        assert cpu_res == dask_res

    def test_cpu_cuda_parity_when_available(self, sample_data):
        if not is_cuda_available():
            pytest.skip("CUDA hardware not available")

        chunks, target = sample_data
        cpu_res = align_chunks(chunks, target)
        cuda_res = cuda_align_chunks(chunks, target)
        cuda_shared_res = cuda_align_chunks_shared(chunks, target)

        assert len(cpu_res) == len(cuda_res)
        assert cpu_res == cuda_res
        assert cpu_res == cuda_shared_res

    def test_empty_input_parity(self):
        target = "ATGCGTACGT"
        cpu_res = align_chunks([], target)
        dask_res = dask_align_chunks([], target, scheduler="sync")
        assert cpu_res == []
        assert dask_res == []

        if is_cuda_available():
            cuda_res = cuda_align_chunks([], target)
            assert cuda_res == []

    def test_short_sequences_and_boundaries(self):
        # 1-base, 2-base, 3-base
        for length in (1, 2, 3, 4):
            chunk = "A" * length
            target = "A" * length
            cpu_res = align_sequence(chunk, target)
            assert cpu_res.matches == length
            assert cpu_res.mismatches == 0
            assert cpu_res.score == 1.0

            if is_cuda_available():
                cuda_res = cuda_align_sequence(chunk, target)
                assert cpu_res == cuda_res

    def test_unequal_tail_handling(self):
        chunks = ["ATGCGTACGT", "ATGC"]
        target = "ATGCGTACGT"

        # skip_unequal=True
        cpu_skip = align_chunks(chunks, target, skip_unequal=True)
        dask_skip = dask_align_chunks(chunks, target, skip_unequal=True, scheduler="sync")
        assert len(cpu_skip) == 1
        assert len(dask_skip) == 1
        assert cpu_skip == dask_skip

        if is_cuda_available():
            cuda_skip = cuda_align_chunks(chunks, target, skip_unequal=True)
            assert cpu_skip == cuda_skip

    def test_scoring_and_visualization_consumption(self, sample_data):
        chunks, target = sample_data
        cpu_res = align_chunks(chunks, target)

        # Biological scoring consumes results correctly
        summary = score_alignment_results(cpu_res)
        assert summary["total_chunks"] == float(len(chunks))
        assert summary["max_score"] == 1.0
        assert summary["total_matches"] > 0

        # Individual scoring
        scores = [score_alignment_result(r) for r in cpu_res]
        assert scores[0] == 10.0  # 10 matches, 0 mismatches

        # Visualizer formats results correctly
        rich_text = format_dna_alignment(chunks, target)
        assert "TARGET  ATGCGTACGT" in rich_text.plain
        assert "Perfect match" in rich_text.plain

        ascii_viz = render_mismatch_ascii(chunks[1], target)
        assert "||||.|||||" in ascii_viz
