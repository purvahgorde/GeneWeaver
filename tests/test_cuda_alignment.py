"""Tests for CUDA DNA sequence alignment implementation.

Tests CUDA results against the already validated CPU reference implementation.
At minimum tests:
- Exact match
- Single mismatch
- Multiple mismatches
- Different sequence lengths where supported
- Small DNA sequences
- Empty sequences
- Batch sequence processing
- Error validation (type errors, invalid base characters, length mismatches)
- Clean handling when CUDA execution is not available in the environment.
"""

from __future__ import annotations

import os
import sys
import pytest

# Ensure src/ is importable
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from alignment.cpu_alignment import (
    AlignmentResult,
    align_sequence,
    align_sequences,
    align_chunks,
)
from alignment.cuda_alignment import (
    cuda_align_sequence,
    cuda_align_sequences,
    cuda_align_sequences_timed,
    cuda_align_chunks,
    is_cuda_available,
    get_cuda_device_info,
    CUDANotAvailableError,
    CUDATimingBreakdown,
    _prepare_host_buffers,
    _encode_sequence_ascii,
)

# Pytest skip marker when CUDA hardware / Numba CUDA is unavailable
skip_if_no_cuda = pytest.mark.skipif(
    not is_cuda_available(),
    reason=f"Numba CUDA is not available: {get_cuda_device_info().get('reason', 'Unknown reason')}",
)


# ═════════════════════════════════════════════════════════════════════════
#  Host-side Preparation & Encoding Tests (Always run on CPU host)
# ═════════════════════════════════════════════════════════════════════════
class TestCUDAHostPreparation:
    """Verifies host-side data packing and validation prior to GPU transfer."""

    def test_encode_sequence_ascii(self):
        arr = _encode_sequence_ascii("ACGT")
        assert list(arr) == [65, 67, 71, 84]

    def test_prepare_host_buffers_empty_pairs(self):
        refs_flat, qrys_flat, seq_len, n_pairs = _prepare_host_buffers([])
        assert len(refs_flat) == 0
        assert len(qrys_flat) == 0
        assert seq_len == 0
        assert n_pairs == 0

    def test_prepare_host_buffers_valid(self):
        pairs = [("ACGT", "ACGT"), ("AGCT", "TCGA")]
        refs_flat, qrys_flat, seq_len, n_pairs = _prepare_host_buffers(pairs)
        assert seq_len == 4
        assert n_pairs == 2
        assert len(refs_flat) == 8
        assert len(qrys_flat) == 8

    def test_prepare_host_buffers_invalid_chars(self):
        with pytest.raises(ValueError, match="invalid characters"):
            _prepare_host_buffers([("ACGTN", "ACGTA")])

    def test_prepare_host_buffers_unequal_pair_length(self):
        with pytest.raises(ValueError, match="equal length"):
            _prepare_host_buffers([("ACG", "ACGT")])

    def test_prepare_host_buffers_inconsistent_batch_length(self):
        with pytest.raises(ValueError, match="equal sequence length across batch"):
            _prepare_host_buffers([("ACGT", "ACGT"), ("ACG", "ACG")])


# ═════════════════════════════════════════════════════════════════════════
#  CUDA Availability and Safe Fallback Behavior
# ═════════════════════════════════════════════════════════════════════════
class TestCUDAAvailabilityAndFallback:
    """Verifies that environment status is reported truthfully without fake execution."""

    def test_get_cuda_device_info_structure(self):
        info = get_cuda_device_info()
        assert isinstance(info, dict)
        assert "available" in info
        if info["available"]:
            assert "device_name" in info
            assert "compute_capability" in info
        else:
            assert "reason" in info

    def test_cuda_not_available_raises_clean_error(self):
        """If CUDA is unavailable, functions must raise CUDANotAvailableError."""
        if not is_cuda_available():
            with pytest.raises(CUDANotAvailableError):
                cuda_align_sequence("ACGT", "ACGT")

            with pytest.raises(CUDANotAvailableError):
                cuda_align_sequences([("ACGT", "ACGT")])

            with pytest.raises(CUDANotAvailableError):
                cuda_align_chunks(["ACGT"], "ACGT")

            with pytest.raises(CUDANotAvailableError):
                cuda_align_sequences_timed([("ACGT", "ACGT")])


# ═════════════════════════════════════════════════════════════════════════
#  CUDA Execution Correctness vs CPU Oracle (Runs when CUDA is available)
# ═════════════════════════════════════════════════════════════════════════
@skip_if_no_cuda
class TestCUDAAlignmentParityWithCPU:
    """Compares CUDA GPU alignment outputs directly against CPU reference."""

    def test_exact_match_small_sequences(self):
        cases = [
            ("A", "A"),
            ("AC", "AC"),
            ("ACGT", "ACGT"),
            ("ATCGATCG", "ATCGATCG"),
        ]
        for ref, qry in cases:
            cpu_res = align_sequence(ref, qry)
            gpu_res = cuda_align_sequence(ref, qry)
            assert gpu_res == cpu_res
            assert gpu_res.matches == len(ref)
            assert gpu_res.mismatches == 0
            assert gpu_res.score == 1.0
            assert gpu_res.mismatch_positions == []

    def test_single_mismatch(self):
        cases = [
            ("ACGT", "TCGT"),  # Mismatch at start (0)
            ("ACGT", "AGGT"),  # Mismatch at index 1
            ("ACGT", "ACCT"),  # Mismatch at index 2
            ("ACGT", "ACGA"),  # Mismatch at end (3)
        ]
        for ref, qry in cases:
            cpu_res = align_sequence(ref, qry)
            gpu_res = cuda_align_sequence(ref, qry)
            assert gpu_res == cpu_res
            assert gpu_res.matches == 3
            assert gpu_res.mismatches == 1
            assert gpu_res.score == 0.75
            assert len(gpu_res.mismatch_positions) == 1

    def test_multiple_mismatches(self):
        cases = [
            ("AAAA", "TTTT"),  # 100% mismatch
            ("ACGTACGT", "TGCAACGT"),  # alternating/partial
            ("CCCCCCCC", "CGCCGCCC"),
        ]
        for ref, qry in cases:
            cpu_res = align_sequence(ref, qry)
            gpu_res = cuda_align_sequence(ref, qry)
            assert gpu_res == cpu_res
            assert gpu_res.matches == cpu_res.matches
            assert gpu_res.mismatches == cpu_res.mismatches
            assert gpu_res.score == cpu_res.score
            assert gpu_res.mismatch_positions == cpu_res.mismatch_positions

    def test_different_sequence_lengths(self):
        """Test varied lengths: 10 bp, 50 bp, 100 bp, 500 bp."""
        for length in [10, 50, 100, 500]:
            ref = "ACGT" * (length // 4) + "A" * (length % 4)
            # Flip every 5th base
            qry_chars = list(ref)
            for i in range(0, length, 5):
                qry_chars[i] = "T" if qry_chars[i] != "T" else "A"
            qry = "".join(qry_chars)

            cpu_res = align_sequence(ref, qry)
            gpu_res = cuda_align_sequence(ref, qry)
            assert gpu_res == cpu_res

    def test_batch_cuda_alignment(self):
        pairs = [
            ("ACGT", "ACGT"),
            ("ACGT", "TCGT"),
            ("ACGT", "TGCA"),
            ("AAAA", "AAAA"),
        ]
        cpu_results = align_sequences(pairs)
        gpu_results = cuda_align_sequences(pairs)
        assert len(gpu_results) == len(cpu_results)
        for cpu_res, gpu_res in zip(cpu_results, gpu_results):
            assert gpu_res == cpu_res

    def test_cuda_align_chunks(self):
        target = "ACGT"
        chunks = ["ACGT", "TCGT", "TGCA", "AAAA"]
        cpu_results = align_chunks(chunks, target)
        gpu_results = cuda_align_chunks(chunks, target)
        assert len(gpu_results) == len(cpu_results)
        for cpu_res, gpu_res in zip(cpu_results, gpu_results):
            assert gpu_res == cpu_res

    def test_empty_sequence_alignment(self):
        cpu_res = align_sequence("", "")
        gpu_res = cuda_align_sequence("", "")
        assert gpu_res == cpu_res
        assert gpu_res.alignment_length == 0
        assert gpu_res.score == 0.0

    def test_batch_cuda_alignment_timed(self):
        pairs = [
            ("ACGT", "ACGT"),
            ("ACGT", "TCGT"),
        ]
        cpu_results = align_sequences(pairs)
        gpu_results, timing = cuda_align_sequences_timed(pairs)
        assert len(gpu_results) == len(cpu_results)
        for cpu_res, gpu_res in zip(cpu_results, gpu_results):
            assert gpu_res == cpu_res
        assert isinstance(timing, CUDATimingBreakdown)
        assert timing.total_wall_time >= 0.0
        assert timing.kernel_execution_time >= 0.0
