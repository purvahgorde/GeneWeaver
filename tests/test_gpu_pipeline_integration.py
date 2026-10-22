from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignment.cpu_alignment import (
    AlignmentResult,
    align_sequence,
    align_sequences,
    align_chunks,
    align_chunk_pairs,
)
from alignment.cuda_alignment import (
    cuda_align_sequence,
    cuda_align_sequences,
    cuda_align_sequences_timed,
    cuda_align_chunks,
    cuda_align_chunks_timed,
    cuda_align_chunk_pairs,
    is_cuda_available,
    get_cuda_device_info,
    CUDANotAvailableError,
    CUDATimingBreakdown,
)
from alignment.dispatcher import (
    align_sequence_dispatch,
    align_sequences_dispatch,
    align_chunks_dispatch,
    align_chunk_pairs_dispatch,
    normalize_backend,
    SUPPORTED_BACKENDS,
)
from alignment.integration import (
    run_pipeline,
    run_pipeline_timed,
    PipelineTimingBreakdown,
)
from parser.fasta_parser import parse_fasta
from chunking.genome_chunker import chunk_sequence

skip_if_no_cuda = pytest.mark.skipif(
    not is_cuda_available(),
    reason=f"CUDA execution is not available: {get_cuda_device_info().get('reason', 'Unknown reason')}",
)


# ═════════════════════════════════════════════════════════════════════════
#  1. Alignment Dispatcher Interface Tests (Always run on CPU host)
# ═════════════════════════════════════════════════════════════════════════
class TestAlignmentDispatcher:
    """Tests the unified dispatch layer interface and input validation."""

    def test_normalize_backend_valid(self):
        assert normalize_backend("cpu") == "cpu"
        assert normalize_backend("CPU") == "cpu"
        assert normalize_backend("cuda") == "cuda"
        assert normalize_backend("CUDA") == "cuda"
        assert normalize_backend("gpu") == "cuda"
        assert normalize_backend("GPU") == "cuda"

    def test_normalize_backend_invalid_raises(self):
        with pytest.raises(ValueError, match="Unsupported backend 'tpu'"):
            normalize_backend("tpu")
        with pytest.raises(TypeError, match="backend must be a string"):
            normalize_backend(123)  # type: ignore

    def test_align_sequence_dispatch_cpu(self):
        res = align_sequence_dispatch("ACGT", "ACGT", backend="cpu")
        assert res.matches == 4
        assert res.mismatches == 0
        assert res.score == 1.0

    def test_align_sequences_dispatch_cpu(self):
        pairs = [("ACGT", "ACGT"), ("ACGT", "AGGT")]
        res = align_sequences_dispatch(pairs, backend="cpu")
        assert len(res) == 2
        assert res[0].matches == 4
        assert res[1].matches == 3

    def test_align_chunks_dispatch_cpu(self):
        chunks = ["ACGT", "ACGA", "TCGT"]
        res = align_chunks_dispatch(chunks, "ACGT", backend="cpu")
        assert len(res) == 3
        assert res[0].matches == 4
        assert res[1].matches == 3
        assert res[2].matches == 3

    def test_align_chunk_pairs_dispatch_cpu(self):
        ref_chunks = ["ACGT", "GGCC"]
        qry_chunks = ["ACGT", "GGAA"]
        res = align_chunk_pairs_dispatch(ref_chunks, qry_chunks, backend="cpu")
        assert len(res) == 2
        assert res[0].matches == 4
        assert res[1].matches == 2

    def test_dispatch_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported backend 'dask'"):
            align_sequence_dispatch("ACGT", "ACGT", backend="dask")

        with pytest.raises(ValueError, match="Unsupported backend 'dask'"):
            align_sequences_dispatch([("ACGT", "ACGT")], backend="dask")

        with pytest.raises(ValueError, match="Unsupported backend 'dask'"):
            align_chunks_dispatch(["ACGT"], "ACGT", backend="dask")

        with pytest.raises(ValueError, match="Unsupported backend 'dask'"):
            align_chunk_pairs_dispatch(["ACGT"], ["ACGT"], backend="dask")

        with pytest.raises(ValueError, match="Unsupported backend 'dask'"):
            run_pipeline(REPO_ROOT / "data" / "mock_genome.fasta", "ACGT", 10, backend="dask")


# ═════════════════════════════════════════════════════════════════════════
#  2. Fallback & Safe Diagnostics When CUDA is Unavailable
# ═════════════════════════════════════════════════════════════════════════
class TestCUDAIntegrationFallback:
    """Verifies pipeline behavior and clean error raising when CUDA is unavailable."""

    @pytest.fixture
    def mock_fasta_path(self):
        path = REPO_ROOT / "data" / "mock_genome.fasta"
        assert path.exists()
        return path

    def test_run_pipeline_cuda_fallback(self, mock_fasta_path):
        if not is_cuda_available():
            with pytest.raises(CUDANotAvailableError):
                run_pipeline(mock_fasta_path, "ATGCGTACGT", chunk_size=10, backend="cuda")

            with pytest.raises(CUDANotAvailableError):
                run_pipeline(mock_fasta_path, "ATGCGTACGT", chunk_size=10, backend="gpu")

            with pytest.raises(CUDANotAvailableError):
                run_pipeline_timed(mock_fasta_path, "ATGCGTACGT", chunk_size=10, backend="cuda")

    def test_dispatch_cuda_fallback(self):
        if not is_cuda_available():
            with pytest.raises(CUDANotAvailableError):
                align_sequence_dispatch("ACGT", "ACGT", backend="cuda")

            with pytest.raises(CUDANotAvailableError):
                align_sequences_dispatch([("ACGT", "ACGT")], backend="cuda")

            with pytest.raises(CUDANotAvailableError):
                align_chunks_dispatch(["ACGT"], "ACGT", backend="cuda")

            with pytest.raises(CUDANotAvailableError):
                align_chunk_pairs_dispatch(["ACGT"], ["ACGT"], backend="cuda")

    def test_cpu_pipeline_unaffected_by_cuda_status(self, mock_fasta_path):
        """CPU pipeline must always function regardless of CUDA availability."""
        results = run_pipeline(mock_fasta_path, "ATGCGTACGT", chunk_size=10, backend="cpu", skip_unequal=True)
        assert len(results) == 8
        assert results[0].matches == 10
        assert results[0].mismatches == 0
        assert results[0].score == 1.0

    def test_timed_pipeline_cpu(self, mock_fasta_path):
        results, timing = run_pipeline_timed(
            mock_fasta_path, "ATGCGTACGT", chunk_size=10, backend="cpu", skip_unequal=True
        )
        assert len(results) == 8
        assert isinstance(timing, PipelineTimingBreakdown)
        assert timing.backend == "cpu"
        assert timing.num_sequences == 2
        assert timing.num_chunks == 10
        assert timing.num_aligned == 8
        assert timing.parse_time >= 0.0
        assert timing.chunk_time >= 0.0
        assert timing.alignment_time >= 0.0
        assert timing.total_pipeline_time >= 0.0
        assert timing.cuda_timing is None


# ═════════════════════════════════════════════════════════════════════════
#  3. Direct Chunk Processing with Member 2 Chunker Output
# ═════════════════════════════════════════════════════════════════════════
class TestGenomeChunkerCUDAIntegration:
    """Tests feeding Member 2 chunker outputs into CUDA alignment routines."""

    def test_chunker_output_contract(self):
        seq = "ATGCGTACGTAGCTAGCTA"  # 19 bp
        chunks = chunk_sequence(seq, 5)
        # Chunks: ['ATGCG', 'TACGT', 'AGCTA', 'GCTA']
        assert len(chunks) == 4
        assert [len(c) for c in chunks] == [5, 5, 5, 4]
        assert "".join(chunks) == seq

    @skip_if_no_cuda
    def test_chunker_to_cuda_exact_match(self):
        seq = "ATGC" * 10  # 40 bp
        chunks = chunk_sequence(seq, 10)  # 4 equal chunks of 10 bp
        assert len(chunks) == 4
        assert all(len(c) == 10 for c in chunks)

        target = "ATGC" * 2 + "AT"  # 10 bp: "ATGCATGCAT"
        cpu_results = align_chunks(chunks, target)
        gpu_results = cuda_align_chunks(chunks, target)

        assert len(gpu_results) == 4
        assert gpu_results == cpu_results
        for res in gpu_results:
            assert isinstance(res, AlignmentResult)
            assert res.alignment_length == 10

    @skip_if_no_cuda
    def test_chunk_boundaries_and_ordering_preserved(self):
        """Verifies chunk order and index alignment match between CPU and CUDA."""
        seq = "AAAAAAAAAA" + "CCCCCCCCCC" + "GGGGGGGGGG" + "TTTTTTTTTT"
        chunks = chunk_sequence(seq, 10)
        assert len(chunks) == 4

        target = "AAAAAAAAAA"
        cpu_results = align_chunks(chunks, target)
        gpu_results = cuda_align_chunks(chunks, target)

        assert gpu_results == cpu_results
        # Chunk 0: all A -> 10 matches
        assert gpu_results[0].matches == 10
        # Chunk 1: all C -> 0 matches
        assert gpu_results[1].matches == 0
        # Chunk 2: all G -> 0 matches
        assert gpu_results[2].matches == 0
        # Chunk 3: all T -> 0 matches
        assert gpu_results[3].matches == 0

    @skip_if_no_cuda
    def test_skip_unequal_tail_chunks(self):
        seq = "ATGCGTACGTA"  # 11 bp
        chunks = chunk_sequence(seq, 5)  # 5, 5, 1
        target = "ATGCG"  # 5 bp

        # skip_unequal=True should omit the 1 bp tail chunk
        cpu_results = align_chunks(chunks, target, skip_unequal=True)
        gpu_results = cuda_align_chunks(chunks, target, skip_unequal=True)

        assert len(gpu_results) == 2
        assert gpu_results == cpu_results

        # skip_unequal=False should raise ValueError on both
        with pytest.raises(ValueError, match="Sequences must have equal length"):
            align_chunks(chunks, target, skip_unequal=False)
        with pytest.raises(ValueError, match="Sequences must have equal length"):
            cuda_align_chunks(chunks, target, skip_unequal=False)


# ═════════════════════════════════════════════════════════════════════════
#  4. End-to-End Pipeline: Parser -> Chunker -> CUDA vs CPU
# ═════════════════════════════════════════════════════════════════════════
@skip_if_no_cuda
class TestEndToEndPipelineCUDA:
    """Verifies FASTA -> Parser -> Chunker -> CUDA Alignment pipeline matches CPU oracle."""

    @pytest.fixture
    def mock_fasta_path(self):
        path = REPO_ROOT / "data" / "mock_genome.fasta"
        assert path.exists()
        return path

    def test_mock_fasta_e2e_parity(self, mock_fasta_path):
        target = "ATGCGTACGT"  # 10 bp
        chunk_size = 10

        cpu_results = run_pipeline(mock_fasta_path, target, chunk_size, backend="cpu", skip_unequal=True)
        gpu_results = run_pipeline(mock_fasta_path, target, chunk_size, backend="cuda", skip_unequal=True)

        assert len(gpu_results) == len(cpu_results) == 8
        for i, (cpu_res, gpu_res) in enumerate(zip(cpu_results, gpu_results)):
            assert gpu_res == cpu_res, f"Mismatch at chunk {i}: CPU={cpu_res} != GPU={gpu_res}"

    def test_mock_fasta_e2e_single_mismatch(self, mock_fasta_path):
        target = "TTGCGTACGT"  # 1 mismatch against start of chr1
        cpu_results = run_pipeline(mock_fasta_path, target, chunk_size=10, backend="cpu", skip_unequal=True)
        gpu_results = run_pipeline(mock_fasta_path, target, chunk_size=10, backend="cuda", skip_unequal=True)

        assert gpu_results == cpu_results
        assert gpu_results[0].matches == 9
        assert gpu_results[0].mismatches == 1
        assert gpu_results[0].mismatch_positions == [0]

    def test_mock_fasta_e2e_multiple_mismatches(self, mock_fasta_path):
        target = "TACGCATGCA"
        cpu_results = run_pipeline(mock_fasta_path, target, chunk_size=10, backend="cpu", skip_unequal=True)
        gpu_results = run_pipeline(mock_fasta_path, target, chunk_size=10, backend="cuda", skip_unequal=True)

        assert gpu_results == cpu_results

    def test_mock_fasta_e2e_even_chunk_size(self, mock_fasta_path):
        target = "ATGCG"  # 5 bp
        cpu_results = run_pipeline(mock_fasta_path, target, chunk_size=5, backend="cpu", skip_unequal=True)
        gpu_results = run_pipeline(mock_fasta_path, target, chunk_size=5, backend="cuda", skip_unequal=True)

        assert len(gpu_results) == len(cpu_results)
        assert gpu_results == cpu_results

    def test_e2e_timed_pipeline_cuda(self, mock_fasta_path):
        target = "ATGCGTACGT"
        gpu_results, timing = run_pipeline_timed(
            mock_fasta_path, target, chunk_size=10, backend="cuda", skip_unequal=True
        )
        assert len(gpu_results) == 8
        assert isinstance(timing, PipelineTimingBreakdown)
        assert timing.backend == "cuda"
        assert timing.cuda_timing is not None
        assert timing.cuda_timing.h2d_transfer_time >= 0.0
        assert timing.cuda_timing.kernel_execution_time >= 0.0
        assert timing.cuda_timing.d2h_transfer_time >= 0.0
        assert timing.cuda_timing.total_gpu_time >= 0.0
        assert timing.total_pipeline_time >= 0.0

    def test_e2e_empty_fasta_cuda(self, tmp_path):
        empty_fasta = tmp_path / "empty.fasta"
        empty_fasta.write_text("")
        results = run_pipeline(empty_fasta, "ACGT", chunk_size=4, backend="cuda")
        assert results == []

    def test_e2e_single_sequence_fasta_cuda(self, tmp_path):
        single_fasta = tmp_path / "single.fasta"
        single_fasta.write_text(">test\nACGTACGT\n")
        cpu_results = run_pipeline(single_fasta, "ACGT", chunk_size=4, backend="cpu")
        gpu_results = run_pipeline(single_fasta, "ACGT", chunk_size=4, backend="cuda")
        assert len(gpu_results) == 2
        assert gpu_results == cpu_results


# ═════════════════════════════════════════════════════════════════════════
#  5. Pipeline Input Validation and Error Handling
# ═════════════════════════════════════════════════════════════════════════
class TestPipelineInputValidation:
    """Verifies edge case and error handling in the integration pipeline."""

    def test_invalid_chunk_size_raises(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">seq\nACGT\n")
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            run_pipeline(fasta, "ACGT", chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            run_pipeline(fasta, "ACGT", chunk_size=-5)
        with pytest.raises(TypeError, match="chunk_size must be an int"):
            run_pipeline(fasta, "ACGT", chunk_size="10")  # type: ignore

    def test_nonexistent_fasta_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_pipeline("nonexistent_path_file.fasta", "ACGT", chunk_size=10)

    def test_invalid_target_bases_raises(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">seq\nACGT\n")
        with pytest.raises(ValueError, match="target contains invalid characters"):
            run_pipeline(fasta, "ACGN", chunk_size=4)


# ═════════════════════════════════════════════════════════════════════════
#  6. Backend Selection, Explicit Reporting, and Fallback Reliability Tests
# ═════════════════════════════════════════════════════════════════════════
class TestBackendSelectionAndFallbackReliability:
    """Verifies Week 4 Day 4 & 5 backend selection, reliability, and fallback reporting."""

    @pytest.fixture
    def mock_fasta_path(self):
        path = REPO_ROOT / "data" / "mock_genome.fasta"
        assert path.exists()
        return path

    def test_normalize_backend_auto(self):
        assert normalize_backend("auto") == "auto"
        assert normalize_backend("AUTO") == "auto"

    def test_align_sequence_dispatch_auto(self):
        res = align_sequence_dispatch("ACGT", "ACCT", backend="auto")
        assert res.matches == 3
        assert res.mismatches == 1
        assert res.score == 0.75

    def test_align_sequences_dispatch_auto(self):
        pairs = [("ACGT", "ACGT"), ("ACGT", "AGGT")]
        res = align_sequences_dispatch(pairs, backend="auto")
        assert len(res) == 2
        assert res[0].matches == 4
        assert res[1].matches == 3

    def test_align_chunks_dispatch_auto(self):
        chunks = ["ACGT", "ACCT", "AAAA"]
        res = align_chunks_dispatch(chunks, "ACGT", backend="auto")
        assert len(res) == 3
        assert res[0].matches == 4
        assert res[1].matches == 3
        assert res[2].matches == 1

    def test_align_chunk_pairs_dispatch_auto(self):
        ref_chunks = ["ACGT", "GGCC"]
        qry_chunks = ["ACGT", "GGAA"]
        res = align_chunk_pairs_dispatch(ref_chunks, qry_chunks, backend="auto")
        assert len(res) == 2
        assert res[0].matches == 4
        assert res[1].matches == 2

    def test_run_pipeline_auto(self, mock_fasta_path):
        results = run_pipeline(mock_fasta_path, "ATGCGTACGT", chunk_size=10, backend="auto", skip_unequal=True)
        assert len(results) == 8
        assert results[0].matches == 10

    def test_run_pipeline_timed_auto_reporting(self, mock_fasta_path):
        results, timing = run_pipeline_timed(
            mock_fasta_path, "ATGCGTACGT", chunk_size=10, backend="auto", skip_unequal=True
        )
        assert len(results) == 8
        assert isinstance(timing, PipelineTimingBreakdown)
        expected_backend = "cuda" if is_cuda_available() else "cpu"
        assert timing.backend == expected_backend

    def test_run_pipeline_cuda_fallback_to_cpu_when_forced_failure(self, mock_fasta_path, monkeypatch):
        """When CUDA fails and fallback_to_cpu=True, pipeline cleanly falls back to CPU."""
        def mock_cuda_fail(*args, **kwargs):
            raise RuntimeError("Simulated GPU driver crash")

        import alignment.integration as integ
        monkeypatch.setattr(integ, "cuda_align_chunks_timed", mock_cuda_fail)

        # fallback_to_cpu=False must propagate error
        with pytest.raises(RuntimeError, match="Simulated GPU driver crash"):
            run_pipeline_timed(
                mock_fasta_path, "ATGCGTACGT", chunk_size=10, backend="cuda", fallback_to_cpu=False
            )

        # fallback_to_cpu=True must fall back to CPU and report backend as 'cpu (fallback)'
        results, timing = run_pipeline_timed(
            mock_fasta_path, "ATGCGTACGT", chunk_size=10, backend="cuda", fallback_to_cpu=True
        )
        assert len(results) == 8
        assert results[0].matches == 10
        assert results[0].mismatches == 0
        assert results[0].score == 1.0
        assert timing.backend == "cpu (fallback)"
        assert timing.cuda_timing is None

    def test_run_pipeline_auto_fallback_when_cuda_fails(self, mock_fasta_path, monkeypatch):
        """When backend='auto' and CUDA fails, pipeline falls back to CPU and reports 'cpu (fallback)'."""
        def mock_cuda_fail(*args, **kwargs):
            raise RuntimeError("Simulated GPU out of memory")

        import alignment.integration as integ
        monkeypatch.setattr(integ, "cuda_align_chunks_timed", mock_cuda_fail)
        monkeypatch.setattr(integ, "is_cuda_available", lambda: True)

        results, timing = run_pipeline_timed(
            mock_fasta_path, "ATGCGTACGT", chunk_size=10, backend="auto", skip_unequal=True
        )
        assert len(results) == 8
        assert results[0].matches == 10
        assert timing.backend == "cpu (fallback)"

    def test_dispatch_cuda_fallback_to_cpu(self, monkeypatch):
        """Dispatch layer fallback_to_cpu handles forced GPU failure cleanly."""
        def mock_fail(*args, **kwargs):
            raise RuntimeError("GPU device execution failure")

        import alignment.dispatcher as disp
        monkeypatch.setattr(disp, "cuda_align_sequence", mock_fail)
        monkeypatch.setattr(disp, "cuda_align_sequences", mock_fail)
        monkeypatch.setattr(disp, "cuda_align_chunks", mock_fail)
        monkeypatch.setattr(disp, "cuda_align_chunk_pairs", mock_fail)

        # Without fallback -> raises
        with pytest.raises(RuntimeError):
            align_sequence_dispatch("ACGT", "ACGT", backend="cuda", fallback_to_cpu=False)

        # With fallback -> CPU handles cleanly
        res = align_sequence_dispatch("ACGT", "ACGT", backend="cuda", fallback_to_cpu=True)
        assert res.matches == 4

        res_seqs = align_sequences_dispatch([("ACGT", "ACGT")], backend="cuda", fallback_to_cpu=True)
        assert len(res_seqs) == 1 and res_seqs[0].matches == 4

        res_chunks = align_chunks_dispatch(["ACGT"], "ACGT", backend="cuda", fallback_to_cpu=True)
        assert len(res_chunks) == 1 and res_chunks[0].matches == 4

        res_pairs = align_chunk_pairs_dispatch(["ACGT"], ["ACGT"], backend="cuda", fallback_to_cpu=True)
        assert len(res_pairs) == 1 and res_pairs[0].matches == 4

    def test_pipeline_entrypoint_with_fallback(self, mock_fasta_path):
        """Top-level src/pipeline.py run_alignment_pipeline accepts all arguments seamlessly."""
        from src.pipeline import run_alignment_pipeline
        results = run_alignment_pipeline(
            mock_fasta_path,
            target="ATGCGTACGT",
            chunk_size=10,
            backend="auto",
            fallback_to_cpu=True,
            use_shared_memory=True,
        )
        assert len(results) == 8
        assert results[0].matches == 10
