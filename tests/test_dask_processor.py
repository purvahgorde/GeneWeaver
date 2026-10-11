"""Comprehensive Unit and Integration Tests for Dask Distributed Processing Engine.

Week 3 Day 1 & Day 2: Dask Setup, Task Processing, and Pipeline Integration.

Verifies:
1. Environment diagnostics and clean fallback when Dask is missing.
2. Resource management and ownership (shared client vs owned cluster).
3. Core chunk alignment (single chunk, multiple chunks, empty input, ordering).
4. Full contract parity between CPU reference baseline and Dask processor.
5. Error handling across task exceptions, type errors, and invalid sequences.
6. End-to-end integration across FASTA parser, genome chunker, and Dask processor.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import pytest

# Ensure repository root and src/ are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alignment.cpu_alignment import (
    AlignmentResult,
    align_sequence,
    align_sequences,
    align_chunks,
    align_chunk_pairs,
)
from alignment.integration import run_pipeline
from src.distributed.dask_processor import (
    DaskGenomeProcessor,
    DaskNotAvailableError,
    DaskTaskError,
    DaskPipelineTiming,
    dask_align_chunks,
    dask_align_chunk_pairs,
    dask_align_sequences,
    process_fasta_dask,
    process_fasta_dask_timed,
    is_dask_available,
    get_dask_info,
)

# Skip entire suite if Dask is not installed
pytestmark = pytest.mark.skipif(
    not is_dask_available(),
    reason="Dask or distributed is not installed in the environment",
)

MOCK_GENOME_PATH = REPO_ROOT / "data" / "mock_genome.fasta"


# =============================================================================
# 1. Environment & Diagnostics Tests
# =============================================================================
class TestDaskEnvironmentAndDiagnostics:
    """Tests environment availability reporting and diagnostic metadata."""

    def test_is_dask_available(self):
        assert is_dask_available() is True

    def test_get_dask_info_structure(self):
        info = get_dask_info()
        assert isinstance(info, dict)
        assert info["available"] is True
        assert info["error"] is None
        assert info["dask_version"] is not None
        assert info["distributed_version"] is not None
        assert "active_client" in info

    def test_dask_not_available_raises_clean_error(self, monkeypatch):
        """Verify DaskNotAvailableError is raised when Dask is marked unavailable."""
        import src.distributed.dask_processor as dp

        monkeypatch.setattr(dp, "is_dask_available", lambda: False)
        with pytest.raises(DaskNotAvailableError, match="Dask is not installed"):
            DaskGenomeProcessor()


# =============================================================================
# 2. Lifecycle, Client Ownership, and Resource Cleanup Tests
# =============================================================================
class TestDaskProcessorLifecycleAndResourceCleanup:
    """Tests client/cluster lifecycle, ownership preservation, and cleanup."""

    def test_processor_context_manager(self):
        with DaskGenomeProcessor(processes=False) as proc:
            assert proc.owns_client is True
            assert proc.is_closed is False
            res = proc.align_chunks(["ACGT"], "ACGT")
            assert len(res) == 1
        assert proc.is_closed is True

    def test_processor_explicit_close(self):
        proc = DaskGenomeProcessor(processes=False)
        assert proc.owns_client is True
        proc.close()
        assert proc.is_closed is True
        # Idempotent close
        proc.close()
        assert proc.is_closed is True

    def test_shared_client_not_closed(self):
        """Verify shared client passed in by caller is preserved upon processor close."""
        from distributed import Client, LocalCluster

        cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False)
        shared_client = Client(cluster)
        try:
            proc = DaskGenomeProcessor(client=shared_client)
            assert proc.owns_client is False
            proc.align_chunks(["ACGT"], "ACGT")
            proc.close()

            # Client must remain active and not closed
            assert shared_client.status != "closed"
            assert proc.is_closed is True
        finally:
            shared_client.close()
            cluster.close()

    def test_repeated_processor_calls(self):
        """Verify a single processor instance can be called repeatedly without leaks."""
        with DaskGenomeProcessor(processes=False) as proc:
            for _ in range(5):
                res = proc.align_chunks(["ACGT", "AGGT"], "ACGT")
                assert len(res) == 2
                assert res[0].matches == 4
                assert res[1].matches == 3


# =============================================================================
# 3. Core Chunk Alignment and Ordering Tests
# =============================================================================
class TestDaskChunkAlignmentCore:
    """Tests chunk alignment correctness, ordering, and CPU parity."""

    def test_empty_chunks_returns_empty_list(self):
        res = dask_align_chunks([], "ACGT")
        assert res == []

    def test_single_chunk_exact_match(self):
        res = dask_align_chunks(["ACGT"], "ACGT")
        assert len(res) == 1
        assert res[0].matches == 4
        assert res[0].mismatches == 0
        assert res[0].score == 1.0
        assert res[0].mismatch_positions == []

    def test_multiple_chunks_equivalence_with_cpu(self):
        chunks = [
            "ACGTACGTAC",
            "ACGTTCGTAC",
            "TTTTTTTTTT",
            "ACGTACGTAT",
            "CCCCCCCCCC",
        ]
        target = "ACGTACGTAC"
        cpu_res = align_chunks(chunks, target)
        dask_res = dask_align_chunks(chunks, target)

        assert len(dask_res) == len(cpu_res)
        assert dask_res == cpu_res

    def test_result_ordering_preserved(self):
        """Verify output result ordering strictly preserves input chunk ordering."""
        # Chunks have varying mismatch patterns
        chunks = [
            "TCGT",  # mismatch at 0
            "AGGT",  # mismatch at 1
            "ACAT",  # mismatch at 2
            "ACGA",  # mismatch at 3
            "ACGT",  # exact match
        ]
        target = "ACGT"
        cpu_res = align_chunks(chunks, target)
        dask_res = dask_align_chunks(chunks, target, batch_size=1)

        assert dask_res == cpu_res
        assert [r.mismatch_positions for r in dask_res] == [
            [0],
            [1],
            [2],
            [3],
            [],
        ]

    def test_different_batch_sizes_produce_identical_results(self):
        chunks = ["ACGT", "AGGT", "TCGT", "ACGA", "ACTT", "AAGT"] * 3
        target = "ACGT"
        ref_results = align_chunks(chunks, target)

        for b_size in [1, 2, 4, 10]:
            d_res = dask_align_chunks(chunks, target, batch_size=b_size)
            assert d_res == ref_results

    def test_skip_unequal_true_omits_boundary_chunks(self):
        chunks = ["ACGT", "ACG", "ACGT", "A", "ACGT"]
        target = "ACGT"
        res = dask_align_chunks(chunks, target, skip_unequal=True)
        assert len(res) == 3
        for r in res:
            assert r.alignment_length == 4

    def test_skip_unequal_false_raises_value_error(self):
        chunks = ["ACGT", "ACG", "ACGT"]
        target = "ACGT"
        with pytest.raises(ValueError, match="Sequences must have equal length"):
            dask_align_chunks(chunks, target, skip_unequal=False)

    def test_small_dna_sequences(self):
        assert dask_align_chunks(["A"], "A") == [align_sequence("A", "A")]
        assert dask_align_chunks(["A"], "C") == [align_sequence("A", "C")]
        assert dask_align_chunks(["AT"], "AT") == [align_sequence("AT", "AT")]
        assert dask_align_chunks(["ATC"], "ATG") == [align_sequence("ATC", "ATG")]


# =============================================================================
# 4. Chunk Pairs and Sequence Batches Tests
# =============================================================================
class TestDaskChunkPairsAndSequences:
    """Tests dask_align_chunk_pairs and dask_align_sequences."""

    def test_align_chunk_pairs_parity_with_cpu(self):
        ref_chunks = ["ACGT", "GGCC", "TTTA"]
        qry_chunks = ["ACGT", "GGAA", "TTTA"]
        cpu_res = align_chunk_pairs(ref_chunks, qry_chunks)
        dask_res = dask_align_chunk_pairs(ref_chunks, qry_chunks)
        assert dask_res == cpu_res

    def test_align_chunk_pairs_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="Chunk lists must have equal length"):
            dask_align_chunk_pairs(["ACGT"], ["ACGT", "ACGT"])

    def test_align_chunk_pairs_empty_lists(self):
        assert dask_align_chunk_pairs([], []) == []

    def test_align_sequences_parity_with_cpu(self):
        pairs = [("ACGT", "ACGT"), ("ACGT", "AGGT"), ("TTTT", "CCCC")]
        cpu_res = align_sequences(pairs)
        dask_res = dask_align_sequences(pairs)
        assert dask_res == cpu_res

    def test_align_sequences_empty(self):
        assert dask_align_sequences([]) == []


# =============================================================================
# 5. Failure Handling and Error Visibility Tests
# =============================================================================
class TestDaskFailureHandling:
    """Tests failure propagation, invalid input rejection, and type safety."""

    def test_invalid_chunk_type_raises_type_error(self):
        with pytest.raises(TypeError, match="chunks must be a list"):
            dask_align_chunks("ACGT", "ACGT")  # type: ignore

    def test_invalid_target_type_raises_type_error(self):
        with pytest.raises(TypeError, match="target must be a string"):
            dask_align_chunks(["ACGT"], 123)  # type: ignore

    def test_invalid_chunk_element_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Each chunk must be a string"):
            dask_align_chunks(["ACGT", 42], "ACGT")  # type: ignore

    def test_invalid_dna_characters_in_target_raises_value_error(self):
        with pytest.raises(ValueError, match="target contains invalid characters"):
            dask_align_chunks(["ACGT"], "ACGN")

    def test_invalid_dna_characters_in_chunk_raises_task_failure(self):
        with pytest.raises(ValueError, match="reference contains invalid characters"):
            dask_align_chunks(["ACGT", "ACGN"], "ACGT")

    def test_invalid_batch_size_raises_value_error(self):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            dask_align_chunks(["ACGT"], "ACGT", batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            dask_align_chunks(["ACGT"], "ACGT", batch_size=-5)

    def test_task_raises_arbitrary_exception_handled_clearly(self, monkeypatch):
        """Simulate unexpected task failure and ensure error is propagated clearly."""
        import src.distributed.dask_processor as dp

        def broken_task(*args, **kwargs):
            raise RuntimeError("Simulated worker compute crash")

        monkeypatch.setattr(dp, "_align_single_chunk_task", broken_task)
        with pytest.raises(RuntimeError, match="Simulated worker compute crash"):
            dask_align_chunks(["ACGT"], "ACGT")


# =============================================================================
# 6. End-to-End Pipeline Integration Tests
# =============================================================================
class TestDaskEndToEndPipelineIntegration:
    """Tests the full pipeline path: FASTA -> Parser -> Chunker -> Dask Tasks -> Results."""

    def test_e2e_mock_genome_pipeline_parity(self):
        """Verify full pipeline on mock_genome.fasta matches CPU baseline exactly."""
        target = "ATGCGTACGT"
        chunk_size = 10

        cpu_results = run_pipeline(
            MOCK_GENOME_PATH, target, chunk_size, skip_unequal=True
        )
        dask_results = process_fasta_dask(
            MOCK_GENOME_PATH, target, chunk_size, skip_unequal=True
        )

        assert len(dask_results) == 8
        assert dask_results == cpu_results

    def test_e2e_pipeline_timed(self):
        target = "ATGCGTACGT"
        chunk_size = 10

        results, timing = process_fasta_dask_timed(
            MOCK_GENOME_PATH, target, chunk_size, skip_unequal=True
        )

        assert len(results) == 8
        assert isinstance(timing, DaskPipelineTiming)
        assert timing.num_sequences == 2
        assert timing.num_chunks == 10
        assert timing.num_aligned == 8
        assert timing.num_tasks > 0
        assert timing.parse_time >= 0.0
        assert timing.chunk_time >= 0.0
        assert timing.dask_align_time >= 0.0
        assert timing.total_pipeline_time >= 0.0

    def test_e2e_missing_fasta_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            process_fasta_dask("non_existent_file.fasta", "ATGCGTACGT", 10)

    def test_e2e_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            process_fasta_dask(MOCK_GENOME_PATH, "ATGCGTACGT", 0)
        with pytest.raises(TypeError, match="chunk_size must be an int"):
            process_fasta_dask(MOCK_GENOME_PATH, "ATGCGTACGT", "10")  # type: ignore

    def test_e2e_custom_fasta_pipeline(self):
        """Verify complete pipeline on dynamically generated multi-record FASTA."""
        with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as tf:
            tf.write(">seq1\nACGTACGTACGT\n>seq2\nACGTACGTAAAA\n")
            temp_path = tf.name

        try:
            target = "ACGT"
            chunk_size = 4
            cpu_results = run_pipeline(temp_path, target, chunk_size)
            dask_results = process_fasta_dask(temp_path, target, chunk_size)

            assert len(dask_results) == 6
            assert dask_results == cpu_results
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_e2e_invariants_verification(self):
        """Verify mathematical invariants hold for every aligned chunk."""
        target = "ATGCGTACGT"
        results = process_fasta_dask(MOCK_GENOME_PATH, target, 10, skip_unequal=True)

        for res in results:
            assert res.matches + res.mismatches == res.alignment_length
            assert res.alignment_length == len(target)
            assert len(res.mismatch_positions) == res.mismatches
            assert 0.0 <= res.score <= 1.0
            assert res.score == pytest.approx(res.matches / res.alignment_length)
