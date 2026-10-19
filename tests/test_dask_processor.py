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
    calculate_optimal_batch_size,
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

    def test_single_chunk_with_mismatches(self):
        res = dask_align_chunks(["ACTG"], "ACGT")
        assert len(res) == 1
        assert res[0].matches == 2
        assert res[0].mismatches == 2
        assert res[0].mismatch_positions == [2, 3]

    def test_empty_target_with_empty_chunk(self):
        """Aligning an empty chunk against an empty target should succeed with zeroed result."""
        res = dask_align_chunks([""], "")
        assert len(res) == 1
        assert res[0] == align_sequence("", "")
        assert res[0].alignment_length == 0
        assert res[0].matches == 0

    def test_empty_target_with_non_empty_chunk_raises(self):
        with pytest.raises(ValueError, match="Sequences must have equal length"):
            dask_align_chunks(["ACGT"], "", skip_unequal=False)

    def test_case_insensitivity_parity(self):
        chunks = ["acgt", "AcGt", "aCgT"]
        target = "ACGT"
        cpu_res = align_chunks(chunks, target)
        dask_res = dask_align_chunks(chunks, target)
        assert dask_res == cpu_res
        assert all(r.score == 1.0 for r in dask_res)

    def test_large_chunk_count_ordering(self):
        """Verify strict ordering is maintained across 50 chunks with diverse mismatches."""
        chunks = [
            f"{'A' * i}{'C' * (10 - i)}"
            for i in range(11)
        ] * 5  # 55 chunks
        target = "AAAAAAAAAA"
        cpu_res = align_chunks(chunks, target)
        dask_res = dask_align_chunks(chunks, target, batch_size=7)
        assert len(dask_res) == len(chunks)
        assert dask_res == cpu_res


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

    def test_align_chunk_pairs_ordering(self):
        ref_chunks = ["AAAA", "CCCC", "GGGG", "TTTT"]
        qry_chunks = ["AAAT", "CCCA", "GGGT", "TTTC"]
        cpu_res = align_chunk_pairs(ref_chunks, qry_chunks)
        dask_res = dask_align_chunk_pairs(ref_chunks, qry_chunks, batch_size=2)
        assert dask_res == cpu_res

    def test_align_chunk_pairs_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="Chunk lists must have equal length"):
            dask_align_chunk_pairs(["ACGT"], ["ACGT", "ACGT"])

    def test_align_chunk_pairs_empty_lists(self):
        assert dask_align_chunk_pairs([], []) == []

    def test_align_chunk_pairs_single_pair(self):
        ref = ["ACGT"]
        qry = ["ACTT"]
        assert dask_align_chunk_pairs(ref, qry) == align_chunk_pairs(ref, qry)

    def test_align_chunk_pairs_element_type_error(self):
        with pytest.raises(TypeError, match="Each reference chunk must be a string"):
            dask_align_chunk_pairs([123], ["ACGT"])  # type: ignore
        with pytest.raises(TypeError, match="Each query chunk must be a string"):
            dask_align_chunk_pairs(["ACGT"], [456])  # type: ignore

    def test_align_sequences_parity_with_cpu(self):
        pairs = [("ACGT", "ACGT"), ("ACGT", "AGGT"), ("TTTT", "CCCC")]
        cpu_res = align_sequences(pairs)
        dask_res = dask_align_sequences(pairs)
        assert dask_res == cpu_res

    def test_align_sequences_different_batch_sizes(self):
        pairs = [("ACGT", "ACGT"), ("ACGT", "AGGT"), ("TTTT", "CCCC")] * 4
        cpu_res = align_sequences(pairs)
        for b_size in [1, 2, 5, 20]:
            dask_res = dask_align_sequences(pairs, batch_size=b_size)
            assert dask_res == cpu_res

    def test_align_sequences_empty(self):
        assert dask_align_sequences([]) == []

    def test_align_sequences_type_errors(self):
        with pytest.raises(TypeError, match="pairs must be a list"):
            dask_align_sequences("not_a_list")  # type: ignore
        with pytest.raises(TypeError, match="Each pair must be a 2-tuple"):
            dask_align_sequences(["not_a_tuple"])  # type: ignore
        with pytest.raises(TypeError, match="Pair elements must be strings"):
            dask_align_sequences([(123, "ACGT")])  # type: ignore


# =============================================================================
# 5. Local Schedulers and Execution Configuration Tests
# =============================================================================
class TestDaskSchedulerConfigurations:
    """Tests Dask execution using different scheduler configurations."""

    def test_align_chunks_threads_scheduler(self):
        chunks = ["ACGT", "AGGT", "TGCA"]
        target = "ACGT"
        res = dask_align_chunks(chunks, target, scheduler="threads")
        assert res == align_chunks(chunks, target)

    def test_align_chunks_sync_scheduler(self):
        chunks = ["ACGT", "AGGT", "TGCA"]
        target = "ACGT"
        res = dask_align_chunks(chunks, target, scheduler="sync")
        assert res == align_chunks(chunks, target)

    def test_align_chunks_single_threaded_scheduler(self):
        chunks = ["ACGT", "AGGT", "TGCA"]
        target = "ACGT"
        res = dask_align_chunks(chunks, target, scheduler="single-threaded")
        assert res == align_chunks(chunks, target)

    def test_processor_with_custom_workers_and_threads(self):
        with DaskGenomeProcessor(
            n_workers=2, threads_per_worker=1, processes=False
        ) as proc:
            assert proc.client is not None
            chunks = ["ACGT", "AGGT", "TGCA"]
            target = "ACGT"
            res = proc.align_chunks(chunks, target)
            assert res == align_chunks(chunks, target)


# =============================================================================
# 6. Failure Handling and Error Visibility Tests
# =============================================================================
class TestDaskFailureHandling:
    """Tests failure propagation, invalid input rejection, and task crash safety."""

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

    def test_task_raises_arbitrary_exception_single_chunk(self, monkeypatch):
        """Simulate unexpected task failure in single task mode."""
        import src.distributed.dask_processor as dp

        def broken_task(*args, **kwargs):
            raise RuntimeError("Simulated worker compute crash")

        monkeypatch.setattr(dp, "_align_single_chunk_task", broken_task)
        with pytest.raises(RuntimeError, match="Simulated worker compute crash"):
            dask_align_chunks(["ACGT"], "ACGT", batch_size=1)

    def test_task_raises_arbitrary_exception_batched(self, monkeypatch):
        """Simulate unexpected task failure in batched task mode."""
        import src.distributed.dask_processor as dp

        def broken_batch_task(*args, **kwargs):
            raise RuntimeError("Simulated worker batch compute crash")

        monkeypatch.setattr(dp, "_align_chunk_batch_task", broken_batch_task)
        with pytest.raises(RuntimeError, match="Simulated worker batch compute crash"):
            dask_align_chunks(["ACGT", "AGGT"], "ACGT", batch_size=2)

    def test_task_failure_does_not_silently_produce_incomplete_results(self, monkeypatch):
        """Ensure failure in partial tasks stops execution and does NOT return partial results."""
        import src.distributed.dask_processor as dp

        original_task = dp._align_single_chunk_task

        def flaky_task(chunk_id, chunk, target):
            if chunk_id == 1:
                raise RuntimeError("Worker fail on chunk 1")
            return original_task(chunk_id, chunk, target)

        monkeypatch.setattr(dp, "_align_single_chunk_task", flaky_task)
        with pytest.raises(RuntimeError, match="Worker fail on chunk 1"):
            dask_align_chunks(["ACGT", "AGGT", "CGTA"], "ACGT", batch_size=1)


# =============================================================================
# 7. End-to-End Pipeline Integration Tests
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

    def test_e2e_pipeline_with_custom_processor(self):
        """Verify passing an explicit DaskGenomeProcessor instance works seamlessly."""
        target = "ATGCGTACGT"
        chunk_size = 10

        with DaskGenomeProcessor(processes=False) as custom_proc:
            results = process_fasta_dask(
                MOCK_GENOME_PATH,
                target,
                chunk_size,
                skip_unequal=True,
                processor=custom_proc,
            )
            assert len(results) == 8

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

    def test_e2e_pipeline_timed_with_custom_processor(self):
        target = "ATGCGTACGT"
        chunk_size = 10

        with DaskGenomeProcessor(processes=False) as custom_proc:
            results, timing = process_fasta_dask_timed(
                MOCK_GENOME_PATH,
                target,
                chunk_size,
                skip_unequal=True,
                processor=custom_proc,
            )
            assert len(results) == 8
            assert timing.num_aligned == 8

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


# =============================================================================
# 7. Optimization, Reusable Contexts, and Local Scheduler Parity Tests
# =============================================================================
class TestDaskOptimizationAndSchedulerModes:
    """Tests for performance optimizations, reusable contexts, and scheduler modes."""

    def test_local_scheduler_threads_parity(self):
        """Verify Dask in-process thread scheduler matches CPU baseline exactly."""
        chunks = ["ACGTACGTAC", "ACGTTCGTAC", "TTTTTTTTTT", "ACGTACGTAT"]
        target = "ACGTACGTAC"
        cpu_res = align_chunks(chunks, target)

        with DaskGenomeProcessor(scheduler="threads") as proc:
            dask_res = proc.align_chunks(chunks, target, batch_size=2)
            assert dask_res == cpu_res

    def test_dask_helpers_reusable_processor_context(self):
        """Verify dask_align_* functional helpers accept and reuse a persistent processor."""
        chunks = ["ACGT", "AGGT", "TCGT"]
        target = "ACGT"
        cpu_chunks = align_chunks(chunks, target)
        cpu_pairs = align_chunk_pairs(chunks, chunks)
        pairs = list(zip(chunks, chunks))
        cpu_seqs = align_sequences(pairs)

        with DaskGenomeProcessor(scheduler="threads") as proc:
            res_chunks = dask_align_chunks(chunks, target, processor=proc)
            res_pairs = dask_align_chunk_pairs(chunks, chunks, processor=proc)
            res_seqs = dask_align_sequences(pairs, processor=proc)

            assert res_chunks == cpu_chunks
            assert res_pairs == cpu_pairs
            assert res_seqs == cpu_seqs

    def test_contiguous_vs_filtered_chunk_ordering(self):
        """Verify chunk ordering is preserved for both contiguous and filtered paths."""
        chunks_mixed = ["ACGT", "ACG", "AGGT", "A", "TCGT", "ACGTA"]
        target = "ACGT"

        with DaskGenomeProcessor(scheduler="threads") as proc:
            # Filtered path
            res_filtered = proc.align_chunks(chunks_mixed, target, skip_unequal=True, batch_size=2)
            assert len(res_filtered) == 3
            assert res_filtered == align_chunks(chunks_mixed, target, skip_unequal=True)

            # Contiguous path (all equal length)
            chunks_equal = [c for c in chunks_mixed if len(c) == len(target)]
            res_contiguous = proc.align_chunks(chunks_equal, target, skip_unequal=False, batch_size=2)
            assert len(res_contiguous) == 3
            assert res_contiguous == align_chunks(chunks_equal, target)
            assert res_filtered == res_contiguous

    def test_batching_equivalence_across_granularities(self):
        """Verify results are strictly identical across varying batch granularities."""
        chunks = ["ACGT" * 5 for _ in range(60)]
        target = "ACGT" * 5
        cpu_res = align_chunks(chunks, target)

        with DaskGenomeProcessor(scheduler="threads") as proc:
            for b in [1, 3, 10, 15, 30, 60]:
                dask_res = proc.align_chunks(chunks, target, batch_size=b)
                assert dask_res == cpu_res


# =============================================================================
# 8. Batch Size Auto-Tuning and Worker Backend Tests
# =============================================================================
class TestDaskBatchAutoTuningAndWorkerBackend:
    """Tests for dynamic batch auto-tuning, worker backend selection, and fallback."""

    def test_calculate_optimal_batch_size_valid_ranges(self):
        # 0 items
        assert calculate_optimal_batch_size(0) == 1
        # Small inputs (<= target_tasks)
        assert calculate_optimal_batch_size(5, n_workers=4, tasks_per_worker=4) == 1
        assert calculate_optimal_batch_size(16, n_workers=4, tasks_per_worker=4) == 4
        # Medium inputs
        # 100 items with 4 workers * 4 tasks = 16 tasks -> (100 + 16 - 1) // 16 = 7 items/batch
        bs_med = calculate_optimal_batch_size(100, n_workers=4, tasks_per_worker=4)
        assert bs_med == 7
        # Large inputs
        # 10,000 items with 8 workers * 4 tasks = 32 tasks -> (10000 + 32 - 1) // 32 = 313
        bs_large = calculate_optimal_batch_size(10000, n_workers=8, tasks_per_worker=4)
        assert bs_large == 313

    def test_calculate_optimal_batch_size_input_validation(self):
        with pytest.raises(ValueError, match="num_items must be a non-negative integer"):
            calculate_optimal_batch_size(-5)
        with pytest.raises(ValueError, match="num_items must be a non-negative integer"):
            calculate_optimal_batch_size(True)  # bool is subclass of int in Python
        with pytest.raises(ValueError, match="num_items must be a non-negative integer"):
            calculate_optimal_batch_size("100")  # type: ignore

        # n_workers fallback to 4 when None or non-positive
        bs1 = calculate_optimal_batch_size(100, n_workers=None)
        bs2 = calculate_optimal_batch_size(100, n_workers=0)
        assert bs1 == bs2 == 7

    def test_processor_auto_batch_size_execution(self):
        """Verify processor and functional APIs execute cleanly with batch_size='auto'."""
        chunks = ["ACGTACGTAC", "ACGTTCGTAC", "TTTTTTTTTT", "ACGTACGTAT"] * 10
        target = "ACGTACGTAC"
        cpu_res = align_chunks(chunks, target)

        with DaskGenomeProcessor(scheduler="threads", default_batch_size="auto") as proc:
            res_chunks = proc.align_chunks(chunks, target)
            assert res_chunks == cpu_res

            pairs = list(zip(chunks, chunks))
            res_pairs = proc.align_sequences(pairs, batch_size="auto")
            assert len(res_pairs) == len(pairs)

            res_chunk_pairs = proc.align_chunk_pairs(chunks, chunks, batch_size="auto")
            assert len(res_chunk_pairs) == len(chunks)

    def test_processor_timed_pipeline_auto_batch(self):
        """Verify process_fasta_dask_timed computes num_tasks properly with batch_size='auto'."""
        target = "ATGCGTACGT"
        with DaskGenomeProcessor(scheduler="threads", default_batch_size="auto") as proc:
            results, timing = process_fasta_dask_timed(
                MOCK_GENOME_PATH,
                target,
                chunk_size=10,
                batch_size="auto",
                processor=proc,
            )
            assert len(results) == 8
            assert timing.num_aligned == 8
            assert timing.num_tasks >= 1

    def test_processor_worker_backend_properties_and_fallback(self):
        """Verify worker_backend configuration and fallback to CPU when CUDA is absent."""
        with DaskGenomeProcessor(scheduler="threads", worker_backend="auto") as proc:
            assert proc.worker_backend == "auto"
            chunks = ["ACGTACGT", "ACGTTCGT"]
            target = "ACGTACGT"
            res = proc.align_chunks(chunks, target)
            assert len(res) == 2
            assert res[0].matches == 8

        with DaskGenomeProcessor(scheduler="threads", worker_backend="cuda") as proc:
            assert proc.worker_backend == "cuda"
            res = proc.align_chunks(chunks, target)
            assert len(res) == 2
            assert res[0].matches == 8

    def test_invalid_batch_size_raises(self):
        """Verify invalid batch_size values raise ValueError."""
        with pytest.raises(ValueError, match="default_batch_size must be a positive integer or 'auto'"):
            DaskGenomeProcessor(scheduler="threads", default_batch_size=0)

        with pytest.raises(ValueError, match="default_batch_size must be a positive integer or 'auto'"):
            DaskGenomeProcessor(scheduler="threads", default_batch_size="invalid")

        with DaskGenomeProcessor(scheduler="threads") as proc:
            with pytest.raises(ValueError, match="batch_size must be a positive integer or 'auto'"):
                proc.align_chunks(["ACGT"], "ACGT", batch_size=-1)



