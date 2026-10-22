from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
from typing import Any, Sequence

# Ensure repo root and src/ are in sys.path (appended so external packages take precedence)
_SRC_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SRC_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))

from alignment.cpu_alignment import (
    AlignmentResult,
    align_sequence,
    align_sequences,
    align_chunks,
    align_chunk_pairs,
    _validate_sequence,
    VALID_BASES,
)
from parser.fasta_parser import parse_fasta
from chunking.genome_chunker import chunk_sequence

# ── Dask import and availability check ───────────────────────────────────────
# Guard against src/distributed shadowing the external pip 'distributed' package
_DASK_AVAILABLE = False
_DASK_ERROR: str | None = None
dask = None  # type: ignore[assignment]
delayed = None  # type: ignore[assignment]
distributed = None  # type: ignore[assignment]
Client = None  # type: ignore[assignment]
LocalCluster = None  # type: ignore[assignment]

try:
    import site
    import importlib.machinery
    import importlib.util

    if "distributed" not in sys.modules or (
        getattr(sys.modules["distributed"], "__file__", "")
        and str(_SRC_DIR) in str(getattr(sys.modules["distributed"], "__file__", ""))
    ):
        for _sp in site.getsitepackages():
            _spec = importlib.machinery.PathFinder.find_spec(
                "distributed", path=[_sp]
            )
            if _spec and _spec.origin and str(_SRC_DIR) not in str(_spec.origin):
                _mod = importlib.util.module_from_spec(_spec)
                sys.modules["distributed"] = _mod
                _spec.loader.exec_module(_mod)
                break

    import dask
    from dask import delayed
    import distributed
    from distributed import Client, LocalCluster

    _DASK_AVAILABLE = True
    _DASK_ERROR = None
except Exception as _import_err:  # pragma: no cover
    _DASK_AVAILABLE = False
    _DASK_ERROR = str(_import_err)


# ── Custom Exceptions ────────────────────────────────────────────────────────
class DaskNotAvailableError(RuntimeError):
    """Raised when Dask operations are requested but Dask is not installed or available."""


class DaskTaskError(RuntimeError):
    """Raised when a Dask distributed task fails during execution."""


# ── Diagnostics ──────────────────────────────────────────────────────────────
def is_dask_available() -> bool:
    """Return True if Dask and distributed are installed and importable."""
    return _DASK_AVAILABLE


def get_dask_info() -> dict[str, Any]:
    """Return diagnostic information about the Dask environment.

    Returns
    -------
    dict[str, Any]
        Dictionary containing Dask availability, versions, and current client state.
    """
    if not _DASK_AVAILABLE:
        return {
            "available": False,
            "error": _DASK_ERROR,
            "dask_version": None,
            "distributed_version": None,
            "active_client": False,
        }

    active_client = False
    client_info = None
    try:
        c = distributed.get_client()
        active_client = True
        client_info = str(c)
    except (ValueError, AttributeError):
        pass

    return {
        "available": True,
        "error": None,
        "dask_version": dask.__version__,
        "distributed_version": distributed.__version__,
        "active_client": active_client,
        "client_info": client_info,
    }


# ── Top-level Task Callables (Picklable for worker threads / processes) ──────
def _align_chunk_fast(chunk: str, target_norm: str, target_len: int) -> AlignmentResult:
    
    if not isinstance(chunk, str):
        raise TypeError(f"reference must be a string, got {type(chunk).__name__}")

    chunk_upper = chunk.upper()
    invalid_chars = set(chunk_upper) - VALID_BASES
    if invalid_chars and chunk_upper != "":
        raise ValueError(
            f"reference contains invalid characters: "
            f"{', '.join(sorted(invalid_chars))}. "
            f"Only A, C, G, T are allowed."
        )

    if len(chunk_upper) != target_len:
        raise ValueError(
            f"Sequences must have equal length: "
            f"reference has {len(chunk_upper)}, query has {target_len}."
        )

    if target_len == 0:
        return AlignmentResult(
            matches=0,
            mismatches=0,
            alignment_length=0,
            score=0.0,
            mismatch_positions=[],
        )

    matches = 0
    mismatches = 0
    mismatch_positions: list[int] = []

    for i in range(target_len):
        if chunk_upper[i] == target_norm[i]:
            matches += 1
        else:
            mismatches += 1
            mismatch_positions.append(i)

    score = matches / target_len
    return AlignmentResult(
        matches=matches,
        mismatches=mismatches,
        alignment_length=target_len,
        score=score,
        mismatch_positions=mismatch_positions,
    )


def _align_single_chunk_task(
    chunk_id: int,
    chunk: str,
    target: str,
) -> tuple[int, AlignmentResult]:
    
    result = _align_chunk_fast(chunk, target, len(target))
    return (chunk_id, result)


def calculate_optimal_batch_size(
    num_items: int,
    n_workers: int | None = None,
    tasks_per_worker: int = 4,
) -> int:
    
    if not isinstance(num_items, int) or isinstance(num_items, bool) or num_items < 0:
        raise ValueError(f"num_items must be a non-negative integer, got {num_items!r}")

    if num_items == 0:
        return 1

    workers = n_workers if (isinstance(n_workers, int) and n_workers > 0) else 4
    target_tasks = max(1, workers * tasks_per_worker)

    if num_items <= target_tasks:
        return max(1, num_items // workers) or 1

    return max(1, (num_items + target_tasks - 1) // target_tasks)


_GPU_TASK_LOCK = threading.Lock()


def _align_chunk_batch_task(
    start_idx: int,
    chunk_indices: list[int] | None,
    chunks: list[str],
    target: str,
    skip_unequal: bool = False,
    worker_backend: str = "cpu",
) -> list[tuple[int, AlignmentResult]]:
    
    target_len = len(target)
    if worker_backend in ("cuda", "auto") and len(chunks) > 0:
        try:
            from alignment.cuda_alignment import is_cuda_available, cuda_align_chunks
            if is_cuda_available():
                with _GPU_TASK_LOCK:
                    cuda_res = cuda_align_chunks(chunks, target, skip_unequal=skip_unequal)
                if chunk_indices is None:
                    return [(start_idx + i, r) for i, r in enumerate(cuda_res)]
                else:
                    return [(idx, r) for idx, r in zip(chunk_indices, cuda_res)]
        except Exception:
            pass  # Fall back cleanly to CPU alignment

    indexed_results: list[tuple[int, AlignmentResult]] = []
    if chunk_indices is None:
        for i, chk in enumerate(chunks):
            if skip_unequal and len(chk) != target_len:
                continue
            res = _align_chunk_fast(chk, target, target_len)
            indexed_results.append((start_idx + i, res))
    else:
        for idx, chk in zip(chunk_indices, chunks):
            if skip_unequal and len(chk) != target_len:
                continue
            res = _align_chunk_fast(chk, target, target_len)
            indexed_results.append((idx, res))
    return indexed_results


def _align_single_pair_task(
    pair_id: int,
    reference: str,
    query: str,
) -> tuple[int, AlignmentResult]:
    """Execute CPU alignment for a single reference-query pair."""
    res = align_sequence(reference, query)
    return (pair_id, res)


def _align_pair_batch_task(
    start_idx: int,
    pair_indices: list[int] | None,
    pairs: list[tuple[str, str]],
) -> list[tuple[int, AlignmentResult]]:
    """Execute CPU alignment for a batch of sequence pairs."""
    results: list[tuple[int, AlignmentResult]] = []
    if pair_indices is None:
        for i, (ref, qry) in enumerate(pairs):
            res = align_sequence(ref, qry)
            results.append((start_idx + i, res))
    else:
        for idx, (ref, qry) in zip(pair_indices, pairs):
            res = align_sequence(ref, qry)
            results.append((idx, res))
    return results


# ── Pipeline Timing Container ────────────────────────────────────────────────
@dataclass(frozen=True)
class DaskPipelineTiming:
    """Detailed timing metrics for the Dask distributed pipeline."""

    parse_time: float
    chunk_time: float
    dask_align_time: float
    total_pipeline_time: float
    num_sequences: int
    num_chunks: int
    num_aligned: int
    num_tasks: int


# ── Main Distributed Processor Class ─────────────────────────────────────────
class DaskGenomeProcessor:
    def __init__(
        self,
        client: Any | None = None,
        *,
        n_workers: int | None = None,
        threads_per_worker: int | None = None,
        processes: bool = False,
        default_batch_size: int | str = 1,
        scheduler: str | None = None,
        auto_start_cluster: bool = True,
        worker_backend: str = "cpu",
    ) -> None:
        self._cluster: Any | None = None
        self._client: Any | None = None
        self._owns_client: bool = False
        self._owns_cluster: bool = False
        self._is_closed: bool = False
        self._default_batch_size: int | str = 1
        self._scheduler: str | None = None
        self._n_workers: int | None = n_workers
        self._worker_backend: str = worker_backend

        if not is_dask_available():
            self._is_closed = True
            raise DaskNotAvailableError(
                f"Dask is not installed or available in this environment. "
                f"Details: {_DASK_ERROR}"
            )

        if default_batch_size != "auto" and (
            not isinstance(default_batch_size, int)
            or isinstance(default_batch_size, bool)
            or default_batch_size < 1
        ):
            self._is_closed = True
            raise ValueError(
                f"default_batch_size must be a positive integer or 'auto', got {default_batch_size}"
            )

        self._default_batch_size = default_batch_size
        self._scheduler = scheduler

        if client is not None:
            self._client = client
            self._owns_client = False
        elif scheduler is not None:
            # Explicit scheduler string requested (e.g. 'threads' or 'sync')
            self._client = None
            self._owns_client = False
        else:
            # Check for existing global client
            try:
                self._client = distributed.get_client()
                self._owns_client = False
            except (ValueError, AttributeError):
                self._client = None
                self._owns_client = False

            # Auto-start LocalCluster if requested and no client is active
            if self._client is None and auto_start_cluster:
                cluster_kwargs: dict[str, Any] = {"processes": processes}
                if n_workers is not None:
                    cluster_kwargs["n_workers"] = n_workers
                if threads_per_worker is not None:
                    cluster_kwargs["threads_per_worker"] = threads_per_worker

                self._cluster = LocalCluster(**cluster_kwargs)
                self._client = Client(self._cluster)
                self._owns_cluster = True
                self._owns_client = True

    @property
    def worker_backend(self) -> str:
        """Return the target worker alignment backend ('cpu', 'cuda', 'auto')."""
        return self._worker_backend

    @property
    def n_workers(self) -> int | None:
        """Return the configured worker count, if specified."""
        return self._n_workers

    @property
    def client(self) -> Any | None:
        """Return the active Dask Client, if any."""
        return self._client

    @property
    def owns_client(self) -> bool:
        """Return True if this processor owns and will shut down the client."""
        return self._owns_client

    @property
    def owns_cluster(self) -> bool:
        """Return True if this processor owns and will shut down the cluster."""
        return self._owns_cluster

    @property
    def is_closed(self) -> bool:
        """Return True if the processor has been closed."""
        return self._is_closed

    def close(self) -> None:
        """Clean up Dask resources owned by this processor.

        Shared clients not owned by this instance are preserved untouched.
        """
        if self._is_closed:
            return

        if self._owns_client and self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover
                pass
            self._client = None

        if self._owns_cluster and self._cluster is not None:
            try:
                self._cluster.close()
            except Exception:  # pragma: no cover
                pass
            self._cluster = None

        self._is_closed = True

    def __enter__(self) -> DaskGenomeProcessor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if not getattr(self, "_is_closed", True) and (
            getattr(self, "_owns_client", False)
            or getattr(self, "_owns_cluster", False)
        ):
            self.close()

    def _execute_tasks(
        self,
        tasks: list[Any],
    ) -> list[Any]:
        if self._is_closed:
            raise RuntimeError("Cannot execute tasks on a closed DaskGenomeProcessor.")

        if not tasks:
            return []

        if self._client is not None:
            try:
                # When client is active, compute tasks using client
                futures = self._client.compute(tasks)
                return self._client.gather(futures)
            except (ValueError, TypeError, FileNotFoundError):
                raise
            except DaskTaskError:
                raise
            except Exception as exc:
                raise DaskTaskError(f"Dask distributed task execution failed: {exc}") from exc
        else:
            sched = self._scheduler or "threads"
            try:
                return list(dask.compute(*tasks, scheduler=sched))
            except (ValueError, TypeError, FileNotFoundError):
                raise
            except DaskTaskError:
                raise
            except Exception as exc:
                raise DaskTaskError(f"Dask task execution failed: {exc}") from exc

    def _resolve_batch_size(
        self,
        num_items: int,
        batch_size: int | str | None,
    ) -> int:
        eff = batch_size if batch_size is not None else self._default_batch_size
        if eff == "auto":
            return calculate_optimal_batch_size(num_items, self._n_workers)
        if not isinstance(eff, int) or isinstance(eff, bool) or eff < 1:
            raise ValueError(
                f"batch_size must be a positive integer or 'auto', got {eff}"
            )
        return eff

    def align_chunks(
        self,
        chunks: list[str],
        target: str,
        *,
        skip_unequal: bool = False,
        batch_size: int | str | None = None,
    ) -> list[AlignmentResult]:
        if self._is_closed:
            raise RuntimeError("Cannot perform operations on a closed DaskGenomeProcessor.")

        if not isinstance(chunks, list):
            raise TypeError(f"chunks must be a list, got {type(chunks).__name__}")
        if not isinstance(target, str):
            raise TypeError(f"target must be a string, got {type(target).__name__}")

        # Upfront validation of target
        target_norm = _validate_sequence(target, "target")
        target_len = len(target_norm)

        # Validate chunk element types
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, str):
                raise TypeError(
                    f"Each chunk must be a string, got {type(chunk).__name__} at index {idx}"
                )

        # Empty chunks handling: support small inputs without unnecessary complexity
        if not chunks:
            return []

        eff_batch_size = self._resolve_batch_size(len(chunks), batch_size)

        # Prepare eligible chunks while tracking their original indices
        valid_items: list[tuple[int, str]] | None = None
        if skip_unequal:
            if any(len(chk) != target_len for chk in chunks):
                valid_items = [
                    (i, chk) for i, chk in enumerate(chunks) if len(chk) == target_len
                ]
                if not valid_items:
                    return []

        # Construct Dask tasks
        task_list: list[Any] = []
        if valid_items is None:
            # Contiguous chunk slices - avoids intermediate index list allocations
            if eff_batch_size == 1:
                for i, chk in enumerate(chunks):
                    task_list.append(
                        delayed(_align_single_chunk_task)(i, chk, target_norm)
                    )
            else:
                for b_start in range(0, len(chunks), eff_batch_size):
                    batch_chunks = chunks[b_start : b_start + eff_batch_size]
                    task_list.append(
                        delayed(_align_chunk_batch_task)(
                            b_start,
                            None,
                            batch_chunks,
                            target_norm,
                            skip_unequal=False,
                            worker_backend=self._worker_backend,
                        )
                    )
        else:
            # Filtered chunks with explicit index tracking
            if eff_batch_size == 1:
                for orig_idx, chk in valid_items:
                    task_list.append(
                        delayed(_align_single_chunk_task)(orig_idx, chk, target_norm)
                    )
            else:
                for b_start in range(0, len(valid_items), eff_batch_size):
                    batch_slice = valid_items[b_start : b_start + eff_batch_size]
                    batch_indices = [item[0] for item in batch_slice]
                    batch_chunks = [item[1] for item in batch_slice]
                    task_list.append(
                        delayed(_align_chunk_batch_task)(
                            b_start,
                            batch_indices,
                            batch_chunks,
                            target_norm,
                            skip_unequal=skip_unequal,
                            worker_backend=self._worker_backend,
                        )
                    )

        # Execute tasks across Dask workers
        raw_results = self._execute_tasks(task_list)

        # Flatten and sort results strictly by original chunk index
        indexed_results: list[tuple[int, AlignmentResult]] = []
        if eff_batch_size == 1:
            indexed_results = list(raw_results)
        else:
            for batch_res in raw_results:
                indexed_results.extend(batch_res)

        expected_count = len(valid_items) if valid_items is not None else len(chunks)
        if len(indexed_results) != expected_count:
            raise DaskTaskError(
                f"Incomplete Dask task execution: expected {expected_count} results, "
                f"but received {len(indexed_results)}. Some worker tasks failed or dropped results."
            )

        indexed_results.sort(key=lambda item: item[0])
        return [res for _, res in indexed_results]

    def align_chunk_pairs(
        self,
        reference_chunks: list[str],
        query_chunks: list[str],
        *,
        batch_size: int | str | None = None,
    ) -> list[AlignmentResult]:
        """Align corresponding pairs of chunks across two lists using Dask tasks."""
        if self._is_closed:
            raise RuntimeError("Cannot perform operations on a closed DaskGenomeProcessor.")

        if not isinstance(reference_chunks, list):
            raise TypeError(
                f"reference_chunks must be a list, got {type(reference_chunks).__name__}"
            )
        if not isinstance(query_chunks, list):
            raise TypeError(
                f"query_chunks must be a list, got {type(query_chunks).__name__}"
            )
        if len(reference_chunks) != len(query_chunks):
            raise ValueError(
                f"Chunk lists must have equal length: "
                f"reference has {len(reference_chunks)}, query has {len(query_chunks)}."
            )

        if not reference_chunks:
            return []

        for idx, (ref, qry) in enumerate(zip(reference_chunks, query_chunks)):
            if not isinstance(ref, str):
                raise TypeError(
                    f"Each reference chunk must be a string, got {type(ref).__name__} at index {idx}"
                )
            if not isinstance(qry, str):
                raise TypeError(
                    f"Each query chunk must be a string, got {type(qry).__name__} at index {idx}"
                )

        eff_batch_size = self._resolve_batch_size(len(reference_chunks), batch_size)

        task_list: list[Any] = []
        n_pairs = len(reference_chunks)

        if eff_batch_size == 1:
            for i in range(n_pairs):
                task_list.append(
                    delayed(_align_single_pair_task)(
                        i, reference_chunks[i], query_chunks[i]
                    )
                )
        else:
            for b_start in range(0, n_pairs, eff_batch_size):
                b_ref = reference_chunks[b_start : b_start + eff_batch_size]
                b_qry = query_chunks[b_start : b_start + eff_batch_size]
                batch_pairs = list(zip(b_ref, b_qry))
                task_list.append(
                    delayed(_align_pair_batch_task)(b_start, None, batch_pairs)
                )

        raw_results = self._execute_tasks(task_list)

        indexed_results: list[tuple[int, AlignmentResult]] = []
        if eff_batch_size == 1:
            indexed_results = list(raw_results)
        else:
            for batch_res in raw_results:
                indexed_results.extend(batch_res)

        if len(indexed_results) != n_pairs:
            raise DaskTaskError(
                f"Incomplete Dask task execution: expected {n_pairs} results, "
                f"but received {len(indexed_results)}. Some worker tasks failed or dropped results."
            )

        indexed_results.sort(key=lambda item: item[0])
        return [res for _, res in indexed_results]

    def align_sequences(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int | str | None = None,
    ) -> list[AlignmentResult]:
        """Align a batch of DNA sequence pairs using Dask tasks."""
        if self._is_closed:
            raise RuntimeError("Cannot perform operations on a closed DaskGenomeProcessor.")

        if not isinstance(pairs, list):
            raise TypeError(f"pairs must be a list, got {type(pairs).__name__}")
        if not pairs:
            return []

        for idx, item in enumerate(pairs):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError(
                    f"Each pair must be a 2-tuple (reference, query), got {item!r} at index {idx}"
                )
            if not isinstance(item[0], str) or not isinstance(item[1], str):
                raise TypeError(
                    f"Pair elements must be strings, got ({type(item[0]).__name__}, {type(item[1]).__name__}) at index {idx}"
                )

        eff_batch_size = self._resolve_batch_size(len(pairs), batch_size)

        task_list: list[Any] = []
        n_pairs = len(pairs)

        if eff_batch_size == 1:
            for i in range(n_pairs):
                ref, qry = pairs[i]
                task_list.append(delayed(_align_single_pair_task)(i, ref, qry))
        else:
            for b_start in range(0, n_pairs, eff_batch_size):
                b_pairs = pairs[b_start : b_start + eff_batch_size]
                task_list.append(
                    delayed(_align_pair_batch_task)(b_start, None, b_pairs)
                )

        raw_results = self._execute_tasks(task_list)

        indexed_results: list[tuple[int, AlignmentResult]] = []
        if eff_batch_size == 1:
            indexed_results = list(raw_results)
        else:
            for batch_res in raw_results:
                indexed_results.extend(batch_res)

        if len(indexed_results) != n_pairs:
            raise DaskTaskError(
                f"Incomplete Dask task execution: expected {n_pairs} results, "
                f"but received {len(indexed_results)}. Some worker tasks failed or dropped results."
            )

        indexed_results.sort(key=lambda item: item[0])
        return [res for _, res in indexed_results]


# ── Functional API Helpers ───────────────────────────────────────────────────
def dask_align_chunks(
    chunks: list[str],
    target: str,
    *,
    skip_unequal: bool = False,
    client: Any | None = None,
    batch_size: int | str = 1,
    scheduler: str | None = None,
    processor: DaskGenomeProcessor | None = None,
    worker_backend: str = "cpu",
) -> list[AlignmentResult]:
    if processor is not None:
        return processor.align_chunks(
            chunks, target, skip_unequal=skip_unequal, batch_size=batch_size
        )
    with DaskGenomeProcessor(
        client=client,
        default_batch_size=batch_size,
        scheduler=scheduler,
        worker_backend=worker_backend,
    ) as proc:
        return proc.align_chunks(
            chunks, target, skip_unequal=skip_unequal, batch_size=batch_size
        )


def dask_align_chunk_pairs(
    reference_chunks: list[str],
    query_chunks: list[str],
    *,
    client: Any | None = None,
    batch_size: int | str = 1,
    scheduler: str | None = None,
    processor: DaskGenomeProcessor | None = None,
    worker_backend: str = "cpu",
) -> list[AlignmentResult]:
    """Align corresponding pairs of chunks using Dask distributed execution."""
    if processor is not None:
        return processor.align_chunk_pairs(
            reference_chunks, query_chunks, batch_size=batch_size
        )
    with DaskGenomeProcessor(
        client=client,
        default_batch_size=batch_size,
        scheduler=scheduler,
        worker_backend=worker_backend,
    ) as proc:
        return proc.align_chunk_pairs(
            reference_chunks, query_chunks, batch_size=batch_size
        )


def dask_align_sequences(
    pairs: list[tuple[str, str]],
    *,
    client: Any | None = None,
    batch_size: int | str = 1,
    scheduler: str | None = None,
    processor: DaskGenomeProcessor | None = None,
    worker_backend: str = "cpu",
) -> list[AlignmentResult]:
    """Align a batch of DNA sequence pairs using Dask distributed execution."""
    if processor is not None:
        return processor.align_sequences(pairs, batch_size=batch_size)
    with DaskGenomeProcessor(
        client=client,
        default_batch_size=batch_size,
        scheduler=scheduler,
        worker_backend=worker_backend,
    ) as proc:
        return proc.align_sequences(pairs, batch_size=batch_size)


# ── Pipeline Integration Functions (FASTA -> Parser -> Chunker -> Dask) ──────
def process_fasta_dask(
    fasta_path: str | Path,
    target: str,
    chunk_size: int,
    *,
    skip_unequal: bool = True,
    batch_size: int | str = 1,
    client: Any | None = None,
    scheduler: str | None = None,
    processor: DaskGenomeProcessor | None = None,
    worker_backend: str = "cpu",
) -> list[AlignmentResult]:

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError(f"chunk_size must be an int, got {type(chunk_size).__name__}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if not isinstance(target, str):
        raise TypeError(f"target must be a string, got {type(target).__name__}")
    _validate_sequence(target, "target")

    sequences = parse_fasta(fasta_path)
    all_chunks: list[str] = []
    for seq in sequences:
        chunks = chunk_sequence(seq, chunk_size)
        all_chunks.extend(chunks)

    if processor is not None:
        return processor.align_chunks(
            all_chunks, target, skip_unequal=skip_unequal, batch_size=batch_size
        )

    with DaskGenomeProcessor(
        client=client,
        scheduler=scheduler,
        default_batch_size=batch_size,
        worker_backend=worker_backend,
    ) as proc:
        return proc.align_chunks(
            all_chunks, target, skip_unequal=skip_unequal, batch_size=batch_size
        )


def process_fasta_dask_timed(
    fasta_path: str | Path,
    target: str,
    chunk_size: int,
    *,
    skip_unequal: bool = True,
    batch_size: int | str = 1,
    client: Any | None = None,
    scheduler: str | None = None,
    processor: DaskGenomeProcessor | None = None,
    worker_backend: str = "cpu",
) -> tuple[list[AlignmentResult], DaskPipelineTiming]:
    """Execute end-to-end Dask pipeline and return granular timing breakdown."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError(f"chunk_size must be an int, got {type(chunk_size).__name__}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if not isinstance(target, str):
        raise TypeError(f"target must be a string, got {type(target).__name__}")
    _validate_sequence(target, "target")

    t_wall_start = time.perf_counter()

    # Stage 1: Parsing
    t_parse_start = time.perf_counter()
    sequences = parse_fasta(fasta_path)
    t_parse = time.perf_counter() - t_parse_start

    # Stage 2: Chunking
    t_chunk_start = time.perf_counter()
    all_chunks: list[str] = []
    for seq in sequences:
        chunks = chunk_sequence(seq, chunk_size)
        all_chunks.extend(chunks)
    t_chunk = time.perf_counter() - t_chunk_start

    # Stage 3: Dask Distributed Alignment
    t_align_start = time.perf_counter()
    if processor is not None:
        results = processor.align_chunks(
            all_chunks, target, skip_unequal=skip_unequal, batch_size=batch_size
        )
    else:
        with DaskGenomeProcessor(
            client=client,
            scheduler=scheduler,
            default_batch_size=batch_size,
            worker_backend=worker_backend,
        ) as proc:
            results = proc.align_chunks(
                all_chunks, target, skip_unequal=skip_unequal, batch_size=batch_size
            )
    t_align = time.perf_counter() - t_align_start

    t_total = time.perf_counter() - t_wall_start

    num_aligned = len(results)
    if isinstance(batch_size, int) and batch_size > 0:
        num_tasks = (num_aligned + batch_size - 1) // batch_size
    elif batch_size == "auto":
        eff_bs = calculate_optimal_batch_size(
            num_aligned, processor.n_workers if processor else None
        )
        num_tasks = (num_aligned + eff_bs - 1) // eff_bs
    else:
        num_tasks = num_aligned

    timing = DaskPipelineTiming(
        parse_time=t_parse,
        chunk_time=t_chunk,
        dask_align_time=t_align,
        total_pipeline_time=t_total,
        num_sequences=len(sequences),
        num_chunks=len(all_chunks),
        num_aligned=num_aligned,
        num_tasks=num_tasks,
    )

    return results, timing
