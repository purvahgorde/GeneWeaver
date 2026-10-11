# Week 3 -- Distributed Sequence Alignment with Dask & GPU Acceleration

## Overview

Week 3 introduces the distributed processing layer to GeneWeaver using **Dask** task scheduling. Building upon the Week 1 single-threaded CPU reference implementation and the Week 2 Numba CUDA GPU acceleration engine, the Dask distributed layer orchestrates parallel genome chunk alignment across multi-threaded or multi-node worker pools.

In accordance with project architecture and pair-programming guidelines:
- The Week 1 CPU baseline is completely preserved without breaking changes and serves as the exact correctness oracle.
- Existing parser (`src/parser/fasta_parser.py`) and chunker (`src/chunking/genome_chunker.py`) implementations are reused directly without duplication.
- No benchmark results or GPU executions are simulated or fabricated.
- Distributed execution guarantees strict result equivalence and ordering parity against the CPU reference baseline.

---

# Day 1 -- Dask Setup and Distributed Processing

## 1. Environment & Dependency Management

The project's dependency configuration (`requirements.txt`) was updated to declare the minimum required Dask dependencies:

```text
biopython>=1.80
pytest>=8.0.0
numpy>=1.24.0
numba>=0.59.0
dask>=2024.1.0
distributed>=2024.1.0
```

### Host Environment Diagnostics:

| Diagnostic Property | Measured Value / Status |
|---|---|
| OS / Platform | Windows 11 Home Single Language (10.0.26200-SP0) |
| Python Version | 3.13.15 (AMD64) |
| Host CPU | Intel64 Family 6 Model 141 Stepping 1 (GenuineIntel) |
| Dask Installed | Dask 2026.8.0 |
| Distributed Installed | Distributed 2026.8.0 |
| Scheduler Mode | LocalCluster (processes=False / thread pool) & Dask Task Graphs |

The local scheduler execution mode was verified and selected to enable self-contained developer machine testing without requiring an external multi-node cluster.

---

## 2. Distributed Architecture & Task Design

The distributed processing engine is implemented in `src/distributed/dask_processor.py` (with package re-exports in `src/distributed/__init__.py`).

```
[Candidate DNA Chunks: list[str]] + [Target Query: str]
                         │
                         ▼
             [Input Validation & Chunk Indexing]
            (Preserves 0-based chunk indices: (i, chunk))
                         │
                         ▼
        ┌─────────────────────────────────────────┐
        │       Dask Task Decomposition           │
        │  (eff_batch_size == 1: 1 task / chunk)  │
        │  (eff_batch_size > 1: batched slices)   │
        └─────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
  Dask Task 0      Dask Task 1      Dask Task N
  (align_sequence) (align_sequence) (align_sequence)
        │                │                │
        └────────────────┼────────────────┘
                         ▼
       [Dask Task Collection & Index Sorting]
     (Guarantees 100% deterministic original order)
                         │
                         ▼
        [list[AlignmentResult]] (Matches CPU Oracle)
```

### Task Input & Output Contract:

1. **Minimal Serialization**:
   - Each distributed task receives strictly the minimal data it requires: `chunk_id: int`, `chunk: str`, and `target: str` (or slice indices and string lists for batched tasks).
   - No GPU contexts, file handles, or heavyweight cluster objects are passed to worker tasks.
2. **Deterministic Task Callables**:
   - `_align_single_chunk_task(chunk_id, chunk, target) -> tuple[int, AlignmentResult]`
   - `_align_chunk_batch_task(start_idx, chunk_indices, chunks, target, skip_unequal) -> list[tuple[int, AlignmentResult]]`
   - `_align_single_pair_task(pair_id, ref, qry) -> tuple[int, AlignmentResult]`
   - `_align_pair_batch_task(start_idx, pair_indices, pairs) -> list[tuple[int, AlignmentResult]]`
3. **Algorithm Reuse**:
   - Tasks call `align_sequence()` and `align_chunks()` from `src/alignment/cpu_alignment.py` directly. Zero duplicate alignment logic was introduced.
4. **Ordering Invariant**:
   - Tasks tag results with original chunk indices `(chunk_id, result)`. Upon collection, results are sorted by `chunk_id` before unpacking, strictly guaranteeing that returned results match the input chunk list sequence regardless of worker retirement order.

---

## 3. Resource Management & Lifecycle Safety

The `DaskGenomeProcessor` class manages cluster and client lifecycles cleanly:

- **Ownership Tracking**:
  - `owns_client = False`: When a caller provides an existing `distributed.Client` or an active global client is discovered, the processor reuses it and will **never** close it upon processor shutdown.
  - `owns_client = True`: When the processor creates its own internal `LocalCluster`, it shuts it down cleanly upon `close()` or context manager exit (`__exit__`).
- **Context Manager Support**:
  ```python
  with DaskGenomeProcessor(processes=False) as processor:
      results = processor.align_chunks(chunks, target)
  ```
- **Safe Del / Cleanup**: Attribute access is guarded to prevent unhandled unraisable exceptions during garbage collection or initialization failures.

---

## 4. Error & Failure Handling

- **Dask Unavailable**: Raises `DaskNotAvailableError` with actionable diagnostic information.
- **Empty Inputs**: Safely returns `[]` immediately without scheduling empty tasks or spinning up unnecessary cluster resources.
- **Invalid Types / Non-canonical Bases**: Validated upfront and by CPU alignment routines, raising `TypeError` or `ValueError` cleanly without swallowing exceptions.
- **Task Failures**: Worker task exceptions are propagated directly to the caller, preventing silent data corruption or masked pipeline failures.

---

## 5. Day 1 Test Suite

33 comprehensive unit and lifecycle tests were added in `tests/test_dask_processor.py`:

- Single chunk alignment.
- Multiple chunks alignment with CPU parity.
- Empty input handling.
- Input order preservation across varying mismatch signatures.
- Varying batch sizes (`batch_size=1`, `2`, `4`, `10`).
- Boundary tail chunks filtering with `skip_unequal=True` and validation with `skip_unequal=False`.
- Chunk pairs (`dask_align_chunk_pairs`) and sequence batches (`dask_align_sequences`).
- Type errors, invalid DNA bases, negative batch sizes, and worker crash propagation.
- Shared client preservation vs owned client cleanup.
- Repeated processor invocation without resource leaks.

---

# Day 2 -- Pipeline Integration and Distributed Regression Testing

## 6. End-to-End Pipeline Integration

The Dask distributed layer was integrated with Member 2's FASTA parser and genome chunker:

```
Mock FASTA (data/mock_genome.fasta)
              │
              ▼  parse_fasta(file_path) [BioPython SeqIO]
Genomic Sequences (2 chromosomes: 45 bp + 41 bp)
              │
              ▼  chunk_sequence(sequence, chunk_size=10)
DNA Chunks (10 total chunks: 8 of 10 bp, 2 residual tail chunks)
              │
              ▼  process_fasta_dask(fasta_path, target="ATGCGTACGT", chunk_size=10, skip_unequal=True)
Dask Tasks Distributed Execution (8 tasks across local cluster)
              │
              ▼  align_sequence(chunk, target) [CPU Reference Oracle]
list[AlignmentResult] (8 results matching CPU baseline 100%)
```

### Pipeline Functions Implemented:

1. `process_fasta_dask(fasta_path, target, chunk_size, skip_unequal=True, batch_size=1, processor=None) -> list[AlignmentResult]`:
   Direct end-to-end distributed execution.
2. `process_fasta_dask_timed(fasta_path, target, chunk_size, skip_unequal=True, batch_size=1, processor=None) -> tuple[list[AlignmentResult], DaskPipelineTiming]`:
   Returns alignment results accompanied by a frozen `DaskPipelineTiming` dataclass:
   - `parse_time`: FASTA file reading and parsing duration.
   - `chunk_time`: Sequence partitioning duration.
   - `dask_align_time`: Total Dask task scheduling, execution, and result collection duration.
   - `total_pipeline_time`: Complete wall-clock duration.
   - Workload metadata: `num_sequences`, `num_chunks`, `num_aligned`, `num_tasks`.

---

## 7. Dask Benchmark & Empirical Measurements

A dedicated multi-round benchmark script was created at `benchmarks/dask_benchmark.py`.

### Methodology:
- **High-Resolution Timing**: `time.perf_counter()`.
- **RNG Seed**: Fixed seed 42 with 10% point mutation rate.
- **Cold-Start Isolation**: A dedicated warm-up alignment (10 chunks of 50 bp) is executed prior to timing collection to separate cluster initialization from steady-state measurements.
- **Repetitions**: 5 independent measurement rounds per configuration reporting Min (peak throughput) and Avg (mean).
- **Correctness Pre-check**: Automated verification ensuring 100% equivalence (`cpu_results == dask_results`) across all chunks before timings are recorded.

### Actual Measured Timings (Windows 11 / Python 3.13 / Intel64):

| Configuration | Chunk Size | Chunks | Total Bases | Sequential CPU (Min) | Sequential CPU (Avg) | Dask Distributed (Min) | Dask Distributed (Avg) | Measured Speedup | Equivalence |
|---|---|---|---|---|---|---|---|---|---|
| **Small** | 100 bp | 500 | 50,000 | **14.14 ms** | 15.81 ms | **51.37 ms** | 73.83 ms | **0.28x** | PASS (100%) |
| **Medium** | 1,000 bp | 500 | 500,000 | **130.77 ms** | 155.85 ms | **178.65 ms** | 209.31 ms | **0.73x** | PASS (100%) |
| **Large** | 5,000 bp | 200 | 1,000,000 | **282.14 ms** | 310.54 ms | **331.99 ms** | 363.84 ms | **0.85x** | PASS (100%) |

#### Cluster Overhead Metrics:
- **Dask LocalCluster startup time**: **231.33 ms** (one-time initialization)
- **Warm-up alignment latency**: **54.40 ms** (initial graph scheduling)

### Architectural Findings & Overhead Analysis:

1. **Task Granularity vs Scheduling Latency (Amdahl's Law in Distributed Systems)**:
   - For short workloads (**Small**, 50k bases), CPU execution completes in only **14.14 ms**. The Dask framework incurs fixed scheduling, graph construction, and thread dispatch latency (~50 ms), resulting in a speedup of **0.28x**.
   - As workload size increases (**Medium** to **Large**, up to 1M bases), CPU compute duration rises from 14 ms to 282 ms. Task scheduling latency becomes amortized, and Dask speedup increases from **0.28x to 0.85x**.
2. **CPython 3.13 GIL & Thread Worker Scaling**:
   - In pure Python string comparison, thread workers on a local cluster contend on the CPython Global Interpreter Lock (GIL). Task batching (`batch_size > 1`) significantly reduces serialization and task queue overhead.
   - For true linear multi-core speedup on CPU, process-based workers (`processes=True`) or compiled Numba/C kernels that release the GIL (`nogil=True`) are required.

---

## 8. Full Regression Testing Results

All existing Week 1 and Week 2 test suites were run alongside the new Dask test suite:

```bash
pytest -v
```

### Test Suite Execution Summary:

| Subsystem / Test Module | Test Focus | Tests Collected | Passed | Skipped | Failed |
|---|---|---|---|---|---|
| `src/tests/test_cpu_alignment.py` | CPU reference alignment oracle unit tests | 72 | 72 | 0 | 0 |
| `src/tests/test_parser.py` | Member 2 FASTA parser contract tests | 3 | 3 | 0 | 0 |
| `src/tests/test_chunking.py` | Member 2 genome chunker contract tests | 5 | 5 | 0 | 0 |
| `src/tests/test_integration.py` | Week 1 integration pipeline tests | 26 | 26 | 0 | 0 |
| `tests/test_cuda_alignment.py` | CUDA host prep, diagnostics, and fallback | 27 | 15 | 12 | 0 |
| `tests/test_gpu_pipeline_integration.py` | Dispatcher & GPU pipeline integration | 25 | 15 | 10 | 0 |
| `tests/test_dask_processor.py` | Dask processor, ordering, tasks, & E2E pipeline | 33 | 33 | 0 | 0 |
| **Total Test Suite** | **Full Project Regression Suite** | **191** | **169** | **22** | **0** |

- **Total Execution Time**: ~13.30 s
- **Pass Rate (Executed Tests)**: **100% (169 / 169)**
- **Skipped Tests**: 22 (Hardware-dependent CUDA kernel execution skipped truthfully due to absence of `nvvm.dll` compiler library on host)
- **Failed Tests**: **0**

---

## 9. Known Limitations

1. **CPython Thread Scheduling Overhead**:
   On small sequence sets (< 100,000 bases), the overhead of Dask task graph construction and task scheduling exceeds the minimal execution time of single-threaded Python sequence comparison. Batching chunk tasks (`batch_size=50` or higher) is recommended for production scale.
2. **Local Worker Mode**:
   Currently validated using local in-process thread pools (`processes=False`) to avoid Windows process spawn latency. Connecting to external distributed Dask clusters is architecturally supported via `DaskGenomeProcessor(client=external_client)` but requires network cluster infrastructure.

---

## 10. Remaining Work for Week 3

- [ ] Multi-node cluster configuration profiles and scaling evaluation.
- [ ] Dask integration with CUDA alignment engine (dispatching GPU tasks across distributed workers where CUDA is available).
- [ ] Evaluation of task batch size auto-tuning based on chunk length and worker count.

**Week 3 Day 1 & Day 2 Status**: **COMPLETE & VALIDATED**
