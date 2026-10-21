# Week 4 -- GPU Shared-Memory Optimization, Pipeline Integration & Interactive Monitoring

## Overview

Week 4 focuses on advanced CUDA GPU optimizations, end-to-end pipeline integration, and robust distributed/fallback execution:
1. **Day 1**: CUDA shared-memory optimization preparation and initial implementation (`src/alignment/cuda_alignment.py`).
2. **Day 2**: CUDA optimization correctness, boundary safety, race condition prevention, and comparative performance profiling (`benchmarks/cuda_benchmark.py`).
3. **Day 3**: End-to-end pipeline integration, explicit backend selection (`cpu`, `cuda`, `gpu`, `auto`), graceful fallback handling, and Dask compatibility.
4. **Day 4**: Member 2 biological scoring & mismatch visualization integration.
5. **Day 5**: Textual interactive terminal dashboard integration.
6. **Day 6**: System-wide performance profiling & memory footprint audits.
7. **Day 7**: Final product validation and documentation.

---

# Day 1 -- CUDA Shared-Memory Optimization Architecture

## 1. Algorithmic Bottleneck Analysis of Baseline CUDA Kernel

In the Week 2 baseline CUDA alignment engine:
- The alignment kernel `dna_align_kernel` assigns one CUDA thread per `(reference, query)` sequence pair.
- In CRISPR and candidate genomic window search workflows (`cuda_align_chunks`), every candidate window (reference chunk) is aligned against the **exact same** target sequence (query).

### Baseline Limitations & Memory Waste:
1. **Redundant Host-to-Device Memory Transfers**:
   - `_prepare_host_buffers` allocates `qrys_buf` of shape $(N, \text{seq\_len})$.
   - For $N = 100,000$ chunks of length $1,000\text{ bp}$, this transfers $100\text{ MB}$ of candidate chunks + $100\text{ MB}$ of redundant replicated target sequences across PCIe, doubling H2D transfer time and device DRAM consumption.
2. **Global DRAM Latency & Read Redundancy**:
   - In `dna_align_kernel`, every thread in a thread block executes:
     ```python
     q_char = qrys_flat[base_offset + i]
     ```
   - For a thread block of 128 threads, the identical byte of the target query is fetched 128 separate times from high-latency global device memory (DRAM: 200–400 cycles latency).

---

## 2. Shared-Memory Optimization Design (`dna_align_shared_kernel`)

### 2.1 Core Strategy
Instead of replicating the target sequence $N$ times across global DRAM, the broadcast target sequence is transferred **once** to device memory (`d_target` of size $\text{seq\_len}$), and cooperatively cached in fast on-chip L1 shared memory (`s_target`) by the threads of each thread block.

```
       Global Device Memory (DRAM)
       ┌───────────────────────────┐
       │ target_flat [seq_len]     │ (Single copy, size seq_len)
       └─────────────┬─────────────┘
                     │ Cooperative strided load
                     ▼
       ┌───────────────────────────┐
       │  cuda.shared.array(1024)  │ (Fast on-chip L1 shared memory)
       │         s_target          │ (1-2 cycles access latency)
       └─────────────┬─────────────┘
                     │ cuda.syncthreads()
         ┌───────────┼───────────┐
         ▼           ▼           ▼
      Thread 0    Thread 1    Thread T (blockDim.x = 128)
     (compare)   (compare)   (compare)
```

### 2.2 Cooperative Strided Loading & Synchronization
Inside `dna_align_shared_kernel`:
```python
s_target = cuda.shared.array(shape=1024, dtype=np.uint8)

tid = cuda.threadIdx.x
bdim = cuda.blockDim.x

# Cooperative strided loading into block shared memory
idx = tid
while idx < seq_len and idx < 1024:
    s_target[idx] = target_flat[idx]
    idx += bdim

# Block-wide barrier synchronization
cuda.syncthreads()
```

---

# Day 2 -- CUDA Optimization Correctness, Boundaries, Safety, and Performance Analysis

## 1. Correctness, Boundary Validation & Concurrency Guarantees

The shared-memory CUDA kernel (`dna_align_shared_kernel`) and dispatcher routines were subjected to rigorous correctness and boundary verification:

1. **Deadlock-Free Barrier Synchronization**:
   - `cuda.syncthreads()` is called unconditionally outside any branch dependent on `pair_idx < n_pairs`. Every active thread in the thread-block participates in the barrier, preventing thread-divergence barrier deadlocks on Ampere/Turing architectures.
2. **Boundary-Safe Strided Loading & Defense in Depth**:
   - The cooperative load loop `while idx < seq_len and idx < 1024:` guards against writing beyond both the sequence boundary and the static shared array bound ($1024\text{ bytes}$).
   - In `dna_align_shared_kernel`, two distinct loop stages guarantee safe memory access even for arbitrarily sized targets:
     ```python
     limit = seq_len if seq_len <= 1024 else 1024
     for i in range(limit):
         r_char = refs_flat[base_offset + i]
         q_char = s_target[i]
         ...
     for i in range(limit, seq_len):
         r_char = refs_flat[base_offset + i]
         q_char = target_flat[i]
         ...
     ```
   - For sequences $\le 1024\text{ bp}$, $100\%$ of target characters are served from fast L1 shared memory.
   - For sequences $> 1024\text{ bp}$, host dispatch automatically routes execution to the baseline linear kernel (`dna_align_kernel`), and the device kernel internally prevents out-of-bounds reads if directly invoked.
3. **Tail Block & Non-Multiple Grid Dimensions**:
   - Handled arbitrary chunk counts that do not align with thread-block dimensions ($N = 1, 7, 31, 33, 63, 65, 127, 129, 255, 257$). Tail threads participate in target loading and synchronization, but safely bypass result write buffers when `pair_idx >= n_pairs`.
4. **Thread-Block Dimension Tuning**:
   - Validated across block dimensions `threads_per_block = 32, 64, 128, 256, 512`, verifying 100% bit-exact equivalence with the CPU reference oracle across all configurations.
5. **Thread Independence & Race-Condition Freedom**:
   - Each thread $i$ computes and writes strictly to its own output row indices (`out_matches[pair_idx]`, `out_mismatches[pair_idx]`, `out_mismatch_flags[pair_idx, :]`). No inter-thread mutable state exists.

---

## 2. Actual Hardware Benchmark Comparison: Baseline CUDA vs Shared-Memory CUDA

Benchmarked on physical hardware: **NVIDIA GeForce RTX 2050 Laptop GPU (Ampere SM 8.6, 4 GB GDDR6)** running CUDA Toolkit 13.4, Driver 610.88, Python 3.13.15.  
Each configuration executed over 5 warm rounds; reported figures represent minimum (peak) elapsed wall and GPU times:

| Target / Chunk Length | Chunks Evaluated | Execution Path | H2D Transfer Time | Kernel Execution Time | D2H Transfer Time | Total GPU Time | Total Wall-Clock Time | Measured Kernel Speedup |
|---|---|---|---|---|---|---|---|---|
| **128 bp** | 10,000 | **Baseline CUDA** | 1.23 ms | 1.86 ms | 585.7 µs | 3.78 ms | 94.08 ms | 1.00x (ref) |
| **128 bp** | 10,000 | **Shared-Memory CUDA** | **928.6 µs** | **1.41 ms** | **517.0 µs** | **2.96 ms** | **66.43 ms** | **1.31x** (+29.4% faster wall-clock) |
| **512 bp** | 10,000 | **Baseline CUDA** | 2.64 ms | 8.35 ms | 1.31 ms | 12.42 ms | 223.00 ms | 1.00x (ref) |
| **512 bp** | 10,000 | **Shared-Memory CUDA** | **1.68 ms** | **6.46 ms** | **1.37 ms** | **9.72 ms** | **172.97 ms** | **1.29x** (+22.4% faster wall-clock) |
| **1024 bp (max)** | 5,000 | **Baseline CUDA** | 2.45 ms | 3.20 ms | 1.30 ms | 7.23 ms | 195.84 ms | 1.00x (ref) |
| **1024 bp (max)** | 5,000 | **Shared-Memory CUDA** | **1.68 ms** | **1.53 ms** | **1.35 ms** | **4.90 ms** | **137.66 ms** | **2.10x** (+29.7% faster wall-clock) |

### Key Benchmark Takeaways:
1. **PCIe Memory Reduction**: Target transfer data across PCIe is reduced by $N\times$ ($10,000\times$ less target data), cutting H2D transfer time by **24.4% to 36.4%**.
2. **Compute Latency Reduction**: Eliminating 128 redundant global DRAM reads per position per block yields up to **2.10x faster kernel execution**.
3. **End-to-End Throughput**: Overall wall-clock speedup reaches **1.29x to 1.42x** (up to **29.7% lower end-to-end latency**) including Python host preparation and result reconstruction.
4. **Parity**: 100% bit-exact equivalence confirmed against the CPU oracle for every trial.

---

# Day 3 -- Pipeline Integration, Robust Fallback, and Regression Testing

## 1. End-to-End Pipeline Architecture

The pipeline seamlessly links:
$$\text{FASTA Input} \longrightarrow \text{BioPython Parsing} \longrightarrow \text{Chunking} \longrightarrow \text{Alignment (CPU/CUDA)} \longrightarrow \text{Scoring} \longrightarrow \text{Results}$$

Integrated modules:
- `src/parser/fasta_parser.py`: BioPython `SeqIO` sequence parser.
- `src/chunking/genome_chunker.py`: Sliding window genomic chunker.
- `src/alignment/cpu_alignment.py`: Deterministic CPU alignment oracle and `AlignmentResult` container.
- `src/alignment/cuda_alignment.py`: Hardware-accelerated baseline and shared-memory CUDA kernels.
- `src/alignment/dispatcher.py`: Unified backend abstraction layer.
- `src/alignment/integration.py`: End-to-end `run_pipeline` and `run_pipeline_timed` orchestrators.
- `src/pipeline.py`: Top-level package entrypoint `run_alignment_pipeline` and `process_genome`.
- `scripts/run_pipeline.py`: Production CLI runner supporting `--backend`, `--shared-mem`, `--fallback-to-cpu`, and `--timed`.

## 2. Explicit Backend Selection and Transparent Fallback Mechanics

Callers can explicitly select execution backends or request automatic hardware probing:

| Backend String | Behavior | Fallback Option |
|---|---|---|
| `backend="cpu"` | Executes single-threaded CPU reference oracle. | N/A |
| `backend="cuda"` (or `"gpu"`) | Executes CUDA GPU acceleration. If CUDA is unavailable and `fallback_to_cpu=False`, raises `CUDANotAvailableError`. If `fallback_to_cpu=True`, falls back to CPU and reports backend as `"cpu (fallback)"`. | Configurable (`fallback_to_cpu`) |
| `backend="auto"` | Automatically checks `is_cuda_available()`. Selects `cuda` if GPU hardware and compiler are functional; selects `cpu` otherwise. Never crashes due to missing GPU. | Automatic |

> [!IMPORTANT]
> The engine **never silently pretends** it ran on GPU when falling back to CPU. The `backend` property on `PipelineTimingBreakdown` explicitly records `"cpu (fallback)"` so downstream callers and loggers are truthfully informed.

## 3. Distributed Compatibility (Dask)

In `src/distributed/dask_processor.py`:
- `DaskGenomeProcessor` supports `worker_backend="cpu"`, `"cuda"`, or `"auto"`.
- Workers dynamically query `is_cuda_available()` on their assigned node. Nodes lacking a GPU cleanly execute CPU alignment without task failure.
- `use_shared_memory=True` is supported across `align_chunks`, `dask_align_chunks`, and `process_fasta_dask_timed`, caching the target sequence in worker GPU L1 shared memory when GPU acceleration is active.

## 4. Full Regression Test Results

Executed complete test suite across physical hardware:
```bash
pytest tests/ src/tests/
```

```text
============================= test session starts =============================
platform win32 -- Python 3.13.15, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Pm\GeneWeaver
configfile: pytest.ini
collected 249 items

tests\test_cuda_alignment.py ........................................... [ 17%]
.                                                                        [ 17%]
tests\test_dask_processor.py ........................................... [ 34%]
..........................                                               [ 45%]
tests\test_gpu_pipeline_integration.py ..............................    [ 57%]
src\tests\test_chunking.py .....                                         [ 59%]
src\tests\test_cpu_alignment.py ........................................ [ 75%]
................................                                         [ 88%]
src\tests\test_integration.py ..........................                 [ 98%]
src\tests\test_parser.py ...                                             [100%]

======================= 249 passed, 7 warnings in 5.59s =======================
```

- **Total Tests Passed**: **249 of 249 (100% pass rate)**.
- **Failures / Regressions**: **0**.

---

# Operational Guide

## 1. Selecting Execution Backend

### CLI Usage (`scripts/run_pipeline.py`)
```bash
# CPU Execution
python scripts/run_pipeline.py --backend cpu --timed

# CUDA Execution with Shared Memory Optimization
python scripts/run_pipeline.py --backend cuda --shared-mem --timed

# Automatic Backend Selection (CUDA if available, else CPU)
python scripts/run_pipeline.py --backend auto --shared-mem --timed

# CUDA with explicit CPU fallback on hardware error
python scripts/run_pipeline.py --backend cuda --fallback-to-cpu --timed
```

### Python API Usage
```python
from alignment.integration import run_pipeline, run_pipeline_timed

# Auto-selection with shared memory
results, timing = run_pipeline_timed(
    "data/mock_genome.fasta",
    target="ATGCGTACGT",
    chunk_size=10,
    backend="auto",
    use_shared_memory=True,
)
print(f"Executed on: {timing.backend}, Duration: {timing.total_pipeline_time*1000:.2f} ms")
```

## 2. Requirements & Setup for Environments Without GPU

In environments without a compatible NVIDIA GPU or CUDA Toolkit:
1. `backend="cpu"` or `backend="auto"` functions out-of-the-box using the standard CPU baseline.
2. GPU unit tests automatically skip without failure via `@skip_if_no_cuda`.
3. To test CUDA kernel logic on CPU systems without GPU hardware, enable Numba's official software simulator:
   ```bash
   # In PowerShell:
   $env:NUMBA_ENABLE_CUDASIM = "1"
   pytest tests/test_cuda_alignment.py
   ```
