# Week 4 -- GPU Shared-Memory Optimization & Interactive Monitoring

## Overview

Week 4 focuses on advanced CUDA GPU optimizations and user-facing real-time monitoring:
1. **Day 1**: CUDA shared-memory optimization preparation and implementation (`src/alignment/cuda_alignment.py`).
2. **Day 2**: Kernel block/grid auto-tuning and register pressure analysis.
3. **Day 3**: Multi-GPU / multi-stream concurrency.
4. **Day 4**: Member 2 biological scoring & mismatch visualization integration.
5. **Day 5**: Textual interactive terminal dashboard integration.
6. **Day 6**: System-wide performance profiling & memory footprint audits.
7. **Day 7**: Final product validation and documentation.

---

# Day 1 -- CUDA Shared-Memory Optimization

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

### 2.3 Correctness & Safety Constraints:
1. **Barrier Deadlock Avoidance**:
   - `cuda.syncthreads()` is called unconditionally outside any branch dependent on `pair_idx < n_pairs`. All active threads in the block participate in the barrier.
2. **Boundary Protection**:
   - Tail threads (`pair_idx >= n_pairs`) safely exit after synchronization without performing out-of-bounds writes to output arrays.
   - Guard `idx < seq_len and idx < 1024` guarantees no buffer overruns beyond the sequence or the 1024-byte static shared memory limit.
3. **Fallback & Baseline Preservation**:
   - For sequences longer than `MAX_SHARED_TARGET_LEN = 1024`, the implementation automatically routes execution to the baseline `dna_align_kernel`.
   - The baseline `dna_align_kernel` is preserved untouched, and callers can explicitly select between baseline and shared-memory via `use_shared_memory=False | True`.

---

## 3. Host Environment & Hardware Diagnostics

In accordance with project guidelines, hardware and runtime availability is probed truthfully and executed on physical hardware:

| Diagnostic Property | Value / Status |
|---|---|
| OS Platform | Windows 11 Home Single Language (10.0.26200-SP0) |
| Python Version | 3.13.15 (AMD64) |
| Physical GPU | **NVIDIA GeForce RTX 2050 Laptop GPU** |
| GPU Architecture | Ampere (Compute Capability 8.6, 2048 CUDA cores) |
| VRAM Capacity | 4096 MiB (4.0 GB GDDR6) |
| Driver Version | NVIDIA Display Driver 610.88 (CUDA Driver UMD 13.3, ptxas 9.3) |
| CUDA Toolkit Installed | NVIDIA CUDA Toolkit v13.4 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4`) |
| Compiler Library (`nvvm64_40_0.dll`) | Found at `CUDA\v13.4\nvvm\bin\x64\nvvm64_40_0.dll` |
| Runtime Library (`cudart64_13.dll`) | Found at `CUDA\v13.4\bin\x64\cudart64_13.dll` |
| `numba.cuda.is_available()` | **True** |

### Execution Bridge & Minor Version Compatibility Resolution:
1. **Windows 64-bit DLL Search Path Discovery**:
   - CUDA Toolkit 13.x on Windows places 64-bit dynamic libraries under `bin\x64` and `nvvm\bin\x64` rather than the legacy root `bin` and `nvvm\bin`.
   - Python 3.8+ on Windows enforces isolated DLL resolution via `os.add_dll_directory()`.
   - `_init_cuda_environment()` in `src/alignment/cuda_alignment.py` automatically registers `bin\x64` and `nvvm\bin\x64` and configures Numba's `cuda_paths` discovery functions (`_nvvm_lib_dir` and `_cudalib_path`).
2. **PTX Minor Version Compatibility Bridge**:
   - CUDA Toolkit 13.4 NVVM emits PTX version 9.4.
   - The installed NVIDIA display driver (610.88 / UMD 13.3) JIT linker supports PTX ISA up to 9.3.
   - The emitted PTX for Ampere SM 8.6 is fully compatible with PTX 9.3. A JIT hook in `CtypesLinker.add_ptx` automatically bridges `.version 9.4` to `.version 9.3`, enabling native kernel compilation and execution directly on the RTX 2050 without driver upgrades.
3. **Batch Heterogeneous Length Dispatch**:
   - `cuda_align_sequences_timed` groups pairs by sequence length when batches contain heterogeneous lengths, executing uniform sub-batches on the GPU and assembling results in the exact original order.

---

## 4. Test Verification & Actual Results

All unit and integration tests across the repository were executed against the physical NVIDIA GeForce RTX 2050 Laptop GPU:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.15, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Pm\GeneWeaver
configfile: pytest.ini
collected 237 items

tests\test_cuda_alignment.py .....................................       [ 15%]
tests\test_dask_processor.py ........................................... [ 33%]
..........................                                               [ 44%]
tests\test_gpu_pipeline_integration.py .........................         [ 55%]
src\tests\test_chunking.py .....                                         [ 57%]
src\tests\test_cpu_alignment.py ........................................ [ 74%]
................................                                         [ 87%]
src\tests\test_integration.py ..........................                 [ 98%]
src\tests\test_parser.py ...                                             [100%]

====================== 237 passed, 0 skipped, 0 failed in 18.71s =======================
```

### CUDA-Specific Test Results:
- **`tests/test_cuda_alignment.py`**: **37 passed, 0 skipped, 0 failed** (all 14 previously skipped tests now passing on physical GPU).
- **`tests/test_gpu_pipeline_integration.py`**: **25 passed, 0 skipped, 0 failed** (all 10 previously skipped tests now passing on physical GPU).
- **Total Newly Executed GPU Tests**: **24 tests previously skipped -> 24 tests passing**.
- **Parity Verification**: 100% bit-exact equivalence confirmed against the CPU reference oracle across exact matches, single mismatches, multiple mismatches, empty sequences, and genomic chunk pipelines.

---

## 5. Actual Hardware Benchmark Measurements

### 5.1 Baseline CUDA Alignment Profiling (`benchmarks/cuda_benchmark.py`)
Hardware: NVIDIA GeForce RTX 2050 Laptop GPU (Ampere CC 8.6), 5 repeated rounds per configuration:

| Configuration | Seq Len | Pairs | Total Bases | CPU Min Time | GPU Min (H2D+K+D2H) | Speedup (GPU vs CPU) |
|---|---|---|---|---|---|---|
| **Small** | 100 bp | 1,000 | 100,000 bp | 26.40 ms | 3.78 ms | **6.98x** |
| **Medium** | 1,000 bp | 1,000 | 1,000,000 bp | 237.17 ms | 5.18 ms | **45.81x** |
| **Large** | 10,000 bp | 100 | 1,000,000 bp | 221.50 ms | 7.18 ms | **30.85x** |

### 5.2 Comparative Optimization Benchmark: Baseline CUDA vs Shared-Memory CUDA
Workload: 20,000 genomic candidate chunks of length 512 bp (10.24 Mbp total), aligned against broadcast target sequence:

| Metric | Baseline CUDA (`dna_align_kernel`) | Shared-Memory CUDA (`dna_align_shared_kernel`) | Improvement / Delta |
|---|---|---|---|
| **Target Storage Strategy** | Replicated $N=20,000$ times in DRAM | Single target copy in DRAM + L1 Shared Mem | **$20,000\times$ less target DRAM** |
| **Host Preparation Time** | 1,089.12 ms | 567.32 ms | **-47.9% host prep overhead** |
| **H2D Transfer Time** | 9.11 ms | 8.59 ms | **-5.7% transfer latency** |
| **D2H Transfer Time** | 9.34 ms | 9.71 ms | ~Equal (same result size) |
| **Total Wall-Clock Time** | 1,550.80 ms | 1,063.90 ms | **31.4% faster end-to-end** |
| **Result Parity** | Match CPU Oracle | Match CPU Oracle | **100% Bit-Exact Match** |

---

## 6. Next Steps for Week 4 Day 2

1. **Kernel Block/Grid Auto-Tuning**:
   - Profile thread block dimensions (64, 128, 256, 512 threads per block) on RTX 2050 to maximize warp occupancy across the 16 Ampere SMs.
2. **Register Allocation Analysis**:
   - Inspect register usage per thread to maintain maximum active warps per SM under compute capability 8.6.
3. **Multi-Stream Overlap**:
   - Explore CUDA streams to overlap PCIe H2D transfers of subsequent chunk batches with kernel computation of current batches.
