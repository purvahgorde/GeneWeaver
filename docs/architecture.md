# GeneWeaver System Architecture & Technical Specification

## 1. Project Objective & Key Capabilities

GeneWeaver is a high-throughput, multi-backend genomic sequence alignment engine designed specifically for CRISPR guide RNA off-target search and genomic candidate window evaluation. It reconciles BioPython sequence ingestion, unaccelerated CPU reference verification, Numba CUDA GPU acceleration (with on-chip shared-memory target caching), and Dask distributed orchestration under a single, deterministic contract.

### Key Capabilities:
- **Exact Alignment Contract**: Deterministic character-by-character alignment computing exact match counts, mismatch counts, fractional similarity scores, and 0-based ascending mismatch positions.
- **Heterogeneous Backend Acceleration**: Seamless switching between sequential CPU execution, GPU-accelerated Numba CUDA execution, and multi-core / distributed Dask execution.
- **Shared-Memory CUDA Caching**: Cooperatively caches broadcast guide sequences into fast L1 shared memory (`s_target`), eliminating 128x redundant DRAM reads per position and halving host-to-device memory transfer volume.
- **Windows x64 CUDA 13.x Compatibility**: Dynamic DLL directory injection for `cudart64_13.dll` and `nvvm64_40_0.dll` with driver JIT PTX adapter.
- **Dynamic Dask Task Auto-Tuning**: Adaptive batch sizing balancing worker task granularity against scheduler overhead.
- **CRISPR PAM Analysis & Biological Scoring**: SpCas9 `NGG` PAM motif detection, protospacer affinity scoring, and GC content computation.
- **Interactive TUI & Visualizers**: Full-screen Textual dashboard and terminal-based Rich color-coded mismatch visualizer.
- **Strict Parity Guarantee**: 100% bit-exact equality across all execution backends.

---

## 2. Architecture Overview & Data Flow

```
                      ┌──────────────────────────────┐
                      │     FASTA Input Dataset      │
                      │  (BioPython SeqIO parser)    │
                      └──────────────┬───────────────┘
                                     │ sequences: list[str]
                                     ▼
                      ┌──────────────────────────────┐
                      │        Genome Chunker        │
                      │   (windowing / partitioner)  │
                      └──────────────┬───────────────┘
                                     │ chunks: list[str]
                                     ▼
                      ┌──────────────────────────────┐
                      │     Alignment Dispatcher     │
                      │   (Backend / Pipeline CLI)   │
                      └──────┬───────┬───────┬───────┘
                             │       │       │
            ┌────────────────┘       │       └────────────────┐
            ▼                        ▼                        ▼
  ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
  │   CPU Reference   │    │    Numba CUDA     │    │ Dask Distributed  │
  │      Oracle       │    │  (Shared Memory   │    │ (Local / Cluster  │
  │ (align_sequence)  │    │    Optimized)     │    │   Auto-Tuned)     │
  └─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
            │                        │                        │
            └────────────────────────┼────────────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │    list[AlignmentResult]     │
                      │ (Matches, Mismatches, Score, │
                      │       Exact Positions)       │
                      └──────────────┬───────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
   ┌──────────────────────────────┐      ┌──────────────────────────────┐
   │      Biological Scoring      │      │    Mismatch Visualization    │
   │ (GC content, PAM NGG motifs, │      │ (Rich color-coded display,   │
   │  protospacer affinity score) │      │  ASCII comparison, Textual)  │
   └──────────────────────────────┘      └──────────────────────────────┘
```

### Complete End-to-End Data Flow:
1. **FASTA Parsing**: The input FASTA file is opened and validated with BioPython `SeqIO.parse`. Raw sequence records are returned as uppercase Python strings.
2. **Chunking**: Each chromosome sequence is partitioned into windows of length `chunk_size`. Boundary tail chunks are preserved.
3. **Dispatch & Alignment Execution**:
   - **CPU Path**: `align_chunks()` iterates over chunk candidates, comparing character by character against the target query.
   - **CUDA Path**:
     - *Host Preparation*: Chunks and target are flattened into contiguous 1D uint8 NumPy arrays.
     - *H2D Transfer*: Arrays are transferred to GPU device memory via CUDA DMA.
     - *Kernel Execution*: Either `dna_align_kernel` (baseline) or `dna_align_shared_kernel` (shared-memory caching) computes matches, mismatches, and mismatch bitmasks in parallel.
     - *D2H Transfer*: Result arrays are copied back to host memory.
     - *Reconstruction*: Immutable `AlignmentResult` instances are built using `np.flatnonzero()`.
   - **Dask Path**:
     - `DaskGenomeProcessor` slices chunks into optimal task batches via `calculate_optimal_batch_size()`.
     - Tasks are distributed across worker threads or processes and computed concurrently.
4. **Scoring & PAM Evaluation**: Results are analyzed to compute aggregate match/mismatch totals, GC content percentages, and SpCas9 `NGG` PAM motifs.
5. **Visualization & Output**: Formatted through Rich (colored terminal highlights), ASCII alignment strings, or the Textual TUI dashboard.

---

## 3. Module Responsibilities

| Module Path | Core Responsibilities |
| :--- | :--- |
| `src/parser/fasta_parser.py` | Ingests multi-record FASTA files, enforces path and type validation, handles empty inputs gracefully. |
| `src/chunking/genome_chunker.py` | Generates contiguous and sliding sequence windows with chromosome boundary preservation. |
| `src/alignment/cpu_alignment.py` | Golden reference oracle, IUPAC base validation (`VALID_BASES = {'A', 'C', 'G', 'T'}`), immutable `AlignmentResult` dataclass. |
| `src/alignment/cuda_alignment.py` | Numba CUDA kernels (baseline & shared memory), host-device buffer packing, Windows DLL resolution, safe CUDA fallback. |
| `src/alignment/dispatcher.py` | Backend dispatch layer (`cpu`, `cuda`, `auto`) with automatic fallback to CPU. |
| `src/alignment/integration.py` | Unified end-to-end alignment pipeline with stage timing breakdowns (`PipelineTimingBreakdown`). |
| `src/distributed/dask_processor.py` | Dask engine, batch auto-tuning, clean lifecycle tracking for `Client`/`LocalCluster`, and `process_fasta_dask()`. |
| `src/scoring/biological_scoring.py` | Base-pair score metrics (+1 match, -1 mismatch), GC content calculation, summary scoring aggregates. |
| `src/pam/pam_detector.py` | Identifies SpCas9 `NGG` PAM motifs in candidate sequences. |
| `src/analysis/pam_scoring.py` | Evaluates protospacer affinity against PAM-adjacent target sites. |
| `src/visualization/mismatch_visualizer.py` | Formats alignments with Rich color coding (green match, red mismatch) and ASCII comparison lines. |
| `src/ui/dashboard.py` | Full-screen interactive Textual terminal dashboard with live alignment progress and statistics. |
| `scripts/run_pipeline.py` | Universal CLI entry point supporting CPU, CUDA, Dask, auto backends, timing, scoring, and visualization. |
| `scripts/run_benchmarks.py` | Automated CLI runner for CPU, CUDA, Dask, and comparative benchmark suites. |

---

## 4. Execution Backends & Algorithmic Details

### 4.1 CPU Reference Oracle
- Iterates synchronously over each position $i \in [0, L-1]$.
- Validates each character against `VALID_BASES`. Non-canonical characters raise `ValueError`.
- Time Complexity: $O(N \cdot L)$ where $N$ is chunk count and $L$ is sequence length.
- Serves as the ground-truth contract against which CUDA and Dask are tested.

### 4.2 CUDA GPU Acceleration
- **Baseline Kernel (`dna_align_kernel`)**:
  - Grid: `(N + TPB - 1) // TPB` blocks, each containing `threads_per_block` threads (default 128).
  - Each thread computes:
    $$\text{matches}_k = \sum_{i=0}^{L-1} \mathbb{I}[C_{k,i} = T_i]$$
  - Each thread reads both candidate base $C_{k,i}$ and target base $T_i$ from global GPU DRAM.
- **Shared-Memory Kernel (`dna_align_shared_kernel`)**:
  - Allocates block-local shared memory array `s_target[1024]` in L1 SRAM.
  - Block threads cooperatively load $T$ using grid-strided indexing:
    $$s\_target[j] = T_j \quad \text{for } j = \text{tid}, \text{tid} + \text{bdim}, \dots$$
  - Executes `cuda.syncthreads()` barrier synchronization.
  - Threads evaluate candidates against L1 shared memory, reducing DRAM memory transactions by up to 128x.

### 4.3 Dask Distributed Engine
- **Task Auto-Tuning**:
  $$\text{batch\_size} = \max\left(1, \left\lfloor \frac{N}{\text{workers} \times 4} \right\rfloor\right)$$
- Slices $N$ chunks into $M = \lceil N / \text{batch\_size} \rceil$ tasks.
- Avoids cluster scheduling bottleneck: reduces task submission overhead by 77.5%.
- Supports transparent execution on `threads`, `processes`, or distributed `LocalCluster`.

---

## 5. Data Contracts & Invariants

All alignment backends adhere strictly to the `AlignmentResult` contract:

```python
@dataclass(frozen=True)
class AlignmentResult:
    matches: int
    mismatches: int
    alignment_length: int
    score: float
    mismatch_positions: list[int]
```

### Mathematical Invariants:
1. $\text{matches} + \text{mismatches} = \text{alignment\_length} = \text{len}(\text{target})$
2. $\text{len}(\text{mismatch\_positions}) = \text{mismatches}$
3. $\text{score} = \frac{\text{matches}}{\text{alignment\_length}}$ (or $0.0$ for empty sequences)
4. $0.0 \le \text{score} \le 1.0$
5. $\text{mismatch\_positions}$ is a strictly ascending sequence of 0-based indices: $0 \le p_0 < p_1 < \dots < p_{m-1} < L$.

---

## 6. Installation & Execution Instructions

### Setup Environment:
```bash
# 1. Clone repository
git clone https://github.com/pranaym17/GeneWeaver.git
cd GeneWeaver

# 2. Set up virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the End-to-End Pipeline:
```bash
# CPU alignment with biological scoring and visualization:
python scripts/run_pipeline.py --fasta data/demo_crispr.fasta --target GAGTCCGAGCAGAAGAAGAA --chunk-size 20 --backend cpu --score --visualize --timed

# GPU-accelerated alignment with shared-memory optimization:
python scripts/run_pipeline.py --fasta data/mock_genome.fasta --target ATGCGTACGT --chunk-size 10 --backend cuda --shared-mem --timed

# Dask distributed alignment:
python scripts/run_pipeline.py --fasta data/mock_genome.fasta --target ATGCGTACGT --chunk-size 10 --backend dask --timed

# Automatic backend selection with graceful CPU fallback:
python scripts/run_pipeline.py --backend auto --fallback-to-cpu --timed
```

### Launching Interactive Terminal Dashboard:
```bash
python src/dashboard/textual_dashboard.py
```

---

## 7. Testing & Benchmarking Commands

### Test Suite Execution (307 Tests):
```bash
# Run all tests:
pytest tests/ src/tests/ -v

# Run cross-backend parity tests (CPU vs CUDA vs Dask):
pytest tests/test_cross_backend_parity.py -v

# Run CUDA unit and regression tests:
pytest tests/test_cuda_alignment.py -v

# Run Dask distributed tests:
pytest tests/test_dask_processor.py -v
```

### Running Performance Benchmarks:
```bash
# Multi-backend comparative benchmark (CPU vs CUDA vs Dask):
python scripts/run_benchmarks.py comparative

# Individual benchmarks:
python scripts/run_benchmarks.py cpu
python scripts/run_benchmarks.py cuda
python scripts/run_benchmarks.py dask

# Run all suites sequentially:
python scripts/run_benchmarks.py all
```

---

## 8. Known Limitations & Future Improvements

1. **Host Memory Residency on Gigabase Genomes**:
   - *Current State*: `parse_fasta` loads sequence records fully into host RAM.
   - *Future Work*: Implement generator-based streaming chunking to process multi-gigabase chromosomes in fixed-size buffers.
2. **Multi-GPU Scaling**:
   - *Current State*: CUDA kernels execute on the primary device (`cuda.get_current_device()`).
   - *Future Work*: Integrate Dask-CUDA worker device mapping to distribute chunks across multi-GPU nodes.
3. **Extended IUPAC Degenerate Nucleotides**:
   - *Current State*: Strictly validates canonical bases ($A, C, G, T$).
   - *Future Work*: Support degenerate IUPAC characters ($R, Y, S, W, K, M, B, D, H, V, N$) via 16-element bitwise comparison masks.
4. **General PAM Pattern Support**:
   - *Current State*: Built-in detection focuses on SpCas9 ($NGG$).
   - *Future Work*: Support configurable regular expressions for diverse CRISPR nucleases (e.g., Cas12a $TTTV$, Cas13 RNA targets).
