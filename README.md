# GeneWeaver

GeneWeaver is a high-performance, multi-backend CRISPR sequence alignment and genome candidate search engine. It combines **BioPython** for FASTA ingestion, an exact **CPU reference oracle**, **Numba CUDA** for GPU-accelerated parallel sequence comparison (with on-chip shared-memory caching), **Dask** for distributed cluster scheduling, and an interactive **Textual TUI** for terminal-based visualization.

---

## Architecture Overview

```
                      ┌──────────────────────────────┐
                      │     FASTA Input Dataset      │
                      │   (BioPython SeqIO parser)   │
                      └──────────────┬───────────────┘
                                     │ sequences: list[str]
                                     ▼
                      ┌──────────────────────────────┐
                      │        Genome Chunker        │
                      │  (windowing / partitioner)   │
                      └──────────────┬───────────────┘
                                     │ chunks: list[str]
                                     ▼
                      ┌──────────────────────────────┐
                      │     Alignment Dispatcher     │
                      │       ('cpu', 'cuda')        │
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
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
  ┌──────────────────────────────┐        ┌──────────────────────────────┐
  │      Biological Scoring      │        │    Mismatch Visualization    │
  │ (GC content, PAM NGG motifs, │        │ (Rich color-coded display,   │
  │   aggregate affinity metrics)│        │  ASCII comparison, Textual)  │
  └──────────────────────────────┘        └──────────────────────────────┘
```

---

## Features & Supported Backends

- **CPU Alignment Core (`src/alignment/cpu_alignment.py`)**: Exact, deterministic baseline computing sequence identity, mismatch counts, and exact mismatch positions with canonical DNA base validation (`A`, `C`, `G`, `T`).
- **CUDA Acceleration (`src/alignment/cuda_alignment.py`)**:
  - *Baseline Kernel*: Grid-strided 1D thread blocks computing sequence comparisons in parallel.
  - *Shared-Memory Kernel*: Cooperatively caches broadcast guide sequences into fast L1 shared memory (`s_target`), eliminating 128x redundant DRAM reads and cutting PCIe host-to-device transfers by 50%.
  - *Windows CUDA 13.x Compatibility*: Automated DLL path resolution for Windows x64 runtime/NVVM libraries and PTX version adapter for seamless driver JIT linking.
  - *Zero-Overhead Transparent Fallback*: Automatic fallback to CPU if CUDA hardware or libraries are missing.
- **Dask Distributed Engine (`src/distributed/dask_processor.py`)**:
  - Scales alignment across multi-core CPUs and distributed worker pools.
  - Dynamic task batch auto-tuning balancing worker task granularity against queue overhead.
  - Clean lifecycle management for both owned and shared `Client`/`LocalCluster` instances.
- **FASTA Parsing & Chunking (`src/parser/`, `src/chunking/`)**:
  - Robust FASTA parsing via BioPython with validation for empty files, invalid paths, and directories.
  - Configurable sliding and contiguous genome chunking with boundary safety.
- **Biological Scoring & PAM Detection (`src/scoring/`, `src/pam/`, `src/analysis/`)**:
  - SpCas9 `NGG` PAM motif detection and target protospacer affinity scoring.
  - Sequence GC content computation and alignment scoring aggregates.
- **Interactive Terminal Dashboard (`src/dashboard/`, `src/ui/`)**:
  - Full-screen terminal dashboard powered by Textual and Rich with an execution progress indicator, alignment statistics, and color-coded DNA base comparisons.
- **Mismatch Visualizer (`src/visualization/`)**:
  - Rich formatted text highlighting matches in green and mismatches in red.
  - Lightweight ASCII alignment rendering (`|` for matches, `.` for mismatches).

---

## Project Structure

```text
GeneWeaver/
├── benchmarks/              # Micro and macro benchmark suites
│   ├── comparative_benchmark.py # Multi-backend benchmark (CPU vs CUDA vs Dask)
│   ├── cpu_benchmark.py         # CPU throughput benchmark
│   ├── cuda_benchmark.py        # CUDA baseline & shared-memory benchmark
│   └── dask_benchmark.py        # Dask scaling & batch size benchmark
├── data/                    # Genomic input datasets
│   ├── mock_genome.fasta    # 2-chromosome synthetic verification genome
│   └── README.md            # Data format documentation
├── docs/                    # System architecture & weekly development logs
│   ├── architecture.md      # Full architecture & invariants specification
│   ├── week1.md             # Week 1: CPU baseline & pipeline foundation
│   ├── week2.md             # Week 2: CUDA acceleration & host-device pipeline
│   ├── week3.md             # Week 3: Dask distributed processing & auto-tuning
│   └── week4.md             # Week 4: Shared-memory optimization & validation
├── results/                 # Reproducible benchmark output artifacts (CSV / JSON)
│   ├── comparative_benchmark_results.csv
│   ├── comparative_benchmark_results.json
│   ├── cuda_benchmark_results.json
│   └── README.md
├── scripts/                 # CLI entry points
│   ├── run_benchmarks.py    # Automated benchmark suite runner
│   └── run_pipeline.py      # End-to-end alignment pipeline CLI
├── src/                     # Application source code
│   ├── alignment/           # CPU and CUDA alignment kernels and dispatcher
│   ├── analysis/            # PAM scoring integration
│   ├── chunking/            # Sequence windowing and chunking
│   ├── dashboard/           # Textual dashboard launcher
│   ├── distributed/         # Dask distributed engine
│   ├── pam/                 # CRISPR PAM motif detector
│   ├── parser/              # BioPython FASTA sequence parser
│   ├── pipeline.py          # Unified genome processing workflow
│   ├── scoring/             # Biological scoring & DNA validation
│   ├── ui/                  # Textual user interface implementation
│   └── visualization/       # Rich & ASCII mismatch visualizer
├── tests/                   # Consolidated regression test suite (315 discovered tests)
├── .gitignore               # Cache, virtualenv, and build exclusions
├── pytest.ini               # Test configuration and pythonpath settings
├── requirements.txt         # Package dependencies
└── README.md                # Project documentation
```

---

## Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows, Linux, or macOS
- *Optional for CUDA GPU acceleration*: NVIDIA GPU with Compute Capability ≥ 5.0 and NVIDIA Display Driver.

### Setup Instructions
1. Clone the repository and navigate into the root directory:
   ```bash
   cd GeneWeaver
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv

   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1

   # On Linux / macOS:
   source .venv/bin/activate
   ```

3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run the Pipeline

Execute the end-to-end genomic alignment pipeline via `scripts/run_pipeline.py`:

```bash
# Basic run using default mock genome with CPU backend:
python scripts/run_pipeline.py

# Run with execution stage timing breakdown:
python scripts/run_pipeline.py --timed

# Run on GPU with CUDA acceleration and shared-memory target caching:
python scripts/run_pipeline.py --backend cuda --shared-mem --timed

# Run with automatic backend detection (CUDA if available, otherwise CPU):
python scripts/run_pipeline.py --backend auto --timed

# Run with explicit fallback to CPU if CUDA is unavailable:
python scripts/run_pipeline.py --backend cuda --fallback-to-cpu --timed

# Run with custom FASTA file, target sequence, and chunk size:
python scripts/run_pipeline.py --fasta data/mock_genome.fasta --target ATGCGTACGT --chunk-size 10 --timed
```

### Example Input and Output

**Sample Command:**
```bash
python scripts/run_pipeline.py --fasta data/mock_genome.fasta --target ATGCGTACGT --chunk-size 10 --timed
```

**Expected Console Output:**
```text
--- GeneWeaver Pipeline Runner ---
FASTA File : C:\Pm\GeneWeaver\data\mock_genome.fasta
Target     : ATGCGTACGT (length: 10)
Chunk Size : 10
Backend    : cpu
Shared Mem : False
Fallback   : False

Completed in 0.315 ms (Backend used: cpu):
  Parse Time    : 0.264 ms
  Chunk Time    : 0.007 ms
  Alignment Time: 0.042 ms
  Aligned Chunks: 8
```

---

## Interactive Terminal Dashboard

GeneWeaver includes an interactive full-screen terminal user interface powered by Textual:

```bash
# Launch interactive dashboard:
python -m src.dashboard.textual_dashboard
```

### Dashboard Features:
- **Pipeline Execution**: Trigger alignments with the `Run Alignment` button and view the resulting statistics.
- **Summary Metrics**: View total sequence count, chunk count, aligned chunks, total matches, total mismatches, and overall percentage similarity.
- **Colored Alignment Inspection**: Visually inspect candidate chunks against the target with matches highlighted in green, mismatches highlighted in red, and exact 1-based mismatch positions reported.
- **Progress & Timing**: Displays execution progress and total pipeline time in milliseconds.

---

## Running Tests

Run the full regression test suite:

```bash
# Run all tests:
python -m pytest -v

# Run cross-backend parity tests (CPU vs CUDA vs Dask):
pytest tests/test_cross_backend_parity.py -v

# Run CUDA unit tests:
pytest tests/test_cuda_alignment.py -v

# Run Dask distributed tests:
pytest tests/test_dask_processor.py -v

# Run biological scoring tests:
pytest tests/test_biological_scoring.py -v

# Run mismatch visualization tests:
pytest tests/test_visualization.py -v

# Run dashboard integration tests:
pytest tests/test_dashboard.py -v
```

> **CUDA Test Environment Handling**: If an NVIDIA GPU is not detected in the environment, CUDA tests automatically skip or fall back safely without test crashes or false reports.

---

## Running Benchmarks

GeneWeaver includes benchmark suites to measure sequential CPU, CUDA GPU, and Dask distributed performance across multiple sequence lengths and chunk volumes:

```bash
# Run multi-backend comparative benchmark (CPU vs CUDA vs Dask):
python scripts/run_benchmarks.py comparative

# Run individual backend benchmarks:
python scripts/run_benchmarks.py cpu
python scripts/run_benchmarks.py cuda
python scripts/run_benchmarks.py dask

# Run all suites sequentially:
python scripts/run_benchmarks.py all
```

---

## Actual Benchmark Results

The following measurements were recorded on a physical test workstation:
- **GPU**: NVIDIA GeForce RTX 2050 Laptop GPU (4 GB GDDR6, 2048 CUDA Cores, Compute Capability 8.6)
- **CPU**: Intel Core i5-11400H / AMD64 Architecture (6 cores / 12 threads)
- **OS**: Windows 11 64-bit
- **Python**: 3.13.15, Numba 0.68.0, Dask 2026.8.0

### Comparative Benchmark Summary (`results/comparative_benchmark_results.json`)

| Workload | Sequence Length | Chunks | Total Bases | CPU Min | CUDA Base | CUDA Shared-Mem | Dask Threads | Dask Distributed | Shared-Mem Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Small** | 100 bp | 1,000 | 100,000 bp | 8.67 ms | 9.29 ms | **7.42 ms** | 15.76 ms | 57.83 ms | **1.17x** |
| **Medium** | 500 bp | 1,000 | 500,000 bp | 37.79 ms | 20.83 ms | **14.63 ms** | 41.40 ms | 77.52 ms | **2.58x** |
| **Large** | 1,000 bp | 1,000 | 1,000,000 bp | 76.92 ms | 37.24 ms | **29.61 ms** | 83.05 ms | 97.60 ms | **2.60x** |

### Key Performance Insights
1. **100% Bit-Exact Determinism**: Every backend (CPU oracle, CUDA baseline, CUDA shared memory, Dask threads, and Dask distributed) produces identical alignment lengths, match counts, mismatch counts, scores, and mismatch positions.
2. **Workload Scaling**:
   - For small sequence batches (≤100 kbp), the unaccelerated CPU oracle is exceptionally competitive (~8.7 ms) because host buffer allocation, PCIe transfer, and thread creation overhead offset raw kernel speed.
   - For medium to large batches (0.5–1.0 Mbp+), CUDA shared-memory caching delivers **2.58x to 2.60x wall-clock speedup** over CPU, and over **34x raw kernel speedup** by amortizing PCIe transfers.
3. **Dask Task Granularity**:
   - Dask thread parallelism scales effectively on large sequences without IPC overhead.
   - For in-memory data, Dask `LocalCluster` introduces inter-process RPC and task-graph scheduling overhead. Batch auto-tuning (`batch_size=125` vs `batch_size=1`) mitigates this, reducing scheduling latency by **77.5%**.

---

## Known Limitations & Recommended Next Steps

1. **Global DRAM Allocation on Large Sequences**: For sequences exceeding $10^6$ base pairs, streaming chunking should be used to avoid loading the entire dataset into RAM simultaneously.
2. **Multi-GPU Orchestration**: The current CUDA backend targets the primary CUDA device (`cuda.get_current_device()`). Integrating Dask-CUDA worker device mapping would enable multi-GPU scaling across multiple cards.
3. **Extended Ambiguity Codes**: The alignment core currently validates canonical DNA bases (`A`, `C`, `G`, `T`). Support for IUPAC ambiguity characters (`R`, `Y`, `S`, `W`, `K`, `M`, `B`, `D`, `H`, `V`, `N`) can be added via custom bitwise lookup tables.

## Streamlit Web Dashboard

GeneWeaver includes a single-page web application for CRISPR DNA alignment.

### Launch the Web Application

Install dependencies:

    python -m pip install -r requirements.txt

Start Streamlit:

    python -m streamlit run src/web/app.py

Open http://localhost:8501 in your browser.

### Web Dashboard Features

- Upload a FASTA file or use the built-in demo dataset.
- Select CPU or Dask execution.
- Configure target DNA sequence and chunk size.
- View alignment statistics and execution time.
- Explore DNA matches and mismatches with color highlighting.
- View biological scoring information.

