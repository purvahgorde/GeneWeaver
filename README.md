# GeneWeaver

GeneWeaver is a high-performance CRISPR alignment engine combining BioPython for FASTA parsing, Numba CUDA for GPU-accelerated sequence comparison, Dask for distributed task scheduling, and Textual for real-time terminal monitoring.

---

## Project Structure

```text
GeneWeaver/
├── benchmarks/              # CPU, CUDA, and Dask benchmark suites
│   ├── cpu_benchmark.py     # CPU alignment benchmark
│   ├── cuda_benchmark.py    # GPU CUDA benchmark & memory transfer timing
│   └── dask_benchmark.py    # Dask distributed throughput & scaling benchmark
├── data/                    # Genomic input datasets & guidance
│   ├── mock_genome.fasta    # Fast verification 2-chromosome synthetic dataset
│   └── README.md            # Data guidance & format specifications
├── docs/                    # Architectural specifications and weekly sprint logs
│   ├── architecture.md      # Comprehensive system architecture & data contracts
│   ├── week1.md             # CPU baseline & chunker integration
│   ├── week2.md             # CUDA kernel development & hardware validation
│   ├── week3.md             # Dask distributed alignment architecture
│   └── week4.md             # CUDA shared-memory optimization & sprint log
├── results/                 # Generated benchmark outputs and execution logs (git-ignored)
│   └── README.md            # Results output policies
├── scripts/                 # CLI entry points and orchestration utilities
│   ├── run_pipeline.py      # End-to-end pipeline CLI
│   └── run_benchmarks.py    # Automated benchmark suite runner
├── src/                     # Core application source code
│   ├── alignment/           # CPU, CUDA, and hybrid dispatcher engines
│   ├── chunking/            # Genome sequence windowing and partitioning
│   ├── distributed/         # Dask distributed genome processing engine
│   ├── parser/              # BioPython FASTA sequence parser
│   └── pipeline.py          # Unified genome processing workflow
├── tests/                   # Consolidated test suite (CUDA, Dask, integration)
│   ├── test_cuda_alignment.py
│   ├── test_dask_processor.py
│   └── test_gpu_pipeline_integration.py
├── .gitignore               # Environment, cache, and artifact exclusions
├── pytest.ini               # Test configuration and pythonpath mappings
├── requirements.txt         # Production and development dependencies
└── README.md                # Project documentation
```

---

## Getting Started

### Prerequisites

- Python 3.10+ (tested on Python 3.13)
- Windows, Linux, or macOS
- *Optional for GPU Acceleration*: NVIDIA GPU with Compute Capability $\ge 5.0$, NVIDIA Display Driver, and CUDA Toolkit (tested on Ampere RTX 2050 Laptop GPU with CUDA 13.4).

### Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Pipeline

Execute the end-to-end alignment pipeline via the entry script:

```bash
# Basic run with default mock genome and CPU backend:
python scripts/run_pipeline.py

# Run with microsecond timings breakdown:
python scripts/run_pipeline.py --timed

# Run with GPU CUDA backend and shared-memory target caching:
python scripts/run_pipeline.py --backend cuda --shared-mem --timed

# Run with auto-backend selection (CUDA if available, otherwise CPU):
python scripts/run_pipeline.py --backend auto --shared-mem --timed

# Run with explicit fallback to CPU if CUDA fails or is unavailable:
python scripts/run_pipeline.py --backend cuda --fallback-to-cpu --timed

# Run with a custom FASTA file and target sequence:
python scripts/run_pipeline.py --fasta data/mock_genome.fasta --target ATGCGTACGT --chunk-size 10 --timed
```

---

## Running Tests

Run the complete test suite across all modules:

```bash
# Run all test suites:
pytest tests/ src/tests/ -v

# Run only CUDA alignment unit tests:
pytest tests/test_cuda_alignment.py -v

# Run pipeline integration tests:
pytest tests/test_gpu_pipeline_integration.py -v

# Run Dask distributed tests:
pytest tests/test_dask_processor.py -v
```

> **Note on CUDA Tests:** If an NVIDIA GPU or compiler toolchain (`nvvm.dll`) is not present in the runtime environment, CUDA-specific kernel tests are safely detected and skipped without failure or false reporting. On systems without GPU hardware, CUDA kernel logic can also be tested using Numba's official CUDA software simulator (`$env:NUMBA_ENABLE_CUDASIM = "1"`).

---

## Running Benchmarks

Run individual benchmark suites or the entire benchmark suite:

```bash
# Run CPU baseline benchmark:
python scripts/run_benchmarks.py cpu

# Run Dask distributed benchmark:
python scripts/run_benchmarks.py dask

# Run CUDA baseline & shared-memory comparative benchmark:
python scripts/run_benchmarks.py cuda

# Run multi-backend comparative benchmark (CPU vs CUDA vs Dask):
python scripts/run_benchmarks.py comparative

# Run all suites sequentially:
python scripts/run_benchmarks.py all
```
