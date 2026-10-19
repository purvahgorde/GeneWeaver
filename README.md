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

# Run with a custom FASTA file and target sequence:
python scripts/run_pipeline.py --fasta data/mock_genome.fasta --target ATGCGTACGT --chunk-size 10 --timed
```

---

## Running Tests

Run the complete test suite across all modules:

```bash
pytest -v
```

> **Note on CUDA Tests:** If an NVIDIA GPU or compiler toolchain (`nvvm.dll`) is not present in the runtime environment, CUDA-specific kernel tests are safely detected and skipped without failure or false reporting.

---

## Running Benchmarks

Run individual benchmark suites or the entire benchmark suite:

```bash
# Run CPU baseline benchmark:
python scripts/run_benchmarks.py cpu

# Run Dask distributed benchmark:
python scripts/run_benchmarks.py dask

# Run CUDA benchmark:
python scripts/run_benchmarks.py cuda

# Run all suites sequentially:
python scripts/run_benchmarks.py all
```
