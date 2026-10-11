# GeneWeaver
GeneWeaver is a GPU-accelerated CRISPR alignment engine for bioinformatics and high-performance computing. It combines BioPython for FASTA parsing, Numba CUDA for parallel sequence alignment, Dask for distributed processing, and Textual for a real-time terminal dashboard and visualization.

## Current Status (Week 3 -- Distributed Dask Processing & Pipeline Integration)

Week 3 establishes the Dask distributed task-scheduling layer for parallel genome chunk alignment, fully integrated with Member 2's FASTA parser and genome chunker while maintaining 100% equivalence with the CPU reference oracle.

### Quickstart

```bash
# Run complete test suite (191 tests: 169 passed, 22 skipped)
pytest -v

# Run CPU alignment benchmark
python benchmarks/cpu_benchmark.py

# Run Dask distributed alignment benchmark
python benchmarks/dask_benchmark.py

# Run end-to-end CPU/CUDA integration pipeline
python src/alignment/integration.py
```
