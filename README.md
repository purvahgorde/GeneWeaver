# GeneWeaver
GeneWeaver is a GPU-accelerated CRISPR alignment engine for bioinformatics and high-performance computing. It combines BioPython for FASTA parsing, Numba CUDA for parallel sequence alignment, Dask for distributed processing, and Textual for a real-time terminal dashboard and visualization.

## Current Status (Week 1 -- CPU Baseline)

Week 1 establishes the single-threaded CPU reference implementation and integration with the FASTA parser and genome chunker. This serves as the correctness oracle for Week 2 CUDA acceleration.

### Quickstart

```bash
# Run complete test suite (106 tests)
pytest -v

# Run CPU alignment benchmark (reproducible multi-round timing)
python benchmarks/cpu_benchmark.py

# Run end-to-end integration pipeline (FASTA -> Chunker -> Alignment)
python src/alignment/integration.py
```
