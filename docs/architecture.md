# GeneWeaver System Architecture

## 1. System Overview

GeneWeaver is a high-throughput, multi-backend genomic sequence alignment system designed specifically for CRISPR guide RNA off-target search and genomic candidate window evaluation. It bridges BioPython sequence ingestion, unaccelerated CPU reference verification, Numba CUDA GPU acceleration, and Dask distributed orchestration under a single, deterministic contract.

```
                      ┌────────────────────────────┐
                      │    FASTA Input Dataset     │
                      │  (BioPython SeqIO parser)  │
                      └─────────────┬──────────────┘
                                    │ sequences: list[str]
                                    ▼
                      ┌────────────────────────────┐
                      │       Genome Chunker       │
                      │ (windowing / partitioner)  │
                      └─────────────┬──────────────┘
                                    │ chunks: list[str]
                                    ▼
                      ┌────────────────────────────┐
                      │    Alignment Dispatcher    │
                      │    ('cpu', 'cuda', 'dask') │
                      └──────┬──────┬──────┬───────┘
                             │      │      │
            ┌────────────────┘      │      └─────────────────┐
            ▼                       ▼                        ▼
  ┌───────────────────┐   ┌────────────────────┐   ┌───────────────────┐
  │   CPU Reference   │   │     Numba CUDA     │   │  Dask Distributed │
  │      Oracle       │   │   (Shared Memory   │   │ (Local / Cluster  │
  │ (align_sequence)  │   │     Optimized)     │   │   Auto-Tuned)     │
  └─────────┬─────────┘   └─────────┬──────────┘   └─────────┬─────────┘
            │                       │                        │
            └───────────────────────┼────────────────────────┘
                                    │
                                    ▼
                      ┌────────────────────────────┐
                      │  list[AlignmentResult]     │
                      │   (Matches, Mismatches,    │
                      │   Score, Exact Positions)  │
                      └────────────────────────────┘
```

---

## 2. Component Architecture

### 2.1 Parser (`src/parser/fasta_parser.py`)
- Reads raw multi-record FASTA files using BioPython `SeqIO.parse`.
- Returns in-memory lists of raw DNA sequence strings.
- Handles empty FASTA files cleanly (`[]`) and raises `FileNotFoundError` on missing paths.

### 2.2 Genome Chunker (`src/chunking/genome_chunker.py`)
- Slices DNA sequences into windows of size `chunk_size`.
- Residual tail chunks at chromosome boundaries are preserved.

### 2.3 CPU Reference Oracle (`src/alignment/cpu_alignment.py`)
- Provides deterministic, character-by-character alignment baseline.
- Validates canonical IUPAC uppercase bases (`A`, `C`, `G`, `T`).
- Serves as the unaccelerated reference oracle that GPU and distributed implementations must match 100%.
- Returns immutable `AlignmentResult` dataclass instances.

### 2.4 CUDA GPU Acceleration Engine (`src/alignment/cuda_alignment.py`)
- **Baseline Kernel (`dna_align_kernel`)**: 1D thread grid where each thread processes one pair.
- **Shared-Memory Kernel (`dna_align_shared_kernel`)**: Cooperatively loads broadcast target sequence into fast L1 shared memory per block (`s_target`), eliminating 128x redundant DRAM reads per position and cutting host-to-device memory transfer volume by 50%.
- **Host Preparation (`_prepare_host_buffers`, `_prepare_host_buffers_target`)**: Packs DNA strings into contiguous ASCII uint8 arrays.
- **Diagnostics & Safe Fallback**: When an NVIDIA GPU is present but `libNVVM/nvvm.dll` is missing, detects the condition and raises `CUDANotAvailableError` truthfully without fake numbers or crashed processes.

### 2.5 Dask Distributed Processor (`src/distributed/dask_processor.py`)
- **Task Decomposition**: Splits chunk batches across worker threads or processes.
- **Dynamic Batch Auto-Tuning (`calculate_optimal_batch_size`)**: Dynamically sizes tasks to match worker pool capacity:
  $$\text{target\_tasks} = \max(1, \text{workers} \times \text{tasks\_per\_worker})$$
- **Resource Management**: Distinguishes between shared clients (`owns_client=False`) and internally spawned `LocalCluster` instances (`owns_client=True`, `owns_cluster=True`). Local schedulers (`scheduler="threads"`) bypass cluster allocation entirely.
- **Reliability & Incomplete-Result Detection**: Tracks `_is_closed` state, wraps worker crashes in `DaskTaskError`, and verifies that returned results strictly match expected counts.

### 2.6 Biological Scoring (`src/scoring/biological_scoring.py` & `src/scoring/dna_scoring.py`)
- **Base & Sequence Scoring**: Validates IUPAC canonical bases, scoring base pairs (+1 match, -1 mismatch) and sequences.
- **CRISPR PAM Detection & Scoring (`src/pam/pam_detector.py`, `src/analysis/pam_scoring.py`)**: Locates SpCas9 `NGG` PAM motifs and computes alignment affinity against target protospacers.
- **Result Metrics & GC Content**: Computes aggregate biological metrics (mean score, percent similarity, total matches/mismatches) and sequence GC content percentages.

### 2.7 Mismatch Visualization (`src/visualization/mismatch_visualizer.py`)
- **Rich Terminal Highlighting (`format_dna_alignment`)**: Produces formatted Rich `Text` objects highlighting matches in green and mismatches in red with 1-based mismatch position summaries.
- **ASCII Rendering (`render_mismatch_ascii`)**: Produces lightweight 3-line terminal alignment displays with `|` matching markers and `.` mismatch markers.

### 2.8 Interactive Terminal Dashboard (`src/dashboard/textual_dashboard.py` & `src/ui/dashboard.py`)
- **Textual UI Application (`GeneWeaverApp`)**: Full-screen TUI providing real-time sequence loading, progress tracking, match/mismatch statistics, alignment score metrics, and colored mismatch visualization.

---

## 3. Data Contracts & Invariants

All alignment engines (CPU, CUDA baseline, CUDA shared memory, and Dask) adhere strictly to the `AlignmentResult` contract:

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
1. `matches + mismatches == alignment_length == len(target)`
2. `len(mismatch_positions) == mismatches`
3. `score == matches / alignment_length` (or `0.0` for empty sequences)
4. `0 <= score <= 1.0`
5. `mismatch_positions` are strictly ascending 0-based indices.

---

## 4. Execution Workflows

### Command Line:
```bash
# Run CPU pipeline:
python scripts/run_pipeline.py --fasta data/mock_genome.fasta --target ATGCGTACGT --chunk-size 10 --timed

# Run Dask benchmarks:
python scripts/run_benchmarks.py dask

# Run CUDA benchmarks / diagnostics:
python scripts/run_benchmarks.py cuda
```

### Programmatic Python API:
```python
from alignment import align_chunks, cuda_align_chunks_shared
from distributed import DaskGenomeProcessor, process_fasta_dask

# CPU alignment
cpu_results = align_chunks(chunks, target)

# Dask distributed processing
with DaskGenomeProcessor(scheduler="threads", default_batch_size="auto") as proc:
    dask_results = proc.align_chunks(chunks, target)

# End-to-end FASTA pipeline
results = process_fasta_dask("data/mock_genome.fasta", "ATGCGTACGT", chunk_size=10)
```
