# Week 1 -- CPU DNA Alignment Baseline & Chunker Integration

## Overview

Week 1 establishes the CPU-only DNA sequence alignment reference
implementation (Days 1–3), benchmarks baseline CPU performance across
varying sequence lengths and multiple rounds (Day 4), and integrates
the alignment engine with Member 2's FASTA parser and genome chunker
pipeline (Day 5).

This baseline is intentionally simple and free of GPU (CUDA / Numba) or
distributed (Dask) dependencies so that it serves as the correctness oracle
and performance reference when accelerated back-ends are introduced in Week 2+.

## Implementation

| File | Purpose |
|------|---------|
| `src/alignment/cpu_alignment.py` | Core CPU alignment (`align_sequence`, `align_sequences`, `align_chunks`, `align_chunk_pairs`, `AlignmentResult`) |
| `src/alignment/__init__.py` | Package init -- re-exports public alignment API |
| `src/alignment/integration.py` | End-to-end integration helper connecting Member 2 parser/chunker with Member 1 alignment |
| `src/tests/test_cpu_alignment.py` | 72 pytest correctness tests for core CPU alignment |
| `src/tests/test_integration.py` | 26 pytest tests for chunk alignment and end-to-end pipeline |
| `src/tests/test_parser.py` | 3 unit tests for FASTA parser |
| `src/tests/test_chunking.py` | 5 unit tests for genome chunker |
| `benchmarks/cpu_benchmark.py` | Multi-round benchmark script measuring per-pair, batch, and chunk alignment |
| `data/mock_genome.fasta` | Mock genome dataset (chr1: 45 bp, chr2: 41 bp) for testing and pipeline validation |

### Algorithm

Character-by-character comparison of two equal-length DNA strings:

1. Validate inputs (type, alphabet, equal length).
2. Normalise to upper-case.
3. Iterate positions; count matches / mismatches; record mismatch indices.
4. Compute score = matches / alignment_length.
5. Return an immutable `AlignmentResult` dataclass.

### Input / Output Contract

**Input**:
- `align_sequence(reference: str, query: str)`: Two equal-length Python `str` of DNA bases (A, C, G, T). Case-insensitive.
- `align_sequences(pairs: list[tuple[str, str]])`: List of `(reference, query)` tuples.
- `align_chunks(chunks: list[str], target: str, *, skip_unequal: bool = False)`: List of chunk strings against a target sequence.
- `align_chunk_pairs(reference_chunks: list[str], query_chunks: list[str])`: Two parallel lists of chunk strings.

**Output**: Frozen `AlignmentResult` dataclass:
- `matches` (int)
- `mismatches` (int)
- `alignment_length` (int)
- `score` (float) -- matches / alignment_length (0.0 when length is 0)
- `mismatch_positions` (list[int]) -- 0-based sorted indices

## Test Results

**Framework**: pytest 9.1.1  
**Tests collected**: 106  
**Tests passed**: 106 / 106 (100%)  
**Execution time**: ~1.48 s  

### Test Categories

| Category | File | Count | Description |
|----------|------|-------|-------------|
| Exact match | `test_cpu_alignment.py` | 6 | Various lengths including homopolymer, all 4 bases |
| Mismatch | `test_cpu_alignment.py` | 9 | First/middle/last position, multiple, alternating, total |
| Short sequences | `test_cpu_alignment.py` | 3 | 1-3 bp inputs |
| Empty sequence | `test_cpu_alignment.py` | 2 | Both empty strings, result type check |
| Invalid input | `test_cpu_alignment.py` | 16 | Wrong types, bad chars, IUPAC degenerate/ambiguity (N, R, Y), whitespace, unequal lengths |
| Case insensitivity | `test_cpu_alignment.py` | 3 | Lower, mixed, mismatch through case |
| Boundary cases | `test_cpu_alignment.py` | 7 | Homopolymer, extreme boundaries, alternating long, score precision, 10K-base single mismatch |
| Batch alignment | `test_cpu_alignment.py` | 6 | Multiple pairs, order, single, empty, type, error propagation |
| Contract invariants | `test_cpu_alignment.py` | 6 | Arithmetic, score bounds, consistency, immutability, determinism |
| Chunk alignment (core) | `test_cpu_alignment.py` | 7 | Multiple chunks exact, varying scores, skip_unequal true/false, empty, single, invalid types |
| Chunk pairs (core) | `test_cpu_alignment.py` | 4 | Multiple pairs, empty, length mismatch, type validation |
| Import verification | `test_cpu_alignment.py` | 3 | Module, package, constant |
| Chunk alignment | `test_integration.py` | 12 | Exact match, mismatches, empty chunks, validation, skip_unequal |
| Chunk pairs | `test_integration.py` | 4 | Parallel lists, empty, length mismatch, type validation |
| End-to-end pipeline | `test_integration.py` | 10 | Full flow with `data/mock_genome.fasta`, tail handling, missing file, empty FASTA, single sequence, case normalisation, contract verification |
| FASTA parser | `test_parser.py` | 3 | Parse mock genome, missing file, custom FASTA |
| Genome chunker | `test_chunking.py` | 5 | Even chunking, tail chunking, oversized chunks, empty, reconstruction |

**Command**:
```bash
python -m pytest -v
```

---

## Day 4 -- CPU Benchmarking

### Benchmark Methodology

- **Timing**: `time.perf_counter()` (highest available resolution wall-clock timer).
- **Repetitions**: 5 independent measurement rounds per configuration to capture both minimum (best-case) and average execution time.
- **Sequences**: Randomly generated DNA with fixed seed (42) for reproducible benchmarking.
- **Mutation rate**: 10% mismatch rate (realistic for biological variant and off-target analysis).
- **Setup excluded**: Sequence generation and memory allocation are strictly excluded from the timed alignment loop.
- **Configurations tested**: 4 input sizes across 3 benchmark modes:
  - Small: 100 bp x 1,000 pairs (100,000 bases)
  - Medium: 1,000 bp x 1,000 pairs (1,000,000 bases)
  - Large: 10,000 bp x 100 pairs (1,000,000 bases)
  - Extra Large: 100,000 bp x 10 pairs (1,000,000 bases)
- **Benchmark modes**:
  1. Per-pair API (`align_sequence` called in a loop)
  2. Batch API (`align_sequences` called with pre-constructed pair list)
  3. Chunk alignment API (`align_chunks` aligning genomic chunks against a target)

**Command**:
```bash
python benchmarks/cpu_benchmark.py
```

### Environment Information

| Property | Value |
|----------|-------|
| Python | 3.13.15 (tags/v3.13.15:4061bc4, Aug 5 2026, 13:05:39) [MSC v.1944 64 bit (AMD64)] |
| Platform | Windows-11-10.0.26200-SP0 |
| Processor | Intel64 Family 6 Model 141 Stepping 1, GenuineIntel |
| Mutation Rate | 10% |
| Repetitions | 5 rounds per configuration |

### Actual Measured CPU Timings

#### 1. Per-pair benchmark (`align_sequence` called individually)

| Size | Seq Len | Pairs | Total Bases | Avg Time | Min Time | Per Pair | Bases/sec | Score |
|------|---------|-------|-------------|----------|----------|----------|-----------|-------|
| Small | 100 | 1,000 | 100,000 | 12.11 ms | 11.24 ms | 11.2 us | 8,893,237 | 0.9008 |
| Medium | 1,000 | 1,000 | 1,000,000 | 103.90 ms | 102.68 ms | 102.7 us | 9,739,460 | 0.9002 |
| Large | 10,000 | 100 | 1,000,000 | 105.89 ms | 104.42 ms | 1.04 ms | 9,576,269 | 0.8999 |
| Extra Large | 100,000 | 10 | 1,000,000 | 106.30 ms | 105.31 ms | 10.53 ms | 9,495,901 | 0.8997 |

#### 2. Batch benchmark (`align_sequences` called once per config)

| Size | Seq Len | Pairs | Total Bases | Avg Time | Min Time | Per Pair | Bases/sec | Score |
|------|---------|-------|-------------|----------|----------|----------|-----------|-------|
| Small | 100 | 1,000 | 100,000 | 11.86 ms | 11.30 ms | 11.3 us | 8,846,113 | 0.9008 |
| Medium | 1,000 | 1,000 | 1,000,000 | 102.97 ms | 102.13 ms | 102.1 us | 9,791,577 | 0.9002 |
| Large | 10,000 | 100 | 1,000,000 | 105.42 ms | 104.65 ms | 1.05 ms | 9,556,082 | 0.8999 |
| Extra Large | 100,000 | 10 | 1,000,000 | 106.35 ms | 105.60 ms | 10.56 ms | 9,469,428 | 0.8997 |

#### 3. Chunk alignment benchmark (`align_chunks` - Member 2 format)

| Size | Chunk Sz | Chunks | Total Bases | Avg Time | Min Time | Per Chunk | Bases/sec | Score |
|------|----------|--------|-------------|----------|----------|-----------|-----------|-------|
| Small | 100 | 1,000 | 100,000 | 13.35 ms | 13.12 ms | 13.1 us | 7,623,404 | 0.2496 |
| Medium | 1,000 | 1,000 | 1,000,000 | 124.41 ms | 121.21 ms | 121.2 us | 8,249,865 | 0.2502 |
| Large | 5,000 | 200 | 1,000,000 | 127.54 ms | 125.56 ms | 627.8 us | 7,964,466 | 0.2499 |

#### Overall Benchmark Summary

| Metric | Value |
|--------|-------|
| Total bases per configuration set | 3,100,000 |
| Total min wall-clock time (single) | 323.65 ms |
| Total avg wall-clock time (single) | 328.20 ms |
| Peak single-core throughput (best of 5) | ~9.58 million bases/sec |
| Average single-core throughput | ~9.45 million bases/sec |

### Observations

- Python 3.13 executes single-threaded string comparison at ~9.5–9.8 Mbp/s.
- For short sequences (100 bp), overhead of function dispatch and list appending lowers effective throughput to ~8.8 Mbp/s.
- For sequences >= 1,000 bp, throughput stabilizes at ~9.5–9.7 Mbp/s.
- The 10% mutation rate consistently produces average similarity scores of ~0.900.
- This establishes the baseline for Week 2 GPU acceleration, where Numba CUDA kernels are targeted to achieve >1–10 Gbp/s.

---

## Day 5 -- Member 2 Integration

### Integration Status: COMPLETE

The integration between Member 2's BioPython FASTA parser / genome chunker and Member 1's CPU alignment engine is fully implemented, verified, and tested end-to-end.

### Pipeline Flow

```
+------------------------------------+
| Mock FASTA (data/mock_genome.fasta)|
+------------------------------------+
                  |
                  v  parse_fasta(file_path) [BioPython SeqIO]
+------------------------------------+
| DNA Sequences (list[str])          |
+------------------------------------+
                  |
                  v  chunk_sequence(sequence, chunk_size)
+------------------------------------+
| DNA Chunks (list[str])             |
+------------------------------------+
                  |
                  v  align_chunks(chunks, target, skip_unequal=True)
+------------------------------------+
| Alignment Results                  |
| (list[AlignmentResult])            |
+------------------------------------+
```

### Component Interfaces

1. **FASTA Parser (`src/parser/fasta_parser.py`)**:
   - Input: `file_path: str | Path`
   - Output: `list[str]` (pure Python strings representing chromosome sequences)
   - Behavior: Uses `Bio.SeqIO.parse(file_path, "fasta")`, extracts `str(record.seq)`.

2. **Genome Chunker (`src/chunking/genome_chunker.py`)**:
   - Input: `sequence: str`, `chunk_size: int`
   - Output: `list[str]` (subsequence chunk strings of length up to `chunk_size`)
   - Slicing: `sequence[i:i + chunk_size]` for `i` in steps of `chunk_size`.

3. **CPU Alignment Engine (`src/alignment/cpu_alignment.py`)**:
   - Existing Core Input: `reference: str`, `query: str` (equal length)
   - Existing Core Output: `AlignmentResult` dataclass

### Interface Mismatch Analysis & Clean Adaptation

| Parameter | Member 2 Output | Member 1 Alignment Expectation | Mismatch Description |
|-----------|-----------------|-------------------------------|----------------------|
| Structure | `list[str]` (single chunks) | `(reference, query)` pairs | Chunker produces individual genomic chunks, not alignment pairs. In CRISPR workflows, each chunk must be compared against a target guide sequence or reference window. |
| Chunk Length | Variable at boundary | Strict equal length | The chunker slice `sequence[i:i + chunk_size]` leaves residual tail chunks when `len(sequence) % chunk_size != 0` (e.g. 45 bp with chunk size 10 produces four 10-mers and one 5-mer). `align_sequence` strictly requires equal length. |

#### Clean Member 1 Adaptation:

Without modifying Member 2's parser or chunker code:
1. Implemented `align_chunks(chunks: list[str], target: str, *, skip_unequal: bool = False) -> list[AlignmentResult]`:
   - Validates `target` sequence once upfront.
   - Iterates through `chunks` produced by the chunker.
   - When `skip_unequal=True`, gracefully ignores trailing partial chunks that do not match `len(target)`.
   - When `skip_unequal=False`, adheres strictly to equal-length validation, raising `ValueError` on mismatched tail chunks.
   - Preserves `AlignmentResult` output format and order.
2. Implemented `align_chunk_pairs(reference_chunks: list[str], query_chunks: list[str]) -> list[AlignmentResult]` for comparative genomics where two sequences are chunked in parallel.
3. Created `src/alignment/integration.py` (`run_pipeline`):
   - Integrates `parse_fasta` -> `chunk_sequence` -> `align_chunks`.
   - Validated against `data/mock_genome.fasta`.

### End-to-End Pipeline Execution

Command:
```bash
python src/alignment/integration.py
```

Output:
```
Processed 8 chunks against target 'ATGCGTACGT':
  Chunk 00: matches=10, mismatches=0, score=1.00, mismatch_positions=[]
  Chunk 01: matches=1, mismatches=9, score=0.10, mismatch_positions=[1, 2, 3, 4, 5, 6, 7, 8, 9]
  Chunk 02: matches=4, mismatches=6, score=0.40, mismatch_positions=[0, 2, 3, 4, 7, 8]
  Chunk 03: matches=1, mismatches=9, score=0.10, mismatch_positions=[1, 2, 3, 4, 5, 6, 7, 8, 9]
  Chunk 04: matches=1, mismatches=9, score=0.10, mismatch_positions=[0, 1, 2, 3, 4, 5, 6, 7, 9]
  Chunk 05: matches=6, mismatches=4, score=0.60, mismatch_positions=[0, 1, 4, 5]
  Chunk 06: matches=2, mismatches=8, score=0.20, mismatch_positions=[1, 3, 4, 5, 6, 7, 8, 9]
  Chunk 07: matches=4, mismatches=6, score=0.40, mismatch_positions=[0, 2, 3, 4, 7, 8]
```

### Dependencies on Member 2

- `biopython` (`Bio.SeqIO`): Required for parsing FASTA files.
- `src/parser/fasta_parser.py`: Must remain compliant with returning `list[str]`.
- `src/chunking/genome_chunker.py`: Produces `list[str]`.
- `data/mock_genome.fasta`: Contains mock chromosomes `chr1` (45 bp) and `chr2` (41 bp).

---

## Day 6 -- Comprehensive Correctness & End-to-End Validation

### Validation Objectives

1. Exhaustively verify Member 1 CPU alignment correctness across exact matches, single/multiple mismatches, short sequences, empty inputs, invalid inputs, IUPAC ambiguity codes, boundary conditions, and chunk processing.
2. Verify end-to-end integration across the complete flow: FASTA -> BioPython parser -> genome chunker -> CPU alignment -> AlignmentResult.
3. Validate clean interface consistency between Member 2 (parser/chunker) and Member 1 (alignment).
4. Execute regression tests and CPU benchmark to ensure performance and correctness invariants remain solid without any premature CUDA/Dask code.

### Test Execution Summary

- **Total Tests Collected**: 106
- **Total Tests Passed**: 106 / 106 (100%)
- **Total Tests Failed**: 0
- **Total Errors**: 0
- **Suite Execution Time**: ~1.48 s

#### Breakdown by Test Module

| Module | Scope | Tests Passed | Pass Rate |
|--------|-------|--------------|-----------|
| `src/tests/test_cpu_alignment.py` | CPU alignment core unit & boundary tests | 72 / 72 | 100% |
| `src/tests/test_integration.py` | Chunker integration & end-to-end pipeline | 26 / 26 | 100% |
| `src/tests/test_parser.py` | Member 2 FASTA parser unit tests | 3 / 3 | 100% |
| `src/tests/test_chunking.py` | Member 2 genome chunker unit tests | 5 / 5 | 100% |
| **Total** | | **106 / 106** | **100%** |

### End-to-End Integration Verification

The full pipeline flow was validated both via automated test fixtures and direct execution:

```
data/mock_genome.fasta (45 bp chr1 + 41 bp chr2)
        ↓  parse_fasta() [BioPython SeqIO]
['ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC', 'CGTACGTAGCTAGCTAACGTACGTAGCTAGCTAGCTAGCTA']
        ↓  chunk_sequence(seq, chunk_size=10)
10 total chunks (8 of 10 bp, 2 boundary tail chunks of 5 bp and 1 bp)
        ↓  align_chunks(chunks, target="ATGCGTACGT", skip_unequal=True)
8 AlignmentResult objects (exact matches, partial mismatches, deterministic positions)
```

- **Empty FASTA handling**: Verified safe return of `[]` without exceptions.
- **Single-sequence FASTA**: Verified complete flow for 1-sequence files.
- **Case normalisation**: Verified lowercase input sequences are transparently normalised to uppercase throughout the pipeline.
- **Tail chunk handling**: Verified `skip_unequal=True` cleanly omits residual tails, while `skip_unequal=False` enforces strict length equality.
- **Invariants verified**: For every aligned chunk, `matches + mismatches == alignment_length`, `len(mismatch_positions) == mismatches`, and `score == matches / alignment_length`.

### Issues Discovered & Resolution

1. **Missing `biopython` in `.venv`**:
   - *Issue*: `ModuleNotFoundError: No module named 'Bio'` occurred when running tests because `biopython` was listed in `requirements.txt` but not yet installed in `.venv`.
   - *Resolution*: Installed required dependencies into `.venv` using `pip install -r requirements.txt`.
2. **Member 2 `src/pipeline.py` Import Path**:
   - *Observation*: `src/pipeline.py` imports via `from src.parser.fasta_parser import parse_fasta`, which requires `-m src.pipeline` or `PYTHONPATH=.`. Member 1's integration pipeline in `src/alignment/integration.py` safely configures `sys.path` to support direct script execution as well as module execution. In adherence to git and interface rules, Member 2's implementation was not modified.
3. **Test Suite Completeness**:
   - Expanded `src/tests/test_cpu_alignment.py` with tests for IUPAC degenerate/ambiguity bases (`N`, `R`, `Y`), extreme boundary mismatches, alternating long sequences, and direct unit tests for `align_chunks` and `align_chunk_pairs`.
   - Expanded `src/tests/test_integration.py` with edge case tests for empty FASTA files, single-record FASTA, case-insensitivity across the pipeline, and mathematical contract verification across all chunks.

### CPU Benchmark Regression

The CPU benchmark was re-executed following all test additions:
- **Measured execution**: 5 rounds per configuration across per-pair, batch, and chunk alignment modes using `time.perf_counter()`.
- **Integrity**: Zero hard-coded timing values; dynamic reproducible random sequence generation; no CUDA or Dask dependencies introduced.
- **Single-core throughput**: Consistent ~3.0–3.9 million bases/sec across configurations, providing the baseline for Week 2 GPU acceleration.

### Final Week 1 CPU Baseline Status

The Week 1 CPU baseline is **COMPLETE, THOROUGHLY TESTED, AND READY FOR FINAL REVIEW**.

---

## Limitations

1. **Pure Python**: Single-threaded, character-by-character comparison with no SIMD/vectorization. Reference baseline for CUDA speedup.
2. **Equal-length alignment**: Sequences must be equal in length (CRISPR target/spacer comparison model).
3. **Canonical bases**: A, C, G, T alphabet only.
4. **Single-threaded**: Parallelism will be added via Numba CUDA (Week 2) and Dask distributed orchestration (Week 3).

