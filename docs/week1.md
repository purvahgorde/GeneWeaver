# Week 1 -- CPU DNA Alignment Baseline

## Overview

Week 1 establishes the CPU-only DNA sequence alignment reference
implementation.  This baseline is intentionally simple and free of
GPU (CUDA / Numba) or distributed (Dask) dependencies so that it can
serve as the correctness oracle when those accelerated back-ends are
introduced in Week 2+.

## Implementation

| File | Purpose |
|------|---------|
| `src/alignment/cpu_alignment.py` | Core CPU alignment (`align_sequence`, `align_sequences`, `AlignmentResult`) |
| `src/alignment/__init__.py` | Package init -- re-exports public API |
| `src/tests/test_cpu_alignment.py` | 56 pytest correctness tests |
| `benchmarks/cpu_benchmark.py` | Benchmark script for baseline timing |

### Algorithm

Character-by-character comparison of two equal-length DNA strings:

1. Validate inputs (type, alphabet, equal length).
2. Normalise to upper-case.
3. Iterate positions; count matches / mismatches; record mismatch indices.
4. Compute score = matches / alignment_length.
5. Return an immutable `AlignmentResult` dataclass.

### Input / Output Contract

**Input**: Two equal-length Python `str` of DNA bases (A, C, G, T).
Lower-case is auto-normalised.

**Output**: Frozen `AlignmentResult` dataclass:
- `matches` (int)
- `mismatches` (int)
- `alignment_length` (int)
- `score` (float) -- matches / alignment_length
- `mismatch_positions` (list[int]) -- 0-based

## Test Results

**Framework**: pytest 9.1.1
**Tests collected**: 56
**Tests passed**: 56 / 56

### Test Categories

| Category | Count | Description |
|----------|-------|-------------|
| Exact match | 6 | Various lengths including homopolymer, all 4 bases |
| Mismatch | 9 | First/middle/last position, multiple, alternating, total |
| Short sequences | 3 | 1-3 bp inputs |
| Empty sequence | 2 | Both empty strings, result type check |
| Invalid input | 13 | Wrong types, bad chars, whitespace, unequal lengths |
| Case insensitivity | 3 | Lower, mixed, mismatch through case |
| Boundary cases | 5 | Homopolymer, score precision, 10K-base single mismatch |
| Batch alignment | 6 | Multiple pairs, order, single, empty, type, error propagation |
| Contract invariants | 6 | Arithmetic, score bounds, consistency, immutability, determinism |
| Import verification | 3 | Module, package, constant |

**Command**:
```
cd src && python -m pytest tests/test_cpu_alignment.py -v --tb=short
```

## Benchmark Methodology

- **Timing**: `time.perf_counter()` (highest-resolution wall-clock timer).
- **Sequences**: Randomly generated DNA with a fixed seed (42) for reproducibility.
- **Mutation rate**: 10% (realistic for variant analysis).
- **Setup excluded**: Sequence generation time is not included in the measured alignment time.
- **Configurations**: 4 input sizes, each measured for both per-pair and batch API.

**Command**:
```
python benchmarks/cpu_benchmark.py
```

## Benchmark Results

### Environment

| Property | Value |
|----------|-------|
| Python | 3.13.15 (MSC v.1944 64 bit AMD64) |
| Platform | Windows-11-10.0.26200-SP0 |
| Processor | Intel64 Family 6 Model 141 Stepping 1, GenuineIntel |

### Per-pair benchmark (align_sequence called individually)

| Seq Length | Pairs | Total Bases | Wall Time | Per Pair | Bases/sec | Avg Score |
|------------|-------|-------------|-----------|----------|-----------|-----------|
| 100 | 1,000 | 100,000 | 8.34 ms | 8.3 us | 11,983,941 | 0.9008 |
| 1,000 | 1,000 | 1,000,000 | 64.10 ms | 64.1 us | 15,601,111 | 0.9002 |
| 10,000 | 100 | 1,000,000 | 69.13 ms | 691.3 us | 14,465,772 | 0.8999 |
| 100,000 | 10 | 1,000,000 | 70.50 ms | 7.05 ms | 14,183,411 | 0.8997 |

### Batch benchmark (align_sequences called once per config)

| Seq Length | Pairs | Total Bases | Wall Time | Per Pair | Bases/sec | Avg Score |
|------------|-------|-------------|-----------|----------|-----------|-----------|
| 100 | 1,000 | 100,000 | 6.83 ms | 6.8 us | 14,648,581 | 0.9008 |
| 1,000 | 1,000 | 1,000,000 | 64.57 ms | 64.6 us | 15,487,452 | 0.9002 |
| 10,000 | 100 | 1,000,000 | 67.53 ms | 675.3 us | 14,807,948 | 0.8999 |
| 100,000 | 10 | 1,000,000 | 66.45 ms | 6.65 ms | 15,048,864 | 0.8997 |

### Summary

| Metric | Value |
|--------|-------|
| Total bases aligned (per-pair) | 3,100,000 |
| Total wall-clock time | 212.08 ms |
| Overall throughput | ~14.6 million bases/sec |

### Observations

- Throughput is ~14-16 million bases/sec on a single CPU core in pure Python.
- The short-sequence (100 bp) configuration shows lower throughput due to
  per-call overhead dominating.  For >= 1000 bp the throughput stabilises.
- The batch API (`align_sequences`) shows marginally better performance for
  short sequences due to reduced Python function-call overhead.
- Average alignment scores (~0.90) are consistent with the 10% mutation rate.

## Integration with Member 2 (Parser / Chunker)

**Status**: Not yet possible.

As of Week 1, no parser, chunker, FASTA reader, or scoring module has been
committed by Member 2.  The CPU alignment module is designed to accept the
simplest possible interface (two equal-length `str`) so that integration
will be straightforward once the parser produces `(reference, query)` pairs.

### Expected Integration Interface

```python
# Member 2 parser produces pairs:
pairs: list[tuple[str, str]] = parser.parse_fasta("input.fasta", reference_seq)

# Member 1 alignment consumes them:
results: list[AlignmentResult] = align_sequences(pairs)
```

If Member 2's parser produces a different data structure (e.g. numpy arrays,
bytes, or variable-length sequences with gap characters), the alignment
module's `_validate_sequence()` function will need to be adapted.

## Limitations

1. **Pure Python**: No C extensions, no vectorisation.  This is the
   unoptimised baseline; the CUDA version (Week 2) will target 100-1000x
   speedup.
2. **Equal-length only**: Sequences must be pre-aligned / pre-chunked to
   the same length.  Gap-aware alignment (Smith-Waterman / Needleman-Wunsch)
   is out of scope for this baseline.
3. **ACGT only**: Ambiguity codes (N, R, Y, ...) are rejected.  Can be
   extended later if needed for CRISPR guide sequences.
4. **Single-threaded**: No multiprocessing.  Parallelism will come via
   CUDA (Week 2) and Dask (Week 3+).
