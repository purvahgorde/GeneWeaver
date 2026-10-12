# Input Data Guidance

This directory contains input datasets and test genomes for GeneWeaver.

## Formats Supported
- **FASTA** (`.fasta`, `.fa`): Standard genomic sequence files parsed using `Bio.SeqIO` in `src.parser.fasta_parser`.

## Included Mock Datasets
- `mock_genome.fasta`: Minimal two-chromosome synthetic test genome used for fast testing, integration checks, and CI validation.
  - `chr1`: 45 bp (`ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC`)
  - `chr2`: 41 bp (`CGTACGTAGCTAGCTAACGTACGTAGCTAGCTAGCTAGCTA`)

## Adding Custom Genomes
When adding custom genome files:
1. Ensure sequences contain standard DNA IUPAC characters (`A`, `C`, `G`, `T`, `N`).
2. Provide valid header lines starting with `>`.
3. Large custom datasets should be excluded from Git version tracking (ignored via `.gitignore`) and stored locally or downloaded via data fetch scripts.
