# Input Data Guidance

This directory contains reproducible input datasets and test genomes for GeneWeaver.

## Formats Supported
- **FASTA** (`.fasta`, `.fa`): Standard genomic sequence files parsed using `Bio.SeqIO` in `src.parser.fasta_parser`.

## Included Mock Datasets

### 1. `mock_genome.fasta`
Minimal two-chromosome synthetic test genome used for fast testing, integration checks, and CI validation.
- `chr1`: 45 bp (`ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC`)
- `chr2`: 41 bp (`CGTACGTAGCTAGCTAACGTACGTAGCTAGCTAGCTAGCTA`)
- Standard Target: `ATGCGTACGT` (10-mer)

### 2. `demo_crispr.fasta`
Small, realistic CRISPR guide RNA candidate evaluation dataset designed for reproducible end-to-end demonstrations across CPU, CUDA, and Dask backends.
- Target Protospacer: `GAGTCCGAGCAGAAGAAGAA` (20-mer, SpCas9 target)
- `chr1_ontarget`: 23 bp (`GAGTCCGAGCAGAAGAAGAATGG`), perfect match with canonical SpCas9 `TGG` PAM.
- `chr2_offtarget_1mm`: 23 bp (`GAGTCCGAGCAGAAGAAGACTGG`), single mismatch at position 20 with `TGG` PAM.
- `chr3_offtarget_2mm`: 23 bp (`GAGTCCGAGCAGAACAAGACTGG`), two mismatches (positions 15 and 20) with `TGG` PAM.
- `chr4_control`: 23 bp (`AAAAAAAAAAAAAAAAAAAAAGG`), non-target negative control with `AGG` PAM.

## Adding Custom Genomes
When adding custom genome files:
1. Ensure sequences contain standard DNA IUPAC characters (`A`, `C`, `G`, `T`).
2. Provide valid header lines starting with `>`.
3. Large custom datasets should be excluded from Git version tracking (ignored via `.gitignore`) and stored locally.
