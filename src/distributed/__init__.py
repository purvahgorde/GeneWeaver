"""GeneWeaver Distributed Processing Module.

Provides distributed DNA sequence alignment using Dask task scheduling.
"""

from .dask_processor import (
    DaskGenomeProcessor,
    DaskNotAvailableError,
    DaskTaskError,
    dask_align_chunks,
    dask_align_chunk_pairs,
    dask_align_sequences,
    process_fasta_dask,
    process_fasta_dask_timed,
    is_dask_available,
    get_dask_info,
)

__all__ = [
    "DaskGenomeProcessor",
    "DaskNotAvailableError",
    "DaskTaskError",
    "dask_align_chunks",
    "dask_align_chunk_pairs",
    "dask_align_sequences",
    "process_fasta_dask",
    "process_fasta_dask_timed",
    "is_dask_available",
    "get_dask_info",
]
