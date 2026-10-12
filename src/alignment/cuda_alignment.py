"""CUDA-accelerated DNA Sequence Alignment Implementation.

Week 2 Day 1: CUDA Environment Setup and Initial GPU Alignment Implementation.

This module provides the first working CUDA version of the DNA sequence
alignment operation using Numba CUDA, conforming to the exact contract
established by the Week 1 CPU reference implementation (AlignmentResult).

Architecture & Execution Flow:
    Host Input DNA Sequences
    -> Host-side Preparation (validation, ASCII/uint8 encoding, batch matrix flattening)
    -> Host-to-Device Memory Transfer (numba.cuda.to_device)
    -> CUDA Kernel Execution (dna_align_kernel grid launch)
    -> Device-to-Host Memory Transfer (copy_to_host)
    -> Result Reconstruction (identical AlignmentResult objects)

Hardware / Environment Fallback:
    If CUDA hardware or Numba CUDA driver support is unavailable or blocked by
    OS application control policy, functions cleanly raise CUDANotAvailableError
    without returning fake/simulated GPU results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import time
from typing import Sequence, Tuple

import numpy as np

# Import the existing AlignmentResult contract to guarantee 100% interoperability
from .cpu_alignment import AlignmentResult, _validate_sequence

# Determine Numba CUDA availability safely
CUDA_AVAILABLE: bool = False
CUDA_IMPORT_ERROR: str | None = None
cuda = None  # type: ignore

try:
    from numba import cuda as _cuda
    if _cuda.is_available():
        cuda = _cuda
        CUDA_AVAILABLE = True
    else:
        # Check if driver is available but compiler library (NVVM) is missing
        if hasattr(_cuda, "cudadrv") and hasattr(_cuda.cudadrv, "driver") and _cuda.cudadrv.driver.driver.is_available:
            CUDA_IMPORT_ERROR = (
                "Numba detected CUDA driver and device, but CUDA compiler library (libNVVM/nvvm.dll) is not available."
            )
        else:
            CUDA_IMPORT_ERROR = "Numba detected no available CUDA devices or driver library."
except Exception as _exc:  # pragma: no cover
    CUDA_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"


@dataclass(frozen=True)
class CUDATimingBreakdown:
    """Detailed timing breakdown for host/device transfers and GPU kernel execution.

    All times are reported in seconds.

    Attributes
    ----------
    host_prep_time : float
        Time spent on host-side validation, ASCII encoding, and array flattening.
    h2d_transfer_time : float
        Time spent copying host memory to GPU device memory (Host-to-Device).
    kernel_execution_time : float
        Time spent executing the CUDA kernel on the GPU (including sync).
    d2h_transfer_time : float
        Time spent copying results back from GPU to host memory (Device-to-Host).
    reconstruction_time : float
        Time spent building AlignmentResult Python objects.
    total_gpu_time : float
        Total time: h2d_transfer + kernel_execution + d2h_transfer.
    total_wall_time : float
        End-to-end wall clock time including host prep and reconstruction.
    """
    host_prep_time: float
    h2d_transfer_time: float
    kernel_execution_time: float
    d2h_transfer_time: float
    reconstruction_time: float
    total_gpu_time: float
    total_wall_time: float


class CUDANotAvailableError(RuntimeError):
    """Raised when CUDA acceleration is requested but CUDA / Numba CUDA is unavailable."""
    pass


def is_cuda_available() -> bool:
    """Check whether Numba CUDA is fully available and initialized."""
    return CUDA_AVAILABLE


def get_cuda_device_info() -> dict[str, str | int | bool]:
    """Retrieve details about the active CUDA environment or failure reason."""
    if not CUDA_AVAILABLE:
        info: dict[str, str | int | bool] = {
            "available": False,
            "reason": CUDA_IMPORT_ERROR or "CUDA not available",
        }
        try:
            from numba import cuda as _nc
            if hasattr(_nc, "cudadrv") and hasattr(_nc.cudadrv, "driver") and _nc.cudadrv.driver.driver.is_available:
                dev = _nc.cudadrv.driver.driver.get_device(0)
                name = dev.name.decode("utf-8") if isinstance(dev.name, bytes) else str(dev.name)
                cc = f"{dev.compute_capability[0]}.{dev.compute_capability[1]}"
                info["driver_available"] = True
                info["detected_device"] = name
                info["compute_capability"] = cc
        except Exception:
            pass
        return info

    try:
        dev = cuda.get_current_device()
        return {
            "available": True,
            "device_name": dev.name.decode("utf-8") if isinstance(dev.name, bytes) else str(dev.name),
            "compute_capability": f"{dev.compute_capability[0]}.{dev.compute_capability[1]}",
        }
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "reason": f"Failed querying device: {exc}",
        }


# ── CUDA Kernel Definition ──────────────────────────────────────────────────
# The kernel is only compiled by Numba when cuda is successfully imported.
if CUDA_AVAILABLE and cuda is not None:  # pragma: no branch

    @cuda.jit
    def dna_align_kernel(
        refs_flat,
        qrys_flat,
        seq_len,
        n_pairs,
        out_matches,
        out_mismatches,
        out_mismatch_flags,
    ):
        """Basic CUDA kernel for character-by-character DNA sequence alignment.

        Thread Decomposition:
            1D Grid mapping where each thread evaluates one (reference, query) pair.
            Thread ID = cuda.grid(1) = blockIdx.x * blockDim.x + threadIdx.x.

        Input Access:
            Sequences are flattened 1D uint8 arrays of size (n_pairs * seq_len).
            Thread `pair_idx` accesses slice `[pair_idx * seq_len : (pair_idx + 1) * seq_len]`.

        Outputs Produced:
            - out_matches[pair_idx]: Total matching base count.
            - out_mismatches[pair_idx]: Total mismatching base count.
            - out_mismatch_flags[pair_idx, i]: 1 if mismatch at index i, else 0.
        """
        pair_idx = cuda.grid(1)
        if pair_idx < n_pairs:
            base_offset = pair_idx * seq_len
            matches = 0
            mismatches = 0

            for i in range(seq_len):
                r_char = refs_flat[base_offset + i]
                q_char = qrys_flat[base_offset + i]
                if r_char == q_char:
                    matches += 1
                    out_mismatch_flags[pair_idx, i] = 0
                else:
                    mismatches += 1
                    out_mismatch_flags[pair_idx, i] = 1

            out_matches[pair_idx] = matches
            out_mismatches[pair_idx] = mismatches
else:
    dna_align_kernel = None  # type: ignore


# ── Host-side Sequence Preparation Helpers ─────────────────────────────────
def _encode_sequence_ascii(seq: str) -> np.ndarray:
    """Convert an uppercase DNA sequence into a 1D uint8 numpy array."""
    return np.frombuffer(seq.encode("ascii"), dtype=np.uint8)


def _prepare_host_buffers(
    pairs: Sequence[Tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Validate and pack input pairs into contiguous host-side numpy arrays.

    Parameters
    ----------
    pairs : Sequence[Tuple[str, str]]
        List or tuple of (reference, query) strings.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int, int]
        (refs_flat, qrys_flat, seq_len, n_pairs)
    """
    if not isinstance(pairs, (list, tuple)):
        raise TypeError(f"pairs must be a list or tuple, got {type(pairs).__name__}")

    n_pairs = len(pairs)
    if n_pairs == 0:
        return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8), 0, 0

    # Validate first pair to establish batch sequence length
    first_item = pairs[0]
    if not isinstance(first_item, (tuple, list)) or len(first_item) != 2:
        raise TypeError(
            f"Each pair must be a 2-element tuple/list of strings, got {type(first_item).__name__}"
        )
    ref0 = _validate_sequence(first_item[0], "reference")
    qry0 = _validate_sequence(first_item[1], "query")
    if len(ref0) != len(qry0):
        raise ValueError(
            f"Sequences must have equal length: "
            f"reference has {len(ref0)}, query has {len(qry0)}."
        )

    seq_len = len(ref0)
    if seq_len == 0:
        return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8), 0, n_pairs

    # Allocate contiguous host memory
    refs_buf = np.empty((n_pairs, seq_len), dtype=np.uint8)
    qrys_buf = np.empty((n_pairs, seq_len), dtype=np.uint8)

    refs_buf[0] = _encode_sequence_ascii(ref0)
    qrys_buf[0] = _encode_sequence_ascii(qry0)

    for idx in range(1, n_pairs):
        item = pairs[idx]
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(
                f"Each pair must be a 2-element tuple/list of strings, got {type(item).__name__} at index {idx}"
            )
        ref_seq = _validate_sequence(item[0], f"reference at pair {idx}")
        qry_seq = _validate_sequence(item[1], f"query at pair {idx}")
        if len(ref_seq) != seq_len or len(qry_seq) != seq_len:
            raise ValueError(
                f"Batch CUDA alignment requires equal sequence length across batch: "
                f"pair 0 has length {seq_len}, but pair {idx} has length "
                f"({len(ref_seq)}, {len(qry_seq)})."
            )
        refs_buf[idx] = _encode_sequence_ascii(ref_seq)
        qrys_buf[idx] = _encode_sequence_ascii(qry_seq)

    # Flatten for linear memory access
    refs_flat = refs_buf.ravel()
    qrys_flat = qrys_buf.ravel()
    return refs_flat, qrys_flat, seq_len, n_pairs


# ── Host -> Device -> Kernel -> Device -> Host Pipeline ────────────────────
def cuda_align_sequences_timed(
    pairs: list[tuple[str, str]],
    threads_per_block: int = 128,
) -> tuple[list[AlignmentResult], CUDATimingBreakdown]:
    """Align a batch of DNA sequence pairs on the GPU and return granular timings.

    Detailed 5-stage measured pipeline:
    1. Host-side Preparation: Input validation, ASCII encoding, 2D packing & flattening.
    2. Host-to-Device Transfer: Allocate GPU buffers & copy inputs via ``cuda.to_device()``.
    3. CUDA Kernel Execution: Launch ``dna_align_kernel`` and synchronize (GPU wall time).
    4. Device-to-Host Transfer: Copy results back via ``copy_to_host()`` and release GPU memory.
    5. Result Reconstruction: Build frozen ``AlignmentResult`` dataclass objects.

    Parameters
    ----------
    pairs : list[tuple[str, str]]
        List of (reference, query) sequence pairs.
    threads_per_block : int, default=128
        CUDA block dimension (threads per block).

    Returns
    -------
    tuple[list[AlignmentResult], CUDATimingBreakdown]
        Tuple of (results list, timing breakdown in seconds).

    Raises
    ------
    CUDANotAvailableError
        If Numba CUDA is not available or device initialization failed.
    TypeError
        If input types are invalid.
    ValueError
        If inputs contain invalid characters or length mismatches.
    """
    if not isinstance(pairs, list):
        raise TypeError(f"pairs must be a list, got {type(pairs).__name__}")
    if not isinstance(threads_per_block, int) or isinstance(threads_per_block, bool):
        raise TypeError(f"threads_per_block must be an integer, got {type(threads_per_block).__name__}")
    if threads_per_block <= 0:
        raise ValueError(f"threads_per_block must be positive, got {threads_per_block}")

    if not CUDA_AVAILABLE or cuda is None:
        raise CUDANotAvailableError(
            f"CUDA execution requested but CUDA is not available in the current environment: "
            f"{CUDA_IMPORT_ERROR}"
        )

    t_wall_start = time.perf_counter()
    n_pairs = len(pairs)
    if n_pairs == 0:
        timing = CUDATimingBreakdown(
            host_prep_time=0.0,
            h2d_transfer_time=0.0,
            kernel_execution_time=0.0,
            d2h_transfer_time=0.0,
            reconstruction_time=0.0,
            total_gpu_time=0.0,
            total_wall_time=time.perf_counter() - t_wall_start,
        )
        return [], timing

    # 1. Host-side preparation
    t_prep_start = time.perf_counter()
    refs_flat, qrys_flat, seq_len, _ = _prepare_host_buffers(pairs)
    t_prep = time.perf_counter() - t_prep_start

    # Handle batch of empty strings
    if seq_len == 0:
        t_recon_start = time.perf_counter()
        results = [
            AlignmentResult(
                matches=0,
                mismatches=0,
                alignment_length=0,
                score=0.0,
                mismatch_positions=[],
            )
            for _ in range(n_pairs)
        ]
        t_recon = time.perf_counter() - t_recon_start
        timing = CUDATimingBreakdown(
            host_prep_time=t_prep,
            h2d_transfer_time=0.0,
            kernel_execution_time=0.0,
            d2h_transfer_time=0.0,
            reconstruction_time=t_recon,
            total_gpu_time=0.0,
            total_wall_time=time.perf_counter() - t_wall_start,
        )
        return results, timing

    # 2. Host-to-Device Memory Transfer
    t_h2d_start = time.perf_counter()
    d_refs = None
    d_qrys = None
    d_matches = None
    d_mismatches = None
    d_mismatch_flags = None
    try:
        d_refs = cuda.to_device(refs_flat)
        d_qrys = cuda.to_device(qrys_flat)
        d_matches = cuda.device_array(n_pairs, dtype=np.int32)
        d_mismatches = cuda.device_array(n_pairs, dtype=np.int32)
        d_mismatch_flags = cuda.device_array((n_pairs, seq_len), dtype=np.uint8)
        cuda.synchronize()
        t_h2d = time.perf_counter() - t_h2d_start

        # 3. Kernel Execution
        t_kernel_start = time.perf_counter()
        blocks_per_grid = math.ceil(n_pairs / threads_per_block)
        dna_align_kernel[blocks_per_grid, threads_per_block](
            d_refs,
            d_qrys,
            seq_len,
            n_pairs,
            d_matches,
            d_mismatches,
            d_mismatch_flags,
        )
        cuda.synchronize()
        t_kernel = time.perf_counter() - t_kernel_start

        # 4. Device-to-Host Memory Transfer
        t_d2h_start = time.perf_counter()
        h_matches = d_matches.copy_to_host()
        h_mismatches = d_mismatches.copy_to_host()
        h_mismatch_flags = d_mismatch_flags.copy_to_host()
        cuda.synchronize()
        t_d2h = time.perf_counter() - t_d2h_start
    finally:
        # Explicitly release GPU device arrays in all cases
        del d_refs, d_qrys, d_matches, d_mismatches, d_mismatch_flags

    # 5. Result Reconstruction
    t_recon_start = time.perf_counter()
    results: list[AlignmentResult] = []
    for i in range(n_pairs):
        m = int(h_matches[i])
        mm = int(h_mismatches[i])
        score = float(m / seq_len) if seq_len > 0 else 0.0
        if mm == 0:
            mismatch_indices: list[int] = []
        elif mm == seq_len:
            mismatch_indices = list(range(seq_len))
        else:
            mismatch_indices = np.flatnonzero(h_mismatch_flags[i]).tolist()

        results.append(
            AlignmentResult(
                matches=m,
                mismatches=mm,
                alignment_length=seq_len,
                score=score,
                mismatch_positions=mismatch_indices,
            )
        )
    t_recon = time.perf_counter() - t_recon_start
    t_wall = time.perf_counter() - t_wall_start

    timing = CUDATimingBreakdown(
        host_prep_time=t_prep,
        h2d_transfer_time=t_h2d,
        kernel_execution_time=t_kernel,
        d2h_transfer_time=t_d2h,
        reconstruction_time=t_recon,
        total_gpu_time=t_h2d + t_kernel + t_d2h,
        total_wall_time=t_wall,
    )

    return results, timing


def cuda_align_sequences(
    pairs: list[tuple[str, str]],
    threads_per_block: int = 128,
) -> list[AlignmentResult]:
    """Align a batch of DNA sequence pairs on the GPU using Numba CUDA.

    Parameters
    ----------
    pairs : list[tuple[str, str]]
        List of (reference, query) sequence pairs.
    threads_per_block : int, default=128
        CUDA block dimension (threads per block).

    Returns
    -------
    list[AlignmentResult]
        Alignment result for each pair, preserving input order.
    """
    results, _ = cuda_align_sequences_timed(pairs, threads_per_block=threads_per_block)
    return results


def cuda_align_sequence(
    reference: str,
    query: str,
) -> AlignmentResult:
    """Align a single pair of equal-length DNA sequences on the GPU.

    Wrapper around ``cuda_align_sequences`` for single-pair invocations.

    Parameters
    ----------
    reference : str
        Reference DNA sequence.
    query : str
        Query DNA sequence.

    Returns
    -------
    AlignmentResult
        Result matching the CPU reference implementation.
    """
    results = cuda_align_sequences([(reference, query)])
    return results[0]


def cuda_align_chunks(
    chunks: list[str],
    target: str,
    *,
    skip_unequal: bool = False,
    threads_per_block: int = 128,
) -> list[AlignmentResult]:
    """Align genomic chunks against a target sequence on the GPU.

    Maintains full compatibility with Member 2's chunking output.

    Parameters
    ----------
    chunks : list[str]
        List of DNA sequence chunks.
    target : str
        Target DNA sequence to align against.
    skip_unequal : bool, default=False
        If True, skips chunks whose length does not match ``len(target)``.
    threads_per_block : int, default=128
        CUDA block dimension (threads per block).

    Returns
    -------
    list[AlignmentResult]
        Alignment result for each chunk.

    Raises
    ------
    CUDANotAvailableError
        If CUDA execution is not available in the environment.
    TypeError
        If input types are invalid.
    ValueError
        If inputs contain invalid characters or length mismatches when
        ``skip_unequal=False``.
    """
    if not isinstance(chunks, list):
        raise TypeError(f"chunks must be a list, got {type(chunks).__name__}")
    if not isinstance(target, str):
        raise TypeError(f"target must be a string, got {type(target).__name__}")

    target_norm = _validate_sequence(target, "target")
    target_len = len(target_norm)

    valid_pairs: list[tuple[str, str]] = []
    for chunk in chunks:
        if not isinstance(chunk, str):
            raise TypeError(f"Each chunk must be a string, got {type(chunk).__name__}")
        if len(chunk) != target_len:
            if skip_unequal:
                continue
            raise ValueError(
                f"Sequences must have equal length: "
                f"reference has {len(chunk)}, query has {target_len}."
            )
        valid_pairs.append((chunk, target_norm))

    return cuda_align_sequences(valid_pairs, threads_per_block=threads_per_block)


def cuda_align_chunks_timed(
    chunks: list[str],
    target: str,
    *,
    skip_unequal: bool = False,
    threads_per_block: int = 128,
) -> tuple[list[AlignmentResult], CUDATimingBreakdown]:
    """Align genomic chunks against a target sequence on the GPU with timing breakdown.

    Parameters
    ----------
    chunks : list[str]
        List of DNA sequence chunks.
    target : str
        Target DNA sequence to align against.
    skip_unequal : bool, default=False
        If True, skips chunks whose length does not match ``len(target)``.
    threads_per_block : int, default=128
        CUDA block dimension (threads per block).

    Returns
    -------
    tuple[list[AlignmentResult], CUDATimingBreakdown]
        Tuple of (results list, timing breakdown in seconds).

    Raises
    ------
    CUDANotAvailableError
        If CUDA execution is not available in the environment.
    TypeError
        If input types are invalid.
    ValueError
        If inputs contain invalid characters or length mismatches when
        ``skip_unequal=False``.
    """
    if not isinstance(chunks, list):
        raise TypeError(f"chunks must be a list, got {type(chunks).__name__}")
    if not isinstance(target, str):
        raise TypeError(f"target must be a string, got {type(target).__name__}")

    target_norm = _validate_sequence(target, "target")
    target_len = len(target_norm)

    valid_pairs: list[tuple[str, str]] = []
    for chunk in chunks:
        if not isinstance(chunk, str):
            raise TypeError(f"Each chunk must be a string, got {type(chunk).__name__}")
        if len(chunk) != target_len:
            if skip_unequal:
                continue
            raise ValueError(
                f"Sequences must have equal length: "
                f"reference has {len(chunk)}, query has {target_len}."
            )
        valid_pairs.append((chunk, target_norm))

    return cuda_align_sequences_timed(valid_pairs, threads_per_block=threads_per_block)


def cuda_align_chunk_pairs(
    reference_chunks: list[str],
    query_chunks: list[str],
    *,
    threads_per_block: int = 128,
) -> list[AlignmentResult]:
    """Align corresponding pairs of chunks from two chunk lists on the GPU.

    Parameters
    ----------
    reference_chunks : list[str]
        Reference sequence chunks.
    query_chunks : list[str]
        Query sequence chunks to compare against reference chunks.
    threads_per_block : int, default=128
        CUDA block dimension.

    Returns
    -------
    list[AlignmentResult]
        One result per chunk pair, in order.

    Raises
    ------
    CUDANotAvailableError
        If CUDA is unavailable.
    TypeError
        If either argument is not a list.
    ValueError
        If the two lists have different lengths.
    """
    if not isinstance(reference_chunks, list):
        raise TypeError(
            f"reference_chunks must be a list, got {type(reference_chunks).__name__}"
        )
    if not isinstance(query_chunks, list):
        raise TypeError(
            f"query_chunks must be a list, got {type(query_chunks).__name__}"
        )
    if len(reference_chunks) != len(query_chunks):
        raise ValueError(
            f"Chunk lists must have equal length: "
            f"reference has {len(reference_chunks)}, query has {len(query_chunks)}."
        )

    for i, ref in enumerate(reference_chunks):
        if not isinstance(ref, str):
            raise TypeError(f"Each chunk in reference_chunks must be a string, got {type(ref).__name__} at index {i}")
    for i, qry in enumerate(query_chunks):
        if not isinstance(qry, str):
            raise TypeError(f"Each chunk in query_chunks must be a string, got {type(qry).__name__} at index {i}")

    pairs = list(zip(reference_chunks, query_chunks))
    return cuda_align_sequences(pairs, threads_per_block=threads_per_block)

