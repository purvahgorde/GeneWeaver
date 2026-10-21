from __future__ import annotations

import math
from dataclasses import dataclass
import time
from typing import Sequence, Tuple

import numpy as np

# Import the existing AlignmentResult contract to guarantee 100% interoperability
from .cpu_alignment import AlignmentResult, _validate_sequence

import os
import sys

# Determine Numba CUDA availability safely
CUDA_AVAILABLE: bool = False
CUDA_IMPORT_ERROR: str | None = None
cuda = None  # type: ignore


def _init_cuda_environment() -> None:
    """Detect and configure CUDA environment on Windows when standard paths differ.

    In CUDA Toolkit 13.x on Windows:
    1. 64-bit runtime DLLs are placed in `<CUDA_PATH>/bin/x64` (cudart64_13.dll).
    2. NVVM compiler DLL is placed in `<CUDA_PATH>/nvvm/bin/x64` (nvvm64_40_0.dll).
    3. Python 3.8+ on Windows requires os.add_dll_directory() for DLL resolution.
    4. Numba's cuda_paths module needs to search the x64 subdirectories.
    5. Driver PTX version compatibility: When NVVM emits .version 9.4 (CUDA 13.4)
       and the installed display driver's JIT linker supports up to PTX 9.3,
       hook CtypesLinker.add_ptx to adapt the version header for SM 8.6 binary emission.
    """
    global CUDA_AVAILABLE, CUDA_IMPORT_ERROR, cuda

    if sys.platform == "win32":
        cuda_candidates = [
            os.environ.get("CUDA_PATH"),
            os.environ.get("CUDA_PATH_V13_4"),
            os.environ.get("CUDA_HOME"),
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4",
        ]
        cuda_root = next((c for c in cuda_candidates if c and os.path.isdir(c)), None)
        if cuda_root:
            x64_bin = os.path.join(cuda_root, "bin", "x64")
            x64_nvvm = os.path.join(cuda_root, "nvvm", "bin", "x64")
            bin_dir = os.path.join(cuda_root, "bin")

            for p in (x64_bin, x64_nvvm, bin_dir):
                if os.path.isdir(p):
                    try:
                        os.add_dll_directory(p)
                    except (AttributeError, OSError):
                        pass
                    if p not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

            try:
                import numba.cuda.cuda_paths as cp
                if os.path.isdir(x64_nvvm):
                    cp._nvvm_lib_dir = lambda: ("nvvm", "bin", "x64")
                if os.path.isdir(x64_bin):
                    cp._cudalib_path = lambda: os.path.join("bin", "x64")
                if hasattr(cp.get_cuda_paths, "_cached_result"):
                    del cp.get_cuda_paths._cached_result
            except Exception:
                pass

            try:
                from numba.cuda.cudadrv.driver import CtypesLinker
                orig_add_ptx = CtypesLinker.add_ptx

                def _hooked_add_ptx(self, ptx, name="<cudapy-ptx>"):
                    if isinstance(ptx, bytes) and b".version 9.4" in ptx:
                        ptx = ptx.replace(b".version 9.4", b".version 9.3")
                    return orig_add_ptx(self, ptx, name=name)

                CtypesLinker.add_ptx = _hooked_add_ptx
            except Exception:
                pass

    try:
        from numba import cuda as _cuda
        if _cuda.is_available():
            cuda = _cuda
            CUDA_AVAILABLE = True
        else:
            # Check if driver is available but compiler library (NVVM) is missing
            if (
                hasattr(_cuda, "cudadrv")
                and hasattr(_cuda.cudadrv, "driver")
                and _cuda.cudadrv.driver.driver.is_available
            ):
                CUDA_IMPORT_ERROR = (
                    "Numba detected CUDA driver and device, but CUDA compiler library (libNVVM/nvvm.dll) is not available."
                )
            else:
                CUDA_IMPORT_ERROR = "Numba detected no available CUDA devices or driver library."
    except Exception as _exc:  # pragma: no cover
        CUDA_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"


_init_cuda_environment()


@dataclass(frozen=True)
class CUDATimingBreakdown:
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


MAX_SHARED_TARGET_LEN: int = 1024

# ── CUDA Kernel Definitions ─────────────────────────────────────────────────
# Kernels are only compiled by Numba when cuda is successfully imported.
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

    @cuda.jit
    def dna_align_shared_kernel(
        refs_flat,
        target_flat,
        seq_len,
        n_pairs,
        out_matches,
        out_mismatches,
        out_mismatch_flags,
    ):
        s_target = cuda.shared.array(shape=1024, dtype=np.uint8)

        tid = cuda.threadIdx.x
        bdim = cuda.blockDim.x

        # Cooperative strided load of broadcast target sequence into block shared memory
        idx = tid
        while idx < seq_len and idx < 1024:
            s_target[idx] = target_flat[idx]
            idx += bdim

        # Synchronize all threads in the block to guarantee target is fully resident in shared memory
        cuda.syncthreads()

        # Thread-level alignment against the cached shared target
        pair_idx = cuda.grid(1)
        if pair_idx < n_pairs:
            base_offset = pair_idx * seq_len
            matches = 0
            mismatches = 0

            for i in range(seq_len):
                r_char = refs_flat[base_offset + i]
                q_char = s_target[i]
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
    dna_align_shared_kernel = None  # type: ignore


# ── Host-side Sequence Preparation Helpers ─────────────────────────────────
def _encode_sequence_ascii(seq: str) -> np.ndarray:
    """Convert an uppercase DNA sequence into a 1D uint8 numpy array."""
    return np.frombuffer(seq.encode("ascii"), dtype=np.uint8)


def _prepare_host_buffers(
    pairs: Sequence[Tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray, int, int]:
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


def _prepare_host_buffers_target(
    chunks: Sequence[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if not isinstance(chunks, (list, tuple)):
        raise TypeError(f"chunks must be a list or tuple, got {type(chunks).__name__}")
    if not isinstance(target, str):
        raise TypeError(f"target must be a string, got {type(target).__name__}")

    target_norm = _validate_sequence(target, "target")
    seq_len = len(target_norm)
    n_pairs = len(chunks)

    if n_pairs == 0 or seq_len == 0:
        return np.empty(0, dtype=np.uint8), _encode_sequence_ascii(target_norm), seq_len, n_pairs

    refs_buf = np.empty((n_pairs, seq_len), dtype=np.uint8)
    for idx, chk in enumerate(chunks):
        if not isinstance(chk, str):
            raise TypeError(f"Each chunk must be a string, got {type(chk).__name__} at index {idx}")
        chk_norm = _validate_sequence(chk, f"chunk at index {idx}")
        if len(chk_norm) != seq_len:
            raise ValueError(
                f"Sequences must have equal length: "
                f"chunk at index {idx} has length {len(chk_norm)}, target has length {seq_len}."
            )
        refs_buf[idx] = _encode_sequence_ascii(chk_norm)

    refs_flat = refs_buf.ravel()
    target_flat = _encode_sequence_ascii(target_norm)
    return refs_flat, target_flat, seq_len, n_pairs


# ── Host -> Device -> Kernel -> Device -> Host Pipeline ────────────────────
def cuda_align_sequences_timed(
    pairs: list[tuple[str, str]],
    threads_per_block: int = 128,
) -> tuple[list[AlignmentResult], CUDATimingBreakdown]:
    
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
    try:
        refs_flat, qrys_flat, seq_len, _ = _prepare_host_buffers(pairs)
    except ValueError as ve:
        if "Batch CUDA alignment requires equal sequence length across batch" in str(ve):
            # Batch has mixed sequence lengths; group pairs by length and execute each sub-batch
            from collections import defaultdict
            groups: dict[int, list[tuple[int, tuple[str, str]]]] = defaultdict(list)
            for orig_idx, pair in enumerate(pairs):
                groups[len(pair[0])].append((orig_idx, pair))

            results: list[AlignmentResult] = [None] * n_pairs  # type: ignore
            t_prep = 0.0
            t_h2d = 0.0
            t_kern = 0.0
            t_d2h = 0.0
            t_recon = 0.0

            for _, items in groups.items():
                sub_pairs = [p for _, p in items]
                sub_res, sub_timing = cuda_align_sequences_timed(
                    sub_pairs, threads_per_block=threads_per_block
                )
                for (orig_idx, _), r in zip(items, sub_res):
                    results[orig_idx] = r
                t_prep += sub_timing.host_prep_time
                t_h2d += sub_timing.h2d_transfer_time
                t_kern += sub_timing.kernel_execution_time
                t_d2h += sub_timing.d2h_transfer_time
                t_recon += sub_timing.reconstruction_time

            t_wall = time.perf_counter() - t_wall_start
            timing = CUDATimingBreakdown(
                host_prep_time=t_prep,
                h2d_transfer_time=t_h2d,
                kernel_execution_time=t_kern,
                d2h_transfer_time=t_d2h,
                reconstruction_time=t_recon,
                total_gpu_time=t_h2d + t_kern + t_d2h,
                total_wall_time=t_wall,
            )
            return results, timing
        raise
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
    results, _ = cuda_align_sequences_timed(pairs, threads_per_block=threads_per_block)
    return results


def cuda_align_sequence(
    reference: str,
    query: str,
) -> AlignmentResult:
    
    results = cuda_align_sequences([(reference, query)])
    return results[0]


def cuda_align_chunks(
    chunks: list[str],
    target: str,
    *,
    skip_unequal: bool = False,
    threads_per_block: int = 128,
    use_shared_memory: bool = False,
) -> list[AlignmentResult]:
    
    results, _ = cuda_align_chunks_timed(
        chunks,
        target,
        skip_unequal=skip_unequal,
        threads_per_block=threads_per_block,
        use_shared_memory=use_shared_memory,
    )
    return results


def cuda_align_chunks_shared(
    chunks: list[str],
    target: str,
    *,
    skip_unequal: bool = False,
    threads_per_block: int = 128,
) -> list[AlignmentResult]:
    
    return cuda_align_chunks(
        chunks,
        target,
        skip_unequal=skip_unequal,
        threads_per_block=threads_per_block,
        use_shared_memory=True,
    )


def cuda_align_chunks_timed(
    chunks: list[str],
    target: str,
    *,
    skip_unequal: bool = False,
    threads_per_block: int = 128,
    use_shared_memory: bool = False,
) -> tuple[list[AlignmentResult], CUDATimingBreakdown]:
    
    if not isinstance(chunks, list):
        raise TypeError(f"chunks must be a list, got {type(chunks).__name__}")
    if not isinstance(target, str):
        raise TypeError(f"target must be a string, got {type(target).__name__}")
    if not isinstance(use_shared_memory, bool):
        raise TypeError(f"use_shared_memory must be a bool, got {type(use_shared_memory).__name__}")
    if not isinstance(threads_per_block, int) or isinstance(threads_per_block, bool):
        raise TypeError(f"threads_per_block must be an integer, got {type(threads_per_block).__name__}")
    if threads_per_block <= 0:
        raise ValueError(f"threads_per_block must be positive, got {threads_per_block}")

    target_norm = _validate_sequence(target, "target")
    target_len = len(target_norm)

    valid_chunks: list[str] = []
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
        valid_chunks.append(chunk)

    if not CUDA_AVAILABLE or cuda is None:
        raise CUDANotAvailableError(
            f"CUDA execution requested but CUDA is not available in the current environment: "
            f"{CUDA_IMPORT_ERROR}"
        )

    # If shared memory is not requested, or target exceeds shared memory capacity, run baseline
    if not use_shared_memory or target_len > MAX_SHARED_TARGET_LEN:
        valid_pairs = [(chunk, target_norm) for chunk in valid_chunks]
        return cuda_align_sequences_timed(valid_pairs, threads_per_block=threads_per_block)

    # Shared memory execution path
    t_wall_start = time.perf_counter()
    n_pairs = len(valid_chunks)

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

    if target_len == 0:
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
            host_prep_time=0.0,
            h2d_transfer_time=0.0,
            kernel_execution_time=0.0,
            d2h_transfer_time=0.0,
            reconstruction_time=t_recon,
            total_gpu_time=0.0,
            total_wall_time=time.perf_counter() - t_wall_start,
        )
        return results, timing

    # 1. Host-side Preparation (target copied once, not replicated)
    t_prep_start = time.perf_counter()
    refs_flat, target_flat, seq_len, _ = _prepare_host_buffers_target(valid_chunks, target_norm)
    t_prep = time.perf_counter() - t_prep_start

    # 2. Host-to-Device Memory Transfer
    t_h2d_start = time.perf_counter()
    d_refs = None
    d_target = None
    d_matches = None
    d_mismatches = None
    d_mismatch_flags = None
    try:
        d_refs = cuda.to_device(refs_flat)
        d_target = cuda.to_device(target_flat)
        d_matches = cuda.device_array(n_pairs, dtype=np.int32)
        d_mismatches = cuda.device_array(n_pairs, dtype=np.int32)
        d_mismatch_flags = cuda.device_array((n_pairs, seq_len), dtype=np.uint8)
        cuda.synchronize()
        t_h2d = time.perf_counter() - t_h2d_start

        # 3. Kernel Execution using cooperative shared-memory target caching
        t_kernel_start = time.perf_counter()
        blocks_per_grid = math.ceil(n_pairs / threads_per_block)
        dna_align_shared_kernel[blocks_per_grid, threads_per_block](
            d_refs,
            d_target,
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
        del d_refs, d_target, d_matches, d_mismatches, d_mismatch_flags

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


def cuda_align_chunk_pairs(
    reference_chunks: list[str],
    query_chunks: list[str],
    *,
    threads_per_block: int = 128,
) -> list[AlignmentResult]:
    
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

