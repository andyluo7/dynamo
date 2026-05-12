# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DRAM staging adapter for RIXL/nixl KV cache transfer on AMD GPUs.

Problem: RIXL's UCX backend cannot register GPU VRAM directly on AMD
ionic NICs (no GPU Direct RDMA support). The `ibv_reg_mr` call for
VRAM addresses fails with NIXL_ERR_BACKEND.

Solution: Allocate pinned host (DRAM) staging buffers, register those
with RIXL, and do explicit GPU<->CPU copies around each transfer.

Data flow:
  Prefill side:  GPU KV cache -> hipMemcpy D2H -> DRAM staging buffer
                 -> RIXL RDMA transfer -> remote DRAM staging buffer
  Decode side:   remote DRAM staging buffer -> hipMemcpy H2D -> GPU KV cache

Performance impact: ~1-2ms extra per transfer for hipMemcpy, but enables
RIXL on AMD ionic NICs where VRAM registration fails.

The main integration is via monkey-patch in nixl_rocm_staging.py
(patches NixlKVManager at runtime, zero changes to sglang-amd source).
This module provides a standalone helper for use outside SGLang.
"""

from __future__ import annotations

import ctypes
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

# hipMemcpy direction constants
_hipMemcpyHostToDevice = 1
_hipMemcpyDeviceToHost = 2

_libhip = None


def _get_libhip():
    global _libhip
    if _libhip is None:
        try:
            _libhip = ctypes.CDLL("libamdhip64.so")
        except OSError:
            raise RuntimeError("libamdhip64.so not found — ROCm HIP runtime required")
    return _libhip


def _hip_memcpy(dst: int, src: int, size: int, kind: int):
    lib = _get_libhip()
    ret = lib.hipMemcpy(
        ctypes.c_void_p(dst),
        ctypes.c_void_p(src),
        ctypes.c_size_t(size),
        ctypes.c_int(kind),
    )
    if ret != 0:
        raise RuntimeError(f"hipMemcpy failed with error code {ret}")


def _hip_sync():
    lib = _get_libhip()
    ret = lib.hipDeviceSynchronize()
    if ret != 0:
        raise RuntimeError(f"hipDeviceSynchronize failed with error code {ret}")


class DramStagingBuffer:
    """Manages pinned host memory for DRAM-staged RDMA transfers.

    Uses mmap+mlock instead of torch.pin_memory() because ionic NICs
    reject hipHostMalloc-backed memory in ibv_reg_mr (returns EINVAL).
    """

    def __init__(self, size: int, device_id: int = 0):
        import mmap as _mmap

        self.size = size
        self.device_id = device_id

        self._mmap = _mmap.mmap(-1, size)
        self._buf = (ctypes.c_char * size).from_buffer(self._mmap)
        self.host_ptr = ctypes.addressof(self._buf)

        libc = ctypes.CDLL("libc.so.6")
        rc = libc.mlock(ctypes.c_void_p(self.host_ptr), ctypes.c_size_t(size))
        if rc != 0:
            logger.warning("mlock failed (rc=%d) for %d MB", rc, size >> 20)

        logger.info(
            "DramStagingBuffer: allocated %d MB mmap+mlock at %s",
            size // (1024 * 1024), hex(self.host_ptr),
        )

    def copy_from_gpu(self, gpu_ptr: int, offset: int, length: int) -> None:
        """Copy data from GPU VRAM to host staging buffer (hipMemcpy D2H)."""
        _hip_memcpy(
            self.host_ptr + offset,
            gpu_ptr,
            length,
            _hipMemcpyDeviceToHost,
        )

    def copy_to_gpu(self, gpu_ptr: int, offset: int, length: int) -> None:
        """Copy data from host staging buffer to GPU VRAM (hipMemcpy H2D)."""
        _hip_memcpy(
            gpu_ptr,
            self.host_ptr + offset,
            length,
            _hipMemcpyHostToDevice,
        )


class DramStagingManager:
    """Manages multiple DRAM staging buffers for KV cache transfer.

    Replaces VRAM registration with DRAM registration in the nixl connector.
    """

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.buffers: dict[int, DramStagingBuffer] = {}
        self._total_allocated = 0

    def get_or_create_buffer(self, buffer_id: int, size: int) -> DramStagingBuffer:
        if buffer_id not in self.buffers:
            buf = DramStagingBuffer(size, self.gpu_id)
            self.buffers[buffer_id] = buf
            self._total_allocated += size
            logger.info(
                f"DramStagingManager: buffer {buffer_id}, "
                f"total={self._total_allocated / (1024**3):.2f} GB"
            )
        return self.buffers[buffer_id]

    def stage_gpu_to_dram(
        self,
        gpu_addrs: List[Tuple[int, int]],
        buffer_id: int,
    ) -> List[Tuple[int, int]]:
        """Stage GPU buffers to DRAM before RDMA transfer.

        Returns list of (dram_ptr, size) pairs for RIXL registration.
        """
        total_size = sum(size for _, size in gpu_addrs)
        buf = self.get_or_create_buffer(buffer_id, total_size)

        dram_addrs = []
        offset = 0
        for gpu_ptr, size in gpu_addrs:
            buf.copy_from_gpu(gpu_ptr, offset, size)
            dram_addrs.append((buf.host_ptr + offset, size))
            offset += size

        _hip_sync()
        return dram_addrs

    def unstage_dram_to_gpu(
        self,
        gpu_addrs: List[Tuple[int, int]],
        buffer_id: int,
    ) -> None:
        """Copy received data from DRAM staging buffer back to GPU."""
        buf = self.buffers.get(buffer_id)
        if buf is None:
            raise ValueError(f"No staging buffer for id={buffer_id}")

        offset = 0
        for gpu_ptr, size in gpu_addrs:
            buf.copy_to_gpu(gpu_ptr, offset, size)
            offset += size

        _hip_sync()

    def cleanup(self) -> None:
        for buf in self.buffers.values():
            del buf._buf
            buf._mmap.close()
        self.buffers.clear()
        self._total_allocated = 0
