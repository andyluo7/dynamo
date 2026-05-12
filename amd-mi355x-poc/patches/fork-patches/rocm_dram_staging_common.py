# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared DRAM staging primitives for ROCm disaggregated serving.

On AMD GPUs with Pensando ionic NICs, RDMA libraries cannot ``ibv_reg_mr()``
on GPU VRAM (no GPUDirect RDMA).  Additionally, ionic's ``ibv_reg_mr()``
rejects ``hipHostMalloc``-backed memory (EINVAL) and has a ~448 MB total
registration limit for regular memory.

This module provides a common ``RocmDramStaging`` class that both the NIXL
and Mooncake monkey-patch modules use to bounce KV data through host memory
allocated via ``mmap`` + ``mlock`` (avoiding ``hipHostMalloc`` entirely).

Data flow::

    GPU VRAM  ─hipMemcpy D2H─▶  DRAM (mmap)  ──transfer──▶  Remote DRAM  ─hipMemcpy H2D─▶  GPU VRAM
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
import threading

logger = logging.getLogger(__name__)

_IS_ROCM = False
try:
    import torch as _torch

    _IS_ROCM = hasattr(_torch.version, "hip") and _torch.version.hip is not None
except ImportError:
    pass


def is_rocm_staging_enabled(env_var: str = "SGLANG_ROCM_DRAM_STAGING") -> bool:
    """Check whether ROCm DRAM staging should be active."""
    if _IS_ROCM:
        return True
    return os.environ.get(env_var, "").lower() in ("1", "true", "yes")


class RocmDramStaging:
    """Thread-safe pinned-host mirrors for GPU VRAM buffers.

    Each GPU buffer gets a same-sized pinned host allocation.  Addresses
    are translated 1:1 ``(gpu_base + offset → host_base + offset)`` so all
    existing index arithmetic in the caller is preserved.
    """

    def __init__(self):
        try:
            self._hip = ctypes.CDLL("libamdhip64.so")
        except OSError:
            raise RuntimeError(
                "libamdhip64.so not found — ROCm DRAM staging requires the HIP runtime"
            )
        self._lock = threading.Lock()
        self.buffers: dict[int, tuple] = {}  # gpu_base → (tensor, host_ptr, size)

        self._stream = ctypes.c_void_p(0)
        err = self._hip.hipStreamCreate(ctypes.byref(self._stream))
        if err != 0:
            logger.warning(
                "hipStreamCreate failed (%d), falling back to device sync", err
            )
            self._stream = None

    # -- buffer management -----------------------------------------------------

    _libc = None

    @classmethod
    def _get_libc(cls):
        if cls._libc is None:
            cls._libc = ctypes.CDLL("libc.so.6")
        return cls._libc

    def create(self, gpu_ptr: int, size: int, allocate: bool = True) -> int:
        """Register a GPU buffer for staging.

        Uses ``mmap`` + ``mlock`` instead of ``torch.pin_memory()`` because
        ionic NICs reject ``hipHostMalloc``-backed memory in ``ibv_reg_mr()``
        (returns EINVAL).

        Args:
            gpu_ptr: GPU buffer base address.
            size: Buffer size in bytes.
            allocate: If ``True``, allocate DRAM now (decode side).
                      If ``False``, record metadata only (prefill lazy mode).

        Returns:
            ``host_ptr`` if allocated, or ``gpu_ptr`` as placeholder if lazy.
        """
        with self._lock:
            if allocate:
                m = mmap.mmap(-1, size)
                buf = (ctypes.c_char * size).from_buffer(m)
                host_ptr = ctypes.addressof(buf)
                libc = self._get_libc()
                rc = libc.mlock(ctypes.c_void_p(host_ptr), ctypes.c_size_t(size))
                if rc != 0:
                    logger.warning("mlock failed (rc=%d) for %d MB — "
                                   "proceeding without locked pages", rc, size >> 20)
                self.buffers[gpu_ptr] = ((m, buf), host_ptr, size)
                logger.info(
                    "DRAM staging: %d MB  GPU %s → host %s (mmap+mlock)",
                    size // (1024 * 1024),
                    hex(gpu_ptr),
                    hex(host_ptr),
                )
                return host_ptr
            else:
                self.buffers[gpu_ptr] = (None, gpu_ptr, size)
                logger.info(
                    "DRAM staging: %d MB GPU %s (lazy)",
                    size // (1024 * 1024),
                    hex(gpu_ptr),
                )
                return gpu_ptr

    def get_staging_region(self, gpu_addr: int, nbytes: int):
        """Allocate a transient mmap region and copy GPU data into it.

        Returns ``(host_ptr, ref)`` — caller must keep *ref* alive
        until the transfer completes.
        """
        m = mmap.mmap(-1, nbytes)
        buf = (ctypes.c_char * nbytes).from_buffer(m)
        host_ptr = ctypes.addressof(buf)
        libc = self._get_libc()
        libc.mlock(ctypes.c_void_p(host_ptr), ctypes.c_size_t(nbytes))
        self.copy_d2h_direct(gpu_addr, host_ptr, nbytes)
        self.sync()
        return host_ptr, (m, buf)

    # -- HIP helpers -----------------------------------------------------------

    def _check_hip(self, err: int, op: str):
        if err != 0:
            raise RuntimeError(f"HIP {op} failed with error code {err}")

    def copy_d2h_direct(self, gpu_addr: int, host_addr: int, nbytes: int):
        if self._stream is not None:
            err = self._hip.hipMemcpyAsync(
                ctypes.c_void_p(host_addr),
                ctypes.c_void_p(gpu_addr),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(2),
                self._stream,
            )
            self._check_hip(err, "hipMemcpyAsync D2H")
        else:
            err = self._hip.hipMemcpy(
                ctypes.c_void_p(host_addr),
                ctypes.c_void_p(gpu_addr),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(2),
            )
            self._check_hip(err, "hipMemcpy D2H")

    def copy_h2d_direct(self, host_addr: int, gpu_addr: int, nbytes: int):
        if self._stream is not None:
            err = self._hip.hipMemcpyAsync(
                ctypes.c_void_p(gpu_addr),
                ctypes.c_void_p(host_addr),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(1),
                self._stream,
            )
            self._check_hip(err, "hipMemcpyAsync H2D")
        else:
            err = self._hip.hipMemcpy(
                ctypes.c_void_p(gpu_addr),
                ctypes.c_void_p(host_addr),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(1),
            )
            self._check_hip(err, "hipMemcpy H2D")

    # -- buffer-aware copies ---------------------------------------------------

    def copy_d2h(self, gpu_addr: int, nbytes: int) -> bool:
        """Copy GPU→Host within a registered buffer. Returns False if unregistered."""
        for gpu_base, (_, host_base, buf_size) in self.buffers.items():
            if gpu_base <= gpu_addr < gpu_base + buf_size:
                offset = gpu_addr - gpu_base
                err = self._hip.hipMemcpy(
                    ctypes.c_void_p(host_base + offset),
                    ctypes.c_void_p(gpu_addr),
                    ctypes.c_size_t(nbytes),
                    ctypes.c_int(2),
                )
                self._check_hip(err, "hipMemcpy D2H")
                return True
        return False

    def copy_h2d(self, gpu_addr: int, nbytes: int) -> bool:
        """Copy Host→GPU within a registered buffer. Returns False if unregistered."""
        for gpu_base, (_, host_base, buf_size) in self.buffers.items():
            if gpu_base <= gpu_addr < gpu_base + buf_size:
                offset = gpu_addr - gpu_base
                err = self._hip.hipMemcpy(
                    ctypes.c_void_p(gpu_addr),
                    ctypes.c_void_p(host_base + offset),
                    ctypes.c_size_t(nbytes),
                    ctypes.c_int(1),
                )
                self._check_hip(err, "hipMemcpy H2D")
                return True
        return False

    def sync(self):
        if self._stream is not None:
            err = self._hip.hipStreamSynchronize(self._stream)
            self._check_hip(err, "hipStreamSynchronize")
        else:
            err = self._hip.hipDeviceSynchronize()
            self._check_hip(err, "hipDeviceSynchronize")

    # -- host registration for async DMA --------------------------------------

    def hip_host_register(self, host_addr: int, size: int) -> bool:
        """Register mmap+mlock'd memory with HIP for true async DMA.

        Without this, hipMemcpyAsync from pageable host memory falls back
        to synchronous internal staging.  With registration, the DMA engine
        can read directly from the pinned pages.
        """
        try:
            err = self._hip.hipHostRegister(
                ctypes.c_void_p(host_addr),
                ctypes.c_size_t(size),
                ctypes.c_int(0),  # hipHostRegisterDefault
            )
            if err != 0:
                logger.warning(
                    "hipHostRegister failed (err=%d) for %d MB — "
                    "async DMA will use internal staging",
                    err, size >> 20,
                )
                return False
            logger.info(
                "hipHostRegister: %d MB at %s", size >> 20, hex(host_addr)
            )
            return True
        except Exception as exc:
            logger.debug("hipHostRegister unavailable: %s", exc)
            return False

    def hip_host_unregister(self, host_addr: int):
        """Unregister previously registered host memory."""
        try:
            self._hip.hipHostUnregister(ctypes.c_void_p(host_addr))
        except Exception:
            pass

    # -- NUMA affinity ---------------------------------------------------------

    @staticmethod
    def numa_bind_buffer(host_addr: int, size: int, rdma_device: str = None):
        """Bind buffer pages to the NUMA node closest to *rdma_device*.

        Uses ``mbind(MPOL_BIND | MPOL_MF_MOVE)`` so existing pages migrate.
        Falls back silently when the syscall or sysfs lookup fails.
        If *rdma_device* is ``None``, auto-detects the first ionic device.
        """
        try:
            if rdma_device is None:
                for i in range(8):
                    if os.path.exists(f"/sys/class/infiniband/ionic_{i}"):
                        rdma_device = f"ionic_{i}"
                        break

            numa_node = 0
            if rdma_device:
                sysfs = f"/sys/class/infiniband/{rdma_device}/device/numa_node"
                if os.path.exists(sysfs):
                    with open(sysfs) as fh:
                        n = int(fh.read().strip())
                        if n >= 0:
                            numa_node = n

            libc = RocmDramStaging._get_libc()
            MPOL_BIND = 2
            MPOL_MF_MOVE = 2
            nodemask = (ctypes.c_ulong * 1)(1 << numa_node)
            rc = libc.mbind(
                ctypes.c_void_p(host_addr),
                ctypes.c_size_t(size),
                ctypes.c_int(MPOL_BIND),
                ctypes.cast(nodemask, ctypes.POINTER(ctypes.c_ulong)),
                ctypes.c_ulong(64),
                ctypes.c_uint(MPOL_MF_MOVE),
            )
            if rc == 0:
                logger.info(
                    "NUMA bind: %d MB → node %d (%s)",
                    size >> 20, numa_node, rdma_device or "default",
                )
            else:
                logger.warning("mbind returned %d for %d MB", rc, size >> 20)
        except Exception as exc:
            logger.debug("NUMA bind skipped: %s", exc)

    # -- address translation ---------------------------------------------------

    def translate_base(self, gpu_ptr: int) -> int:
        e = self.buffers.get(gpu_ptr)
        return e[1] if e else gpu_ptr

    def translate_ptrs(self, ptrs: list) -> list:
        return [self.translate_base(p) for p in ptrs]

    def translate_addr(self, addr: int) -> int:
        for gpu_base, (_, host_base, buf_size) in self.buffers.items():
            if gpu_base <= addr < gpu_base + buf_size:
                return addr - gpu_base + host_base
        return addr

    def translate_addrs(self, addrs: list) -> list:
        return [self.translate_addr(a) for a in addrs]

    def translate_reqs(self, reqs):
        """Translate GPU addrs in an (N,3) array or list of tuples."""
        import numpy as np

        if isinstance(reqs, np.ndarray):
            if reqs.size == 0:
                return reqs
            result = reqs.copy()
            for gpu_base, (_, host_base, buf_size) in self.buffers.items():
                mask = (result[:, 0] >= gpu_base) & (result[:, 0] < gpu_base + buf_size)
                result[mask, 0] = result[mask, 0] - gpu_base + host_base
                result[mask, 2] = 0
            return result

        out = []
        for item in reqs:
            addr, length, dev = item[0], item[1], item[2]
            for gpu_base, (_, host_base, buf_size) in self.buffers.items():
                if gpu_base <= addr < gpu_base + buf_size:
                    addr = addr - gpu_base + host_base
                    dev = 0
                    break
            out.append(
                (addr, length, dev) if len(item) == 3 else (addr, length, dev, item[3])
            )
        return out
