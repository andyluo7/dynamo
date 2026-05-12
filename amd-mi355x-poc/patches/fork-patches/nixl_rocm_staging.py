# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ROCm DRAM staging monkey-patch for SGLang nixl connector.

On AMD GPUs with Pensando ionic NICs, RIXL/nixl cannot ibv_reg_mr() on
GPU VRAM (no GPU Direct RDMA).  Additionally, ionic has hard limits:
  - Single ibv_reg_mr call rejects buffers > ~199 MB
  - Total registered memory per RDMA device is ~250 MB
  - DeepSeek-R1 needs ~1.7 GB KV data per TP worker

This module patches NixlKVManager / NixlKVReceiver / NixlKVSender at
runtime to work within these constraints:

  1. KV/state buffers are mirrored in host DRAM (mmap+mlock, NOT
     hipHostMalloc — ionic rejects hipHostMalloc in ibv_reg_mr).
  2. Before each RDMA send, GPU blocks are hipMemcpy'd to DRAM,
     then registered, transferred, and deregistered in chunks that
     fit within the ionic per-device MR limit (~250 MB).
  3. The decode side pre-registers a single 200 MB receive slab.
     Prefill RDMA-writes into the slab, then decode copies
     slab → DRAM staging → GPU.
  4. A tiny LD_PRELOAD-style interposer strips IBV_ACCESS_REMOTE_ATOMIC
     (0x8) from ibv_reg_mr() access flags.  UCX hardcodes
     UCP_FEATURE_AMO32|AMO64 in the UCP context which causes
     ibv_reg_mr(access=0xf); ionic rejects that with EINVAL.
     RIXL never uses atomic ops, so stripping is safe.
     The interposer is compiled on-demand and loaded with
     ctypes.CDLL(RTLD_GLOBAL) before UCX initialises its
     IB transport.

Nothing in sglang-amd source is modified — all hooks are injected here.

Usage:
    from dynamo.sglang.nixl_rocm_staging import patch_nixl_for_rocm
    patch_nixl_for_rocm()       # idempotent

Or:  export SGLANG_NIXL_ROCM_STAGING=1

Data flow:
    Prefill GPU KV ─hipMemcpy D2H─▶ Prefill DRAM staging
        ──── RIXL RDMA WRITE (chunked MR) ────▶
    Decode slab ─hipMemcpy H2D─▶ Decode GPU KV   (direct, no staging memcpy)

ionic ibv_reg_mr() limits handled by chunked registration:
  - ~199 MB max single MR, ~250 MB total per device
  - Register/transfer/deregister in chunks ≤ 190 MB so the MR slot
    is reused for every chunk
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
import struct
import subprocess
import tempfile
import threading
from typing import List, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ibv_reg_mr interposer for ionic REMOTE_ATOMIC compatibility
# ---------------------------------------------------------------------------
# UCX hardcodes UCP_FEATURE_AMO32|AMO64 in the UCP context features
# (ucx_utils.cpp:428).  This causes ibv_reg_mr(access=0xf) which includes
# IBV_ACCESS_REMOTE_ATOMIC (0x8).  Ionic NICs reject that flag with EINVAL.
# RIXL never uses atomic ops (only RMA read/write), so stripping the flag
# is safe.  The interposer is loaded with RTLD_GLOBAL before UCX loads its
# IB transport, so the dynamic linker resolves ibv_reg_mr to our wrapper.
#
# UCX upstream PR #10341 would fix this by retrying with reduced access
# flags, but it remains unmerged as of 2026-04.
_IBV_INTERPOSER_SRC = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define IBV_ACCESS_REMOTE_ATOMIC 0x8

void *ibv_reg_mr(void *pd, void *addr, size_t length, int access) {
    typedef void *(*fn_t)(void *, void *, size_t, int);
    static fn_t real_fn;
    if (__builtin_expect(!real_fn, 0)) {
        real_fn = (fn_t)dlsym(RTLD_NEXT, "ibv_reg_mr");
        if (!real_fn) abort();
    }
    access &= ~IBV_ACCESS_REMOTE_ATOMIC;
    return real_fn(pd, addr, length, access);
}

void *ibv_reg_mr_iova2(void *pd, void *addr, size_t length,
                        uint64_t iova, int access) {
    typedef void *(*fn_t)(void *, void *, size_t, uint64_t, int);
    static fn_t real_fn;
    if (__builtin_expect(!real_fn, 0)) {
        real_fn = (fn_t)dlsym(RTLD_NEXT, "ibv_reg_mr_iova2");
        if (!real_fn) abort();
    }
    access &= ~IBV_ACCESS_REMOTE_ATOMIC;
    return real_fn(pd, addr, length, iova, access);
}
"""

_IBV_INTERPOSER_LOADED = False


def _ensure_ibv_ionic_interposer() -> bool:
    """Build (if needed) and load the ibv_reg_mr interposer for ionic.

    Strips IBV_ACCESS_REMOTE_ATOMIC from all ibv_reg_mr calls so ionic
    NICs accept the memory registration.  Returns True if the interposer
    is active in the current process.
    """
    global _IBV_INTERPOSER_LOADED
    if _IBV_INTERPOSER_LOADED:
        return True

    so_path = os.path.join(tempfile.gettempdir(), "ibv_ionic_compat.so")

    if not os.path.exists(so_path):
        c_path = so_path.replace(".so", ".c")
        try:
            with open(c_path, "w") as f:
                f.write(_IBV_INTERPOSER_SRC)
            result = subprocess.run(
                ["gcc", "-shared", "-fPIC", "-O2", "-o", so_path, c_path, "-ldl"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                logger.warning(
                    "Failed to compile ibv_reg_mr interposer: %s", result.stderr
                )
                return False
            logger.info("Built ibv_reg_mr interposer: %s", so_path)
        except Exception as e:
            logger.warning("Failed to build ibv_reg_mr interposer: %s", e)
            return False

    existing = os.environ.get("LD_PRELOAD", "")
    if so_path not in existing:
        os.environ["LD_PRELOAD"] = f"{so_path}:{existing}" if existing else so_path
        logger.info("Set LD_PRELOAD=%s for forked scheduler processes", so_path)

    # Load in current process too (for threads that don't fork).
    # Use mode=0 (not RTLD_GLOBAL) to avoid conflicting with
    # already-loaded libibverbs; LD_PRELOAD handles the forked children.
    try:
        ctypes.CDLL(so_path, mode=os.RTLD_LAZY)
    except OSError:
        pass

    _IBV_INTERPOSER_LOADED = True
    logger.info(
        "ibv_reg_mr interposer ready — "
        "IBV_ACCESS_REMOTE_ATOMIC stripped for ionic compat"
    )

    return True

# ---------------------------------------------------------------------------
# Auto-detect ROCm
# ---------------------------------------------------------------------------
_IS_ROCM = False
try:
    import torch as _torch

    _IS_ROCM = hasattr(_torch.version, "hip") and _torch.version.hip is not None
except ImportError:
    pass

_STAGING_ENABLED = _IS_ROCM or os.environ.get(
    "SGLANG_NIXL_ROCM_STAGING", ""
).lower() in ("1", "true", "yes")

_PATCHED = False  # guard against double-patch


# ---------------------------------------------------------------------------
# DRAM staging pool — shared implementation (after _IS_ROCM detection)
# ---------------------------------------------------------------------------
from dynamo.sglang.rocm_dram_staging_common import (  # noqa: E402
    RocmDramStaging as _RocmDramStaging,
)


# ---------------------------------------------------------------------------
# ionic MR constraints
# ---------------------------------------------------------------------------
_IONIC_MR_DEVICE_LIMIT_MB = int(os.environ.get("NIXL_IONIC_MR_DEVICE_LIMIT_MB", "250"))
_IONIC_MR_CHUNK = int(os.environ.get("NIXL_IONIC_MR_CHUNK_MB", "190")) * 1024 * 1024
_SLAB_MB = int(os.environ.get("NIXL_RECV_SLAB_MB", "200"))


def _safe_chunk_size(tp_size: int) -> int:
    """Per-worker MR chunk that keeps all TP workers within ionic per-device limit.

    All TP workers may register MRs on the same ionic device simultaneously.
    Each device has ~250MB total MR capacity, so:
      chunk_per_worker = (device_limit - margin) / tp_size
    For TP=8: (250-10)/8 = 30MB per worker.
    """
    if tp_size <= 1:
        return _IONIC_MR_CHUNK
    safe = ((_IONIC_MR_DEVICE_LIMIT_MB - 10) // tp_size) * 1024 * 1024
    safe = max(safe, 10 * 1024 * 1024)  # floor at 10MB
    if safe < _IONIC_MR_CHUNK:
        return safe
    return _IONIC_MR_CHUNK


# Management subnets that lack RDMA L2 connectivity (pre-configured by OS,
# NOT added by setup_network.sh).  UCX must avoid these GIDs.
_MGMT_SUBNETS = frozenset(os.environ.get(
    "NIXL_IONIC_MGMT_SUBNETS", "192.168.1,192.168.48"
).split(","))


def _parse_ipv4_gid(gid_hex: str) -> Optional[str]:
    """Extract dotted-quad IPv4 from an IPv4-mapped IPv6 GID string.

    GID format: ``0000:0000:0000:0000:0000:ffff:AABB:CCDD``
    Returns ``"A.B.C.D"`` or None.
    """
    if "ffff:" not in gid_hex:
        return None
    parts = gid_hex.strip().split(":")
    if len(parts) < 8:
        return None
    try:
        hi = int(parts[6], 16)
        lo = int(parts[7], 16)
        return f"{hi >> 8}.{hi & 0xFF}.{lo >> 8}.{lo & 0xFF}"
    except (ValueError, IndexError):
        return None


def _get_ionic_device_subnet_map(devs: list[str]) -> dict[str, tuple[str, int]]:
    """Build device→(subnet, gid_idx) map from sysfs GID tables.

    Only includes devices with IPv4-mapped GIDs that are NOT management subnets.
    """
    dev_map: dict[str, tuple[str, int]] = {}
    for dev in devs:
        gid_dir = f"/sys/class/infiniband/{dev}/ports/1/gids"
        if not os.path.isdir(gid_dir):
            continue
        for idx in range(16):
            try:
                with open(f"{gid_dir}/{idx}") as f:
                    gid_hex = f.read().strip()
            except (OSError, IOError):
                break
            if not gid_hex or gid_hex == "0000:0000:0000:0000:0000:0000:0000:0000":
                break
            ipv4 = _parse_ipv4_gid(gid_hex)
            if ipv4 is None:
                continue
            subnet = ".".join(ipv4.split(".")[:3])
            if subnet in _MGMT_SUBNETS:
                continue
            dev_map[dev] = (subnet, idx)
            break
    return dev_map


def _get_ionic_rdma_config() -> tuple[Optional[str], Optional[int]]:
    """Auto-detect ionic devices with RDMA-capable GIDs.

    For UCX-based backends (RIXL), restricts device_list to a SINGLE device
    matching the current GPU's subnet affinity. This prevents UCX from
    advertising connections on mismatched subnets (ionic same-subnet
    requirement). Both nodes must have IPv4 assigned via setup_ionic_network.sh.

    Returns:
        (device_list, gid_index) — device name for UCX's
        ``create_backend("UCX", {"device_list": ...})``,
        and the GID index to use (for ``UCX_IB_GID_INDEX``).
        Returns (None, None) if no ionic devices found.
    """
    ib_dev_env = os.environ.get("SGLANG_DISAGGREGATION_IB_DEVICE", "")
    if not ib_dev_env:
        import sys
        for i, arg in enumerate(sys.argv):
            if arg == "--disaggregation-ib-device" and i + 1 < len(sys.argv):
                ib_dev_env = sys.argv[i + 1]
                break
    if not ib_dev_env:
        if os.path.exists("/sys/class/infiniband/ionic_0"):
            ib_dev_env = ",".join(f"ionic_{i}" for i in range(8))
        else:
            return None, None

    devs = [d.strip() for d in ib_dev_env.split(",") if d.strip()]
    dev_map = _get_ionic_device_subnet_map(devs)

    if not dev_map:
        logger.warning("No ionic devices with RDMA-capable GIDs found")
        return None, None

    # Select a single device for this GPU process.
    # With TP=1, each process uses one GPU (CUDA_VISIBLE_DEVICES).
    # Pick device by round-robin over sorted subnets to ensure both nodes
    # select devices on the SAME subnet for the same GPU index.
    try:
        import torch
        gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
    except Exception:
        gpu_id = 0

    sorted_devs = sorted(dev_map.keys(), key=lambda d: dev_map[d][0])
    selected = sorted_devs[gpu_id % len(sorted_devs)]
    subnet, gid_idx = dev_map[selected]

    logger.info(
        "Ionic RDMA: GPU %d → %s (subnet %s, GID[%d]) — "
        "single device for UCX (avoids cross-subnet connections)",
        gpu_id, selected, subnet, gid_idx,
    )

    # Log full map for debugging
    all_devs = ", ".join(f"{d}={dev_map[d][0]}" for d in sorted_devs)
    logger.info("Ionic subnet map: %s", all_devs)

    return selected, gid_idx


class _CompletedTransfer:
    """Sentinel returned by synchronous chunked transfers.

    NixlKVSender.poll() checks isinstance() on xfer handles; this
    sentinel signals "already done" without needing a real RIXL handle.
    """

    pass


# ---------------------------------------------------------------------------
# Vectorized D2H copy helper
# ---------------------------------------------------------------------------
def _batch_d2h_copy(staging: _RocmDramStaging, src_reqs: np.ndarray):
    """Async D2H copy grouped by staging buffer (one hipMemcpyAsync per buffer)."""
    if len(src_reqs) == 0:
        return
    addrs = src_reqs[:, 0].astype(np.int64)
    sizes = src_reqs[:, 1].astype(np.int64)

    for gpu_base, (_, host_base, buf_size) in staging.buffers.items():
        mask = (addrs >= gpu_base) & (addrs < gpu_base + buf_size)
        if not mask.any():
            continue
        offsets = addrs[mask] - gpu_base
        ends = offsets + sizes[mask]
        lo = int(offsets.min())
        hi = int(ends.max())
        staging.copy_d2h_direct(gpu_base + lo, host_base + lo, hi - lo)

    staging.sync()


# ---------------------------------------------------------------------------
# Chunked register → RDMA WRITE → deregister
# ---------------------------------------------------------------------------
def _chunked_rixl_write(
    agent,
    staging: _RocmDramStaging,
    src_reqs: np.ndarray,
    dst_reqs: np.ndarray,
    peer_name: str,
    notif_msg: bytes,
    mr_lock: threading.Lock,
    chunk_size: int = None,
):
    """Register→RDMA WRITE→deregister in ionic-safe chunks.

    ``src_reqs`` / ``dst_reqs`` are Nx3 int64 numpy arrays
    ``(addr, size, dev_id)``.  Source addresses must already be DRAM
    staging addresses (post-translate, post-D2H).

    Groups source descriptors by staging buffer, registers bounding MR
    ranges in chunks whose total registered bytes ≤ ``chunk_size``,
    transfers synchronously, then deregisters.  Only the **last** chunk
    carries ``notif_msg``; earlier chunks use empty notifications.
    """
    if chunk_size is None:
        chunk_size = _IONIC_MR_CHUNK
    n = len(src_reqs)
    if n == 0:
        return _CompletedTransfer()

    src_addrs = src_reqs[:, 0].astype(np.int64)
    src_sizes = src_reqs[:, 1].astype(np.int64)

    # 1. Compute bounding MR region per staging buffer (keyed by host_base)
    mr_regions = {}  # host_base → (min_addr, max_end_addr)
    desc_buf_base = np.full(n, -1, dtype=np.int64)

    for gpu_base, (_, host_base, buf_size) in staging.buffers.items():
        mask = (src_addrs >= host_base) & (src_addrs < host_base + buf_size)
        if not mask.any():
            continue
        desc_buf_base[mask] = host_base
        ends = src_addrs[mask] + src_sizes[mask]
        lo = int(src_addrs[mask].min())
        hi = int(ends.max())
        if host_base in mr_regions:
            prev_lo, prev_hi = mr_regions[host_base]
            mr_regions[host_base] = (min(prev_lo, lo), max(prev_hi, hi))
        else:
            mr_regions[host_base] = (lo, hi)

    # Handle orphan descriptors (not in any known buffer)
    orphan_mask = desc_buf_base == -1
    if orphan_mask.any():
        for i in np.where(orphan_mask)[0]:
            addr = int(src_addrs[i])
            size = int(src_sizes[i])
            desc_buf_base[i] = addr
            if addr in mr_regions:
                lo, hi = mr_regions[addr]
                mr_regions[addr] = (min(lo, addr), max(hi, addr + size))
            else:
                mr_regions[addr] = (addr, addr + size)

    # 2. Group MR regions into chunks ≤ chunk_size
    mr_list = list(mr_regions.items())
    chunks: list[list[tuple]] = []
    cur_chunk: list[tuple] = []
    cur_bytes = 0
    for host_base, (lo, hi) in mr_list:
        region_sz = hi - lo
        if cur_bytes + region_sz > chunk_size and cur_chunk:
            chunks.append(cur_chunk)
            cur_chunk = []
            cur_bytes = 0
        cur_chunk.append((host_base, lo, hi))
        cur_bytes += region_sz
    if cur_chunk:
        chunks.append(cur_chunk)

    num_chunks = len(chunks)
    logger.debug(
        "Chunked RIXL write: %d descs, %d MR regions, %d chunks → %s",
        n,
        len(mr_regions),
        num_chunks,
        peer_name,
    )

    # 3. For each chunk: register → transfer → poll → deregister
    for cidx, chunk_mrs in enumerate(chunks):
        is_last = cidx == num_chunks - 1
        chunk_bases = np.array([base for base, _, _ in chunk_mrs], dtype=np.int64)

        desc_mask = np.isin(desc_buf_base, chunk_bases)
        chunk_src = src_reqs[desc_mask]
        chunk_dst = dst_reqs[desc_mask]

        if len(chunk_src) == 0:
            continue

        reg_addrs = [(lo, hi - lo, 0, "") for _, lo, hi in chunk_mrs]

        with mr_lock:
            reg_descs = agent.register_memory(reg_addrs, "DRAM")
            try:
                src_descs = agent.get_xfer_descs(chunk_src, "DRAM")
                dst_descs = agent.get_xfer_descs(chunk_dst, "DRAM")

                chunk_notif = notif_msg if is_last else b""
                xfer_handle = agent.initialize_xfer(
                    "WRITE",
                    src_descs,
                    dst_descs,
                    peer_name,
                    chunk_notif,
                )
                if not xfer_handle:
                    raise Exception(
                        f"Chunked RIXL write: create xfer failed "
                        f"(chunk {cidx + 1}/{num_chunks})"
                    )

                state = agent.transfer(xfer_handle)
                if state == "ERR":
                    raise Exception(
                        f"Chunked RIXL write: post xfer failed "
                        f"(chunk {cidx + 1}/{num_chunks})"
                    )

                while True:
                    state = agent.check_xfer_state(xfer_handle)
                    if state == "DONE":
                        break
                    if state == "ERR":
                        raise Exception(
                            f"Chunked RIXL write: xfer error "
                            f"(chunk {cidx + 1}/{num_chunks})"
                        )

                agent.release_xfer_handle(xfer_handle)
            finally:
                agent.deregister_memory(reg_descs)

    return _CompletedTransfer()


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def patch_nixl_for_rocm():
    """Monkey-patch SGLang's NixlKVManager / NixlKVReceiver / NixlKVSender
    for ROCm DRAM staging with chunked MR and receive slab.

    Safe to call multiple times (idempotent).  Does nothing when not on ROCm
    and ``SGLANG_NIXL_ROCM_STAGING`` is not set.
    """
    global _PATCHED
    if _PATCHED:
        return
    if not _STAGING_ENABLED:
        logger.debug("ROCm staging not enabled, skipping nixl patch")
        return

    try:
        from sglang.srt.disaggregation.nixl.conn import (
            GUARD,
            NixlKVManager,
            NixlKVReceiver,
            NixlKVSender,
        )
    except ImportError:
        logger.warning("sglang nixl connector not available, skipping ROCm patch")
        return

    from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous

    _required_attrs = [
        (NixlKVManager, "__init__"),
        (NixlKVManager, "register_buffer_to_engine"),
        (NixlKVManager, "_send_kvcache_generic"),
        (NixlKVReceiver, "_register_kv_args"),
        (NixlKVReceiver, "init"),
        (NixlKVReceiver, "poll"),
        (NixlKVSender, "poll"),
    ]
    for cls, attr in _required_attrs:
        if not hasattr(cls, attr):
            logger.error(
                "ROCm DRAM staging patch ABORTED: %s.%s not found. "
                "SGLang API may have changed — update the monkey-patch.",
                cls.__name__,
                attr,
            )
            return

    logger.info(
        "Applying ROCm DRAM staging monkey-patch to NixlKVManager "
        "(chunked MR %d MB, slab %d MB)",
        _IONIC_MR_CHUNK // (1024 * 1024),
        _SLAB_MB,
    )

    # ---- save originals -----------------------------------------------------
    _orig_mgr_init = NixlKVManager.__init__
    _orig_register = NixlKVManager.register_buffer_to_engine
    _orig_recv_register_kv = NixlKVReceiver._register_kv_args
    _orig_recv_init = NixlKVReceiver.init
    _orig_recv_poll = NixlKVReceiver.poll
    _orig_sender_poll = NixlKVSender.poll

    # ---- 1. NixlKVManager.__init__ ------------------------------------------
    def _patched_mgr_init(self, *args, **kwargs):
        self._rocm_staging = _RocmDramStaging()
        self._mr_lock = threading.Lock()
        self._ionic_chunk_size = None  # set after init when tp_size is known

        # Auto-detect ionic RDMA devices and GID index.
        # Both sides must restrict the UCX context to RDMA-capable GIDs
        # so worker addresses in metadata only advertise reachable devices.
        ionic_devs, gid_idx = _get_ionic_rdma_config()
        if ionic_devs:
            # Load ibv_reg_mr interposer BEFORE UCX loads its IB transport.
            # This strips IBV_ACCESS_REMOTE_ATOMIC (0x8) from ibv_reg_mr
            # access flags — ionic rejects that flag with EINVAL because
            # it doesn't support RDMA atomics, but UCX sets it due to
            # hardcoded UCP_FEATURE_AMO32/64 in RIXL.
            if not _ensure_ibv_ionic_interposer():
                logger.warning(
                    "ibv_reg_mr interposer not loaded — ionic MR registration "
                    "may fail with EINVAL (IBV_ACCESS_REMOTE_ATOMIC).  "
                    "Fallback: set LD_PRELOAD to the interposer .so before "
                    "launching the process."
                )

            # Set UCX env vars BEFORE backend creation (ucp_config_read)
            if gid_idx is not None:
                os.environ.setdefault("UCX_IB_GID_INDEX", str(gid_idx))
            os.environ.setdefault("UCX_TLS", "rc_v,tcp")
            os.environ.setdefault("UCX_IB_REG_METHODS", "direct")
            # Performance tuning for ionic:
            os.environ.setdefault("UCX_RNDV_THRESH", "1048576")  # eager up to 1MB
            os.environ.setdefault("UCX_RC_TIMEOUT", "100ms")
            os.environ.setdefault("UCX_RC_RETRY_COUNT", "12")
            # Reduce RC path setup overhead
            os.environ.setdefault("UCX_RC_PATH_MTU", "4096")
            # Reuse RC connections aggressively
            os.environ.setdefault("UCX_RC_FC_WND_SIZE", "512")
            os.environ.setdefault("UCX_RC_TX_NUM_GET_BYTES", "4194304")
            # Pre-allocate larger TX/RX buffers to avoid resizing
            os.environ.setdefault("UCX_RC_TX_QUEUE_LEN", "256")
            # NOTE: UCX_IB_DISABLE_ATOMIC is NOT set here — it would break
            # RCCL/NCCL which also uses UCX for inter-GPU communication.
            # The patched libuct_ib.so reads this env var in uct_ib_reg_mr()
            # but it must only be set for RIXL processes, not globally.
            # Instead, the uct_ib_reg_mr patch should detect ionic vendor
            # and skip IBV_ACCESS_REMOTE_ATOMIC automatically.

            from nixl._api import nixl_agent as _nixl_agent_cls

            _real_create_backend = _nixl_agent_cls.create_backend

            def _filtered_create_backend(agent_self, backend, initParams=None):
                if initParams is None:
                    initParams = {}
                if backend == "UCX" and "device_list" not in initParams:
                    initParams["device_list"] = ionic_devs
                    logger.info(
                        "ROCm staging: UCX backend device_list='%s', "
                        "GID_INDEX=%s, TLS=%s",
                        ionic_devs,
                        os.environ.get("UCX_IB_GID_INDEX", "default"),
                        os.environ.get("UCX_TLS", "default"),
                    )
                return _real_create_backend(agent_self, backend, initParams)

            _nixl_agent_cls.create_backend = _filtered_create_backend

        _orig_mgr_init(self, *args, **kwargs)

        if ionic_devs:
            _nixl_agent_cls.create_backend = _real_create_backend

        tp_size = getattr(self, 'tp_size', 8)
        if not hasattr(self, 'tp_size'):
            try:
                tp_size = self.server_args.tp_size
            except:
                tp_size = 8
        self._ionic_chunk_size = _safe_chunk_size(tp_size)
        logger.info(
            "ROCm staging: NixlKVManager initialized with chunked MR "
            "(chunk %d MB, TP=%d)",
            self._ionic_chunk_size // (1024 * 1024), tp_size,
        )

    NixlKVManager.__init__ = _patched_mgr_init

    # ---- 2. register_buffer_to_engine — DRAM mirrors + slab -----------------
    def _patched_register(self):
        staging: _RocmDramStaging = self._rocm_staging
        is_decode = self.disaggregation_mode.value == "decode"

        # Allocate DRAM staging mirrors for KV buffers (both sides)
        for gpu_ptr, data_len in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
        ):
            staging.create(gpu_ptr, data_len, allocate=True)

        # Allocate DRAM staging mirrors for state buffers
        if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
            for gpu_ptr, data_len in zip(
                self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
            ):
                staging.create(gpu_ptr, data_len, allocate=True)

        if is_decode:
            # -- State buffers: register as DRAM in chunks --
            if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
                state_addrs = []
                for gpu_ptr, data_len in zip(
                    self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
                ):
                    host_ptr = staging.translate_base(gpu_ptr)
                    offset = 0
                    while offset < data_len:
                        chunk = min(_IONIC_MR_CHUNK, data_len - offset)
                        state_addrs.append((host_ptr + offset, chunk, 0, ""))
                        offset += chunk
                self.state_descs = self.agent.register_memory(state_addrs, "DRAM")
                if not self.state_descs:
                    raise Exception(
                        "NIXL DRAM registration failed for state tensors (decode)"
                    )

            # -- Receive slab for KV data --
            slab_size = _SLAB_MB * 1024 * 1024
            slab_mmap = mmap.mmap(-1, slab_size)
            slab_buf = (ctypes.c_char * slab_size).from_buffer(slab_mmap)
            slab_ptr = ctypes.addressof(slab_buf)
            libc = _RocmDramStaging._get_libc()
            libc.mlock(ctypes.c_void_p(slab_ptr), ctypes.c_size_t(slab_size))

            # NUMA-bind slab pages to the ionic NIC's node, then register
            # with HIP so hipMemcpyAsync can DMA directly from the slab.
            _ionic_dev, _ = _get_ionic_rdma_config()
            _RocmDramStaging.numa_bind_buffer(
                slab_ptr, slab_size,
                _ionic_dev.split(",")[0].strip() if _ionic_dev else None,
            )
            staging.hip_host_register(slab_ptr, slab_size)

            self.kv_descs = self.agent.register_memory(
                [(slab_ptr, slab_size, 0, "")], "DRAM"
            )
            if not self.kv_descs:
                raise Exception(
                    "NIXL DRAM registration failed for receive slab (decode)"
                )
            self._recv_slab = (slab_mmap, slab_buf, slab_ptr, slab_size)

            logger.info(
                "ROCm staging DECODE: %d MB slab at %s, "
                "%d KV + %d state DRAM mirrors",
                _SLAB_MB,
                hex(slab_ptr),
                len(self.kv_args.kv_data_ptrs),
                len(self.kv_args.state_data_ptrs)
                if self.kv_args.state_data_ptrs
                else 0,
            )
        else:
            # PREFILL: no MR pre-registration — chunked per-transfer
            self.kv_descs = True  # placeholder
            if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
                self.state_descs = True
            logger.info(
                "ROCm staging PREFILL: %d KV + %d state DRAM mirrors (no pre-reg)",
                len(self.kv_args.kv_data_ptrs),
                len(self.kv_args.state_data_ptrs)
                if self.kv_args.state_data_ptrs
                else 0,
            )

        # Aux buffers — small CPU/DRAM, always register directly
        aux_addrs = [
            (ptr, length, 0, "")
            for ptr, length in zip(
                self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
            )
        ]
        self.aux_descs = self.agent.register_memory(aux_addrs, "DRAM")
        if not self.aux_descs:
            raise Exception("NIXL DRAM registration failed for aux tensors")

    NixlKVManager.register_buffer_to_engine = _patched_register

    # ---- 3. Helper: D2H + translate + chunked write -------------------------
    def _do_rocm_staged_write(self, src_reqs_gpu, dst_reqs, peer_name, notif_msg):
        """D2H copy, translate GPU→DRAM addresses, chunked MR transfer."""
        staging: _RocmDramStaging = self._rocm_staging
        if len(src_reqs_gpu) == 0:
            return _CompletedTransfer()

        _batch_d2h_copy(staging, src_reqs_gpu)
        dram_src = staging.translate_reqs(src_reqs_gpu)

        dst_dram = dst_reqs.copy()
        dst_dram[:, 2] = 0  # ensure DRAM device id

        return _chunked_rixl_write(
            self.agent,
            staging,
            dram_src,
            dst_dram,
            peer_name,
            notif_msg,
            self._mr_lock,
            chunk_size=getattr(self, '_ionic_chunk_size', _IONIC_MR_CHUNK),
        )

    NixlKVManager._do_rocm_staged_write = _do_rocm_staged_write

    # ---- 4. _send_kvcache_generic — chunked DRAM staging --------------------
    def _patched_send_kvcache_generic(
        self,
        peer_name,
        src_data_ptrs,
        dst_data_ptrs,
        item_lens,
        prefill_data_indices,
        dst_data_indices,
        dst_gpu_id,
        notif,
    ):
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_data_indices, dst_data_indices
        )

        logger.debug(
            "sending kvcache to %s with notif %s (ROCm staged)", peer_name, notif
        )

        if self.is_mla_backend:
            src_kv_ptrs, dst_kv_ptrs, layers_pp = self.get_mla_kv_ptrs_with_pp(
                src_data_ptrs, dst_data_ptrs
            )
            layers_params = [
                (src_kv_ptrs[lid], dst_kv_ptrs[lid], item_lens[lid])
                for lid in range(layers_pp)
            ]
        else:
            src_k, src_v, dst_k, dst_v, layers_pp = self.get_mha_kv_ptrs_with_pp(
                src_data_ptrs, dst_data_ptrs
            )
            layers_params = [
                (src_k[lid], dst_k[lid], item_lens[lid]) for lid in range(layers_pp)
            ] + [
                (src_v[lid], dst_v[lid], item_lens[lid]) for lid in range(layers_pp)
            ]

        src_addrs: list = []
        src_lens: list = []
        dst_addrs: list = []
        dst_lens: list = []

        prefill_starts = np.fromiter(
            (b[0] for b in prefill_kv_blocks), dtype=np.int64
        )
        dst_starts = np.fromiter((b[0] for b in dst_kv_blocks), dtype=np.int64)
        block_lens = np.fromiter(
            (len(b) for b in prefill_kv_blocks), dtype=np.int64
        )

        for src_ptr, dst_ptr, item_len in layers_params:
            lengths = item_len * block_lens
            src_addrs.append(src_ptr + prefill_starts * item_len)
            src_lens.append(lengths)
            dst_addrs.append(dst_ptr + dst_starts * item_len)
            dst_lens.append(lengths)

        def make_req_array(addr_chunks, len_chunks, gpu):
            if not addr_chunks:
                return np.empty((0, 3), dtype=np.int64)
            flat_addrs = np.concatenate(addr_chunks)
            flat_lens = np.concatenate(len_chunks)
            return np.column_stack(
                (flat_addrs, flat_lens, np.full_like(flat_addrs, gpu))
            )

        src_reqs = make_req_array(src_addrs, src_lens, self.kv_args.gpu_id)
        dst_reqs = make_req_array(dst_addrs, dst_lens, dst_gpu_id)

        return self._do_rocm_staged_write(
            src_reqs, dst_reqs, peer_name, notif.encode("ascii")
        )

    NixlKVManager._send_kvcache_generic = _patched_send_kvcache_generic

    # ---- 5. send_kvcache_slice — chunked DRAM staging -----------------------
    if hasattr(NixlKVManager, "send_kvcache_slice"):

        def _patched_send_kvcache_slice(
            self,
            peer_name,
            prefill_kv_indices,
            dst_kv_ptrs,
            dst_kv_indices,
            dst_gpu_id,
            notif,
            prefill_tp_size,
            decode_tp_size,
            decode_tp_rank,
            dst_kv_item_len,
        ):
            local_tp_rank = self.kv_args.engine_rank % prefill_tp_size
            dst_tp_rank = decode_tp_rank % decode_tp_size
            num_kv_heads = self.kv_args.kv_head_num

            src_heads_per_rank = num_kv_heads
            dst_heads_per_rank = num_kv_heads * prefill_tp_size // decode_tp_size

            src_kv_item_len = self.kv_args.kv_item_lens[0]
            page_size = self.kv_args.page_size

            bytes_per_head_slice = dst_kv_item_len // page_size // dst_heads_per_rank

            if prefill_tp_size > decode_tp_size:
                src_head_start = 0
                num_heads_to_send = src_heads_per_rank
                dst_head_start = local_tp_rank * src_heads_per_rank
            else:
                src_head_start = (
                    dst_tp_rank * dst_heads_per_rank
                ) % src_heads_per_rank
                num_heads_to_send = dst_heads_per_rank
                dst_head_start = 0

            src_k, src_v, dst_k, dst_v, layers_pp = self.get_mha_kv_ptrs_with_pp(
                self.kv_args.kv_data_ptrs, dst_kv_ptrs
            )

            src_head_slice_off = src_head_start * bytes_per_head_slice
            dst_head_slice_off = dst_head_start * bytes_per_head_slice
            heads_bytes_to_send = num_heads_to_send * bytes_per_head_slice

            src_dst_pairs = [
                (src_k[lid], dst_k[lid]) for lid in range(layers_pp)
            ] + [(src_v[lid], dst_v[lid]) for lid in range(layers_pp)]

            prefill_idx = np.asarray(prefill_kv_indices, dtype=np.int64)
            dst_idx = np.asarray(dst_kv_indices, dtype=np.int64)
            bytes_per_token_src = src_kv_item_len // page_size
            bytes_per_token_dst = dst_kv_item_len // page_size
            token_offsets = np.arange(page_size, dtype=np.int64)

            src_addr_chunks: list = []
            dst_addr_chunks: list = []

            for src_ptr, dst_ptr in src_dst_pairs:
                src_page_bases = src_ptr + prefill_idx * src_kv_item_len
                dst_page_bases = dst_ptr + dst_idx * dst_kv_item_len

                src_all = (
                    src_page_bases[:, None]
                    + token_offsets[None, :] * bytes_per_token_src
                    + src_head_slice_off
                ).ravel()
                dst_all = (
                    dst_page_bases[:, None]
                    + token_offsets[None, :] * bytes_per_token_dst
                    + dst_head_slice_off
                ).ravel()

                src_addr_chunks.append(src_all)
                dst_addr_chunks.append(dst_all)

            def make_req_array(addr_chunks, size, gpu):
                if not addr_chunks:
                    return np.empty((0, 3), dtype=np.int64)
                flat_addrs = np.concatenate(addr_chunks)
                return np.column_stack(
                    (
                        flat_addrs,
                        np.full_like(flat_addrs, size),
                        np.full_like(flat_addrs, gpu),
                    )
                )

            src_reqs = make_req_array(
                src_addr_chunks, heads_bytes_to_send, self.kv_args.gpu_id
            )
            dst_reqs = make_req_array(
                dst_addr_chunks, heads_bytes_to_send, dst_gpu_id
            )

            return self._do_rocm_staged_write(
                src_reqs, dst_reqs, peer_name, notif.encode("ascii")
            )

        NixlKVManager.send_kvcache_slice = _patched_send_kvcache_slice

    # ---- 6. _send_mamba_state — chunked DRAM staging ------------------------
    if hasattr(NixlKVManager, "_send_mamba_state"):

        def _patched_send_mamba_state(
            self,
            peer_name,
            prefill_state_indices,
            dst_state_data_ptrs,
            dst_state_indices,
            dst_gpu_id,
            notif,
        ):
            assert len(prefill_state_indices) == 1, (
                "Mamba should have single state index"
            )
            assert len(dst_state_indices) == len(prefill_state_indices), (
                "State indices count mismatch"
            )

            src_tuples = []
            dst_tuples = []

            for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
                length = self.kv_args.state_item_lens[i]
                src_addr = self.kv_args.state_data_ptrs[i] + length * int(
                    prefill_state_indices[0]
                )
                dst_addr = dst_state_ptr + length * int(dst_state_indices[0])
                src_tuples.append((src_addr, length, self.kv_args.gpu_id))
                dst_tuples.append((dst_addr, length, dst_gpu_id))

            src_reqs = np.array(src_tuples, dtype=np.int64)
            dst_reqs = np.array(dst_tuples, dtype=np.int64)

            return self._do_rocm_staged_write(
                src_reqs, dst_reqs, peer_name, notif.encode("ascii")
            )

        NixlKVManager._send_mamba_state = _patched_send_mamba_state

    # ---- 7. _staging_post_copy_h2d (new helper) ----------------------------
    def _staging_post_copy_h2d(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        """Copy received slots from DRAM staging → GPU after RDMA receive."""
        staging: _RocmDramStaging = self._rocm_staging

        for buf_idx, (gpu_ptr, data_len) in enumerate(
            zip(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens)
        ):
            if gpu_ptr not in staging.buffers:
                continue
            if self.is_mla_backend:
                item_len = self.kv_args.kv_item_lens[buf_idx]
            else:
                item_len = self.kv_args.kv_item_lens[buf_idx // 2]
            for idx in kv_indices:
                offset = int(idx) * item_len
                if offset + item_len <= data_len:
                    staging.copy_h2d(gpu_ptr + offset, item_len)

        if (
            state_indices
            and self.kv_args.state_data_ptrs
            and self.kv_args.state_item_lens
        ):
            for buf_idx, gpu_ptr in enumerate(self.kv_args.state_data_ptrs):
                if gpu_ptr not in staging.buffers:
                    continue
                _, _, buf_size = staging.buffers[gpu_ptr]
                item_len = self.kv_args.state_item_lens[buf_idx]
                for idx in state_indices:
                    offset = int(idx) * item_len
                    if offset + item_len <= buf_size:
                        staging.copy_h2d(gpu_ptr + offset, item_len)

        staging.sync()

    NixlKVManager._staging_post_copy_h2d = _staging_post_copy_h2d

    # ---- 8. NixlKVReceiver._register_kv_args — slab addressing -------------
    def _patched_register_kv_args(self):
        staging: _RocmDramStaging = self.kv_mgr._rocm_staging

        if hasattr(self.kv_mgr, "_recv_slab"):
            _, _, slab_ptr, slab_size = self.kv_mgr._recv_slab
            num_kv = len(self.kv_mgr.kv_args.kv_data_ptrs)
            layer_slab = slab_size // max(num_kv, 1)
            kv_ptrs = [slab_ptr + i * layer_slab for i in range(num_kv)]
            logger.debug(
                "ROCm staging: advertising slab addresses for %d KV layers "
                "(%.1f MB/layer)",
                num_kv,
                layer_slab / (1024 * 1024),
            )
        else:
            kv_ptrs = staging.translate_ptrs(self.kv_mgr.kv_args.kv_data_ptrs)

        state_ptrs = staging.translate_ptrs(
            self.kv_mgr.kv_args.state_data_ptrs or []
        )
        advertised_gpu_id = 0  # DRAM

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)

            packed_kv = b"".join(struct.pack("Q", p) for p in kv_ptrs)
            packed_aux = b"".join(
                struct.pack("Q", p) for p in self.kv_mgr.kv_args.aux_data_ptrs
            )
            packed_state = b"".join(struct.pack("Q", p) for p in state_ptrs)

            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent.name.encode("ascii"),
                        self.kv_mgr.agent.get_agent_metadata(),
                        packed_kv,
                        packed_aux,
                        packed_state,
                        str(advertised_gpu_id).encode("ascii"),
                        str(self.kv_mgr.attn_tp_size).encode("ascii"),
                        str(self.kv_mgr.kv_args.engine_rank).encode("ascii"),
                        str(self.kv_mgr.kv_args.kv_item_lens[0]).encode("ascii"),
                    ]
                )

    NixlKVReceiver._register_kv_args = _patched_register_kv_args

    # ---- 9. NixlKVReceiver.init — store indices for post-copy ---------------
    def _patched_recv_init(self, kv_indices, aux_index=None, state_indices=None):
        self._staging_kv_indices = kv_indices.copy()
        self._staging_state_indices = (
            list(state_indices) if state_indices is not None else None
        )
        return _orig_recv_init(self, kv_indices, aux_index, state_indices)

    NixlKVReceiver.init = _patched_recv_init

    # ---- 10. NixlKVReceiver.poll — slab→GPU (direct) or DRAM→GPU -----------
    def _patched_recv_poll(self):
        from sglang.srt.disaggregation.base.conn import KVPoll

        result = _orig_recv_poll(self)
        if result == KVPoll.Success and hasattr(self, "_staging_kv_indices"):
            staging: _RocmDramStaging = self.kv_mgr._rocm_staging
            state_indices = getattr(self, "_staging_state_indices", None)

            if hasattr(self.kv_mgr, "_recv_slab"):
                # Slab mode: slab → GPU directly via hipMemcpy (KV data)
                _, _, slab_ptr, slab_size = self.kv_mgr._recv_slab
                num_layers = len(self.kv_mgr.kv_args.kv_data_ptrs)
                layer_slab = slab_size // max(num_layers, 1)

                for buf_idx, gpu_ptr in enumerate(
                    self.kv_mgr.kv_args.kv_data_ptrs
                ):
                    if gpu_ptr not in staging.buffers:
                        continue
                    _, host_base, buf_size = staging.buffers[gpu_ptr]
                    if self.kv_mgr.is_mla_backend:
                        item_len = self.kv_mgr.kv_args.kv_item_lens[buf_idx]
                    else:
                        item_len = self.kv_mgr.kv_args.kv_item_lens[buf_idx // 2]
                    slab_layer_base = slab_ptr + buf_idx * layer_slab

                    for idx in self._staging_kv_indices:
                        offset = int(idx) * item_len
                        if (
                            offset + item_len <= buf_size
                            and offset + item_len <= layer_slab
                        ):
                            # slab → GPU directly (skip DRAM staging memmove)
                            staging.copy_h2d_direct(
                                slab_layer_base + offset,
                                gpu_ptr + offset,
                                item_len,
                            )

                # State: DRAM → GPU (state uses direct DRAM mirrors, not slab)
                if (
                    state_indices
                    and self.kv_mgr.kv_args.state_data_ptrs
                    and self.kv_mgr.kv_args.state_item_lens
                ):
                    for buf_idx, gpu_ptr in enumerate(
                        self.kv_mgr.kv_args.state_data_ptrs
                    ):
                        if gpu_ptr not in staging.buffers:
                            continue
                        _, _, s_buf_size = staging.buffers[gpu_ptr]
                        s_item_len = self.kv_mgr.kv_args.state_item_lens[buf_idx]
                        for idx in state_indices:
                            offset = int(idx) * s_item_len
                            if offset + s_item_len <= s_buf_size:
                                staging.copy_h2d(gpu_ptr + offset, s_item_len)

                staging.sync()
            else:
                # Non-slab: DRAM → GPU directly
                self.kv_mgr._staging_post_copy_h2d(
                    self._staging_kv_indices,
                    state_indices=state_indices,
                )

            del self._staging_kv_indices
            if hasattr(self, "_staging_state_indices"):
                del self._staging_state_indices

        return result

    NixlKVReceiver.poll = _patched_recv_poll

    # ---- 11. NixlKVSender.poll — handle _CompletedTransfer ------------------
    def _patched_sender_poll(self):
        from sglang.srt.disaggregation.base.conn import KVPoll

        if not self.has_sent:
            return self.kv_mgr.check_status(self.bootstrap_room)

        for x in self.xfer_handles:
            if isinstance(x, _CompletedTransfer):
                continue
            state = self.kv_mgr.agent.check_xfer_state(x)
            if state == "ERR":
                raise Exception("KVSender transfer encountered an error.")
            if state != "DONE":
                return KVPoll.WaitingForInput  # type: ignore

        return KVPoll.Success  # type: ignore

    NixlKVSender.poll = _patched_sender_poll

    _PATCHED = True
    logger.info(
        "ROCm DRAM staging patch (chunked MR + slab) applied successfully"
    )


# ---------------------------------------------------------------------------
# Auto-patch on import when env var is set
# ---------------------------------------------------------------------------
if _STAGING_ENABLED:
    try:
        patch_nixl_for_rocm()
    except Exception:
        logger.debug("Auto-patch deferred (sglang not yet importable)", exc_info=True)
