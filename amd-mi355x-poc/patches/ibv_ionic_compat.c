// SPDX-License-Identifier: Apache-2.0
//
// LD_PRELOAD interposer for AMD Pensando "ionic" RoCE NICs.
//
// Strips IBV_ACCESS_REMOTE_ATOMIC (0x8) from the access flags passed to
// ibv_reg_mr / ibv_reg_mr_iova2 / ibv_reg_dmabuf_mr.
//
// Why:
//   UCX hardcodes UCP_FEATURE_AMO32|AMO64 in the UCP context features,
//   which causes ibv_reg_mr(access=0xf). Pensando ionic NICs reject the
//   REMOTE_ATOMIC bit with EINVAL. RIXL never uses RDMA atomic ops (only
//   read/write), so stripping the flag is safe.
//
// The fork's nixl_rocm_staging.py (https://github.com/JohnQinAMD/dynamo
// branch amd-dynamo) provides this same wrapper for ibv_reg_mr and
// ibv_reg_mr_iova2 only. We extended it with ibv_reg_dmabuf_mr because
// UCX uses dmabuf-based MR registration for GPU VRAM, and that path was
// not covered by the original interposer — without this wrapper the
// vLLM disagg path fails at `register_kv_caches → register_memory(VRAM
// descs)` with `nixlBackendError: NIXL_ERR_BACKEND`.
//
// Build:
//   gcc -shared -fPIC -O2 -o ibv_ionic_compat.so ibv_ionic_compat.c -ldl
//
// Use:
//   export LD_PRELOAD=$(pwd)/ibv_ionic_compat.so
//   python3 -m dynamo.vllm ...
//
// UCX upstream PR #10341 would fix this by retrying with reduced access
// flags, but it remains unmerged as of 2026-04. AMD's vendor-aware UCX
// patch (in JohnQinAMD's private ucx-1.12/build_ionic) achieves the same
// end by stripping REMOTE_ATOMIC at the UCX source level for vendor_id
// 0x1dd8 only — but that build isn't publicly available, and our
// LD_PRELOAD approach is functionally equivalent for processes that
// don't share libuct_ib.so with RCCL (which uses XGMI on intra-node and
// is unaffected on cross-node).
//
// Set fprintf to /dev/null in production; useful only for debugging.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define IBV_ACCESS_REMOTE_ATOMIC 0x8

void *ibv_reg_mr(void *pd, void *addr, size_t length, int access) {
    typedef void *(*fn_t)(void *, void *, size_t, int);
    static fn_t real_fn;
    if (__builtin_expect(!real_fn, 0)) {
        real_fn = (fn_t)dlsym(RTLD_NEXT, "ibv_reg_mr");
        if (!real_fn) abort();
    }
    int orig = access; access &= ~IBV_ACCESS_REMOTE_ATOMIC;
    void *r = real_fn(pd, addr, length, access);
    if (!r) fprintf(stderr, "[ibv_compat] ibv_reg_mr len=%zu access=%x->%x failed\n",
                    length, orig, access);
    return r;
}

void *ibv_reg_mr_iova2(void *pd, void *addr, size_t length,
                       uint64_t iova, int access) {
    typedef void *(*fn_t)(void *, void *, size_t, uint64_t, int);
    static fn_t real_fn;
    if (__builtin_expect(!real_fn, 0)) {
        real_fn = (fn_t)dlsym(RTLD_NEXT, "ibv_reg_mr_iova2");
        if (!real_fn) abort();
    }
    int orig = access; access &= ~IBV_ACCESS_REMOTE_ATOMIC;
    void *r = real_fn(pd, addr, length, iova, access);
    if (!r) fprintf(stderr, "[ibv_compat] ibv_reg_mr_iova2 len=%zu access=%x->%x failed\n",
                    length, orig, access);
    return r;
}

/* dmabuf-based MR registration (used by UCX for GPU VRAM via libdrm).
 * Critical for vLLM disagg path on ionic — the fork's interposer doesn't
 * wrap this, but ionic dmabuf RDMA fails anyway, and stripping the
 * REMOTE_ATOMIC flag lets RIXL's higher layers detect the failure
 * cleanly and fall back through ROCm copy + TCP transport. */
void *ibv_reg_dmabuf_mr(void *pd, uint64_t offset, size_t length,
                        uint64_t iova, int fd, int access) {
    typedef void *(*fn_t)(void *, uint64_t, size_t, uint64_t, int, int);
    static fn_t real_fn;
    if (__builtin_expect(!real_fn, 0)) {
        real_fn = (fn_t)dlsym(RTLD_NEXT, "ibv_reg_dmabuf_mr");
        if (!real_fn) {
            fprintf(stderr, "[ibv_compat] ibv_reg_dmabuf_mr not in libibverbs\n");
            return NULL;
        }
    }
    int orig = access; access &= ~IBV_ACCESS_REMOTE_ATOMIC;
    void *r = real_fn(pd, offset, length, iova, fd, access);
    if (!r) fprintf(stderr, "[ibv_compat] ibv_reg_dmabuf_mr len=%zu fd=%d access=%x->%x failed\n",
                    length, fd, orig, access);
    else    fprintf(stderr, "[ibv_compat] ibv_reg_dmabuf_mr len=%zu fd=%d access=%x->%x OK\n",
                    length, fd, orig, access);
    return r;
}
