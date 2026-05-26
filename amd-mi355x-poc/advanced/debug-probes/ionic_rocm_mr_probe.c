/*
 * ionic_rocm_mr_probe: probe ionic's per-MR ceiling for ROCm/VRAM (the path
 * that actually fails for DSR1+vLLM+RIXL, not the DRAM path that succeeds at
 * 4 GiB).
 *
 * Allocates VRAM via hipMalloc, then registers it with ibv_reg_mr on ionic_0.
 * Sweeps sizes coarsely then binary-searches the boundary.
 *
 * Build (on a node with ROCm + libibverbs):
 *   /opt/rocm/bin/hipcc -O2 -o ionic_rocm_mr_probe ionic_rocm_mr_probe.c -libverbs
 * Run:
 *   LD_PRELOAD=/tmp/ibv_ionic_compat.so ./ionic_rocm_mr_probe
 */
#include <hip/hip_runtime.h>
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

static struct ibv_context *open_ionic(void) {
    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list || n == 0) return NULL;
    struct ibv_context *ctx = NULL;
    for (int i = 0; i < n; i++) {
        const char *name = ibv_get_device_name(list[i]);
        if (name && strstr(name, "ionic")) {
            ctx = ibv_open_device(list[i]);
            fprintf(stderr, "opened device: %s -> ctx=%p\n", name, (void*)ctx);
            break;
        }
    }
    ibv_free_device_list(list);
    return ctx;
}

static int try_reg(struct ibv_pd *pd, void *addr, size_t bytes) {
    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    struct ibv_mr *mr = ibv_reg_mr(pd, addr, bytes, access);
    if (!mr) {
        fprintf(stderr, "  reg %.2f MiB FAIL errno=%d (%s)\n",
                bytes / (1024.0 * 1024.0), errno, strerror(errno));
        return 0;
    }
    fprintf(stderr, "  reg %.2f MiB OK   lkey=0x%x rkey=0x%x\n",
            bytes / (1024.0 * 1024.0), mr->lkey, mr->rkey);
    ibv_dereg_mr(mr);
    return 1;
}

int main(void) {
    struct ibv_context *ctx = open_ionic();
    if (!ctx) { fprintf(stderr, "no ionic device\n"); return 1; }
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    if (!pd) { fprintf(stderr, "alloc_pd FAIL\n"); return 2; }

    /* Try up to 8 GiB VRAM (fits MI355X 288 GB easily) */
    size_t alloc_bytes = (size_t)8 * 1024 * 1024 * 1024;
    void *vram = NULL;
    hipError_t herr = hipMalloc(&vram, alloc_bytes);
    if (herr != hipSuccess) {
        fprintf(stderr, "hipMalloc %.2f GiB FAIL: %s\n",
                alloc_bytes / 1024.0 / 1024.0 / 1024.0, hipGetErrorString(herr));
        return 3;
    }
    /* Touch to ensure the allocation is materialised */
    hipMemset(vram, 0, alloc_bytes);
    hipDeviceSynchronize();
    fprintf(stderr, "allocated %.2f GiB VRAM at %p\n",
            alloc_bytes / 1024.0 / 1024.0 / 1024.0, vram);

    fprintf(stderr, "\n=== coarse sweep (ROCm/VRAM) ===\n");
    for (size_t mb = 16; mb <= 4096; mb *= 2) {
        try_reg(pd, vram, mb * 1024 * 1024);
    }

    fprintf(stderr, "\n=== fine sweep (ROCm/VRAM) ===\n");
    size_t fine[] = { 1, 4, 8, 16, 32, 64, 128, 192, 200, 220, 240, 250, 256, 272, 300, 384, 500, 512, 768, 1024, 1536, 1750, 2048, 2560, 3072 };
    for (size_t i = 0; i < sizeof(fine) / sizeof(fine[0]); i++) {
        try_reg(pd, vram, fine[i] * 1024 * 1024);
    }

    fprintf(stderr, "\n=== binary search (ROCm/VRAM) ===\n");
    size_t lo = 1, hi = 8192; /* MiB */
    while (lo < hi) {
        size_t mid = (lo + hi + 1) / 2;
        if (try_reg(pd, vram, mid * 1024 * 1024)) lo = mid;
        else hi = mid - 1;
    }
    fprintf(stderr, "\nlargest successful VRAM MR: %zu MiB (= %zu bytes)\n",
            lo, lo * 1024 * 1024);

    hipFree(vram);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    return 0;
}
