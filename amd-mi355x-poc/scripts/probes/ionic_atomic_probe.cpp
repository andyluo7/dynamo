/*
 * ionic_atomic_probe: does ionic accept ibv_reg_mr with REMOTE_ATOMIC (0x8)?
 * If this fails with EINVAL, the actual root cause of the DSR1+vLLM+RIXL
 * crash is "LD_PRELOAD interposer not applied to vLLM workers", not any
 * ionic MR-size limit.
 */
#include <hip/hip_runtime.h>
#include <infiniband/verbs.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

int main(int argc, char **argv) {
    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    struct ibv_context *ctx = NULL;
    for (int i = 0; i < n; i++) {
        const char *name = ibv_get_device_name(list[i]);
        if (name && strstr(name, "ionic")) { ctx = ibv_open_device(list[i]); break; }
    }
    ibv_free_device_list(list);
    if (!ctx) { fprintf(stderr, "no ionic\n"); return 1; }
    struct ibv_pd *pd = ibv_alloc_pd(ctx);

    void *vram;
    size_t bytes = (size_t)2638 * 1024 * 1024; /* match DSR1 size */
    hipMalloc(&vram, bytes);
    hipMemset(vram, 0, bytes);
    hipDeviceSynchronize();

    fprintf(stderr, "buffer: %.2f GiB ROCm/VRAM at %p\n", bytes / 1024.0 / 1024.0 / 1024.0, vram);

    /* Try every access combination */
    const char *names[] = { "LOCAL_WRITE only (0x1)", "LW|RW|RR (0x7)", "LW|RW|RR|ATOMIC (0xf)", "ATOMIC only (0x8)" };
    int flags[]        = { 0x1,                         0x7,             0xf,                       0x8 };
    for (int i = 0; i < 4; i++) {
        struct ibv_mr *mr = ibv_reg_mr(pd, vram, bytes, flags[i]);
        if (mr) {
            fprintf(stderr, "  access=%-25s OK  lkey=0x%x rkey=0x%x\n", names[i], mr->lkey, mr->rkey);
            ibv_dereg_mr(mr);
        } else {
            fprintf(stderr, "  access=%-25s FAIL errno=%d (%s)\n", names[i], errno, strerror(errno));
        }
    }

    hipFree(vram);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    return 0;
}
