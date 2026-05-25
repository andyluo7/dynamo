/*
 * ionic_concurrent_probe: how many concurrent MRs can ionic_0 hold?
 *
 * Loops registering 2 GiB ROCm MRs (matches DSR1's per-region size class)
 * without deregistering until ibv_reg_mr fails. Reports the count and the
 * cumulative bytes at the failure point.
 */
#include <hip/hip_runtime.h>
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <vector>

static struct ibv_context *open_ionic_idx(int idx) {
    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list) return NULL;
    struct ibv_context *ctx = NULL;
    int seen = 0;
    for (int i = 0; i < n; i++) {
        const char *name = ibv_get_device_name(list[i]);
        if (name && strstr(name, "ionic")) {
            if (seen == idx) {
                ctx = ibv_open_device(list[i]);
                fprintf(stderr, "opened device %d: %s -> ctx=%p\n", idx, name, (void*)ctx);
                break;
            }
            seen++;
        }
    }
    ibv_free_device_list(list);
    return ctx;
}

int main(int argc, char **argv) {
    size_t mr_mb = (argc > 1) ? atol(argv[1]) : 2048;  /* default 2 GiB per MR */
    size_t mr_bytes = mr_mb * 1024 * 1024;
    int max_iters = (argc > 2) ? atoi(argv[2]) : 64;
    int dev_idx = (argc > 3) ? atoi(argv[3]) : 0;

    fprintf(stderr, "config: mr=%zu MiB, max_iters=%d, ionic_idx=%d\n",
            mr_mb, max_iters, dev_idx);

    struct ibv_context *ctx = open_ionic_idx(dev_idx);
    if (!ctx) { fprintf(stderr, "no ionic\n"); return 1; }
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    if (!pd) { fprintf(stderr, "alloc_pd FAIL\n"); return 2; }

    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    std::vector<struct ibv_mr*> mrs;
    std::vector<void*> bufs;
    size_t total_bytes = 0;

    for (int i = 0; i < max_iters; i++) {
        void *vram = NULL;
        hipError_t herr = hipMalloc(&vram, mr_bytes);
        if (herr != hipSuccess) {
            fprintf(stderr, "iter %d: hipMalloc FAIL: %s (cumulative VRAM=%.2f GiB)\n",
                    i, hipGetErrorString(herr), total_bytes / 1024.0 / 1024.0 / 1024.0);
            break;
        }
        hipMemset(vram, 0, mr_bytes);
        hipDeviceSynchronize();
        struct ibv_mr *mr = ibv_reg_mr(pd, vram, mr_bytes, access);
        if (!mr) {
            fprintf(stderr, "iter %d: ibv_reg_mr FAIL errno=%d (%s) "
                            "cumulative_MR_bytes=%.2f GiB n_mrs=%zu\n",
                    i, errno, strerror(errno),
                    total_bytes / 1024.0 / 1024.0 / 1024.0, mrs.size());
            hipFree(vram);
            break;
        }
        mrs.push_back(mr);
        bufs.push_back(vram);
        total_bytes += mr_bytes;
        fprintf(stderr, "iter %d: reg OK n_mrs=%zu cumulative=%.2f GiB lkey=0x%x rkey=0x%x\n",
                i, mrs.size(), total_bytes / 1024.0 / 1024.0 / 1024.0,
                mr->lkey, mr->rkey);
    }

    fprintf(stderr, "\n=== summary ===\n");
    fprintf(stderr, "concurrent MRs held: %zu\n", mrs.size());
    fprintf(stderr, "total registered:   %.2f GiB\n",
            total_bytes / 1024.0 / 1024.0 / 1024.0);

    for (auto m : mrs) ibv_dereg_mr(m);
    for (auto b : bufs) hipFree(b);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    return 0;
}
