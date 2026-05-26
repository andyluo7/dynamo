/*
 * ionic_mr_probe: binary-search the per-MR byte limit on AMD Pensando ionic.
 *
 * Opens the first ionic IB device, allocates a contiguous DRAM buffer, and
 * tries ibv_reg_mr() with progressively larger sizes (binary search) until
 * it finds the largest size that succeeds.  Then sweeps a few finer grain
 * points around the boundary to confirm.
 *
 * Build: gcc -O2 -o ionic_mr_probe ionic_mr_probe.c -libverbs
 * Run:   LD_PRELOAD=/path/ibv_ionic_compat.so ./ionic_mr_probe
 */
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>

static struct ibv_context *open_ionic(void) {
    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list || n == 0) {
        fprintf(stderr, "no IB devices\n");
        return NULL;
    }
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
    if (!ctx) return 1;
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    if (!pd) { fprintf(stderr, "alloc_pd FAIL\n"); return 2; }

    /* Try up to 4 GiB DRAM buffer */
    size_t alloc_bytes = (size_t)4 * 1024 * 1024 * 1024;
    void *buf = mmap(NULL, alloc_bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) {
        fprintf(stderr, "mmap %.2f GiB FAIL: %s\n",
                alloc_bytes / (1024.0 * 1024.0 * 1024.0), strerror(errno));
        return 3;
    }
    /* Touch one byte per page to materialise pages */
    long ps = sysconf(_SC_PAGESIZE);
    for (size_t off = 0; off < alloc_bytes; off += ps) {
        ((volatile char *)buf)[off] = 0;
    }
    fprintf(stderr, "allocated + touched %.2f GiB DRAM at %p\n",
            alloc_bytes / (1024.0 * 1024.0 * 1024.0), buf);

    /* Coarse sweep: powers of 2 from 16 MiB to 2 GiB */
    fprintf(stderr, "\n=== coarse sweep ===\n");
    for (size_t mb = 16; mb <= 2048; mb *= 2) {
        try_reg(pd, buf, mb * 1024 * 1024);
    }

    /* Fine sweep around 256 MiB: 200, 220, 240, 250, 256, 260, 280, 300, 320 */
    fprintf(stderr, "\n=== fine sweep around suspected boundary ===\n");
    size_t fine[] = { 200, 220, 240, 248, 250, 252, 254, 256, 258, 260, 270, 280, 300, 320, 384, 400, 448, 480, 500, 512, 768, 1024 };
    for (size_t i = 0; i < sizeof(fine) / sizeof(fine[0]); i++) {
        try_reg(pd, buf, fine[i] * 1024 * 1024);
    }

    /* Binary search to nail the exact boundary in MiB */
    fprintf(stderr, "\n=== binary search ===\n");
    size_t lo = 1, hi = 4096; /* MiB */
    while (lo < hi) {
        size_t mid = (lo + hi + 1) / 2;
        if (try_reg(pd, buf, mid * 1024 * 1024)) lo = mid;
        else hi = mid - 1;
    }
    fprintf(stderr, "\nlargest successful MR: %zu MiB (= %zu bytes)\n",
            lo, lo * 1024 * 1024);

    /* Try byte-granular near the MiB boundary */
    fprintf(stderr, "\n=== byte-granular at MiB boundary ===\n");
    size_t base = lo * 1024 * 1024;
    size_t bytes_lo = base, bytes_hi = (lo + 1) * 1024 * 1024;
    while (bytes_lo < bytes_hi - 1) {
        size_t mid = (bytes_lo + bytes_hi) / 2;
        if (try_reg(pd, buf, mid)) bytes_lo = mid;
        else bytes_hi = mid;
    }
    fprintf(stderr, "\nlargest successful MR: %zu bytes (= %.4f MiB)\n",
            bytes_lo, bytes_lo / (1024.0 * 1024.0));

    munmap(buf, alloc_bytes);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    return 0;
}
