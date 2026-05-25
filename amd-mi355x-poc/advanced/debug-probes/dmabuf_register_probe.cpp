// dmabuf_register_probe.cpp
//
// Verifies the full chain Mooncake needs for ROCm dmabuf MR registration:
//
//   hipMalloc -> hsa_amd_portable_export_dmabuf -> ibv_reg_dmabuf_mr
//
// This is the standalone reproducer for what Mooncake issue #751 currently
// doesn't support. If this probe passes on your host, the dmabuf code path
// we're proposing to add to Mooncake (parallel to the existing CUDA path
// at rdma_context.cpp:311) will work in your environment.
//
// Build:
//   g++ -std=c++17 -O2 dmabuf_register_probe.cpp \
//     -I/opt/rocm/include -L/opt/rocm/lib \
//     -lhsa-runtime64 -lamdhip64 -libverbs \
//     -o dmabuf_register_probe
//
// Run:
//   ./dmabuf_register_probe              # auto-pick first RDMA device
//   ./dmabuf_register_probe rocep28s0    # specific device
//
// Exit codes:
//   0 = PASS, dmabuf path works
//   1 = FAIL with diagnostic on stderr

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <infiniband/verbs.h>

#define HIP_CHECK(call)                                                      \
    do {                                                                     \
        hipError_t _e = (call);                                              \
        if (_e != hipSuccess) {                                              \
            fprintf(stderr, "FAIL HIP %s:%d %s -> %s\n", __FILE__, __LINE__, \
                    #call, hipGetErrorString(_e));                           \
            return 1;                                                        \
        }                                                                    \
    } while (0)

int main(int argc, char** argv) {
    const char* target_device = (argc > 1) ? argv[1] : nullptr;
    constexpr size_t BUF_SIZE = 256ULL * 1024 * 1024;  // 256 MiB

    printf("=== dmabuf_register_probe ===\n");
    printf("    target: hsa_amd_portable_export_dmabuf -> ibv_reg_dmabuf_mr\n");
    printf("    buffer: %zu MiB on GPU 0\n\n", BUF_SIZE / (1 << 20));

    // 1. HSA init -------------------------------------------------------
    hsa_status_t hs = hsa_init();
    if (hs != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL hsa_init -> %d\n", hs);
        return 1;
    }
    printf("[ok]   hsa_init\n");

    // 2. HIP device + allocation ---------------------------------------
    int hip_devs = 0;
    HIP_CHECK(hipGetDeviceCount(&hip_devs));
    if (hip_devs == 0) {
        fprintf(stderr, "FAIL no HIP devices visible\n");
        hsa_shut_down();
        return 1;
    }
    HIP_CHECK(hipSetDevice(0));
    printf("[ok]   hipSetDevice(0) of %d devices\n", hip_devs);

    void* gpu_buf = nullptr;
    HIP_CHECK(hipMalloc(&gpu_buf, BUF_SIZE));
    printf("[ok]   hipMalloc(%zu MiB) -> %p\n", BUF_SIZE / (1 << 20), gpu_buf);

    // 3. Export dmabuf fd ----------------------------------------------
    int dmabuf_fd = -1;
    uint64_t dmabuf_offset = 0;
    hs = hsa_amd_portable_export_dmabuf(gpu_buf, BUF_SIZE, &dmabuf_fd,
                                        &dmabuf_offset);
    if (hs != HSA_STATUS_SUCCESS) {
        const char* msg = nullptr;
        hsa_status_string(hs, &msg);
        fprintf(stderr, "FAIL hsa_amd_portable_export_dmabuf -> %s (%d)\n",
                msg ? msg : "unknown", hs);
        fprintf(stderr,
                "       Likely cause: kernel amdgpu does not implement\n"
                "       dmabuf-fd export. Required: CONFIG_DMA_SHARED_BUFFER=y\n"
                "       and a recent-enough amdgpu (DKMS or in-tree).\n");
        hipFree(gpu_buf);
        hsa_shut_down();
        return 1;
    }
    printf("[ok]   hsa_amd_portable_export_dmabuf -> fd=%d offset=%lu\n",
           dmabuf_fd, dmabuf_offset);

    // 4. Open the RDMA device ------------------------------------------
    int num_devs = 0;
    struct ibv_device** dev_list = ibv_get_device_list(&num_devs);
    if (!dev_list || num_devs == 0) {
        fprintf(stderr, "FAIL ibv_get_device_list returned no RDMA devices\n");
        close(dmabuf_fd);
        hipFree(gpu_buf);
        hsa_shut_down();
        return 1;
    }
    printf("[ok]   ibv_get_device_list -> %d device(s):\n", num_devs);
    struct ibv_device* picked = nullptr;
    for (int i = 0; i < num_devs; i++) {
        const char* name = ibv_get_device_name(dev_list[i]);
        printf("       [%d] %s\n", i, name);
        if (target_device && strcmp(name, target_device) == 0) {
            picked = dev_list[i];
        }
    }
    if (!picked) picked = dev_list[0];
    printf("[info] using device: %s\n", ibv_get_device_name(picked));

    struct ibv_context* ctx = ibv_open_device(picked);
    if (!ctx) {
        fprintf(stderr, "FAIL ibv_open_device(%s): %s\n",
                ibv_get_device_name(picked), strerror(errno));
        ibv_free_device_list(dev_list);
        close(dmabuf_fd);
        hipFree(gpu_buf);
        hsa_shut_down();
        return 1;
    }
    printf("[ok]   ibv_open_device\n");

    struct ibv_pd* pd = ibv_alloc_pd(ctx);
    if (!pd) {
        fprintf(stderr, "FAIL ibv_alloc_pd: %s\n", strerror(errno));
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        close(dmabuf_fd);
        hipFree(gpu_buf);
        hsa_shut_down();
        return 1;
    }
    printf("[ok]   ibv_alloc_pd\n");

    // 5. The actual test: ibv_reg_dmabuf_mr ----------------------------
    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                       IBV_ACCESS_REMOTE_WRITE;
    struct ibv_mr* mr =
        ibv_reg_dmabuf_mr(pd, dmabuf_offset, BUF_SIZE,
                          reinterpret_cast<uint64_t>(gpu_buf), dmabuf_fd,
                          access);
    if (!mr) {
        int err = errno;
        fprintf(stderr, "FAIL ibv_reg_dmabuf_mr: %s (errno=%d)\n",
                strerror(err), err);
        if (err == EINVAL) {
            fprintf(stderr,
                    "       EINVAL typically means one of:\n"
                    "       - NIC driver does not accept this dmabuf fd\n"
                    "       - access flag rejected by the NIC firmware\n"
                    "         (e.g. Pensando ionic rejects IBV_ACCESS_REMOTE_ATOMIC)\n"
                    "       - kernel dmabuf fd is not routable to the NIC\n"
                    "         (peermem-equivalent path not enabled)\n");
        }
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        close(dmabuf_fd);
        hipFree(gpu_buf);
        hsa_shut_down();
        return 1;
    }
    printf("[ok]   ibv_reg_dmabuf_mr -> lkey=0x%x rkey=0x%x length=%zu\n",
           mr->lkey, mr->rkey, mr->length);

    printf("\n=== PASS ===\n");
    printf("Full chain works on this host. The Mooncake HIP dmabuf path\n");
    printf("we are proposing (parallel to rdma_context.cpp:311) will\n");
    printf("succeed against the registered NIC.\n");

    // 6. Cleanup -------------------------------------------------------
    ibv_dereg_mr(mr);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(dev_list);
    // ibv_reg_dmabuf_mr dup's the fd in-kernel, so closing the userspace
    // fd here is correct (matches Mooncake's pattern at
    // mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp:331).
    close(dmabuf_fd);
    hipFree(gpu_buf);
    hsa_shut_down();
    return 0;
}
