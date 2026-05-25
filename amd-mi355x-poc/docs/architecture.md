# Architecture

How the pieces fit together on AMD MI355X.

```
                  ┌────────────────────────────────────────────────┐
                  │  Dynamo control plane                          │
                  │   • dynamo.frontend (HTTP)                     │
                  │   • etcd  (service discovery)                  │
                  │   • NATS  (request/response bus)               │
                  └─────────────────┬──────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
 ┌─────────────┐           ┌─────────────┐           ┌─────────────────┐
 │ dynamo.sglang│           │  dynamo.vllm│           │ dynamo.vllm     │
 │  (agg)      │           │  (agg)      │           │  (prefill/decode│
 │             │           │             │           │   disagg)       │
 └──────┬──────┘           └──────┬──────┘           └────────┬────────┘
        │                         │                           │
        ▼                         ▼                ┌──────────┴──────────┐
   ┌────────┐               ┌──────────┐           ▼                     ▼
   │ SGLang │               │ vLLM     │      ┌─────────┐         ┌─────────┐
   │ engine │               │ engine   │      │ vLLM    │  KV     │ vLLM    │
   │        │               │          │      │ prefill ├────────►│ decode  │
   │  MI355X│               │ MI355X   │      │ MI355X  │ over    │ MI355X  │
   └────────┘               └──────────┘      └─────────┘ RIXL    └─────────┘
                                                          (UCX
                                                           + ROCm
                                                           + ionic)

   For SGLang 2-node disagg (not pictured):
   prefill ─── Mooncake (Python API, DRAM-staged on ROCm) ───► decode
```

## Pieces

### Dynamo control plane (NV-side, unchanged)

`dynamo.frontend`, `etcd`, `NATS` are all language-agnostic. They run
unchanged on AMD inside a small Python container. No GPU dependency.

### Backend workers

- **`dynamo.sglang`** drives SGLang engines, both agg and prefill/decode.
- **`dynamo.vllm`** drives vLLM engines, same shape.

Both use SGLang/vLLM's standard ROCm builds — no fork required.

### KV transport (the AMD-specific part)

This is where the divergence from CUDA Dynamo lives.

**vLLM disagg path — RIXL via UCX:**

- vLLM's `NixlConnector` calls `nixl_agent.register_memory(KV blocks)`.
- NIXL is provided by [ROCm/RIXL](https://github.com/ROCm/RIXL), a ROCm
  build that re-exports the NIXL ABI on top of UCX with ROCm/HIP support.
- UCX runs over the ionic libibverbs driver. UCX hardcodes
  `IBV_ACCESS_REMOTE_ATOMIC` in MR registrations; ionic NICs reject that
  bit with `EINVAL`. The [LD_PRELOAD interposer in
  `patches/ibv_ionic_compat.c`](../patches/README.md#ibv_ionic_compatc--ld_preload-interposer-for-amd-pensando-ionic-nics)
  strips the bit before forwarding.
- Result: RIXL's UCX-plugin C++ transfer pipeline scales cleanly past
  where SGLang+Mooncake saturates the ionic firmware's QP-setup queue.

**SGLang disagg path — Mooncake with ROCm DRAM staging:**

- SGLang's Mooncake transport posts KV blocks directly from device
  memory. On ROCm, the ionic driver rejects MRs registered against
  `hipHostMalloc`-allocated regions.
- The [`patches/fork-patches/mooncake_rocm_staging.py`](../patches/fork-patches/)
  adapter wraps Mooncake's Python API and adds a `RocmDramStagingCommon`
  helper that mirrors KV blocks into mmap+mlock'd host DRAM, registers
  that, and posts the RDMA WRITE/READ.
- Chunked MR registration (≤190 MB) fits ionic's per-device MR limit
  (~250 MB).
- Saturates at c=32 on DSR1 because Mooncake's Python-level chunked
  register/transfer/deregister cycle exhausts the ionic firmware's
  QP-setup queue. See [`findings.md`](findings.md) for the side-by-side.

### Why two transports

`dynamo.sglang` uses Mooncake (SGLang's native transport); `dynamo.vllm`
uses RIXL/UCX (vLLM's NIXL connector). The AMD-side adapter shape is
different for each, but the principle is the same: bridge the
CUDA-assumed transport API onto ROCm + ionic.

### Network behavior

AAC1 ionic runs lossy RoCEv2 with **ECN/DCQCN** congestion control, not
lossless PFC. This matters for transport choice:

- MoRI (lossless-PFC-optimized) hits a congestion wall at c≥4 on AAC1
  because ECN backpressure shrinks the congestion window faster than the
  transport's recovery loop can grow it back. The fork's published MoRI
  numbers (178 / 672 / 2196 tok/s at c=4/16/128) are measured on a
  lossless-PFC cluster.
- Mooncake (chunked register/transfer/deregister) and RIXL (UCX C++
  pipeline) both back-pressure naturally against ECN, so they work on
  AAC1.

Full root-cause: [`network-debug.md`](network-debug.md).

## What's left as future work

- **PR #9929** (lazy nixl import + `typing.Self` compat) is the upstream
  side of this. Once merged, the older runtime shims in this repo
  (`nixl_stub/`, `zzz_typing_self_compat.pth` — already removed from
  this directory) are no longer needed.
- **PR3 RFC** at `sgl-project/sglang` will propose the Mooncake ROCm
  staging adapter be absorbed upstream. If accepted, `fork-patches/`
  disappears.
- **DSR1 vLLM+RIXL startup failure** is the only known broken path.
  Root cause is vLLM-internal, not transport. See
  [`archive/12-dsr1-vllm-rixl-cross-validation.md`](archive/12-dsr1-vllm-rixl-cross-validation.md).
- **PFC enablement on AAC1 ionic + switch ports** would let us run MoRI
  here at fork-published speeds. Cluster-admin work; not a code change.
