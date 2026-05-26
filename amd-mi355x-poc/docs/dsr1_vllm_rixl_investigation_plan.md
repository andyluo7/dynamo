# DSR1 vLLM+RIXL `register_memory` — investigation plan

The vLLM disaggregated path on AMD MI355X works on MiniMax-M2.5 (TP=4, ~730 tok/s
@ c=32, 320/320 success) but fails to start on DeepSeek-R1-0528 FP8 (TP=8) with
`ibv_reg_mr ... access=0xf failed: Invalid argument` → `NIXL_ERR_BACKEND`. Eleven
transport-layer hypotheses already eliminated (see
[`archive/12-dsr1-vllm-rixl-cross-validation.md`](archive/12-dsr1-vllm-rixl-cross-validation.md)).
This doc captures the recon needed to move from "transport layer is clean" to
"vLLM-internal root cause identified," scoped for a future debugging session.

## Current call site

vLLM 0.19.1.dev0+g2a69949bd (rocm/atom-dev container):

  `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py:1709`

```python
descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
logger.debug("Registering descs: %s", caches_data)
self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)   # ← fails
logger.debug("Done registering descs")
self._registered_descs.append(descs)
```

Where `caches_data` is a list of per-layer `(base_addr, size_bytes, device_id, "")`
tuples. Note: this is a moved call site — earlier PoC docs reference
`worker.py:929`, which no longer exists in current vLLM.

## Targeted instrumentation patch

Paste before line 1709. Logs everything that distinguishes vLLM's call from
our standalone `nixl_pytorch_probe.py`, which succeeds at the same per-region
sizes:

```python
# DSR1-DEBUG block
import os
import traceback
logger.warning(
    "DSR1-DEBUG before register_memory: "
    "num_regions=%d, nixl_memory_type=%s, nixl_backends=%s, "
    "tp_rank=%s/%s, device_id=%s, total_size=%d, "
    "UCX_TLS=%s, UCX_ROCM_COPY_D2H_THRESH=%s, "
    "HIP_VISIBLE_DEVICES=%s, ROCR_VISIBLE_DEVICES=%s",
    len(caches_data), self.nixl_memory_type, self.nixl_backends,
    self.tp_rank, getattr(self, "tp_size", "?"), self.device_id,
    sum(c[1] for c in caches_data),
    os.environ.get("UCX_TLS", "(unset)"),
    os.environ.get("UCX_ROCM_COPY_D2H_THRESH", "(unset)"),
    os.environ.get("HIP_VISIBLE_DEVICES", "(unset)"),
    os.environ.get("ROCR_VISIBLE_DEVICES", "(unset)"),
)
for i, (addr, size, dev, _) in enumerate(caches_data):
    logger.warning("DSR1-DEBUG caches_data[%d]: addr=0x%x size=%d dev=%d",
                   i, addr, size, dev)
try:
    self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
except Exception as e:
    logger.error("DSR1-DEBUG register_memory FAILED: %s\n%s",
                 e, traceback.format_exc())
    raise
```

Apply the same logging pattern (UCX/HIP env + sizes + dev) inside
`advanced/debug-probes/nixl_pytorch_probe.py` so the two logs are line-diff'able.

## Top remaining hypotheses to test

Per
[`archive/12-dsr1-vllm-rixl-cross-validation.md`](archive/12-dsr1-vllm-rixl-cross-validation.md),
the standalone probes ruled out everything in the ionic NIC, ibv MR limits,
UCX/RIXL/NIXL Python wrappers, multi-process spawning, multi-region
registration, PyTorch allocator, and weight-allocation memory pressure
layers. What the probes do NOT cover is the **integrated vLLM worker state**
present when `register_memory` runs:

1. **RCCL / `torch.distributed` TP=8 group active.** vLLM workers are members
   of a torch.distributed group for TP collectives by the time KV registration
   happens. RCCL opens its own HIP streams that may interfere with UCX's ROCm
   transport. Our probes never call `torch.distributed.init_process_group`.
2. **HIP graph capture state.** vLLM has captured CUDA graphs for the model
   forward pass before KV registration — many HIP streams + graphs allocated.
   Probes have only the implicit default stream.
3. **vLLM NixlConnector role config.** Prefill vs decode workers may set
   agent name / role specific to a side-channel handshake; we should diff
   `nixl_wrapper` init params between vLLM and the probe.
4. **PyTorch tensor allocation pool.** vLLM allocates KV cache through its
   own cache spec layers, not direct `torch.zeros` — may use a non-default
   caching allocator pool whose backing memory has different mmap flags.

## End-to-end test environment to reproduce

| Component | Detail |
|---|---|
| Nodes | 2× MI355X (AAC1 reservation, ideally `g12-{26,34}` or similar idle pair) |
| Container | `localhost/dynamo-vllm-rixl:latest` (build from [`container/Dockerfile.rocm-vllm-rixl`](../container/Dockerfile.rocm-vllm-rixl), ~20–30 min one-time) |
| Launcher | [`advanced/sweeps/phase4_dsr1_disagg.sh`](../advanced/sweeps/phase4_dsr1_disagg.sh) (edit `PREFILL_NODE` / `DECODE_NODE` to current allocation) |
| Probe baseline | [`advanced/debug-probes/nixl_pytorch_probe.py`](../advanced/debug-probes/nixl_pytorch_probe.py) (works at 8 procs × 1×2.638 GiB on the same hardware) |
| Model | DSR1 671B FP8 (cached at `/shared/amdgpu/home/anluo/.cache/huggingface/`) |

## Step-by-step (estimated time)

| # | Step | Notes |
|---|---|---|
| 1 | Allocate 2 idle MI355X nodes (8h) | salloc |
| 2 | Build `dynamo-vllm-rixl:latest` on one node | ~25 min, one-time per release |
| 3 | Apply instrumentation patch to `nixl_connector.py:1709` inside the image | small in-container sed or rebuild |
| 4 | Edit launcher's `PREFILL_NODE` / `DECODE_NODE` | minutes |
| 5 | Launch DSR1 1P1D disagg | ~10 min model load + ~30 s before crash |
| 6 | Capture `dynamo-vllm-prefill` worker log | grep `DSR1-DEBUG` |
| 7 | Apply same DSR1-DEBUG log lines to `nixl_pytorch_probe.py` | small diff |
| 8 | Run `nixl_pytorch_probe.py` with TP=8 process pattern | matches vLLM TP=8 process count |
| 9 | Line-diff the two log captures | identify divergence |
| 10 | Hypothesis → minimal fix attempt | open-ended |

## Likely next steps after the divergence is found

| If divergence is in… | Probable owner |
|---|---|
| UCX-side env vars (UCX_TLS, UCX_ROCM_*) | vLLM container env config — likely a quick `Dockerfile.rocm-vllm-rixl` change |
| NIXL agent init params (name, role, backend list) | vLLM `nixl_connector.py` — likely an upstream vLLM PR |
| ibv access flags propagation | UCX ROCm backend (`uct/rocm/base/rocm_base.c`) — upstream UCX PR or RIXL fork |
| RCCL/torch.distributed interaction | Investigate RCCL + UCX coexistence on ROCm; may need ROCm/UCX coordination |
| HIP graph capture state | vLLM init order change — upstream vLLM PR |

## Cross-reference

- [`archive/12-dsr1-vllm-rixl-cross-validation.md`](archive/12-dsr1-vllm-rixl-cross-validation.md) — the 11-hypothesis elimination matrix that scoped this investigation
- [`advanced/debug-probes/`](../advanced/debug-probes/) — standalone reproducers + the new `dmabuf_register_probe.cpp`
- [Mooncake PR #2225](https://github.com/kvcache-ai/Mooncake/pull/2225) — adds an analogous ROCm dmabuf path to Mooncake; relevant because the failure mode (`ibv_reg_mr ... EINVAL` on GPU memory) is the same class of bug. If UCX is similarly missing a ROCm dmabuf code path inside `uct/rocm/base/rocm_base.c`, the fix shape may mirror the Mooncake PR.
