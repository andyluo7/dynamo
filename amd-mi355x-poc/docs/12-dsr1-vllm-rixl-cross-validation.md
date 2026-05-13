# DSR1 on vLLM+RIXL — cross-validation (correction follows)

Date: 2026-05-12
Cluster: AAC1, prefill=`smci355-ccs-aus-g12-06`, decode=`smci355-ccs-aus-g12-30`
Model: `deepseek-ai/DeepSeek-R1-0528` FP8, TP=8 per node
Setup: identical RIXL+UCX-ROCm stack as the M2.5 sweep in `11-`

## TL;DR (revised after deeper probing)

Initial result: vLLM+RIXL fails to start DSR1 disagg with `NIXL_ERR_BACKEND` and
UCX log lines like:

```
ib_md.c:287 UCX ERROR ibv_reg_mr(... length=2763307008, access=0xf) failed: Invalid argument
ucp_mm.c:76 UCX ERROR failed to register address ... (rocm) length 2763307008 on md[4]=ionic_0: Input/output error (md supports: host|rocm)
```

**First-pass diagnosis (now retracted): "ionic per-MR size ceiling around 250 MB."**

After running standalone libibverbs+HIP probes on the same node, image, and
LD_PRELOAD environment, this diagnosis is **wrong**. ionic's MR ceiling, both
DRAM and ROCm/VRAM paths, is much higher than DSR1 needs:

| Probe | Result |
|---|---|
| Single 4 GiB DRAM MR on ionic_0 (interposer on) | OK |
| Single 8 GiB ROCm/VRAM MR on ionic_0 (interposer on) | OK |
| 32 × 2 GiB ROCm/VRAM MRs in one process on ionic_0 = 64 GiB cumulative | all OK |
| 8 processes × 2.58 GiB ROCm/VRAM MR each on ionic_0 (matches DSR1 TP=8 size) | all 8 OK |
| 2.58 GiB ROCm/VRAM MR with `access=0xf` (REMOTE_ATOMIC included), no interposer | OK |
| 2.58 GiB ROCm/VRAM MR with `access=0x8` (ATOMIC only) | FAIL EINVAL |

So:
- ionic's MR-size limit on this firmware is at least 8 GiB for both DRAM and ROCm.
- ionic accepts `access=0xf` (LW|RW|RR|ATOMIC) for normal-sized MRs — the
  LD_PRELOAD `IBV_ACCESS_REMOTE_ATOMIC` strip we documented in PR 5 is **not
  required for ionic_0 on this firmware revision**. It rejects only
  `access=0x8` (ATOMIC standalone).
- Single-process and 8-process concurrent 2.58 GiB ROCm registrations on
  ionic_0 all succeed.

**The actual vLLM+RIXL failure is therefore something more specific than
"ionic can't take a 2.5 GiB MR".** Candidate causes that the standalone probes
do NOT cover:

1. PyTorch's caching allocator returns VRAM allocated via `hipExtMallocWithFlags`
   or similar non-default HIP flags — that allocation may interact with
   ionic's dmabuf path differently than plain `hipMalloc`.
2. RIXL/UCX maps the same VRAM region on **all 9 ionic devices**
   (ionic_0..ionic_8) per rank. With TP=8 that's 72 (rank, device) registration
   pairs. Failure on `md[4]=ionic_0` may be order-dependent: by the time
   md[4] is reached, an earlier device's registration has consumed a shared
   kernel resource.
3. vLLM holds the MRs through subsequent QP creation. The crash may not be
   at registration itself but at a downstream step that the standalone probe
   doesn't perform.
4. ROCm dmabuf handle exhaustion across many concurrent registrations on
   the same VRAM region.

Distinguishing among (1)-(4) requires a probe that mirrors RIXL's full
`ucp_mem_map` flow on PyTorch-allocated memory — not in scope for this
PoC, but worth flagging to whoever picks this up next.

## Numbers (the failure observation, kept for completeness)

vLLM+RIXL with DSR1 TP=8, two configs attempted:

| Attempt | max-num-seqs | gpu-mem-util | Per-rank registration size | Result |
|---|---:|---:|---:|---|
| 1 | 32 | 0.85 | 2,763,307,008 B (2.57 GiB) | `ibv_reg_mr ... access=0xf failed: Invalid argument` → `NIXL_ERR_BACKEND` |
| 2 | 4 | 0.65 | 1,748,551,680 B (1.63 GiB) | same EINVAL → same `NIXL_ERR_BACKEND` |

Both failed at the same code path. Reducing per-region size by ~40% did not
change the outcome — consistent with the "not a size limit" finding from the
probes.

## What this means for the PoC scorecard

| Phase | Stack | Model | Result |
|---|---|---|---|
| 3 (SGLang+Mooncake) | fork's chunked staging | DSR1 | 527 tok/s @ c=16, crash at c=32 (transport retry) |
| 4 (vLLM+RIXL) | public RIXL | M2.5 | **730 tok/s @ c=32**, 320/320 success |
| 4-DSR1 (this doc) | public RIXL | DSR1 | **startup failure on `ibv_reg_mr` for ROCm/VRAM, root cause not a per-MR size limit** |

The takeaway for the email/upstream story:

- M2.5 numbers stand: vLLM+RIXL is production-ready on AMD ionic for that
  model.
- DSR1 disagg with public RIXL is **broken in a way that the standalone
  probe does not reproduce**. The first-pass "ionic per-MR ceiling"
  characterization was wrong; the right framing is "needs more
  investigation — likely an interaction between PyTorch VRAM allocation
  patterns and UCX multi-MD registration across 9 ionic devices."
- The PR 5 framing should NOT promise "all models" but also should NOT
  blame an ionic ceiling that doesn't exist at the size we hit. Honest
  framing: "validated on M2.5 class; DSR1 needs follow-up."

## Probes (for reproduction)

Three standalone probes are checked into `scripts/probes/`:

- `ionic_mr_probe.c` — DRAM MR size sweep (binary search to 4 GiB, succeeds)
- `ionic_rocm_mr_probe.cpp` — ROCm/VRAM MR size sweep (succeeds to 8 GiB)
- `ionic_concurrent_probe.cpp` — repeat-registration loop (32 × 2 GiB OK)
- `ionic_atomic_probe.cpp` — access-flag combinations (0xf OK, 0x8 fails)
- `ionic_multiproc_probe.sh` — N parallel processes each registering 1 MR

All four use plain `hipMalloc` for VRAM, run under the same image and
LD_PRELOAD setup as the failing vLLM container.

## Reproducer for the failure itself

`scripts/phase4_dsr1_disagg.sh` — same launcher as `phase4_m25_disagg.sh`
with `MODEL=deepseek-ai/DeepSeek-R1-0528`, `TP=8`,
`--max-model-len 4096 --max-num-seqs 4 --gpu-memory-utilization 0.65`.

The failure is reproducible at startup. Look for `ibv_reg_mr ... length=<N>
... failed: Invalid argument` followed by `NIXL_ERR_BACKEND` in the worker
log. The standalone probes show the failure mode is NOT what the surface
log suggests; deeper investigation is required.
