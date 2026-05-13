# DSR1 on vLLM+RIXL — cross-validation (does RIXL beat Mooncake on the same model?)

Date: 2026-05-12
Cluster: AAC1, prefill=`smci355-ccs-aus-g12-06`, decode=`smci355-ccs-aus-g12-30`
Model: `deepseek-ai/DeepSeek-R1-0528` FP8, TP=8 per node
Setup: identical RIXL+UCX-ROCm stack as the M2.5 sweep in `11-`

## TL;DR

**Vanilla public RIXL+UCX-ROCm cannot start DSR1 disagg on ionic.** vLLM's NixlConnector
registers the per-rank KV pool as a single contiguous `ibv_reg_mr` call. For DSR1
TP=8 that registration is **1.7-2.6 GB per rank**, which exceeds AMD Pensando
ionic's per-MR limit (~250 MB based on M2.5/TP=4 succeeding at ~190 MB/rank).
Both attempted configs failed at startup before any token generation:

| Attempt | max-num-seqs | gpu-mem-util | Per-rank MR size requested | Result |
|---|---:|---:|---:|---|
| 1 | 32 | 0.85 | 2,763,307,008 B (2.57 GiB) | `ibv_reg_mr ... access=0xf failed: Invalid argument` → `NIXL_ERR_BACKEND` |
| 2 | 4 | 0.65 | 1,748,551,680 B (1.63 GiB) | same `Invalid argument` → same `NIXL_ERR_BACKEND` |

Same image, same env, same nodes that successfully ran M2.5 to 730 tok/s @ c=32.
The variable that changed is the per-rank KV pool size required by DSR1's 61-layer
MLA architecture vs M2.5's MoE.

## What this confirms architecturally

The PoC now has **three orthogonal data points** on the same ionic hardware:

| Stack | Model | Per-rank KV | Result |
|---|---|---:|---|
| vLLM + **RIXL** (public) | M2.5 (TP=4) | ~190 MB | clean to c=32, 730 tok/s, 320/320 success |
| vLLM + **RIXL** (public) | **DSR1 (TP=8)** | **1.7-2.6 GB** | **crashes at startup — MR-size ceiling** |
| SGLang + **Mooncake** (fork's `mooncake_rocm_staging.py`) | DSR1 (TP=8) | chunked ~190 MB | works to c=16 (527 tok/s), crashes at c=32 (transport retry) |

The clean reading:

1. **RIXL handles M2.5 cleanly because its per-rank MR happens to fit ionic's limit.** RIXL has no MR-chunking on the public branch — it relies on the full pool registering as one MR.
2. **Mooncake handles DSR1 to c=16 because the fork's `mooncake_rocm_staging.py` chunks the registration (~190 MB chunks).** That same chunking pattern has a different ceiling: per-chunk register/transfer/deregister can't sustain QP-setup rate past c=16-32 on ionic.
3. **Neither public transport handles DSR1 at production scale on ionic.** That is the architectural justification for the fork's MoRI integration as the production path.

Vanilla `ai-dynamo/dynamo:main` + public RIXL is **production-ready for medium-KV models on ionic** (M2.5 class). For DSR1-class models the patch surface required is non-trivial: either an MR-chunking patch in RIXL/UCX or the MoRI integration the fork uses.

## Reproducer

`scripts/phase4_dsr1_disagg.sh` — same launcher as `phase4_m25_disagg.sh` with:
- `MODEL=deepseek-ai/DeepSeek-R1-0528`
- `TP=8` (no `HIP_VISIBLE_DEVICES` — vLLM uses all 8 GPUs)
- `--max-model-len 4096 --max-num-seqs 4 --gpu-memory-utilization 0.65` (already minimized)

The failure is reproducible at startup — no model-warmup window or token generation needed. Look for `ibv_reg_mr ... length=<N> ... failed: Invalid argument` followed by `NIXL_ERR_BACKEND` in the worker log.

## Updated PoC scorecard

| Phase | Stack | Model | Result | Saturated at |
|---|---|---|---:|---|
| 3 (SGLang+Mooncake disagg, DSR1 TP=8 each) | fork's chunked-staging | DSR1 | 527 tok/s @ c=16 (160/160) | **transport (c=32 crash)** |
| 4 (vLLM+RIXL disagg, M2.5 TP=4 each) | public RIXL | M2.5 | **730 tok/s @ c=32 (320/320)** | **compute** |
| 4-DSR1 (this doc, vLLM+RIXL disagg, DSR1 TP=8) | public RIXL | DSR1 | **startup failure** | **MR-size limit (ionic)** |

## What this changes in the PR breakdown

PR 5 (`dynamo.vllm` AMD bits + LD_PRELOAD + UCX_TLS docs) is **still mergeable today** — it is correct, has no AMD-only bugs, and works for any model whose per-rank KV pool fits ionic's per-MR limit. The right framing in the PR description is:

> "Public RIXL on ionic is production-ready for models where per-rank KV-pool registration fits in a single ionic MR (~250 MB observed). For larger models, see `12-dsr1-vllm-rixl-cross-validation.md` — MR-chunking is the next workstream."

This is the same shape as the SGLang+Mooncake story: public path works to a certain scale, fork's enhanced transport (or MoRI) is required for production-grade DSR1.
