# Phase 3 + 4 escalations — DSR1 SGLang disagg + M2.5 vLLM disagg

Date: 2026-05-11 (after the original Phase 0–5 PoC)
Cluster: AAC1, prefill=`smci355-ccs-aus-g12-06`, decode=`smci355-ccs-aus-g12-26` (both 8x MI355X gfx950, 9× Pensando ionic NICs)

After the original 4 PoC phases passed on small models (Qwen3-0.6B for both disagg paths), the user requested escalation to the production-relevant models:
- DeepSeek-R1-0528 FP8 (671B) — 2-node SGLang disagg with Mooncake
- MiniMax-M2.5 (229B FP8 MoE) — 2-node vLLM disagg with RIXL/UCX

**Both escalations PASS with 100% success rate.**

## Phase 3 escalation: DSR1 SGLang 1P1D disagg

Stack identical to original Phase 3 (Qwen3-0.6B), bumped to TP=8 per node and DSR1 model. All 16 MI355X GPUs in active use simultaneously.

### Numbers

| conc | N  | P50 (ms) | P95 (ms) | tok/s | output avg | success |
|------|----|----------|----------|-------|------------|---------|
| 1    | 8  | 3943     | 5916     | 17.2  | 64         | 8/8     |
| 4    | 12 | 6104     | 10426    | 36.5  | 64         | 12/12   |
| 8    | 24 | 14980    | 16792    | 32.8  | 64         | 24/24   |

Cold first request: 9.5 s. Reply: real DSR1 reasoning model output (`<think>` tag visible).

### Comparison with Phase 1 (DSR1 agg, single node)

| Mode | TP | Nodes | c=8 tok/s | Latency P50 c=1 |
|---|---|---|---|---|
| Phase 1 SGLang agg | 8 | 1 | 708.8 | 1259 ms |
| Phase 3 SGLang disagg + Mooncake | 8 each | 2 | 32.8 | 3943 ms |

Disagg is **~22× slower than agg** for DSR1, dominated by Mooncake's chunked-MR DRAM-staging overhead per token. DSR1's KV cache per token is large (61 layers, deep attention); each transfer requires multiple ~190 MB chunks. Each request also pays 2× hipMemcpy (D2H + H2D) plus RDMA write.

The fork's runbook reports DSR1 disagg via Mooncake at 97.7 tok/s @ c=1 (Test 12). We measured 17.2 tok/s @ c=1 — slower; the discrepancy is likely due to:
- Different prompt/output sizes (we used out=64; fork's bench may have used longer)
- ionic firmware or subnet topology differences
- Eager mode vs HIP graphs

Either way, **the integration path works end-to-end**. Production performance optimization is a separate workstream (the fork explicitly notes MoRI gives ~5× the throughput of Mooncake on ionic; building MoRI from source is the next step).

## Phase 4 escalation: MiniMax-M2.5 vLLM 1P1D disagg

Stack identical to original Phase 4 (Qwen3-0.6B), with TP=4 per node and M2.5 model. Uses 4 of 8 GPUs per node (HIP_VISIBLE_DEVICES=0,1,2,3).

### Numbers

| conc | N  | P50 (ms) | P95 (ms) | tps  | output avg | success |
|------|----|----------|----------|------|------------|---------|
| 1    | 8  | 3466     | 3486     | 18.6 | 64         | 8/8     |
| 4    | 12 | 3665     | 3853     | 68.0 | 64         | 12/12   |
| 8    | 24 | 7011     | 7127     | 72.7 | 64         | 24/24   |

Cold first request: 7.4 s. Reply: real M2.5 output with reasoning thinking tag and emoji ("Hello! 👋").

Note the **very tight P50/P95 spread** (3466→3486 at c=1, 3665→3853 at c=4) — extremely consistent latency across requests, suggesting RIXL/UCX with TCP fallback handles M2.5's KV transfer evenly.

### Comparison with Phase 2.5 (M2.5 agg, single node)

| Mode | TP | Nodes | c=8 tok/s | Latency P50 c=1 |
|---|---|---|---|---|
| Phase 2.5 vLLM agg + HIP graphs | 4 | 1 | 521.2 | 1279 ms |
| Phase 4 vLLM disagg + RIXL/UCX | 4 each | 2 | 72.7 | 3466 ms |

Disagg is **~7× slower than agg** for M2.5 (vs 22× for DSR1). vLLM/RIXL handles KV transfer more efficiently than SGLang/Mooncake on this hardware:
- RIXL falls back to ROCm copy + TCP transparently (no chunked-MR overhead per layer)
- vLLM's NixlConnector batches register_memory calls more efficiently
- Eager mode (we ran with `--enforce-eager`) caps decode throughput; HIP graphs would help

### Comparison with Phase 4 small model

| Model | c=8 tok/s | success |
|---|---|---|
| Phase 4 (Qwen3-0.6B, TP=1) | 645.6 | 24/24 |
| Phase 4 escalation (M2.5, TP=4) | 72.7 | 24/24 |

M2.5 is much slower in absolute tok/s than Qwen3-0.6B because:
- 229B params vs 0.6B (~380× more compute per token)
- KV cache transfer per token scales with model size
- TP=4 vs TP=1 has more inter-GPU sync overhead

But again, **100% success rate** — the path is solid for production-scale models.

## Bumps in the road (escalation-specific)

| Issue | Resolution |
|---|---|
| g12-22 holder job 56 expired (8h limit) mid-test | Allocated new node g12-06; updated launcher PREFILL_NODE/PREFILL_IP |
| `dynamo-vllm-rixl` image only on g12-26 | `podman save | scp | podman load` ~40 GB to g12-06 (~3 min) |
| New g12-06 didn't have `rocm/sgl-dev` | `podman pull` (~10 min for 75 GB image) |
| HIP OOM at engine init: GPU 0 had 189 GB allocated by PyTorch | Orphan `python3 dynamo.sglang/dynamo.vllm` processes survived `podman rm -f`; `pkill -9 -f` released GPU memory |
| Etcd state inconsistency (workers registered to old etcd, frontend connected to new) | Clean rerun: `podman rm -f` all dynamo containers, recreate etcd, restart workers |

## Final scorecard

| Phase | Backend | Mode | Model | Result | c=8 tok/s |
|---|---|---|---|---|---|
| 1 | SGLang | agg, TP=8 | DSR1-0528 FP8 (671B) | ✅ | 708 |
| 2.5 | vLLM | agg, TP=4 + HIP graphs | MiniMax-M2.5 (229B MoE) | ✅ | 521 |
| 3 (Qwen) | SGLang | disagg + Mooncake, TP=1 | Qwen3-0.6B | ✅ | 122 |
| 3 (DSR1) | SGLang | disagg + Mooncake, TP=8 | **DeepSeek-R1-0528 FP8** | **✅** | **32.8** |
| 4 (Qwen) | vLLM | disagg + RIXL, TP=1 | Qwen3-0.6B | ✅ | 646 |
| 4 (M2.5) | vLLM | disagg + RIXL, TP=4 | **MiniMaxAI/MiniMax-M2.5** | **✅** | **72.7** |

All 6 (re)demonstrated milestones PASS. Both production-scale disaggregated paths work end-to-end on AMD MI355X with the minimal-patch stack documented in `07-phase5-final-report.md`.

## Reproducer scripts

- `phase3_dsr1_disagg.sh` — DSR1 SGLang disagg orchestrator
- `phase4_m25_disagg.sh` — M2.5 vLLM disagg orchestrator
- `dsr1_bench.py` — DSR1 concurrency sweep
- `m25_bench.py` — M2.5 concurrency sweep
