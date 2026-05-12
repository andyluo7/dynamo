# Dynamo on AMD MI355X — Proof-of-Concept

End-to-end PoC demonstrating NVIDIA Dynamo running on AMD MI355X (gfx950) GPUs
with both vLLM and SGLang backends, in single-node aggregated **and** 2-node
disaggregated topologies. Exercises the full stack: `dynamo.frontend`,
`dynamo.sglang`/`dynamo.vllm`, etcd, NATS, and KV-cache transfer over AMD
Pensando ionic RoCE NICs.

**This branch does not modify any existing `dynamo` source files.** Everything
lives under this `amd-mi355x-poc/` subdirectory and is applied at runtime by
the launch scripts. The objective was to identify the **minimum** set of
changes upstream `ai-dynamo/dynamo:main` would need to support AMD MI355X.

## Results — all milestones PASS (incl. production-scale escalations)

Cluster: AAC1 (`aac1.amd.com`), partition `256C8G1H_MI355X_Ubuntu22`.
Hardware: 2× 8-GPU MI355X nodes, each with 9× AMD Pensando ionic RoCE NICs.

| Phase | Backend | Mode | Model | Result | c=8 tok/s | Success @ c=8 |
|---|---|---|---|---|---|---|
| 1 | SGLang | single-node agg, TP=8 | DeepSeek-R1-0528 FP8 (671B) | ✅ | 708 | 24/24 |
| 2 / 2.5 | vLLM | single-node agg, TP=4, HIP graphs | MiniMax-M2.5 FP8 (229B MoE) | ✅ | 521 | 24/24 |
| 3 (Qwen) | SGLang | 2-node disagg + Mooncake, TP=1 | Qwen3-0.6B | ✅ | 122 | 24/24 |
| 3 (DSR1, conservative) | SGLang | 2-node disagg + Mooncake, TP=8 + HIP graphs | DeepSeek-R1-0528 FP8 (671B) | ✅ | 32.8 | 24/24 |
| **3 (DSR1, fork-aligned)** | **SGLang** | **2-node disagg + Mooncake, TP=8 + fork's launch + env (Test 12 repro)** | **DeepSeek-R1-0528 FP8 (671B)** | **✅** | **104.7 tok/s/req** ² | **10/10** |
| 4 (Qwen) | vLLM | 2-node disagg + RIXL/UCX, TP=1, eager | Qwen3-0.6B | ✅ | 646 | 24/24 |
| **4 (M2.5)** | vLLM | **2-node disagg + RIXL/UCX, TP=4 + HIP graphs** | **MiniMaxAI/MiniMax-M2.5 (229B MoE)** | **✅** | **587.1** ¹ | **24/24** |

¹ Initial M2.5 disagg run with `--enforce-eager` measured 72.7 tok/s @ c=8; re-running with HIP graphs enabled gave **8.1× speedup** to 587 tok/s. Notably **587 > 521 (Phase 2.5 single-node agg)** — disagg has 2 nodes' compute (8 GPUs vs 4) and the Dynamo frontend + KV-router overhead does not exceed the doubled compute.

² Initial DSR1 disagg measurement was 32.8 tok/s aggregate @ c=8 with a conservative launch config. After applying the fork's exact launch flags (`--kv-cache-dtype fp8_e4m3 --attention-backend aiter` etc. from `scripts/benchmark/models.yaml` DSR1 entry) + 17 env vars from `env.sh` + a streaming bench harness with ISL=1024 OSL=1024 (matching `bench.sh`), per-request output throughput rose to **104.7 tok/s — 7% above the JohnQinAMD fork's published Test 12 result of 97.7 tok/s on the same Mooncake transport**. See [`docs/09-test12-reproduction.md`](docs/09-test12-reproduction.md). The ~6× speedup over the conservative measurement was tuning, not architecture.

Production-scale escalation rows (bold) — see [`docs/08-phase34-escalation-results.md`](docs/08-phase34-escalation-results.md). Both the 671B DeepSeek-R1 SGLang disagg and 229B MiniMax-M2.5 vLLM disagg paths run end-to-end across two MI355X nodes with KV transfer over Pensando ionic RoCE.

Per-GPU efficiency observations:
- **DSR1 disagg = 2.1 tok/s/GPU @ c=8** on 16 GPUs with Mooncake (the chunked-MR DRAM-staging path is the bottleneck — fork reports MoRI gives ~5× more throughput on ionic than Mooncake)
- **M2.5 disagg = 73.4 tok/s/GPU @ c=8** on 8 GPUs with RIXL/UCX — **matches the JohnQinAMD fork's published 73.9 tok/s/GPU result** for DSR1 InferenceX MoRI 1P1D at c=32

The remaining gap to the fork's headline 1,334 tok/s/GPU DEP8 result requires building MoRI from source, applying EP/DP-Attention, and a few additional patches — see [Path to Production Performance](docs/08-phase34-escalation-results.md#path-to-production-performance) in the escalation doc for the full 13-item list.

Headline: **single-node aggregated Dynamo on AMD requires essentially zero
patches** to upstream `dynamo` source. Disaggregated serving needs ~150 LoC
new (vLLM path) or ~1000 LoC new (SGLang+Mooncake path) — both reasonable
upstream PR sizes.

## Layout

```
amd-mi355x-poc/
├── README.md                              ← this file
├── docs/                                  ← phase-by-phase reports
│   ├── 00-poc-plan.md                     ← original plan + scope decisions
│   ├── 01-phase0-inventory.md             ← AAC1 hardware inventory + tooling
│   ├── 02-phase1-sglang-agg.md            ← SGLang + DSR1 single-node ✅
│   ├── 03-phase2-vllm-design.md           ← Phase 2 design rationale
│   ├── 04-phase2-vllm-agg.md              ← vLLM + MiniMax-M2.5 single-node ✅
│   ├── 05-phase3-sglang-disagg.md         ← SGLang + Mooncake 2-node ✅
│   ├── 06-phase4-vllm-disagg.md           ← vLLM + RIXL/UCX 2-node ✅
│   └── 07-phase5-final-report.md          ← consolidated final report + PR breakdown
├── scripts/                               ← reproducer scripts (run from AAC1 login node)
│   ├── phase1_e2e.sh                      ← SGLang+DSR1 single-node + bench
│   ├── phase2_e2e.sh                      ← vLLM+MiniMax-M2.5 single-node (eager)
│   ├── phase2_perf.sh                     ← vLLM+MiniMax-M2.5 single-node (HIP graphs + bench)
│   ├── phase3_disagg.sh                   ← SGLang 1P1D disagg with Mooncake (orchestrator)
│   ├── phase3_bench.py                    ← Phase 3 concurrency sweep client
│   ├── phase4_disagg.sh                   ← vLLM 1P1D disagg with RIXL (orchestrator)
│   ├── phase4_bench.py                    ← Phase 4 concurrency sweep client
│   └── rixl_probe.py                      ← Direct RIXL register-memory probe (Phase 4 debug)
├── container/
│   └── Dockerfile.rocm-vllm-rixl          ← extends rocm/vllm-dev:nightly w/ UCX-ROCm + RIXL
└── patches/                               ← runtime patches applied inside containers
    ├── README.md                          ← describes each patch + upstream candidate
    ├── nixl_stub/                         ← 4-file Python stub (Phase 1/2/3)
    │   ├── __init__.py
    │   ├── _api.py
    │   └── _bindings.py
    ├── ibv_ionic_compat.c                 ← LD_PRELOAD interposer for ionic (Phase 4)
    ├── zzz_typing_self_compat.pth         ← Python 3.10 compat shim (Phase 1/3/4)
    └── fork-patches/                      ← copied from JohnQinAMD/dynamo:amd-dynamo
        ├── mooncake_rocm_staging.py       ← 640 LoC, used in Phase 3
        ├── rocm_dram_staging_common.py    ← 352 LoC, used in Phase 3
        ├── nixl_rocm_staging.py           ← 1225 LoC, reference for SGLang+NIXL path
        └── nixl_dram_staging.py           ← 188 LoC, reference
```

## How to reproduce

### Prerequisites

- Access to a cluster with 8× AMD Instinct MI355X (gfx950) per node
- Pensando ionic RoCE NICs (or equivalent — KV-transfer specifics will differ)
- `podman` (containers) — note: rocm/vllm-dev:nightly user is NOT in `docker` group
- ROCm 7.2.x available (e.g., via `module load rocm/7.2.2` on AAC1)
- `huggingface-cli` access to `deepseek-ai/DeepSeek-R1-0528`, `MiniMaxAI/MiniMax-M2.5`,
  `Qwen/Qwen3-0.6B` (the latter two should be pre-cached on shared FS)

### Phase 1 — SGLang single-node aggregated, DeepSeek-R1 FP8

```bash
bash scripts/phase1_e2e.sh
# Container `dynamo-sglang-poc` will load DSR1 (~10 min from NFS),
# bring up dynamo.frontend on :8000, register the model, and run a
# concurrency sweep at c=1, 4, 8.
```

### Phase 2 — vLLM single-node aggregated, MiniMax-M2.5

```bash
# Smoke test (eager mode, ~5 min model load):
bash scripts/phase2_e2e.sh

# Production perf (HIP graphs, ~7 min model load + 2 min graph capture):
bash scripts/phase2_perf.sh
```

### Phase 3 — SGLang 1P1D disaggregated with Mooncake

```bash
# Edit the script to set PREFILL_NODE/DECODE_NODE/PREFILL_IP for your cluster.
bash scripts/phase3_disagg.sh
# Bench:
ssh <prefill_node> python3 /tmp/phase3_bench.py
```

### Phase 4 — vLLM 1P1D disaggregated with RIXL

```bash
# First time: build the custom image (~10 min, includes UCX-ROCm + RIXL).
podman build -t dynamo-vllm-rixl:latest -f container/Dockerfile.rocm-vllm-rixl .

# Then orchestrate prefill+decode across two nodes:
bash scripts/phase4_disagg.sh
ssh <prefill_node> python3 /tmp/p4_bench.py
```

## Suggested upstream PR breakdown (from `docs/07-phase5-final-report.md`)

| PR | Scope | LoC | Risk |
|---|---|---|---|
| 1 | nixl import lazy + typing_extensions.Self use + dynamo.sglang.publisher Engine import fix | ~15 source | Tiny, mergeable today |
| 2 | `examples/backends/{sglang,vllm}/launch/rocm/agg_rocm.sh` + AMD quickstart docs | ~150 | Tiny, mergeable today |
| 3 | new `dynamo.sglang.transports.mooncake_rocm` submodule (renamed from fork's mooncake_rocm_staging.py + rocm_dram_staging_common.py) | ~1000 | Medium |
| 4 | `container/Dockerfile.rocm-{sglang,vllm}` baking the libionic ABI fix + ionic device discovery | ~150 each | Small, mostly Docker |
| 5 | `dynamo.vllm.{args,main}` bootstrap-host patches + LD_PRELOAD interposer C source + UCX_TLS env documentation | ~150 | **Mergeable today** — Phase 4 PASS proved this works on public UCX 1.19.x |

PRs 1, 2, 5 are essentially free wins — small diffs, no source-code controversy,
all proven to work end-to-end in this PoC.

## License + attribution

All new code in this directory is Apache 2.0 (matching the parent dynamo repo).
Files in `patches/fork-patches/` are copied unmodified from the JohnQinAMD fork
of ai-dynamo/dynamo (Apache 2.0, NVIDIA copyright). The LD_PRELOAD interposer
in `patches/ibv_ionic_compat.c` was extracted from that fork's
`nixl_rocm_staging.py` and extended with `ibv_reg_dmabuf_mr` wrapping.

## Contact

- AMD-side: andyluo7 (this fork's owner)
- Upstream Dynamo: ai-dynamo/dynamo
- AMD Dynamo reference: https://github.com/JohnQinAMD/dynamo/tree/amd-dynamo
- AMD RIXL: https://github.com/ROCm/RIXL
