# Dynamo on AMD MI355X — Proof-of-Concept

End-to-end PoC demonstrating NVIDIA Dynamo running on AMD MI355X (gfx950) GPUs
with both vLLM and SGLang backends, in single-node aggregated **and** 2-node
disaggregated topologies. Exercises the full stack: `dynamo.frontend`,
`dynamo.sglang` / `dynamo.vllm`, etcd, NATS, and KV-cache transfer over AMD
Pensando ionic RoCE NICs.

**This branch does not modify any existing `dynamo` source files.** Everything
new lives under this `amd-mi355x-poc/` subdirectory and is applied at runtime
by the launch scripts. The objective was to identify the **minimum** set of
changes upstream `ai-dynamo/dynamo:main` would need to support AMD MI355X.

## Four findings to take away

1. **Single-node aggregated Dynamo on AMD requires essentially zero patches**
   to upstream `dynamo` source. PRs 1, 2 in the breakdown below are mergeable
   today.

2. **vLLM + RIXL disaggregated serving is production-ready on AMD ionic
   hardware for medium-KV models.** Saturates cleanly at the **compute**
   ceiling on M2.5 (730 tok/s aggregate at c=32, 320/320 success). PR 5
   captures the dynamo-side bits — also mergeable today.

3. **SGLang + Mooncake disaggregated has a transport ceiling at c=32 on
   ionic.** Matches the JohnQinAMD fork's documented finding. We exceed the
   fork's published Mooncake numbers at low concurrency (+8.2% @ c=1, +47% @
   c=4) and reach 527 tok/s @ c=16, but Mooncake crashes with cascading
   `transport retry counter exceeded` at c=32. The fork uses MoRI for c>16
   production benchmarks; that's the next workstream, not a PoC blocker.

4. **Public RIXL also has a ceiling on DSR1-class models — at startup, not
   under load.** Same RIXL+UCX-ROCm stack that runs M2.5 cleanly fails to
   `ibv_reg_mr` DSR1's 1.7-2.6 GB per-rank KV pool against ionic
   (~250 MB per-MR limit). The MR-chunking that Mooncake's fork-staging code
   does (and that MoRI does properly) is what's missing from public RIXL.
   See [`docs/12-dsr1-vllm-rixl-cross-validation.md`](docs/12-dsr1-vllm-rixl-cross-validation.md). This sharpens the upstream story:
   PR 5 is correct and mergeable today, but **the right scope-statement for
   public RIXL on ionic is "medium-KV per rank"**, not "all models".

## Results — all milestones PASS

Cluster: AAC1 (`aac1.amd.com`), partition `256C8G1H_MI355X_Ubuntu22`.
Hardware: 2× 8-GPU MI355X nodes, each with 9× AMD Pensando ionic RoCE NICs.

### Best result per configuration

| # | Backend | Topology | Model | Best aggregate tok/s | Saturated at | Success |
|---|---|---|---|---:|---:|---:|
| 1 | SGLang | single-node agg, TP=8 + HIP graphs | DeepSeek-R1-0528 FP8 (671B) | **708** @ c=8 | compute | 24/24 |
| 2.5 | vLLM | single-node agg, TP=4 + HIP graphs | MiniMax-M2.5 FP8 (229B MoE) | **521** @ c=8 | compute | 24/24 |
| 3 | SGLang | 2-node disagg + Mooncake, TP=8 each | DeepSeek-R1-0528 FP8 (671B) | **527** @ c=16 ¹ | **transport (Mooncake @ c=32)** | 160/160 |
| 4 | vLLM | 2-node disagg + RIXL/UCX, TP=4 each | MiniMaxAI/MiniMax-M2.5 (229B MoE) | **730** @ c=32 ² | compute | 320/320 |
| 4-DSR1 | vLLM | 2-node disagg + RIXL/UCX, TP=8 each | DeepSeek-R1-0528 FP8 (671B) | n/a — startup fails | **MR-size limit (ionic)** ³ | n/a |

### Side-by-side: same hardware, different transport

| Concurrency | SGLang+Mooncake (DSR1) | **vLLM+RIXL (M2.5)** |
|---:|---:|---:|
| 4 | 261.7 tok/s | 385.4 tok/s |
| 8 | 456.7 | 718.2 |
| 16 | 527.4 | 727.3 |
| **32** | **CRASH** (transport retry exceeded) | **730.4 tok/s, 320/320 success** |

This is the architectural difference at the heart of finding #3. **RIXL's UCX-plugin C++ DRAM staging handles ionic's MR limits gracefully where Mooncake's Python-level chunked-transfer pattern saturates the firmware's QP-setup queue.**

### vs JohnQinAMD fork's published numbers

| Comparison point | Fork | Us | Verdict |
|---|---:|---:|---|
| Mooncake DSR1 @ c=1 (Test 12) | 97.7 tok/s/req | **105.7** | **+8.2% ahead** with stock public Mooncake |
| Mooncake DSR1 TPOT @ c=1 | 7.11 ms | 9.46 ms | -33% (residual aiter MoE preshuffle tuning) |
| MoRI+EP/DP-Attn DSR1 @ c=4 | 178 tok/s | **261.7** (Mooncake!) | **+47% ahead** |
| MoRI+EP/DP-Attn DSR1 @ c=16 | 672 tok/s | 527.4 (Mooncake) | -22% behind |
| MoRI+EP/DP-Attn DSR1 @ c=128 | 2,196 | n/a (Mooncake crashed at c=32) | needs MoRI to compete |
| DEP8 DSR1 @ c=1024 | 16,011 (1,334/GPU) | n/a | needs MoRI + EP/DP-Attn (Tier 3 in path-to-prod) |

¹ DSR1 SGLang+Mooncake disagg headline: **527.4 tok/s @ c=16 with 160/160 success — 16× over the conservative 32.8 tok/s baseline measured before fork-aligned tuning**. Crashed at c=32 (transport retry exceeded). See [`docs/10-dsr1-concurrency-sweep.md`](docs/10-dsr1-concurrency-sweep.md). Initial M2.5 disagg run with `--enforce-eager` measured 72.7 tok/s @ c=8; HIP graphs gave 8.1× speedup to 587. Initial DSR1 disagg measurement was 32.8 tok/s aggregate @ c=8 with a conservative launch config; applying the fork's exact launch flags + 9 env vars + matching bench harness raised per-request output throughput to **105.7 tok/s — 8.2% above the fork's published Test 12 result of 97.7 tok/s on the same Mooncake transport**, see [`docs/09-test12-reproduction.md`](docs/09-test12-reproduction.md).

² M2.5 vLLM+RIXL disagg headline: **730.4 tok/s @ c=32 with 320/320 success**. Performance saturates at compute (c=8/16/32 all hit ~720-730 tok/s aggregate; TTFT grows from 131ms to 33.6s but throughput plateaus). To go higher requires more decode GPUs or a smaller model. Detailed sweep + RIXL-vs-Mooncake architectural analysis in [`docs/11-m25-vllm-rixl-sweep.md`](docs/11-m25-vllm-rixl-sweep.md).

³ DSR1 vLLM+RIXL cross-validation: same RIXL+UCX-ROCm stack that runs M2.5 cleanly cannot complete `ibv_reg_mr` for DSR1's 1.7-2.6 GB per-rank KV-pool MR against ionic (~250 MB per-MR limit). Two attempts (max-num-seqs=32 and =4) both `NIXL_ERR_BACKEND` at startup. This is the fundamental MR-size ceiling for vanilla public RIXL on ionic — completes the architectural story: M2.5 fits, DSR1 doesn't, neither public transport handles DSR1 at production scale on this NIC. Full reproducer + log excerpts in [`docs/12-dsr1-vllm-rixl-cross-validation.md`](docs/12-dsr1-vllm-rixl-cross-validation.md).

The remaining gap to the fork's headline 1,334 tok/s/GPU DEP8 result requires building MoRI from source, applying EP/DP-Attention, and a few additional patches — see [Path to Production Performance](docs/08-phase34-escalation-results.md#path-to-production-performance) in the escalation doc for the full 13-item list.

## Suggested upstream PR breakdown

| PR | Scope | LoC | Status |
|---|---|---|---|
| 1 | `dynamo.nixl_connect.import nixl._api` lazy + `typing_extensions.Self` use + `dynamo.sglang.publisher.py` Engine import path fix | ~15 source | **Mergeable today** |
| 2 | `examples/backends/{sglang,vllm}/launch/rocm/agg_rocm.sh` + AMD quickstart docs | ~150 | **Mergeable today** |
| 3 | New `dynamo.sglang.transports.mooncake_rocm` submodule (renamed from JohnQinAMD fork's `mooncake_rocm_staging.py` + `rocm_dram_staging_common.py`); opt-in via `SGLANG_MOONCAKE_ROCM_STAGING=1` | ~1000 | Needs design discussion |
| 4 | `container/Dockerfile.rocm-{sglang,vllm}` with libionic ABI fix + ionic device discovery | ~150 each | Small, mostly Docker |
| 5 | `dynamo.vllm.{args,main}` bootstrap-host patches + LD_PRELOAD interposer C source + UCX_TLS env var docs + libionic-rdmav34 mount instructions | ~150 | **Mergeable today** — Phase 4 PASS proved this works on public UCX 1.19.x |

PRs 1, 2, 5 are essentially free wins — small diffs, no source-code controversy, all proven to work end-to-end in this PoC. PR 3 is the bulk of the SGLang disagg work and would benefit from a quick design conversation about transport-adapter location before sending.

## Layout

```
amd-mi355x-poc/
├── README.md                              ← this file
├── docs/                                  ← phase-by-phase reports (read in order)
│   ├── 00-poc-plan.md                     ← original plan + scope decisions
│   ├── 01-phase0-inventory.md             ← AAC1 hardware inventory + tooling
│   ├── 02-phase1-sglang-agg.md            ← SGLang + DSR1 single-node ✅
│   ├── 03-phase2-vllm-design.md           ← Phase 2 design rationale
│   ├── 04-phase2-vllm-agg.md              ← vLLM + MiniMax-M2.5 single-node ✅
│   ├── 05-phase3-sglang-disagg.md         ← SGLang + Mooncake 2-node ✅
│   ├── 06-phase4-vllm-disagg.md           ← vLLM + RIXL/UCX 2-node ✅
│   ├── 07-phase5-final-report.md          ← consolidated final report + PR breakdown
│   ├── 08-phase34-escalation-results.md   ← DSR1 + M2.5 production-scale escalation
│   ├── 09-test12-reproduction.md          ← reproducing fork's Test 12 (+8.2% vs fork)
│   ├── 10-dsr1-concurrency-sweep.md       ← DSR1 sweep, hits Mooncake ionic ceiling
│   ├── 10-dsr1-sweep-results.csv          ← raw per-c CSV from the DSR1 sweep
│   ├── 11-m25-vllm-rixl-sweep.md          ← M2.5 sweep, RIXL has no transport ceiling
│   ├── 11-m25-sweep-results.csv           ← raw per-c CSV from the M2.5 sweep
│   └── 12-dsr1-vllm-rixl-cross-validation.md  ← DSR1+RIXL hits ionic MR-size limit at startup
├── scripts/                               ← reproducer scripts (run from AAC1 login node)
│   ├── phase1_e2e.sh                      ← SGLang+DSR1 single-node + bench
│   ├── phase2_e2e.sh                      ← vLLM+MiniMax-M2.5 single-node (eager)
│   ├── phase2_perf.sh                     ← vLLM+MiniMax-M2.5 single-node (HIP graphs + bench)
│   ├── phase3_disagg.sh                   ← SGLang 1P1D Qwen3-0.6B disagg with Mooncake
│   ├── phase3_dsr1_disagg.sh              ← SGLang 1P1D DSR1 disagg, conservative config
│   ├── phase3_test12_repro.sh             ← SGLang 1P1D DSR1 disagg, fork-aligned config
│   ├── phase3_bench.py                    ← Phase 3 single-c bench
│   ├── dsr1_bench.py                      ← DSR1 single-c bench
│   ├── test12_bench.py                    ← streaming bench matching fork's bench.sh
│   ├── test12_sweep.sh                    ← DSR1 concurrency sweep (calls test12_bench.py)
│   ├── phase4_disagg.sh                   ← vLLM 1P1D Qwen3-0.6B disagg with RIXL
│   ├── phase4_m25_disagg.sh               ← vLLM 1P1D M2.5 disagg with RIXL + HIP graphs
│   ├── phase4_dsr1_disagg.sh              ← vLLM 1P1D DSR1 disagg attempt (MR-size cross-validation)
│   ├── phase4_bench.py                    ← Phase 4 single-c bench
│   ├── m25_bench.py                       ← M2.5 single-c bench
│   ├── m25_sweep.sh                       ← M2.5 concurrency sweep
│   └── rixl_probe.py                      ← Direct RIXL register-memory probe (Phase 4 debug)
├── container/
│   └── Dockerfile.rocm-vllm-rixl          ← extends rocm/vllm-dev:nightly w/ UCX-ROCm + RIXL
└── patches/                               ← runtime patches applied inside containers
    ├── README.md                          ← describes each patch + upstream candidate
    ├── nixl_stub/                         ← 4-file Python stub (Phase 1/2/3)
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
- `podman` (containers) — note: rocm/vllm-dev:nightly user is NOT in `docker` group on AAC1
- ROCm 7.2.x available (e.g., via `module load rocm/7.2.2` on AAC1)
- HuggingFace access to `deepseek-ai/DeepSeek-R1-0528`, `MiniMaxAI/MiniMax-M2.5`,
  `Qwen/Qwen3-0.6B`

### Single-node aggregated

```bash
# Phase 1 — SGLang + DSR1 (TP=8, HIP graphs, ~10 min model load)
bash scripts/phase1_e2e.sh

# Phase 2 — vLLM + MiniMax-M2.5 single-node
bash scripts/phase2_e2e.sh    # eager (smoke test)
bash scripts/phase2_perf.sh   # HIP graphs (production)
```

### 2-node disaggregated

```bash
# Phase 3 — SGLang + Mooncake (Qwen3-0.6B small model)
bash scripts/phase3_disagg.sh

# Phase 3 — DSR1 with fork-aligned config (Test 12 reproduction)
bash scripts/phase3_test12_repro.sh
ssh <prefill_node> python3 /tmp/test12_bench.py --isl 1024 --osl 1024 --conc 16 --num-prompts 160 --warmup 32 --ignore-eos
# Or run the full sweep:
bash scripts/test12_sweep.sh    # c=1, 4, 8, 16, 32, 64

# Phase 4 — vLLM + RIXL (Qwen3-0.6B)
podman build -t dynamo-vllm-rixl:latest -f container/Dockerfile.rocm-vllm-rixl .  # ~10 min, first time
bash scripts/phase4_disagg.sh

# Phase 4 — vLLM + RIXL with M2.5
bash scripts/phase4_m25_disagg.sh
ssh <prefill_node> bash /tmp/m25_sweep.sh    # c=1, 4, 8, 16, 32
```

### Cluster-ops gotchas (learned the hard way)

- The SLURM partition `256C8G1H_MI355X_Ubuntu22` on AAC1 is **NOT enforced exclusive**. Multiple users can land on the same node. Pick an explicit idle node via `--nodelist=` in your sbatch and verify with `rocm-smi --showmemuse` before launching.
- ROCm/HIP doesn't always release GPU memory immediately on container kill. Use `pkill -9 -f VLLM::Worker` / `pkill -9 -f sglang.srt` to force-release if `podman rm -f` leaves orphans.
- The dynamo-prefill container's `--log-level warning` suppresses progress lines for HIP graph capture and KV transfer — looks "stuck" for the first ~10 min after launch but is fine. Check `rocm-smi --showpids` if uncertain.

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
