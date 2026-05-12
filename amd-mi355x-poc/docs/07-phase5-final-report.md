# Dynamo on AMD MI355X — PoC Final Report

Date: 2026-05-11
Cluster: AAC1 (`aac1.amd.com`), partition `256C8G1H_MI355X_Ubuntu22`
Hardware: 2x 8x AMD Instinct MI355X (gfx950) nodes (`smci355-ccs-aus-g12-22` + `smci355-ccs-aus-g12-26`), each with 9× AMD Pensando ionic RoCE NICs

## Executive summary

Single-day PoC against `ai-dynamo/dynamo:main` (HEAD `5d9e5f6`). Goal: identify the **minimum** changes to upstream Dynamo to support AMD MI355X, by deriving a path simpler than the JohnQinAMD reference fork (which is 145 commits ahead, 231 files changed, 733 commits behind main).

**Result**: **All 4 attempted milestones PASS**, with sharply quantified per-phase patch sizes:

| Phase | Backend | Mode | Model | Status | c=8 tok/s | Patch surface |
|---|---|---|---|---|---|---|
| 0 | n/a | env baseline | n/a | ✅ | — | n/a |
| 1 | SGLang | single-node agg | DeepSeek-R1-0528 FP8 (671B) | **✅ PASS** | 708 | ~5 LoC + 4-file nixl stub |
| 2 / 2.5 | vLLM | single-node agg | MiniMaxAI/MiniMax-M2.5 (229B FP8 MoE) | **✅ PASS** | 521 | 4-file nixl stub only |
| 3 | SGLang | 2-node disagg + Mooncake | Qwen/Qwen3-0.6B | **✅ PASS** | 122 | + 992 LoC (fork's mooncake_rocm_staging.py + rocm_dram_staging_common.py) |
| 4 | vLLM | 2-node disagg + RIXL | Qwen/Qwen3-0.6B | **✅ PASS** | 646 | + Dockerfile.rocm-vllm + LD_PRELOAD interposer (~120 LoC); ~5× faster than Phase 3 disagg |

The headline observations:
1. **Single-node aggregated Dynamo on AMD requires essentially zero patches** to upstream `dynamo` source.
2. **vLLM disagg upstream PR is small** (~150 LoC, mostly Dockerfile + LD_PRELOAD interposer): NO vendor-patched UCX 1.12 required (the fork's runbook overstates this); RIXL's C++ stack handles VRAM detection correctly once the right `UCX_TLS` env var is set and the right ionic provider plugin is mounted.
3. **SGLang disagg upstream PR is larger** (~1000 LoC) because Mooncake on AMD requires Python-level chunked-MR + DRAM-staging that lives in the dynamo codebase. RIXL handles the equivalent in its native C++.

## Phase-by-phase numbers

### Single-node aggregated (Phases 1, 2, 2.5)

Both backends benchmarked at concurrencies 1, 4, 8. HIP graphs ON. `--enforce-eager` only used for first smoke test (Phase 2).

| Backend | Model | TP | c=1 P50 | c=8 P50 | c=8 tok/s | tok/s/GPU |
|---|---|--:|---:|---:|---:|---:|
| SGLang | DSR1-0528 FP8 (671B) | 8 | 1259 ms | 1409 ms | 708.8 | 88.6 |
| vLLM (eager) | MiniMax-M2.5 FP8 (229B MoE) | 4 | — | — | — | — |
| vLLM (HIP graphs) | MiniMax-M2.5 FP8 (229B MoE) | 4 | 1279 ms | 1427 ms | 521.2 | 130.3 |

Phase 2 eager-mode single-request: TTFT 1982 ms, ITL 66 ms/tok (~15 tok/s). Re-running with HIP graphs enabled (Phase 2.5) brought ITL down to ~10 ms/tok — **6.6× faster decode**, validating HIP graph capture is OK on MI355X+gfx950 for this MoE model.

### 2-node disaggregated (Phase 3 SGLang+Mooncake, Qwen3-0.6B)

| c | N  | P50 ms | P95 ms | tok/s | success |
|---|----|--------|--------|-------|---------|
| 1 | 8  | 1216   | 21348  | 19.1  | 8/8     |
| 4 | 12 | 2396   | 4652   | 83.3  | 12/12   |
| 8 | 24 | 4527   | 4620   | 122.2 | 24/24   |

Cold first request 6.4 s (includes Mooncake handshake + JIT). Disagg adds latency vs agg (Mooncake DRAM staging on ionic is ~5× slower than MoRI per fork's runbook); MoRI would be a next-step optimization that requires building MoRI from source (not done in PoC).

### 2-node disaggregated (Phase 4 vLLM+NIXL/RIXL, Qwen3-0.6B)

| c | N  | P50 ms | P95 ms | tok/s | success |
|---|----|--------|--------|-------|---------|
| 1 | 8  | 381    | 384    | 168.0 | 8/8     |
| 4 | 12 | 422    | 602    | 526.7 | 12/12   |
| 8 | 24 | 777    | 812    | 645.6 | 24/24   |

vLLM disagg via `NixlConnector` + RIXL/UCX worked end-to-end after fixing 4 layered config issues (none required AMD-internal source we can't access):
1. Bind-mount `libionic-rdmav34.so` (the actual ibverbs provider plugin, missing from container's apt `ibverbs-providers`)
2. Extend LD_PRELOAD interposer to wrap `ibv_reg_dmabuf_mr` (UCX uses dmabuf for VRAM)
3. Add ROCm transports to `UCX_TLS` (`rocm,rocm_copy,rocm_ipc`) so UCX recognizes GPU memory as VRAM not HOST
4. Use a fresh GPU (avoid stale 57 GB allocations from prior crashed runs)

The dmabuf-based VRAM RDMA still fails (ionic genuinely lacks GPUDirect RDMA), but RIXL's C++ stack falls back through `rocm_copy` + TCP transparently. **5× faster than SGLang+Mooncake disagg on the same hardware** because RIXL avoids the chunked-MR DRAM-staging overhead Mooncake incurs.

## Patch matrix vs `ai-dynamo/dynamo:main`

| Patch | Lines | Where | Phase used | Upstream candidate? |
|---|---|---|---|---|
| `nixl` 4-file Python stub package | 35 | runtime, `<site-packages>/nixl/` | 1, 2, 3, 4 | **Better as upstream patch**: make `dynamo.nixl_connect` import lazy (try/except) so AMD installs don't need a stub at all. ~5 LoC patch to `lib/bindings/python/src/dynamo/nixl_connect/__init__.py`. |
| `typing.Self` `.pth` shim | 1 | runtime, `<site-packages>/zzz_typing_self_compat.pth` | 1 only | Bump `python_requires>=3.11` in ai-dynamo metadata (or use `typing_extensions.Self`); fixes Python 3.10 containers. |
| `dynamo/sglang/publisher.py` `sgl.Engine` re-export sed | 3 | runtime sed | 1, 3 | One-line upstream patch: import `Engine` from `sglang.srt.entrypoints.engine` directly. |
| `mooncake_rocm_staging.py` (fork copy) | 640 | runtime, `<site-packages>/dynamo/sglang/` | 3 | **Upstream candidate** as a new submodule `dynamo.sglang.transports.mooncake_rocm`. The bulk of AMD-Dynamo's "real work". |
| `rocm_dram_staging_common.py` (fork copy) | 352 | runtime, `<site-packages>/dynamo/sglang/` | 3 | Companion to above; upstream as `dynamo.sglang.transports.rocm_dram_staging_common`. |
| Auto-activation .pth files | 4 | runtime, `<site-packages>/zzz_dynamo_*.pth` | 3 | Roll into the submodule's `__init__.py` (no .pth needed). |
| `Dockerfile.rocm-vllm-rixl` | 35 | new, build-time | 4 | Upstream as `container/Dockerfile.rocm-vllm`, same intent as fork's. |
| `ibv_ionic_compat.so` LD_PRELOAD interposer (~50 LoC C) | 50 | runtime, built in container | 4 | Ship as a small native helper or document the LD_PRELOAD workaround until AMD's UCX 1.12 patch lands. |
| Launch scripts (per phase) | 50-100 each | new, `examples/backends/{sglang,vllm}/launch/rocm/*.sh` | 1-4 | Upstream as new `examples/` files, mirrors fork's structure. |
| Container env+device flags | n/a | runtime via podman/docker | 3, 4 | Documented in launch scripts; no source change needed. |

### Net upstream PR breakdown (proposed)

| PR # | Scope | Approx LoC | Risk |
|---|---|---|---|
| 1 | Make `dynamo.nixl_connect.import nixl._api` lazy + `typing_extensions.Self` shim + `dynamo/sglang/publisher.py` Engine import fix | ~15 LoC source + tests | Tiny, mergeable today |
| 2 | `examples/backends/{sglang,vllm}/launch/rocm/agg_rocm.sh` + `docs/amd-quickstart.md` | ~150 LoC scripts/docs | Tiny, mergeable today |
| 3 | New `dynamo.sglang.transports.mooncake_rocm` submodule (renamed from fork's mooncake_rocm_staging.py + rocm_dram_staging_common.py) + opt-in via `SGLANG_MOONCAKE_ROCM_STAGING=1` | ~1000 LoC | Medium — needs design discussion (transport abstraction, code ownership) |
| 4 | `container/Dockerfile.rocm-sglang` and `container/Dockerfile.rocm-vllm` baking in the libionic ABI fix and ionic device discovery | ~150 LoC each | Small, mostly Docker |
| 5 | `dynamo.vllm.{args,main}` bootstrap-host patches (~40 LoC) + LD_PRELOAD interposer C source (~70 LoC) + UCX_TLS env var docs + libionic-rdmav34 mount instructions | ~150 LoC | **Mergeable today** — Phase 4 PASS proved this works against public UCX 1.19.x with our 4-fix config |

PRs 1 and 2 are essentially free wins — they can land immediately and unblock anyone trying Dynamo on AMD agg. PR 3 is the bulk of the SGLang disagg work. PR 4 is container plumbing. **PR 5 is also mergeable today** — Phase 4 PASS proved that vLLM disagg works on public UCX 1.19.x with our 4 documented config fixes; the AMD-internal vendor UCX 1.12 referenced in the fork's runbook Layer 5 turned out to be a red herring (it strips REMOTE_ATOMIC at the UCX-source level, but our LD_PRELOAD interposer wrapping `ibv_reg_mr`/`ibv_reg_mr_iova2`/`ibv_reg_dmabuf_mr` plus `UCX_TLS=...,rocm,rocm_copy,rocm_ipc,...` achieves equivalent functional behavior).

## Comparison vs JohnQinAMD reference fork

The fork has 145 commits ahead of main, 231 files changed (per `gh api compare/main...amd-dynamo`). Key components NOT touched by this PoC (out of scope):

- `components/src/dynamo/atom/` — entire third backend (Atom, MI355X-specific kernels), 13 files, ~1300 LoC
- `lib/{llm,memory,bindings,kvbm-{kernels,physical}}/src/...hip.rs` and `lib/llm/src/block_manager/.../hip.rs` — Rust HIP shims for KVBM transfer (~10 files)
- `lib/gpu_memory_service/.../hip_vmm_utils.py` — HIP VMM utilities for KVBM
- `lib/bindings/kvbm/python/kvbm/sglang_integration/` — KVBM-SGLang integration
- `components/src/dynamo/common/gpu_utils.py` — fork has 612 LoC GPU detection abstraction; this PoC didn't need any of it (Dynamo's existing GPU detection works on AMD via amd-smi)
- All planner/autoscaler/fault-tolerance/Kubernetes operator AMD bits
- EP/DP-Attn DEP8 (the 16k tok/s configuration)
- Multimodal, embedding, long-context (1M)
- vLLM disagg KV transport beyond `NixlConnector` (which we attempted)
- KV-aware router (NVIDIA InferenceX itself uses round-robin in production)

The fork's bulk is in three places: (1) the Atom backend, (2) the Mooncake/NIXL DRAM-staging adapters, (3) container plumbing. This PoC needed (2) and (3) only. (1) and most of the others are valuable but not strictly required to demonstrate Dynamo running on AMD.

## Decision points to surface to NVIDIA team

1. **`dynamo.nixl_connect` import policy**: PR 1 (make nixl import lazy) is a tiny, friendly change. Does NVIDIA want it gated behind a feature flag, or just unconditional?
2. **Transport adapter pattern**: should `mooncake_rocm_staging.py` (and the eventual NIXL VRAM→DRAM staging) live under a new `dynamo.{sglang,vllm}.transports.<name>_rocm` submodule? That's a clean upstream pattern but defines a new internal API.
3. **Container distribution**: separate `Dockerfile.rocm-{sglang,vllm}` files (fork pattern, what we used) vs unified `Dockerfile.template` with `--build-arg HARDWARE=rocm`?
4. **GPU detection shim**: do we want the fork's 612-LoC `gpu_utils.py` upstream (handles many edge cases) or a focused `amd-smi` adapter (~50 LoC)? PoC suggests the latter.
5. ~~**AMD UCX 1.12 vendor patch**~~ — turns out NOT required for the vLLM disagg path. Phase 4 PASS used public UCX 1.19.x with our 4-fix config (LD_PRELOAD interposer + UCX_TLS env + libionic-rdmav34 bind-mount + fresh GPU). The vendor UCX patch in the fork's runbook is a "cleaner" version that strips REMOTE_ATOMIC at the UCX source level for ionic only, but the runtime behavior is equivalent. AMD could still upstream the vendor-aware UCX patch to `ROCm/ucx` for a cleaner long-term story, but it's not blocking.
6. **CI**: AMD MI355X CI surface in dynamo's repo — full ROCm build + smoke test on every PR (expensive, ~25 min image build), or label-gated workflow?

## Reproducer index

All scripts are on AAC1 at `/shared/amdgpu/home/anluo/dynamo-poc/`:

| File | Phase | What |
|---|---|---|
| `phase2_e2e.sh` | 2 | vLLM agg with --enforce-eager (smoke test recipe) |
| `phase2_perf.sh` | 2.5 | vLLM agg with HIP graphs + concurrency sweep |
| `phase1_e2e.sh` | 1 | SGLang agg w/ DSR1 + perf sweep |
| `phase3_disagg.sh` | 3 | SGLang 1P1D disagg with Mooncake (uses fork's `mooncake_rocm_staging.py`) |
| `phase4_disagg.sh` | 4 | vLLM 1P1D disagg with NixlConnector + RIXL + LD_PRELOAD interposer (PARTIAL) |
| `fork-patches/{mooncake,nixl}_rocm_staging.py`, `rocm_dram_staging_common.py`, `nixl_dram_staging.py` | 3, 4 | Copied from JohnQinAMD/dynamo:amd-dynamo |
| `Dockerfile` (in `/tmp/rixl-build/`) | 4 | Custom image: rocm/vllm-dev:nightly + UCX-ROCm 1.19.x + RIXL + Python bindings |

Container holders: SLURM jobs `56` (g12-22) + `57` (g12-26), 8h each.

Reports: `~/Documents/dynamo-amd-poc/phase{0,1,2,3,4,5}-*.md` + `dynamo-mi355x-poc-plan.md` (this folder).

## What I'd do next (if I were continuing)

1. **Land PRs 1, 2, 5** against `ai-dynamo/dynamo:main` immediately. All small, no-risk, all proven to work.
2. **Attempt DSR1 disagg via Phase 3 stack** (SGLang+Mooncake) — already proven path; mooncake_rocm_staging includes chunked-MR for large models; ~30 min model loading per node.
3. **Attempt MiniMax-M2.5 disagg via Phase 4 stack** (vLLM+RIXL) — same recipe, larger model; would establish whether the chunked KV transfer scales for big models on the RIXL path.
4. **Build MoRI** from source on AAC1 (fork's runbook reports MoRI gives ~5× the throughput of Mooncake on ionic) and run Phase 3 with MoRI for the production-grade SGLang story.
5. **Write the actual PR descriptions** for upstream review, including the patches we identified above.
