# Dynamo on AMD MI355X — PoC Plan

## Context

NVIDIA Dynamo (`ai-dynamo/dynamo`) is a Datacenter-Scale Distributed Inference Serving Framework. Following a meeting between AMD and the NVIDIA Dynamo team, both sides agreed to scope a PoC that demonstrates Dynamo running on AMD MI355X (gfx950) with both vLLM and SGLang backends, serving DeepSeek-R1 (FP8). The objective is to **identify the minimal set of changes from current Dynamo `main`** required to land AMD support upstream.

A reference implementation already exists: [`JohnQinAMD/dynamo:amd-dynamo`](https://github.com/JohnQinAMD/dynamo/tree/amd-dynamo). It is comprehensive (145 commits ahead of `main`, 231 files changed) but is also 733 commits behind `main` and includes substantial features that are out of PoC scope (a third "Atom" backend, KVBM HIP transfer, full benchmarking suite, planner/autoscaler AMD bits, ionic-NIC RDMA patches for MoRI/Mooncake/RIXL, etc.). The PoC's value is to distill this fork into a small, reviewable diff against the current `main`.

**Outcome we want:** a minimal PoC fork + per-file diff list that NVIDIA can use as the basis for upstream AMD support PRs, plus reproducer scripts and a short comparison report.

## User-confirmed scope

| Decision | Value |
|---|---|
| Backends | Both **SGLang** and **vLLM** |
| Models | SGLang: **DeepSeek-R1 FP8** (`deepseek-ai/DeepSeek-R1-0528`, ~640 GB, will be downloaded fresh — matches the fork's proven DEP8 path). vLLM: **cached `MiniMaxAI/MiniMax-M2.5`** at `/shared/amdgpu/home/anluo/inferencex-agentic-test/hf-cache/hub/models--MiniMaxAI--MiniMax-M2.5` (216 GB, already validated TP=4 on AAC1 MI355X by prior InferenceX agentx-v0.2 work — zero download). The fork's vLLM examples only use Qwen3-0.6B, so M2.5 is still a fresh vLLM-on-Dynamo path but with a derisked weight set. |
| Topologies | Single-node aggregated **and** 2-node disaggregated |
| Hardware | **AAC1 MI355X** (`256C8G1H_MI355X_Ubuntu22` partition, 5 idle 8x-MI355X nodes available, no reservation needed for this account). Tensorwave `amd-aim` (4 nodes, all busy) as a secondary if AAC1 is congested. |

## Hardware availability (verified 2026-05-11)

**Primary cluster: AAC1 (`aac1.amd.com`)**, partition `256C8G1H_MI355X_Ubuntu22`. Snapshot at planning time:
- 5 nodes **idle**: `smci355-ccs-aus-g12-22`, `g12-26`, `g12-30`, `g12-34`, `g12-38` — each 8x MI355X
- 3 nodes allocated (`g12-06`, `g12-14`, `g12-18` — one of which is running our existing job #55)
- 1 node down (`g12-10`)
- No reservation required for our `anluo` account on this partition

This is sufficient headroom for 1-node and 2-node milestones simultaneously, and leaves room to scale to multi-node disagg if the PoC expands. Workflow follows the `slurm-aac` skill: `salloc -p 256C8G1H_MI355X_Ubuntu22 -N1 --gres=gpu:8 -t <hours>:00:00`, then `ssh <nodename>` to land on the worker; podman (not docker) for containers; ROCm via `module load rocm/7.2.2`.

**Secondary cluster: Tensorwave** (`amd-aim` partition, 4 nodes, all currently allocated/mixed; usable only if a slot opens or for low-priority background runs).

**Phase 0 must verify** (a) NIC type on AAC1 MI355X (Pensando ionic vs commodity RoCE/Ethernet — decides whether Phase 3 must absorb the fork's ionic-specific patches: libionic ABI, MoRI CQE fix, IPv4 GID setup, SearchBySubnet, vs. relying on a simpler TCP/standard-RoCE KV transfer path), and (b) that `MiniMaxAI/MiniMax-M2.7` weights (~230 GB FP8) are present or downloadable to a shared FS path mounted on the AAC1 worker nodes.

## Out of scope (explicit)

To keep the PoC small and reviewable, these are deferred:
- Atom backend (separate, MI355X-specific)
- KVBM HIP transfer / `gpu_memory_service` HIP / `kvbm-kernels` HIP
- Planner, autoscaler, fault tolerance, Kubernetes operator AMD bits
- KV-aware router (NVIDIA InferenceX itself uses round-robin in production benchmarks — confirmed by fork's runbook)
- EP/DP-Attn DEP8 (the 16k tok/s configuration; pulls in bootstrap-port race fix, MoRI EP env vars, ionic CQE patch — defer to a follow-up)
- Multimodal, embeddings, long-context (1M)
- Anthropic protocol additions (`lib/llm/src/protocols/anthropic/*`)

## Phases

### Phase 0 — Environment baseline (no Dynamo yet)
1. On AAC1, allocate one MI355X node via `salloc -p 256C8G1H_MI355X_Ubuntu22 -N1 --gres=gpu:8 -t 4:00:00`, then `ssh <nodename>`. Use podman (user is not in `docker` group); `module load rocm/7.2.2`.
2. Inventory: ROCm version, kernel, container runtime (podman vs docker), NIC vendor (`lspci | grep -i ethernet`, `ibv_devinfo`), shared FS for model weights, DeepSeek-R1 FP8 weight location.
3. Confirm raw SGLang serves DeepSeek-R1 FP8 on a single MI355X (`sglang.launch_server`, no Dynamo). Document any backend-side patches required (out of PoC scope but listed as dependencies).
4. Confirm raw vLLM serves `MiniMaxAI/MiniMax-M2.7` on a single MI355X (`vllm serve`, no Dynamo). 229B FP8 ≈ 230 GB weights + KV cache fits one MI355X (288 GB HBM3e). Plan for ~230 GB download to shared FS before allocation.
5. **Decision gate:** ionic vs non-ionic interconnect → choose disaggregation transport for Phase 3/4.

### Phase 1 — Single-node aggregated, SGLang
Pin Dynamo to current `main` SHA. Cherry-pick the **minimum** subset of fork files needed to bring up `python3 -m dynamo.frontend` + `python3 -m dynamo.sglang` end-to-end. Candidate subset (drawn from fork delta — confirm each is required, not nice-to-have):
- `container/Dockerfile.rocm-sglang` (~135 LoC, NEW) and `container/templates/sglang_runtime.Dockerfile` (12 LoC) and `container/constraints-rocm.txt` (17 LoC)
- `components/src/dynamo/sglang/args.py` (+31/-4) — disaggregation arg additions
- `components/src/dynamo/sglang/init_llm.py` (+184/-4) — SGLang Engine bring-up tweaks
- `components/src/dynamo/common/gpu_utils.py` — likely **trim** the fork's 612-LoC version to the minimal `amd-smi` shim needed for GPU detection
- `examples/common/rocm_utils.sh` and `examples/backends/sglang/launch/rocm/agg_rocm.sh` — reproducer scripts
- Any tiny Rust hooks: `lib/llm/src/hip.rs` (NEW), `lib/memory/src/gpu/hip.rs` (NEW). **Verify these are reachable from agg path** — they may only be needed for KVBM/disagg.

Acceptance: `curl /v1/chat/completions` returns DeepSeek-R1 FP8 generation through Dynamo frontend → SGLang worker on one MI355X node.

### Phase 2 — Single-node aggregated, vLLM
Same approach for the vLLM backend, model = `MiniMaxAI/MiniMax-M2.7`. Candidate subset:
- `container/Dockerfile.rocm-vllm` (148 LoC, NEW) and `container/templates/vllm_runtime.Dockerfile` (16 LoC)
- `components/src/dynamo/vllm/args.py` (+6), `handlers.py` (+10/-5), `main.py` (+34), `worker_factory.py` (+10/-5) — total ~50 LoC of vLLM-component AMD bits
- `examples/backends/vllm/launch/rocm/agg_rocm.sh` (adapt fork's script: replace `Qwen/Qwen3-0.6B` default with `MiniMaxAI/MiniMax-M2.7`, raise `MAX_MODEL_LEN`/`MAX_CONCURRENT_SEQS` defaults, drop `--enforce-eager` once HIP graph capture is verified working)

Acceptance: chat-completion `curl` against the Dynamo frontend → vLLM worker returns generated text from MiniMax-M2.7. Note: this is a **fresh path** — the fork only proved vLLM-on-Dynamo with Qwen3-0.6B, so expect to debug FP8 weight loading, HIP graph capture, and any 229B-scale memory issues that did not surface in fork testing. Phase 0 step 4 must succeed before this phase starts.

### Phase 3 — 2-node disaggregated, SGLang
Pick the simplest KV transfer path that works on Tensorwave hardware:
- **If non-ionic** (commodity RoCE/Ethernet): use NIXL with C++ DRAM staging or TCP fallback. Adds modest patch surface (`nixl_dram_staging.py`, `nixl_rocm_staging.py` — though the fork's 1225-LoC version is likely overkill; aim for a smaller staging shim).
- **If ionic**: pull in the libionic ABI fix mechanism + minimum subset of MoRI patches. This significantly enlarges the diff; document it as a "ionic-tax" delta separate from the core PoC.

Probable additions regardless of NIC choice:
- `components/src/dynamo/sglang/rocm_dram_staging_common.py` (352 LoC) — pinned-DRAM staging helper
- `components/src/dynamo/sglang/init_llm.py` — disaggregation init paths

Acceptance: 1P1D DeepSeek-R1 FP8 across two MI355X nodes, end-to-end chat completion succeeds, P50 latency reasonable at c=1.

### Phase 4 — 2-node disaggregated, vLLM
Same with vLLM, model = `MiniMaxAI/MiniMax-M2.7`. Smaller patch surface than SGLang (vLLM's NIXL connector path is shorter). Reference fork's `examples/backends/vllm/launch/rocm/disagg_rocm.sh` for the `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'` pattern; adapt to MiniMax-M2.7 (one full model copy per role on each MI355X node). Acceptance: 1P1D vLLM disagg returns valid completion, KV transfer logs visible.

### Phase 5 — Diff compaction + report
1. Produce `git diff --stat` against `ai-dynamo/dynamo:main` HEAD; verify total LoC is small enough for a few logical PRs (target: < ~3k net LoC, excluding new Dockerfiles and reproducer scripts).
2. Group the diff into upstreamable chunks:
   - **Container/build**: `Dockerfile.rocm-*`, `constraints-rocm.txt`, dockerignore
   - **GPU detection / common**: `gpu_utils.py` minimal shim, `rocm_utils.sh`
   - **SGLang component AMD bits**: `args.py`, `init_llm.py`, staging helpers
   - **vLLM component AMD bits**: 4 files, ~50 LoC
   - **Rust HIP shims** (if reached): `lib/llm/src/hip.rs`, `lib/memory/src/gpu/hip.rs`, `Cargo.toml` feature flags
   - **CI**: `.github/workflows/rocm-build.yml`, `rocm-test.yml`
3. Write `docs/poc-amd-mi355x-report.md` covering: minimal patch list (per-file LoC), reproducer commands, perf numbers vs raw-backend baseline, comparison vs JohnQinAMD's full fork (what was excluded and why), and open questions for NVIDIA review.

## Critical files (for reference during execution)

From the fork (`JohnQinAMD/dynamo:amd-dynamo`):
- `docs/amd-feature-test-runbook.md` — proven recipes for every test in the fork
- `docs/amd-rocm-build.md` — build-from-source docs
- `docs/ionic-rdma-fixes.md` — comprehensive ionic NIC issue catalog (only relevant if Phase 0 finds ionic NICs)
- `docs/perf-optimization-plan.md` — bootstrap port race + GIL + concurrency analysis (relevant only for EP/DP-Attn follow-up, out of PoC)
- `scripts/setup_ionic_network.sh`, `scripts/preflight_check.sh` — ionic-only utilities
- `scripts/run_benchmark.sh`, `scripts/benchmark_lib.sh` — benchmarking harness (use for Phase 5 perf measurements)

From `ai-dynamo/dynamo:main`:
- `components/src/dynamo/sglang/` — SGLang backend (target of Phase 1/3 patches)
- `components/src/dynamo/vllm/` — vLLM backend (target of Phase 2/4 patches)
- `components/src/dynamo/common/` — shared utilities (target of `gpu_utils` shim)
- `container/Dockerfile.template`, `container/render.py`, `container/templates/` — container build system to extend
- `lib/llm/`, `lib/memory/` — Rust core (touched only if HIP shims are reachable from agg/disagg paths)

## Decision points to surface to NVIDIA team

Captured from the fork's structure; flag these in the Phase 5 report:
1. **HIP integration in Rust core**: parallel module to CUDA (`lib/llm/src/hip.rs`, fork approach) vs feature-flag merge with `cuda.rs`?
2. **Container strategy**: separate `Dockerfile.rocm-*` files (fork) vs unified `Dockerfile.template` with `--build-arg HARDWARE=rocm`?
3. **Non-NIXL KV transports**: where should MoRI / Mooncake plug into the KV-transfer abstraction? The fork patches them in via SGLang directly; a clean abstraction would be a separate, larger upstream design discussion.
4. **GPU detection shim**: do we want the fork's 612-LoC `gpu_utils.py` upstream, or a focused `amd-smi` adapter?
5. **CI**: what does NVIDIA want as the AMD CI surface — full ROCm build + smoke test on every PR, or a separate workflow gated on a label?

## Verification (end-to-end test plan)

Per phase, the acceptance check:

| Phase | Command | Expected |
|---|---|---|
| 0 | `python3 -m sglang.launch_server --model-path <DSR1>` then chat-completion `curl` | 200 OK, generated text |
| 0 | `vllm serve MiniMaxAI/MiniMax-M2.7` then chat-completion `curl` | 200 OK, generated text |
| 1 | `python3 -m dynamo.frontend &` + `python3 -m dynamo.sglang --model-path <DSR1>` + `curl` | 200 OK |
| 2 | `python3 -m dynamo.frontend &` + `python3 -m dynamo.vllm --model MiniMaxAI/MiniMax-M2.7` + `curl` | 200 OK |
| 3 | Prefill node + decode node with `--disaggregation-mode {prefill,decode}`, `--disaggregation-transfer-backend <chosen>`, then `curl` from frontend | 200 OK; KV transfer logs show traffic |
| 4 | Same as 3 but with `dynamo.vllm` | 200 OK |
| 5 | `git diff --stat ai-dynamo/dynamo/main...HEAD` and compare against fork | Net LoC well under fork's 231 files |

For each "200 OK" milestone, also run a small concurrency sweep (c=1, 4, 8) using a copy of the benchmark snippet from the runbook (`docs/amd-feature-test-runbook.md` § Benchmarking) and record P50/throughput. Compare PoC numbers vs raw-backend baseline from Phase 0 — Dynamo overhead should be small.

## Deliverables

1. PoC repo (fork of `ai-dynamo/dynamo:main`) with the minimal cherry-picked diff, branch named `amd-mi355x-poc`.
2. `docs/poc-amd-mi355x-report.md` summarizing: minimal patch list, reproducer commands, perf vs baseline, comparison vs JohnQinAMD fork, decision points for NVIDIA.
3. Reproducer scripts under `examples/backends/{sglang,vllm}/launch/rocm/` (single-node + 2-node).
4. Suggested upstream PR breakdown (5 logical PRs as listed in Phase 5 step 2).
