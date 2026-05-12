# Phase 3 result — 2-node SGLang disaggregated PASS

Date: 2026-05-11
Cluster: AAC1, prefill=`smci355-ccs-aus-g12-22`, decode=`smci355-ccs-aus-g12-26`, both 8x MI355X (gfx950) with 9× AMD Pensando ionic NICs each
Model: `Qwen/Qwen3-0.6B` (TP=1; small model chosen to derisk Mooncake's ionic MR limits before attempting DSR1)

## Outcome: **PASS**

End-to-end Dynamo-orchestrated 1-prefill-1-decode (1P1D) serving across 2 MI355X nodes with **Mooncake RDMA over Pensando ionic NICs**, with KV cache transferred prefill→decode for every request.

## Stack assembled

| Layer | Source | Patches |
|---|---|---|
| Container | `docker.io/rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503` | none |
| Discovery / Event | etcd 3.5.21 + nats 2.10.28 (off-the-shelf, on prefill node) | none |
| Dynamo Python | `pip install --no-deps ai-dynamo==1.1.1 ai-dynamo-runtime==1.1.1 ...` (same as Phase 1) | none |
| `nixl` package | 4-file Python stub (~35 LoC) — same as Phase 1/2 | new |
| `typing.Self` shim | `.pth` (1 LoC) — same as Phase 1 | new |
| `sgl.Engine` compat | sed-patch on `dynamo/sglang/publisher.py` (3 LoC) — same as Phase 1 | 3 LoC |
| **Mooncake DRAM staging + chunked MR + subnet-aware NIC pick** | Copied as-is from JohnQinAMD fork: `mooncake_rocm_staging.py` (640 LoC) + `rocm_dram_staging_common.py` (352 LoC) | **+992 LoC** of fork code, unmodified |
| **libionic ABI fix** | `cp host:/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184 → container:/usr/lib/x86_64-linux-gnu/libionic.so.1` (per fork runbook) | 1 line in launcher |
| Auto-activation | 2 .pth files: `os.environ.setdefault("SGLANG_MOONCAKE_ROCM_STAGING","1")` and `import dynamo.sglang.mooncake_rocm_staging` | 2 LoC |
| Container devices | `--device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add keep-groups --security-opt seccomp=unconfined --network=host --ipc=host` | n/a |
| Disagg flags | prefill: `--disaggregation-mode prefill --disaggregation-transfer-backend mooncake --host 0.0.0.0`; decode: same with `decode` | n/a |
| Required env | `SGLANG_MOONCAKE_ROCM_STAGING=1`, `MC_MAX_SGE=2` | n/a |

Inside-container verification at startup:
```
ibv_devinfo count: 9          ← ionic NICs visible (libionic ABI fix worked)
mooncake_rocm_staging imported OK
```

## Numbers (1P1D, Qwen3-0.6B, c=1/4/8)

| conc | N  | P50 (ms) | P95 (ms) | tok/s | output avg | success |
|------|----|----------|----------|-------|------------|---------|
| 1    | 8  | 1216     | 21348    | 19.1  | 64         | 8/8     |
| 4    | 12 | 2396     | 4652     | 83.3  | 64         | 12/12   |
| 8    | 24 | 4527     | 4620     | 122.2 | 64         | 24/24   |

Warm requests after first cold one: 251 / 197 / 1167 ms (the third was an MR-recycle stall).

Comparison: fork's runbook reports 81.7 req/s on Qwen3-0.6B with MoRI at c=8 (5× faster than Mooncake here). MoRI requires extra build steps the fork's `Dockerfile.rocm-sglang` does at image-build time. **For PoC purposes, Mooncake proves disagg works; MoRI would be a follow-up perf optimization.**

## Patch surface vs Phase 1/2 (cumulative)

| Phase | Net new patches/files |
|---|---|
| 2 (vLLM agg) | nixl stub (~35 LoC) |
| 1 (SGLang agg) | + typing.Self .pth (1 LoC), + sgl.Engine sed (3 LoC) |
| 3 (SGLang disagg) | + Mooncake staging (992 LoC fork code, unmodified) + libionic ABI fix (1 cmd) + 2 activation .pth (2 LoC) + 6 launcher env/device flags |

**Most of the disagg "weight" is in `mooncake_rocm_staging.py` (640 LoC) and `rocm_dram_staging_common.py` (352 LoC)** — these two files contain the AMD-specific Mooncake adaptation (DRAM staging since ionic can't `ibv_reg_mr` GPU VRAM, chunked MR ≤190 MB to fit ionic's per-device limit, mmap+mlock instead of pinned host memory, subnet-aware device selection). They're entirely net-new code from the JohnQinAMD fork. Upstream candidate: a `dynamo.sglang.transports.mooncake_rocm` submodule that ships with Dynamo when ROCm support is enabled.

## Bumps in the road

| Issue | Fix |
|---|---|
| Variables `$MODEL`, `$TP` not visible inside container | passed via `-e MODEL=$MODEL -e TP=$TP` |
| Decode reported "a different model 'deepseek-ai/DeepSeek-R1-0528' is already registered" | leftover registration in etcd from Phase 1 container; cleared with `etcdctl del --prefix dynamo/` |
| Container saw "No IB devices found" | needed `--device /dev/infiniband` + libionic ABI fix |

## Reproducer

- Orchestrator: `/shared/amdgpu/home/anluo/dynamo-poc/phase3_disagg.sh` — runs from login node, ssh's to prefill+decode nodes
- Inside-container script: `/tmp/inside_p3.sh` (generated) on each worker
- Bench client: `/tmp/phase3_bench.py` on prefill node (curls localhost:8000)
- Containers `dynamo-prefill` (g12-22) + `dynamo-decode` (g12-26) + `dynamo-frontend`/`etcd`/`nats` (g12-22) left running for further testing.

## Implications for upstream

For SGLang **disaggregated** (vs agg), the upstream change set against `ai-dynamo/dynamo:main` becomes substantially larger:

1. Everything Phase 1/2 added (nixl-import-lazy, typing.Self use, publisher.py engine import, ~10 LoC + 1 launch script)
2. **Add `dynamo.sglang.mooncake_rocm_staging` (~640 LoC) and `dynamo.sglang.rocm_dram_staging_common` (~352 LoC)** — the AMD-specific Mooncake adapter
3. Document libionic ABI fix in container build/launch docs
4. Document required env vars (`SGLANG_MOONCAKE_ROCM_STAGING=1`, `MC_MAX_SGE=2`)
5. Optional: ship `Dockerfile.rocm-sglang` that bakes in the libionic fix at image-build time

**Net upstream surface for SGLang disagg: ~1000 LoC of new ROCm-staging code + minor docs.** This is where the JohnQinAMD fork's bulk lives. The corresponding upstream PR can be either:
- **Two PRs**: agg-only (10 LoC + script, lands easily) followed by disagg (1000 LoC + design discussion about adapter pattern)
- **One PR** with a feature flag (`--disaggregation-transfer-backend mooncake_rocm`)

## DSR1 escalation (not attempted)

The fork reports DSR1 disagg works with Mooncake using the chunked-MR patch already in `mooncake_rocm_staging.py`. We verified the patch loads cleanly. Running 1P1D on DSR1 would consume both nodes' 8 GPUs each (TP=8) and add ~10 minutes for model loading on each node. Skipped for time; logically the path should work given Phase 1 (DSR1 agg) and Phase 3 (small-model disagg) both pass.

## Side-by-side (summary across all phases)

| Phase | What | Stack | PASS? | Notes |
|---|---|---|---|---|
| 0 | Env baseline | AAC1 MI355X, ionic NICs, ROCm 7.2.2 | ✅ | |
| 1 | SGLang agg (DSR1) | rocm/sgl-dev:v0.5.10.post1 + 4-LoC patches | ✅ | 708 tok/s @ c=8 |
| 2 | vLLM agg (M2.5, eager) | rocm/vllm-dev:nightly + nixl stub | ✅ | 99 tok/s @ c=1 (eager) |
| 2.5 | vLLM agg (M2.5, HIP graphs) | same as 2, drop --enforce-eager | ✅ | 521 tok/s @ c=8 |
| 3 | SGLang disagg (Qwen3-0.6B, Mooncake) | + 992 LoC from fork (mooncake staging) + libionic fix | ✅ | 122 tok/s @ c=8, all 24/24 |
| 4 | vLLM disagg (M2.5) | TBD (not yet attempted) | ⏳ | Smaller patch surface than Phase 3 expected |
| 5 | Diff + report | TBD | ⏳ | |
