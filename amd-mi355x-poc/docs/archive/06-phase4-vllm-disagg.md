# Phase 4 result — vLLM 2-node disaggregated PASS

Date: 2026-05-11 (revised after additional debugging found the actual fixes)
Cluster: AAC1, prefill=`smci355-ccs-aus-g12-22`, decode=`smci355-ccs-aus-g12-26`, both 8x MI355X (gfx950) with 9× AMD Pensando ionic NICs
Model: `Qwen/Qwen3-0.6B` (TP=1; same model used by JohnQinAMD fork's vLLM disagg test)

## Outcome: **PASS** (after applying 4 layered fixes)

End-to-end Dynamo-orchestrated 1-prefill-1-decode (1P1D) serving across 2 MI355X nodes with **vLLM `NixlConnector` + RIXL + UCX-ROCm 1.19**, KV transfer over TCP/rocm_copy fallback path (ionic doesn't support GPUDirect RDMA so dmabuf-based VRAM registration fails as expected; RIXL falls back successfully).

## Numbers (1P1D, Qwen3-0.6B, c=1/4/8)

| conc | N  | P50 (ms) | P95 (ms) | tok/s | output avg | success |
|------|----|----------|----------|-------|------------|---------|
| 1    | 8  | 381      | 384      | 168.0 | 64         | 8/8     |
| 4    | 12 | 422      | 602      | 526.7 | 64         | 12/12   |
| 8    | 24 | 777      | 812      | 645.6 | 64         | 24/24   |

Warm warmup runs: 266 / 198 / 197 ms.

**~5× faster than Phase 3** (SGLang+Mooncake disagg, same model+hardware: 122 tok/s @ c=8). RIXL/UCX with TCP+rocm_copy avoids the per-request chunked-MR DRAM-staging overhead Mooncake incurs.

## Stack assembled

| Layer | Source | Patches |
|---|---|---|
| Custom Dockerfile | `Dockerfile.rocm-vllm-rixl` (~30 LoC) extending `rocm/vllm-dev:nightly` with UCX-ROCm 1.19.x + RIXL + Python bindings | new |
| UCX-ROCm | `ROCm/ucx:v1.19.x`, built from source (modules: rc_v, tcp, **rocm, rocm_copy, rocm_ipc**, etc.) | none |
| RIXL | `ROCm/RIXL:master` (HEAD as of 2026-05-11), built via meson | none |
| `nixl→rixl` Python shim | 4-file `<site-packages>/nixl/{__init__,_api,_bindings}.py` | new |
| `IBV_ACCESS_REMOTE_ATOMIC` strip | LD_PRELOAD interposer (~60 LoC C) wrapping `ibv_reg_mr`, `ibv_reg_mr_iova2`, `ibv_reg_dmabuf_mr` (extended from fork's nixl_rocm_staging.py with the dmabuf wrapper added by us — fork's version doesn't wrap dmabuf) | new |
| ionic visibility in container | bind-mount: `/etc/libibverbs.d/ionic.driver`, `libionic.so*` chain, **and** `/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so` (the actual ibverbs provider plugin — easy to miss because the container's apt `ibverbs-providers` package doesn't include it) | n/a |
| ai-dynamo Python | `pip install --no-deps ai-dynamo==1.1.1 ai-dynamo-runtime==1.1.1 blake3 kubernetes msgpack msgspec prometheus-client pyzmq` | none |
| Container devices | `--device /dev/kfd /dev/dri /dev/infiniband --group-add keep-groups --security-opt seccomp=unconfined --network=host --ipc=host` | n/a |
| Required env | `HIP_VISIBLE_DEVICES=4` (avoid dirty GPU 0), `HF_HUB_OFFLINE=1`, `LD_PRELOAD=/tmp/ibv_ionic_compat.so`, `UCX_MEMTYPE_CACHE=y`, **`UCX_TLS=rc_v,tcp,rocm,rocm_copy,rocm_ipc,self,sm`** (the ROCm transports are what makes UCX recognize VRAM correctly) | n/a |

## The four fixes that turned PARTIAL into PASS

These were not in the fork's runbook for vLLM disagg; we discovered them by debugging:

1. **Mount `libionic-rdmav34.so` (the ibverbs provider plugin)**, not just `libionic.so*`. The container's `ibverbs-providers` apt package (Mellanox-flavored in `rocm/vllm-dev:nightly`) does **not** ship the ionic provider; it must be bind-mounted from the host's `/usr/lib/x86_64-linux-gnu/libibverbs/`.
2. **Extended LD_PRELOAD interposer to wrap `ibv_reg_dmabuf_mr`** (UCX uses dmabuf for GPU VRAM registration; the fork's nixl_rocm_staging.py wraps only `ibv_reg_mr`/`ibv_reg_mr_iova2`).
3. **Include ROCm transports in `UCX_TLS`** (`rocm,rocm_copy,rocm_ipc`). Without them, UCX's `ucp_mem_query` returns `UCS_MEMORY_TYPE_HOST` for GPU pointers → RIXL refuses with `"VRAM memory is detected as host by UCX. UCX is likely not configured with CUDA support."` This was the actual root cause of the `nixlBackendError`, **NOT** the missing AMD-internal vendor UCX 1.12.
4. **Use a fresh GPU** (`HIP_VISIBLE_DEVICES=4`) — GPU 0 had stale 57 GB allocations from prior crashed runs that caused HIP OOM on retry. amd-smi showed only GPU[3] dirty but new processes still hit OOM on GPU 0.

The interposer's `ibv_reg_dmabuf_mr` calls all return `failed (fd=-1)` (ionic does not support dmabuf RDMA at all), but RIXL handles this gracefully by falling back through ROCm copy paths. Cross-node KV transfer flows over TCP (visible in UCX log).

## Updated PoC story

**Earlier I incorrectly concluded** Phase 4 was blocked on the AMD-internal vendor-patched UCX 1.12. The actual story is more nuanced:

- The vendor UCX 1.12 patch in the fork's `docs/ionic-rdma-fixes.md` Layer 5 **vendor-aware-strips REMOTE_ATOMIC for ionic only** (so RCCL stays unaffected). Our LD_PRELOAD interposer is functionally equivalent for the dynamo-side processes (RCCL is in a different process and unaffected).
- The vendor UCX is NOT required for vLLM disagg to PASS on Qwen3-0.6B. It would matter only when `ibv_reg_mr`/`ibv_reg_dmabuf_mr` is actually invoked from a process that ALSO uses RCCL with REMOTE_ATOMIC needs.
- The PoC bottleneck was actually four mundane container-config issues (missing provider .so, missing dmabuf wrap, wrong UCX_TLS, dirty GPU), not a fundamental UCX kernel.

## Bumps in the road (full timeline)

| Issue | Resolution |
|---|---|
| RIXL Python build (`Cannot import 'mesonpy'`) | Add `meson-python` to image's pip install |
| `nixl` package missing post-RIXL build | RIXL installs as `rixl`; add 4-file `nixl→rixl` shim |
| `apt-get install` failed (`Unable to locate package iproute2`) | Run `apt-get update` first |
| `ibv_devinfo: command not found` after install | Was OK; the test script bashed it from a stale path |
| `ibv_devinfo count: 0` despite host having 9 ionic | Container missing `libionic-rdmav34.so` (provider plugin, distinct from `libionic.so`) → bind-mount it |
| `HIP out of memory` despite 287 GB GPU | Stale allocations from prior crashed runs on GPU 0 → use HIP_VISIBLE_DEVICES=4 |
| `Lock acquisition failed` on safetensors NFS | Both workers tried to fetch simultaneously → set `HF_HUB_OFFLINE=1` (model already cached) |
| First `nixlBackendError: NIXL_ERR_BACKEND` | Misdiagnosed as needing vendor UCX 1.12 |
| **Real cause**: `VRAM memory is detected as host by UCX. UCX is likely not configured with CUDA support.` (RIXL `ucx_utils.cpp:571`) | Add ROCm transports to `UCX_TLS` so UCX loads `libucm_rocm.so` for memory-type detection |

## Reproducer

- Image: `localhost/dynamo-vllm-rixl:latest` (39.9 GB) on g12-22 + g12-26 (tagged identically; built once on g12-22, exported via `podman save`/`load`)
- Dockerfile: `/tmp/Dockerfile.rocm-vllm-rixl` (and copy on AAC1 at `/shared/amdgpu/home/anluo/dynamo-poc/`)
- RIXL source: cloned from `https://github.com/ROCm/RIXL.git` to `/tmp/rixl-build/RIXL` on g12-22
- Orchestrator: `/shared/amdgpu/home/anluo/dynamo-poc/phase4_disagg.sh` (incorporates all 4 fixes)
- Bench client: `/tmp/p4_bench.py` (mirrored to g12-22)
- Containers `dynamo-vllm-prefill` (g12-22) + `dynamo-vllm-decode` (g12-26) left running for further testing (sleep 7200)

## Implications for upstream

For vLLM disaggregated, the upstream change set against `ai-dynamo/dynamo:main` is now **dramatically smaller than Phase 3 (SGLang disagg)**:

1. Everything from Phase 1/2 (nixl import lazy, typing.Self compat, etc.) ~10 LoC source
2. **`Dockerfile.rocm-vllm`** — the heavy image (UCX-ROCm + RIXL build), ~50 LoC
3. **LD_PRELOAD interposer C source** + activation in launch script (~70 LoC)
4. **Bind-mount documentation** for `/etc/libibverbs.d/ionic.driver`, `libionic*` chain, and `libionic-rdmav34.so` (the gotcha)
5. **Required env vars** documentation: `UCX_TLS=...,rocm,rocm_copy,rocm_ipc,...`, `UCX_MEMTYPE_CACHE=y`, `LD_PRELOAD`
6. **(Already in fork) Bootstrap host/port patches** to `dynamo.vllm.{args,main}.py` (~40 LoC) — only matters when running outside `localhost`

**Net new code for vLLM disagg upstream: ~150 LoC** (mostly Dockerfile + interposer + launch script). No vLLM Python source patches needed in the dynamo codebase.

This is **substantially smaller** than the SGLang disagg path (~1000 LoC of mooncake_rocm_staging.py + rocm_dram_staging_common.py). The asymmetry exists because RIXL handles the heavy lifting in C++ (via UCX's ROCm transport selection), whereas Mooncake on AMD requires Python-level chunked-MR + DRAM-staging that lives in the dynamo codebase.

## Side-by-side (final, all phases)

| Phase | What | PASS? | c=8 tok/s | Patches against `dynamo:main` |
|---|---|---|---|---|
| 1 | SGLang agg (DSR1, TP=8) | ✅ | 708 | nixl stub + 1-LoC typing.Self + 3-LoC sgl.Engine compat |
| 2.5 | vLLM agg (M2.5, TP=4, HIP graphs) | ✅ | 521 | nixl stub only |
| 3 | SGLang disagg (Qwen3-0.6B, Mooncake) | ✅ | 122 | + 992 LoC fork code (mooncake_rocm_staging.py + rocm_dram_staging_common.py) + libionic ABI fix |
| 4 | **vLLM disagg (Qwen3-0.6B, RIXL+UCX)** | **✅** | **646** | + Dockerfile.rocm-vllm + LD_PRELOAD interposer (~120 LoC) + container mount/env docs |

All four targeted milestones PASS with sharply quantified upstream-PR sizes.
