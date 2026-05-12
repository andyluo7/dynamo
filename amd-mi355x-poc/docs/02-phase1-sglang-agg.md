# Phase 1 result — SGLang single-node aggregated PASS

Date: 2026-05-11
Cluster: AAC1, node `smci355-ccs-aus-g12-22`, 8x MI355X (gfx950)
Model: `deepseek-ai/DeepSeek-R1-0528` (671B, FP8, 163 safetensors, ~644 GB)

## Outcome: **PASS**

End-to-end Dynamo + SGLang + DeepSeek-R1 chat completion via HTTP.

Sample reply (DSR1 is a reasoning model, replies start with `<think>` tag):
> `<think>` Okay, the user just greeted me and asked for a short reply. Hmm, they specifically emphasized "short," so they probably want something quick and concise—no fluff. Let me think about why they'd request that... (cut at 48 tokens)

## Stack assembled

| Layer | Source | Patches |
|---|---|---|
| Container base | `docker.io/rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503` (75.1 GB; sglang `0.5.10.post1.dev20260503+g44ca2d01f`; Python 3.10.12) | none |
| Discovery / Event plane | etcd 3.5.21 + nats 2.10.28 (off-the-shelf, reused from Phase 2 containers) | none |
| Dynamo Python | `pip install --no-deps ai-dynamo==1.1.1 ai-dynamo-runtime==1.1.1 blake3 kubernetes msgpack msgspec prometheus-client pyzmq uvloop typing_extensions` | none |
| `nixl` package | 4-file Python stub (~35 LoC) — same as Phase 2 | new |
| **`typing.Self` shim** | `.pth` file, 1 line: `import typing, typing_extensions; (not hasattr(typing, "Self")) and setattr(typing, "Self", typing_extensions.Self)`. Required because the sgl-dev image ships Python 3.10 but ai-dynamo 1.1.1 uses `from typing import Self` (3.11+). | new |
| **`sgl.Engine` compat shim** | sed-patch on `dynamo/sglang/publisher.py`: insert `from sglang.srt.entrypoints.engine import Engine as _SglEngine; sgl.Engine = _SglEngine` after the `import sglang as sgl` line. Required because the rocm/sgl-dev container's `sglang` package no longer re-exports `Engine` at top level. | 3 LoC |
| Workload | `--tp-size 8 --trust-remote-code --mem-fraction-static 0.85` | n/a |

## Numbers (HIP graphs ON, c=1/4/8)

| conc | N  | P50 (ms) | P95 (ms) | tok/s total | input avg | output avg | success |
|------|----|----------|----------|-------------|-----------|------------|---------|
| 1    | 8  | 1259     | 1570     | 98.3        | 23        | 128        | 8/8     |
| 4    | 12 | 1327     | 1346     | 384.0       | 23        | 128        | 12/12   |
| 8    | 24 | 1409     | 1515     | 708.8       | 23        | 128        | 24/24   |

- Cold first request: 15235 ms (aiter JIT compile + first kernel launches)
- Warm warmup runs (3): 382 / 451 / 370 ms
- ~89 tok/s/GPU at c=8 (8x MI355X) for a 671B FP8 reasoning model
- HIP graph capture per worker: 90 s (8 workers in parallel; total +90s on top of model load)

## Stack timeline

| Stage | Wall time |
|---|---|
| Model load (163 safetensors from NFS w/ page-cache miss) | 60 s |
| `Detected fp8 checkpoint` + Shared experts fusion + tokenizer | 5 s |
| Worker spawn + aiter JIT | ~140 s |
| HIP graph capture (8 workers in parallel) | 90 s |
| Engine init total | ~9 minutes |
| Warm request | ~400 ms |

## Bumps in the road (and what they teach us about upstream patches)

| Issue | Fix applied | Upstream patch suggestion |
|---|---|---|
| `module 'sglang' has no attribute '__version__'` | dropped the print | n/a (was just my probe code) |
| `from typing import Self` fails on Python 3.10 | `.pth` file injects Self via typing_extensions | ai-dynamo: declare `python_requires>=3.11` in metadata, OR use `typing_extensions.Self` directly so 3.10 works |
| `sgl.Engine` AttributeError | sed-patch adds re-export | dynamo/sglang/publisher.py: import from `sglang.srt.entrypoints.engine` directly (or guard with version check) |
| `--model-path /local/snapshot/...` interpreted as HF model ID → 404 | use HF model ID; HF cache resolves locally | n/a (user error from initial Phase 1 script — upstream behavior is correct) |
| `.pth` shim for `import sglang` caused fork bomb (rocm_agent_enumerator is itself Python; recursive import) | dropped the .pth shim, used sed-patch instead | document gotcha; do NOT pre-import heavy modules in .pth files |

## Implications for upstream

For SGLang single-node aggregated, the cleanest upstream change set against `ai-dynamo/dynamo:main` is:
1. **Make `nixl._api` import lazy** (same patch as vLLM, ~5 LoC in `dynamo.nixl_connect`)
2. **Make `typing_extensions.Self` the import** (~1 LoC) OR bump `python_requires` to 3.11+
3. **Update `dynamo.sglang.publisher` to use `sglang.srt.entrypoints.engine.Engine`** (~1 LoC)
4. **One launch script** (`examples/backends/sglang/launch/rocm/agg_rocm.sh`)

Net upstream surface for SGLang agg: ~10 LoC of source changes + ~50 LoC launch script. Tiny.

## Comparison to JohnQinAMD fork

The fork has 184 changed LoC in `components/src/dynamo/sglang/init_llm.py` and 31 LoC in `args.py`, plus a 135-LoC `Dockerfile.rocm-sglang` and 12-LoC `sglang_runtime.Dockerfile` template. **None of those changes were strictly required for the agg-only PoC** — they're all about disaggregated serving (KV transport, bootstrap port handling, RIXL/MoRI/Mooncake integration). Phase 3 (SGLang disagg) will exercise that machinery.

## Reproducer

- Script: `/shared/amdgpu/home/anluo/dynamo-poc/phase1_e2e.sh`
- Log: `/shared/amdgpu/home/anluo/dynamo-poc/phase1-launch.log`
- Container `dynamo-sglang-poc` left running (sleep 7200) on g12-22

## Side-by-side: Phase 1 vs Phase 2 (both single-node agg, on identical hardware)

| Metric | Phase 1 (SGLang+DSR1, TP=8) | Phase 2.5 (vLLM+M2.5, TP=4) |
|---|---|---|
| Model size | 671B FP8 | 229B FP8 (M2 hybrid attn) |
| GPUs used | 8 | 4 |
| First (cold) request | 15235 ms | 5095 ms (eager) |
| P50 c=1 | 1259 ms | 1279 ms |
| P50 c=8 | 1409 ms | 1427 ms |
| Tok/s c=8 | 708.8 | 521.2 |
| Tok/s/GPU c=8 | 88.6 | 130.3 |
| Patches needed | nixl stub + 1-line typing.Self shim + 3-line sgl.Engine compat | nixl stub only |

Both phases prove the same headline: **Dynamo runs on AMD MI355X with serving-grade backends and minimal patches**.
