# Phase 2 result — vLLM single-node aggregated PASS

Date: 2026-05-11
Cluster: AAC1, node `smci355-ccs-aus-g12-22`, 8x MI355X (gfx950)
Model: `MiniMaxAI/MiniMax-M2.5` (216 GB FP8, MoE, 126 safetensors files)

## Outcome: **PASS**

End-to-end Dynamo + vLLM + MiniMax-M2.5 chat completion via HTTP, **with zero patches to Dynamo source code**.

## Stack assembled

| Layer | Source | Patches |
|---|---|---|
| Container base | `docker.io/rocm/vllm-dev:nightly` (38.8 GB, 19h old, vllm 0.20.2rc1.dev203+g21943d4c2.rocm722) | none |
| Discovery | `quay.io/coreos/etcd:v3.5.21` (off-the-shelf) | none |
| Event plane | `docker.io/library/nats:2.10.28 -p 4222 -js` (off-the-shelf) | none |
| Dynamo Python | `pip install --no-deps ai-dynamo==1.1.1 ai-dynamo-runtime==1.1.1 blake3 kubernetes msgpack msgspec prometheus-client pyzmq` (PyPI) | none |
| `nixl` package | 4-file Python stub at `<site-packages>/nixl/` (35 LoC total: `__init__.py`, `_api.py` with raise-on-use stubs for `nixl_agent`/`nixl_xfer_handle`/etc., `_bindings.py`) | new |
| Workload config | TP=4 (matches prior InferenceX agentx-v0.2 validation; TP=8 fails the FP8 block-divisibility check at `vllm/model_executor/layers/quantization/fp8.py:630` with `output_size=192 not divisible by block_n=128`) | n/a |
| Engine flags | `--max-model-len 8192 --max-num-seqs 8 --enforce-eager --trust-remote-code` | n/a |

The nixl stub exists because `dynamo.vllm.__init__` chain eagerly imports `dynamo.nixl_connect.__init__.py` line 42: `import nixl._api as nixl_api`. The PyPI `nixl` package is CUDA-only (`nixl-cu12`), so on AMD we either need the stub or a small upstream patch making the import lazy.

## Numbers (eager mode, c=1)

| Metric | Value |
|---|---|
| Model weight load | 4:10 (125 shards from NFS, throughput ~870 MB/s) |
| Engine init (profile + KV alloc + warmup) | 100 s |
| KV cache available | 208.68 GiB → 3,527,168 tokens |
| Max concurrency at 8K context | 430.56× |
| First chat completion (47 in / 48 out tokens) | 5095 ms total |
| TTFT | 1982 ms |
| Decode avg ITL | 66.24 ms/token (~15 tok/s) |

Eager mode disables HIP graph capture, so decode is ~3-5× slower than expected production. Production rerun (Phase 2.5) followed.

## Phase 2.5 — production numbers (HIP graphs ON)

Same stack, removed `--enforce-eager`, raised `--max-num-seqs 64` for graph amortization. **HIP graph capture worked fine on MI355X/gfx950** — the fork's runbook warning ("aiter JIT segfaults on gfx950") did NOT materialize for this config (MiniMax-M2.5 + TP=4 + max-num-seqs 64).

| conc | N  | P50 (ms) | P95 (ms) | tok/s total | input avg | output avg | success |
|------|----|----------|----------|-------------|-----------|------------|---------|
| 1    | 8  | 1279     | 1295     | 99.9        | 58        | 128        | 8/8     |
| 4    | 12 | 1427     | 1647     | 341.9       | 58        | 128        | 12/12   |
| 8    | 24 | 1427     | 3043     | 521.2       | 58        | 128        | 24/24   |

Decode latency: **~10 ms/token vs 66 ms/token in eager mode (6.6× speedup)**. At c=8, sustained throughput is ~130 tok/s/GPU across 4 MI355X.

Aiter JIT compile (one-time, first request) takes ~3 minutes per worker. Once compiled (cached in container), reuse is instant. NFS page cache also makes the 2nd model load ~4× faster (60 s vs 4:10).

## Implications for the upstream story (final)

For vLLM single-node aggregated, the cleanest upstream change set against `ai-dynamo/dynamo:main` is:

## Sample reply

```
The user wants a short reply. They say "Hello, please reply with one short sentence."
So I should respond with a short sentence: perhaps "Hello! How can I help you today?"
That's one sentence. That meets "one short
```

(Cut at 48 tokens by max_tokens; finish_reason=length.)

## Implications for the upstream story

For vLLM single-node aggregated, the cleanest upstream change set against `ai-dynamo/dynamo:main` is:

1. **Make `nixl._api` import lazy** in `lib/bindings/python/src/dynamo/nixl_connect/__init__.py` (~5 LoC try/except, falls back to a no-op stub when nixl is absent). Eliminates the need for the stub package at runtime.
2. **One launch script**: `examples/backends/vllm/launch/rocm/agg_rocm.sh` that documents the `rocm/vllm-dev:nightly` + pip install path (no Dockerfile required).
3. **One docs page**: AMD-on-vLLM quickstart pointing at the above.

This is dramatically smaller than the JohnQinAMD fork's 148-line `Dockerfile.rocm-vllm` + UCX/RIXL build chain — that machinery only matters for **disaggregated** serving (Phase 4).

## Reproducer (for the report)

Files left on AAC1 for re-runnability:
- `/shared/amdgpu/home/anluo/dynamo-poc/phase2_e2e.sh` — full end-to-end launcher (etcd + NATS + dynamo-vllm container)
- `/shared/amdgpu/home/anluo/dynamo-poc/phase2-launch.log` — successful run log
- Container `dynamo-vllm-poc` is left running (sleep 7200) for further `podman exec` testing

Tear-down: `podman rm -f dynamo-vllm-poc dynamo-nats dynamo-etcd` on the worker node.
