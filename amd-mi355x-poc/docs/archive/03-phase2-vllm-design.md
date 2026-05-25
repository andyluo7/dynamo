# Phase 2 design notes — vLLM single-node aggregated on AMD MI355X

Date: 2026-05-11

## Headline insight

Reading the JohnQinAMD fork's actual vLLM Python diffs (`gh api compare/main...amd-dynamo --jq '.files'`) reveals the fork's vLLM-component changes are **all either disagg-only or generic resilience**, **not** ROCm-required for the aggregated path.

| File | LoC change | Purpose | Phase 2 (agg)? | Phase 4 (disagg)? |
|---|---|---|---|---|
| `args.py` | +6 | `ensure_side_channel_host()` for non-NIXL prefill workers | ❌ no | ✅ yes |
| `main.py` | +34 | Bootstrap host/port publication for PrefillRouter | ❌ no | ✅ yes |
| `handlers.py` | +10/-5 | `try/except ImportError` around `dynamo.common.multimodal.embedding_transfer` | optional (only matters if multimodal pkg missing) | optional |
| `worker_factory.py` | +10/-5 | `try/except ImportError` around `multimodal_handlers` | optional | optional |

→ For **vLLM single-node aggregated**, the fork required **zero ROCm-specific Python patches**.

## Implication for the PoC

The Phase 2 minimal-diff hypothesis to test: **stock `pip install ai-dynamo[vllm] ai-dynamo-runtime` from PyPI, inside a `rocm/vllm-dev:nightly` container, will work end-to-end** to serve `MiniMaxAI/MiniMax-M2.5` via Dynamo, with no Python changes to Dynamo at all.

If true, the entire Phase 2 patch surface against `ai-dynamo/dynamo:main` is:
- 1 launch script (`examples/backends/vllm/launch/rocm/agg_rocm.sh`, ~50 LoC, NEW)
- 0 component code changes
- 0 Dockerfile changes (use stock `rocm/vllm-dev:nightly` + pip install)

This would be an extremely clean upstream story for vLLM agg.

## Why this is plausible

- `ai-dynamo` Python package is pure-Python (1.7 MB wheel)
- `ai-dynamo-runtime` is a Rust binary built against `manylinux_2_28_x86_64` — cpu-arch agnostic
- The `[vllm]` extra only adds `blake3` (no CUDA deps)
- `dynamo.frontend` is HTTP / etcd / NATS — no GPU code path
- `dynamo.vllm --model X` mostly proxies through to `vllm.LLMEngine` which the vLLM-ROCm install handles

## Why it might fail

- The published `ai-dynamo-runtime` wheel may be built with `--features cuda` and crash on import if it tries to dlopen `libcuda.so` eagerly (versus lazily). If so, we need either a ROCm-features build of the runtime wheel (the lib/llm/src/hip.rs work in the fork) or a runtime feature flag to disable CUDA paths.
- KV-event ZMQ publisher in vLLM uses the prefill bootstrap port logic (only matters at disagg time, but `--enable-kv-cache-events` could be enabled even in agg)
- vLLM-ROCm container's `vllm` version (whatever nightly snapshot) may be incompatible with `ai-dynamo`'s `vllm` API surface — Dynamo typically pins specific versions

## Phase 2 import-test result (2026-05-11)

**Test**: inside `docker.io/rocm/vllm-dev:nightly` (vllm `0.20.2rc1.dev203+g21943d4c2.rocm722`, 38.8 GB), with:
1. 4-file `nixl` stub package created at `<site-packages>/nixl/` (35 LoC total — `__init__.py`, `_api.py` with raise-on-use stubs for `nixl_agent`/`nixl_xfer_handle`/etc., `_bindings.py`)
2. `pip install --no-deps ai-dynamo==1.1.1 ai-dynamo-runtime==1.1.1 blake3 kubernetes msgpack msgspec prometheus-client pyzmq`

**Result**:
- `import dynamo.frontend` ✅
- `import dynamo.vllm` ✅
- `python3 -m dynamo.vllm --help` ✅ (full arg parser including `--dyn-tool-call-parser minimax_m2`, `--dyn-reasoning-parser minimax_append_think`, `--disaggregation-mode {agg,prefill,decode,encode}`)
- `python3 -m dynamo.frontend --help` ✅
- One harmless `dynamo.nixl_connect: Failed to load CuPy for GPU acceleration, utilizing numpy to provide CPU based operations.` info line at startup

**Implication**: For vLLM single-node aggregated, the upstream Dynamo PR can be:
1. **One launch script** (`examples/backends/vllm/launch/rocm/agg_rocm.sh`)
2. **One small patch to `dynamo.nixl_connect.__init__.py`** to make the `nixl._api` import lazy/optional (~5 LoC try/except), eliminating the need for the stub package on AMD installs
3. **Optional CI**: `Dockerfile.rocm-vllm` or just documentation pointing at `rocm/vllm-dev:nightly` + `pip install`

That is **dramatically smaller** than the JohnQinAMD fork's 148-line `Dockerfile.rocm-vllm` + UCX/RIXL build chain for the agg-only case. The fork's Dockerfile machinery only matters for **disaggregated** serving.

## Test plan (now): full end-to-end agg with MiniMax-M2.5

```bash
# On smci355-ccs-aus-g12-22:
podman run --rm -it --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined \
  --network=host --ipc=host \
  -v /shared/amdgpu/home/anluo/inferencex-agentic-test/hf-cache:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  --entrypoint bash \
  docker.io/rocm/vllm-dev:nightly

# Inside the container:
pip install --quiet "ai-dynamo[vllm]==1.1.1" "ai-dynamo-runtime==1.1.1"

# Smoke test 1: import only
python -c "import dynamo.frontend, dynamo.vllm; print('imports OK')"

# Smoke test 2: stand up etcd+nats (Dynamo deps)
# (need to start etcd and NATS — they ARE in the rocm-sglang/rocm-vllm fork images
#  but probably not in stock rocm/vllm-dev:nightly — install separately or use podman containers)

# Smoke test 3: end-to-end
python -m dynamo.frontend --http-port 8000 &
python -m dynamo.vllm --model MiniMaxAI/MiniMax-M2.5 \
  --max-model-len 8192 --max-num-seqs 4 &
sleep 600  # MiniMax-M2.5 load time
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniMaxAI/MiniMax-M2.5","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

### Failure modes to watch

| Symptom | Likely cause | Mitigation |
|---|---|---|
| `ImportError: libcuda.so.1: cannot open shared object` on `import dynamo.runtime` | Runtime wheel hard-links CUDA at startup | Need to build runtime wheel from source with HIP feature, or ship a stub `libcuda.so.1` |
| `Connection refused` on etcd/NATS | Not started | Run `etcd` + `nats-server` from podman or apt-get install |
| MiniMaxM2 architecture not registered in vLLM | Older nightly | Pull `rocm/vllm-dev:nightly` from a date after MiniMax-M2 added |
| KV cache OOM | 229B params + KV cache too big for one MI355X | Lower `max-model-len` / `max-num-seqs`; consider `--enforce-eager` to skip CUDA graphs |

## What this means for the report (Phase 5)

If Phase 2 ends up being "stock PyPI wheels work", the upstream story for vLLM AMD support is:
- **1 new file**: `examples/backends/vllm/launch/rocm/agg_rocm.sh`
- **0 source changes**
- **0 Dockerfile changes**

That makes the vLLM upstream PR a single ~50-line script change, which is essentially uncontroversial.

The "real work" then concentrates in Phase 4 (vLLM disagg) where the bootstrap-host patches and KV-transport (RIXL or NIXL-with-ROCm) actually matter.
