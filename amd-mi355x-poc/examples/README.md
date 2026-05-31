# Examples — Dynamo on AMD Instinct MI355X

Three entry-point scripts that launch the container, the Dynamo control plane
(`etcd`, `nats`), the engine, and a small benchmark, then print the headline
tok/s. Each script is self-contained and parameterized via env vars.

| Script | Backend | Topology | Model |
|---|---|---|---|
| [`sglang/agg_rocm.sh`](sglang/agg_rocm.sh) | SGLang | single-node aggregated, TP=8 + HIP graphs | DeepSeek-R1-0528 FP8 |
| [`sglang/disagg_rocm.sh`](sglang/disagg_rocm.sh) | SGLang | 2-node prefill/decode via Mooncake, TP=8 | DeepSeek-R1-0528 FP8 |
| [`vllm/agg_rocm.sh`](vllm/agg_rocm.sh) | vLLM | single-node aggregated, TP=4 + HIP graphs | MiniMax-M2.5 FP8 |

A vLLM disaggregated variant (RIXL/UCX over Pensando AINIC) will follow in a
later PR once the matching container image is published upstream.

## Prerequisites

- One or two AMD Instinct MI355X nodes (`gfx950`); examples also run on MI350X
  with the same scripts.
- `podman` available on each node; the container image is pulled from
  `docker.io/rocm/sgl-dev`.
- ROCm 7.2.x kernel modules loaded on the host.
- HuggingFace credentials configured for the model downloads (DSR1 is ~700 GB,
  MiniMax-M2.5 is ~230 GB).
- For `sglang/disagg_rocm.sh`: SSH access from the runner host to both worker
  nodes, with ionic NICs on a shared L2 RoCE fabric.

## Environment overrides

All scripts pick up defaults from environment variables; override before
invoking:

| Variable | Default | Purpose |
|---|---|---|
| `DYNAMO_VERSION` | `1.3.0` | `ai-dynamo` release; must contain [#9929](https://github.com/ai-dynamo/dynamo/pull/9929) for the nixl lazy-import proxy + `typing_extensions.Self`. |
| `HF_CACHE` | `$HOME/.cache/huggingface` | Host path mounted into the container for model weights. |
| `MODEL` | per script | HF model ID. |
| `IMAGE` | `docker.io/rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503` | Container image. |
| `LAUNCH_LOG` | `/tmp/dynamo-*.log` | Where to tee the engine launch output. |
| `PREFILL_NODE`, `DECODE_NODE`, `PREFILL_IP` | required for disagg | SSH-reachable hostnames + mgmt IP of the two MI355X nodes. |

Example:

```bash
DYNAMO_VERSION=1.3.0 HF_CACHE=/data/hf ./sglang/agg_rocm.sh
```

## Expected output

Each script prints a `[date]` timeline of container start → model load →
benchmark, ending with a single-line summary such as:

```
[2026-05-30 14:02:11] tok/s=97.7  TPOT=7.11 ms  c=1  ISL=1024 OSL=1024
```

First run includes a one-time model download (DSR1 is ~700 GB, MiniMax-M2.5
is ~230 GB), so budget ~30–90 min before the engine launches if the HF cache
is empty. Subsequent runs reuse the cached weights and start in under a
minute.
