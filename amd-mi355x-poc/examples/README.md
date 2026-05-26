# Examples

Four supported entry points. Each is a single bash script that launches
the container, the Dynamo control plane (etcd, NATS), the engine, and a
small benchmark, then prints the headline tok/s.

| Script | Backend | Topology | Model |
|---|---|---|---|
| [`sglang/agg_rocm.sh`](sglang/agg_rocm.sh) | SGLang | single-node aggregated, TP=8 + HIP graphs | DeepSeek-R1-0528 FP8 |
| [`sglang/disagg_rocm.sh`](sglang/disagg_rocm.sh) | SGLang | 2-node prefill/decode via Mooncake, TP=8 | DeepSeek-R1-0528 FP8 |
| [`vllm/agg_rocm.sh`](vllm/agg_rocm.sh) | vLLM | single-node aggregated, TP=4 + HIP graphs | MiniMax-M2.5 FP8 |
| [`vllm/disagg_rocm.sh`](vllm/disagg_rocm.sh) | vLLM | 2-node prefill/decode via RIXL/UCX, TP=4 | MiniMax-M2.5 FP8 |

## Prerequisites

See the [top-level README](../README.md#prerequisites) — MI355X node(s),
ionic NICs, podman, ROCm 7.2.x, HuggingFace access.

## Reproducibility

These scripts were promoted from the original PoC's `phase{1,2,3,4}_*.sh`
working drivers (now in [`../advanced/sweeps/`](../advanced/sweeps/)).
Each is the best-performing variant we found for its (backend, topology)
combination, with comments explaining non-obvious flags.

For full concurrency sweeps, side-by-side transport comparisons, or to
reproduce the matrix in [`../docs/findings.md`](../docs/findings.md),
use the scripts in `../advanced/sweeps/` and `../advanced/benchmarks/`.

## Expected outcomes

See [`../docs/findings.md`](../docs/findings.md) for headline numbers.
First run includes a one-time model download (DSR1 is ~700 GB, M2.5 is
~230 GB), so budget the model load if HF cache is empty.
