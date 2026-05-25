# Dynamo on AMD MI355X

End-to-end working setup for running NVIDIA Dynamo on AMD MI355X (gfx950)
GPUs, with both vLLM and SGLang backends in single-node aggregated and
2-node disaggregated topologies. Validated on AAC1 (`aac1.amd.com`)
with 8× MI355X per node and AMD Pensando ionic RoCE NICs.

> The four scripts in [`examples/`](examples/) are the supported entry
> points. The original phase-by-phase development reports are in
> [`docs/archive/`](docs/archive/) for historical context.

## Quickstart

Pick a topology + backend, run one script. Each example launches the
container, the Dynamo control plane (etcd, NATS), the engine, and a small
benchmark — full round-trip in 10–20 min depending on model load.

```bash
# 1. Clone + cd
git clone https://github.com/andyluo7/dynamo.git
cd dynamo/amd-mi355x-poc

# 2. (vLLM disagg only) build the RIXL container — ~10 min, first time
podman build -t dynamo-vllm-rixl:latest -f container/Dockerfile.rocm-vllm-rixl .

# 3. Run one of the four examples
bash examples/sglang/agg_rocm.sh       # SGLang single-node aggregated (DSR1)
bash examples/sglang/disagg_rocm.sh    # SGLang 2-node disagg via Mooncake (DSR1)
bash examples/vllm/agg_rocm.sh         # vLLM single-node aggregated (MiniMax-M2.5)
bash examples/vllm/disagg_rocm.sh      # vLLM 2-node disagg via RIXL (MiniMax-M2.5)

# 4. (Optional) full concurrency sweeps and reproduction harnesses
ls advanced/sweeps/                    # see advanced/README.md
```

## What works

| Backend | Topology | Model | Best aggregate tok/s | Saturated at |
|---|---|---|---:|---|
| SGLang | single-node agg, TP=8 + HIP graphs | DeepSeek-R1-0528 FP8 | **708** @ c=8 | compute |
| vLLM   | single-node agg, TP=4 + HIP graphs | MiniMax-M2.5 FP8     | **521** @ c=8 | compute |
| SGLang | 2-node disagg + Mooncake, TP=8     | DeepSeek-R1-0528 FP8 | **527** @ c=16 | transport (Mooncake @ c=32) |
| vLLM   | 2-node disagg + RIXL, TP=4         | MiniMax-M2.5 FP8     | **730** @ c=32 | compute |

Full results table, side-by-side transport comparison, and vs-fork
benchmarks: [`docs/findings.md`](docs/findings.md).

## What's known broken

DSR1 vLLM+RIXL (TP=8) currently fails at startup with `NIXL_ERR_BACKEND`
from inside vLLM's `register_memory`. **Eleven** transport-layer
hypotheses have been eliminated — the remaining root cause is vLLM-internal.
Full elimination matrix:
[`docs/archive/12-dsr1-vllm-rixl-cross-validation.md`](docs/archive/12-dsr1-vllm-rixl-cross-validation.md).
Tracked as a follow-up.

## Prerequisites

- 1–2 nodes with 8× AMD Instinct MI355X (gfx950) per node
- Pensando ionic RoCE NICs (or equivalent — KV-transfer specifics will
  differ on other RDMA hardware)
- `podman` (the AAC1 user is not in the `docker` group)
- ROCm 7.2.x available on the host (`module load rocm/7.2.2` on AAC1)
- HuggingFace access for the models the examples pull

## Repository layout

```
amd-mi355x-poc/
├── README.md                        ← this file
├── container/
│   └── Dockerfile.rocm-vllm-rixl    ← extends rocm/vllm-dev:nightly w/ UCX-ROCm + RIXL
├── examples/                        ← 4 supported entry points
│   ├── sglang/{agg,disagg}_rocm.sh
│   └── vllm/{agg,disagg}_rocm.sh
├── patches/                         ← runtime patches applied inside containers
│   ├── README.md                    ← what each patch does + build/install
│   ├── ibv_ionic_compat.c           ← LD_PRELOAD interposer for ionic NIC quirks
│   └── fork-patches/                ← Mooncake ROCm DRAM staging (used by sglang disagg)
├── docs/
│   ├── README.md                    ← doc index
│   ├── architecture.md              ← stack overview + how the pieces fit
│   ├── findings.md                  ← full results + vs-fork comparison
│   ├── network-debug.md             ← ionic NIC root-cause notes (ECN/DCQCN, PFC)
│   └── archive/                     ← original phase-by-phase reports (14 files)
└── advanced/                        ← for users reproducing the PoC sweeps
    ├── benchmarks/                  ← single-c benchmark drivers
    ├── sweeps/                      ← full concurrency sweep scripts
    └── debug-probes/                ← standalone RDMA + MR probes
```

## Common gotchas

- The SLURM partition `256C8G1H_MI355X_Ubuntu22` on AAC1 is **not enforced
  exclusive**. Multiple users can land on the same node. Pick an explicit
  idle node via `--nodelist=` and verify with `rocm-smi --showmemuse`.
- ROCm/HIP doesn't always release GPU memory on container kill.
  `pkill -9 -f VLLM::Worker` / `pkill -9 -f sglang.srt` to force-release.
- The dynamo-prefill container's `--log-level warning` suppresses progress
  during HIP graph capture and KV transfer — looks "stuck" for ~10 min
  after launch but is fine. `rocm-smi --showpids` if uncertain.

## License + attribution

All new code in this directory is Apache 2.0 (matching the parent dynamo
repo). Files in `patches/fork-patches/` are copied unmodified from the
JohnQinAMD fork of ai-dynamo/dynamo (Apache 2.0, NVIDIA copyright). The
LD_PRELOAD interposer in `patches/ibv_ionic_compat.c` was extracted from
that fork's `nixl_rocm_staging.py` and extended with `ibv_reg_dmabuf_mr`
wrapping.

## Links

- Upstream: [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)
- This fork: [andyluo7/dynamo](https://github.com/andyluo7/dynamo)
- Reference AMD Dynamo work: [JohnQinAMD/dynamo `amd-dynamo`](https://github.com/JohnQinAMD/dynamo/tree/amd-dynamo)
- AMD RIXL: [ROCm/RIXL](https://github.com/ROCm/RIXL)
- AMD-side upstream-compat PR: [ai-dynamo/dynamo#9929](https://github.com/ai-dynamo/dynamo/pull/9929)
