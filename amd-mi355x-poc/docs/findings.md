# Findings — AMD MI355X validation

Headline numbers and what they mean. Cluster: AAC1 (`aac1.amd.com`),
partition `256C8G1H_MI355X_Ubuntu22`. Hardware: 2× 8-GPU MI355X nodes,
each with 9× AMD Pensando ionic RoCE NICs.

## Best result per configuration

| # | Backend | Topology | Model | Best aggregate tok/s | Saturated at | Success |
|---|---|---|---|---:|---|---:|
| 1 | SGLang | single-node agg, TP=8 + HIP graphs | DeepSeek-R1-0528 FP8 (671B) | **708** @ c=8 | compute | 24/24 |
| 2 | vLLM | single-node agg, TP=4 + HIP graphs | MiniMax-M2.5 FP8 (229B MoE) | **521** @ c=8 | compute | 24/24 |
| 3 | SGLang | 2-node disagg + Mooncake, TP=8 each | DeepSeek-R1-0528 FP8 (671B) | **527** @ c=16 ¹ | transport (Mooncake @ c=32) | 160/160 |
| 4 | vLLM | 2-node disagg + RIXL/UCX, TP=4 each | MiniMax-M2.5 FP8 (229B MoE) | **730** @ c=32 ² | compute | 320/320 |
| 4-DSR1 | vLLM | 2-node disagg + RIXL/UCX, TP=8 each | DeepSeek-R1-0528 FP8 (671B) | n/a — startup fails | `ibv_reg_mr` EINVAL ³ | n/a |

## Side-by-side: same hardware, different transport

| Concurrency | SGLang+Mooncake (DSR1) | **vLLM+RIXL (M2.5)** |
|---:|---:|---:|
| 4 | 261.7 tok/s | 385.4 tok/s |
| 8 | 456.7 | 718.2 |
| 16 | 527.4 | 727.3 |
| **32** | **CRASH** (transport retry exceeded) | **730.4 tok/s, 320/320 success** |

This is the architectural difference at the heart of finding #3.
**RIXL's UCX-plugin C++ transfer pipeline scales cleanly past where
Mooncake's Python-level chunked register/transfer/deregister pattern
saturates the ionic firmware's QP-setup queue.** (Note: RIXL doesn't
"chunk MRs better" — RIXL doesn't chunk at all; the difference is in
the UCX plugin's C++ transfer scheduling.)

## vs JohnQinAMD fork's published numbers

| Comparison point | Fork | Us | Verdict |
|---|---:|---:|---|
| Mooncake DSR1 @ c=1 (Test 12) | 97.7 tok/s/req | **105.7** | **+8.2% ahead** with stock public Mooncake |
| Mooncake DSR1 TPOT @ c=1 | 7.11 ms | 9.46 ms | -33% (residual aiter MoE preshuffle tuning) |
| MoRI+EP/DP-Attn DSR1 @ c=4 | 178 tok/s | **261.7** (Mooncake) | **+47% ahead** |
| MoRI+EP/DP-Attn DSR1 @ c=16 | 672 tok/s | 527.4 (Mooncake) | -22% behind |
| MoRI+EP/DP-Attn DSR1 @ c=128 | 2,196 | n/a (Mooncake crashed at c=32) | needs MoRI to compete |
| DEP8 DSR1 @ c=1024 | 16,011 (1,334/GPU) | n/a | needs MoRI + EP/DP-Attn (Tier 3 in path-to-prod) |

The remaining gap to the fork's 1,334 tok/s/GPU DEP8 result requires
running MoRI on lossless PFC + applying EP/DP-Attention. AAC1 ionic is
ECN/DCQCN (not PFC), which is the wall described in
[`network-debug.md`](network-debug.md). On a lossless-PFC cluster MoRI
should hit fork's published numbers using the same Dynamo + SGLang +
AMD MI355X stack.

## Footnotes

¹ **DSR1 SGLang+Mooncake disagg headline: 527.4 tok/s @ c=16 with
160/160 success — 16× over the conservative 32.8 tok/s baseline measured
before fork-aligned tuning.** Crashed at c=32 (transport retry exceeded).
See [`archive/10-dsr1-concurrency-sweep.md`](archive/10-dsr1-concurrency-sweep.md).
Initial M2.5 disagg run with `--enforce-eager` measured 72.7 tok/s @
c=8; HIP graphs gave 8.1× speedup to 587. Initial DSR1 disagg
measurement was 32.8 tok/s aggregate @ c=8 with a conservative launch
config; applying the fork's exact launch flags + 9 env vars + matching
bench harness raised per-request output throughput to 105.7 tok/s —
**8.2% above the fork's published Test 12 result of 97.7 tok/s** on the
same Mooncake transport. See
[`archive/09-test12-reproduction.md`](archive/09-test12-reproduction.md).

² **M2.5 vLLM+RIXL disagg headline: 730.4 tok/s @ c=32 with 320/320
success.** Performance saturates at compute (c=8/16/32 all hit
~720–730 tok/s aggregate; TTFT grows from 131 ms to 33.6 s but throughput
plateaus). To go higher requires more decode GPUs or a smaller model.
Detailed sweep + RIXL-vs-Mooncake architectural analysis:
[`archive/11-m25-vllm-rixl-sweep.md`](archive/11-m25-vllm-rixl-sweep.md).

³ **DSR1 vLLM+RIXL cross-validation: vLLM crashes at startup with
`ibv_reg_mr ... access=0xf failed: Invalid argument` → `NIXL_ERR_BACKEND`.**
The initial "ionic per-MR size ceiling" diagnosis is **retracted.**
Eleven standalone probes on the same image/node/env disprove every
infrastructure-layer hypothesis:

- ionic DRAM MR up to 4 GiB OK
- ROCm/VRAM MR up to 8 GiB OK
- 32×2 GiB concurrent in one process (64 GiB) OK
- 8 procs × 2.58 GiB libibverbs OK
- `access=0xf` (REMOTE_ATOMIC included) OK on ionic
- NIXL+PyTorch 1×2.638 GiB OK
- NIXL+PyTorch 16×1 GiB OK
- `UCX_RCACHE_MAX_UNRELEASED=4` OK
- 8 procs × 1×2.638 GiB NIXL+PyTorch (matches vLLM TP=8 exactly) OK
- 8 procs × 60 regions × 256 MiB (120 GiB total) OK
- 80 GiB pre-allocated weights then 2.638 GiB NIXL register OK

**Whatever causes vLLM to fail is vLLM-internal — not ionic, not MR
limits, not UCX/RIXL/NIXL, not PyTorch, not multiprocessing, not memory
pressure.** Right next step: direct instrumentation inside
`vllm/worker.py:929` (`register_memory` call site), not another
transport-layer probe. Full elimination matrix:
[`archive/12-dsr1-vllm-rixl-cross-validation.md`](archive/12-dsr1-vllm-rixl-cross-validation.md).
All eight reproducer probes in [`../advanced/debug-probes/`](../advanced/debug-probes/).
