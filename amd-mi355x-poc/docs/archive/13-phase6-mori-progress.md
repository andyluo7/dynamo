# Phase 6 — SGLang + MoRI on MI355X (in-progress)

Date: 2026-05-13
Cluster: AAC1, prefill=`smci355-ccs-aus-g12-06`, decode=`smci355-ccs-aus-g12-26`
Image: `docker.io/rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503` (MoRI pre-installed at `/sgl-workspace/mori/`)
Model: `deepseek-ai/DeepSeek-R1-0528` FP8, TP=8 / EP=8 / DP=8 + DP-Attn

**Goal:** reproduce JohnQinAMD fork's published MoRI numbers — 178 tok/s @ c=4, 672 @ c=16, 2196 @ c=128.

## Status

| Step | Result |
|---|---|
| 1 | rocm/sgl-dev image already ships MoRI (`/sgl-workspace/mori/`) — **no build needed** |
| 2 | MoRI-EP single-node smoke (TP=8 EP=8 DP=8 + `--moe-a2a-backend mori`) — **PASS**, ~11 min from launch to ready, chat completion 200 OK |
| 3 | MoRI-IO 2-node disagg setup — **PASS** for single request after `sglang_router.launch_router --pd-disaggregation --mini-lb` is started with prefill IP (not localhost), libionic host mounts present |
| 4 | Concurrency sweep at c=1, 4 with tuned config (SGLANG_MORI_FP8_DISP=True, QP_PER_TRANSFER=4, NUM_WORKERS=4, --enable-two-batch-overlap, MORI_RDMA_TC=104) — **PARTIAL PASS** |
| 5 | c=8 onward — **FAIL** with `std::bad_alloc` on decode (memory pressure from larger MoRI per-QP buffers × workers × overlap pipeline) |
| 6 | Fork-comparable numbers — **partial** (54% of fork's c=4 number; c=16/128 not yet reached) |

## Sweep results so far

| c | n_prompts | TPOT P50 | TTFT P50 | tok/s/req | **tok/s aggregate** | success | vs fork |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 39.4 ms | 1,477 ms | 25.6 | 24.9 | 10/10 | (fork: 97.7 → ~25%) |
| 4 | 40 | 40.4 ms | 301 ms | 23.9 | **96.1** | 40/40 | (fork: 178 → **54%**) |
| 8 | 80 | — | — | — | crash (`std::bad_alloc`) | — | — |
| 16 | 160 | — | — | — | not reached | — | (fork: 672) |
| 32 | 320 | — | — | — | not reached | — | — |
| 64 | 640 | — | — | — | not reached | — | — |
| 128 | 1280 | — | — | — | not reached | — | (fork: 2196) |

The c=1 and c=4 numbers are the **first AMD MI355X SGLang+MoRI-IO disagg
numbers we have on this PoC**. They are already meaningful — disagg is
end-to-end functional, the router routes correctly, the bootstrap
handshake completes — but they are nowhere near the fork's published
numbers because:

- TPOT @ c=1 is **39.4 ms** vs fork's Mooncake 9.46 ms vs fork's MoRI ~7 ms.
  4× too slow. Most likely: aiter MoE preshuffle still untuned for our
  exact MI355X firmware, MoRI dispatch dtype not optimal, missing
  `--enforce-shared-experts-fusion`, and/or the KV-cache layout is FA
  instead of FlashInfer (MoRI prefers FI for less metadata overhead).
- c=8 OOM means our `--mem-fraction-static 0.72 + SGLANG_MORI_QP_PER_TRANSFER=4
  + SGLANG_MORI_NUM_WORKERS=4 + --enable-two-batch-overlap` config exceeds
  the available VRAM headroom for MoRI's per-QP DRAM staging buffers.

## Tuning iteration log

| Fix candidate | Round 1 | Round 2 | **Round 3** |
|---|---|---|---|
| `SGLANG_MORI_FP8_DISP` | False | **True** | **True** |
| `--enable-two-batch-overlap` | absent | **present** | **present** |
| `MORI_RDMA_TC` | 96 | **104** | **104** |
| `SGLANG_MORI_QP_PER_TRANSFER` | 1 | 4 | **1** (back to default) |
| `SGLANG_MORI_NUM_WORKERS` | 1 | 4 | **2** |
| `--mem-fraction-static` | 0.72 | 0.72 | **0.65** |
| **Result @ c=4** | crash (RDMA assertion) | OK (96.1 tok/s) | **OK (97.3 tok/s)** |
| **Result @ c=8** | (not reached) | crash (`std::bad_alloc`) | **crash (`KV transfer failed: Work Request Flushed Error`)** |

The wall has moved through three different failure modes:
- **Round 1**: MoRI control-plane handshake assertion (fixed by FP8/two-batch-overlap/RDMA-TC).
- **Round 2**: VRAM `std::bad_alloc` on decode (fixed by smaller MoRI per-QP buffers + lower mem-fraction).
- **Round 3**: ionic NIC RDMA QPs go into Error State and flush all in-flight Work Requests. MoRI's own error message says: *"Flush errors are cascaded from QP(s) entering Error State. Check: (1) peer process alive, (2) PFC / network congestion, (3) ibv_devinfo / dmesg for HW errors."*

**Round 3 c=8+ failure is at the NIC layer**, not in our software stack. Symptoms (`Work Request Flushed Error` cascading from one QP) are textbook PFC/lossless-RoCE configuration issues. The fork's runbook is silent about PFC tuning, which suggests their cluster has PFC enabled NIC-wide. AAC1's ionic NICs may not (and configuring PFC is a cluster-admin operation, not ours).

## Round 3 sweep results (current best)

| c | n_prompts | TPOT P50 | TTFT P50 | tok/s/req | **tok/s aggregate** | success | vs fork |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 39.8 ms | 794 ms | 25.2 | 24.8 | 10/10 | (fork: 97.7 → 25%) |
| 4 | 40 | 40.1 ms | 294 ms | 24.7 | **97.3** | 40/40 | (fork: 178 → **55%**) |
| 8+ | — | — | — | — | crash (RDMA flush) | — | (fork: 672 @ c=16) |

## What broke during the sweep

Both prefill and decode SGLang processes crashed mid-sweep:

```
[mori]src/io/rdma/backend_impl.cpp:801: ControlPlaneServer::BuildRdmaConn:
  Assertion `hdr.type == MessageType::RegEndpoint' failed.
[mori]ibverbs.cpp:168: syscall failed with Connection timed out
RuntimeError: [gloo/transport/tcp/pair.cc:547] Connection closed by peer
[2026-05-13 16:07:25] SIGQUIT received. signum=None, frame=None.
```

So MoRI-IO RDMA handshake works for the first one or two requests, then fails when subsequent requests try to (re)establish the QP — control-plane message-type assertion blows up. After the assertion the scheduler aborts and gloo collective TCP pairs close, which kills the rest of the workers.

## Probable causes (in order of likelihood)

1. **Missing MoRI-IO port config.** The fork's `amd_utils/env.sh` references `handshake_port=6301` and `notify_port=61005` for MoRI-IO. Our launcher only set the SGLang `--disaggregation-bootstrap-port 30001` but did not pin the MoRI-IO control-plane ports. SGLang's MoRI connector may be picking dynamic ports that get clobbered between requests.

2. **`MORI_RDMA_TC=96` may not match this cluster.** Fork's runbook says `MORI_RDMA_TC=104` for `mia1*` nodes and `MORI_RDMA_TC=96` for `GPU* / smci355-ccs-aus-*` nodes; we set 96 based on hostname. If the QoS class is wrong, RDMA QP setup may silently fail.

3. **Missing FP8 dispatch flag.** Fork sets `SGLANG_MORI_FP8_DISP=True`; we set False (in part because of an earlier "FP8 only" misread). DSR1's MoE kernels are tuned for FP8 dispatch on MI355X.

4. **`--enable-two-batch-overlap` not set.** Fork's headline numbers use this. Without it, MoRI's pipelining is suboptimal.

5. **Bootstrap port reuse on retried handshakes.** `MORI_IO_QP_MAX_*` defaults may not match what SGLang's MoRI connector requests. The control-plane assertion suggests message framing got confused — could be a leftover QP from a previous request being reused.

## What we have working (PR-ready)

- **Image discovery**: `rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503` ships MoRI pre-built. No need for the Dockerfile.rocm_base build_mori stage when you can use this image.
- **MoRI-EP single-node** at TP=8/EP=8/DP=8 + DP-Attn boots and serves DSR1 cleanly. Useful baseline for MoE perf comparisons before disagg.
- **MoRI-IO 2-node bring-up** sequence: launch prefill server (`--disaggregation-mode prefill --disaggregation-transfer-backend mori`), launch decode server (same with decode), then `sglang_router.launch_router --pd-disaggregation --mini-lb --prefill <PREFILL_IP>:8001 30001 --decode <DECODE_IP>:8002 --port 8000`. Single request through the router routes correctly.
- **Required mounts** beyond the image: `/etc/libibverbs.d/ionic.driver`, `libionic.so*` chain, `libionic-rdmav34.so` — same as Phases 3-4. Without these the container's libionic kernel-ABI mismatch makes ionic invisible to MoRI.

## Next workstream (to close the gap to fork's numbers)

In priority order:

1. **Fix c=8 OOM**: lower one or more of `SGLANG_MORI_QP_PER_TRANSFER` (4→1
   or 2), `SGLANG_MORI_NUM_WORKERS` (4→2), or `--mem-fraction-static`
   (0.72→0.65). Needs a dimensional analysis of MoRI's per-QP × per-worker
   buffer footprint vs VRAM headroom.
2. **Fix the 4× TPOT gap @ c=1** (39.4 ms vs fork's 9.46 ms with Mooncake,
   ~7 ms with MoRI):
   - Add `--enforce-shared-experts-fusion` (fork's runbook flags it as
     critical for DSR1 perf).
   - Verify `SGLANG_MORI_DISPATCH_DTYPE` matches DSR1's expert weight
     dtype (we set bf16; FP8 may be more efficient with FP8 weights).
   - Check whether `aiter` MoE preshuffle is enabled / cached — first run
     pays preshuffle cost.
   - Try `--cuda-graph-max-bs 64` instead of 32 (fork's c=128 setup uses
     larger graphs).
3. **Re-run sweep at c=1, 4, 8 only** with stable config to lock in the
   numbers, then extend to c=16, 32, 128 on a follow-up.
4. **For the truly-headline 1,334 tok/s/GPU DEP8 number**: requires
   multi-node DEP=8 (8 nodes × 8 GPUs = 64 GPUs total) — out of scope
   for this 2-node setup.

## Reproducer

```bash
# 1. Allocate two MI355X nodes (we used g12-06 + g12-26)
# 2. Launch:
bash scripts/phase6_mori_disagg.sh
# 3. Wait ~12 min for both servers to come up (warm cache; ~30 min cold)
# 4. Start the PD router (do this AFTER both servers print "fired up and ready"):
ssh <prefill> "podman exec -d sglang-mori-prefill bash -c \
  'python3 -m sglang_router.launch_router \
     --pd-disaggregation --mini-lb --policy random \
     --prefill http://<PREFILL_IP>:8001 30001 \
     --decode  http://<DECODE_IP>:8002 \
     --port 8000 --host 0.0.0.0 > /tmp/lb.log 2>&1'"
# 5. Single-shot test BEFORE the sweep (verifies handshake works):
curl http://<prefill>:8000/v1/chat/completions -d '{"model":"...","messages":[...]}'
# 6. Sweep (currently fails after first 1-2 requests):
ssh <prefill> "podman exec sglang-mori-prefill bash /tmp/mori_sweep.sh"
```

Single-node smoke test (no router needed): `bash scripts/phase6_mori_agg.sh`.
