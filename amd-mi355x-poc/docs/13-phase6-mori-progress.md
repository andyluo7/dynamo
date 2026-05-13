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
| 4 | Concurrency sweep at c=1,4,8,16,32,64,128 — **FAIL across the board** with `ChunkedEncodingError: Response ended prematurely` |
| 5 | Fork-comparable numbers (178/672/2196 tok/s) — **not yet reached** |

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

## Next workstream

To get the fork's numbers reproduced:

1. Re-run with `SGLANG_MORI_FP8_DISP=True` and `--enable-two-batch-overlap`.
2. Pin `MORI_IO_HANDSHAKE_PORT` and `MORI_IO_NOTIFY_PORT` if the env-var names exist (otherwise look at fork's launch script for the actual override path).
3. Try lower concurrency first (c=1, c=4 only) and verify each level is stable for 1-2 minutes before stepping up.
4. If RDMA assertion still fires, instrument MoRI's `BuildRdmaConn` (it's open-source at `/sgl-workspace/mori/src/io/rdma/backend_impl.cpp:801`) to log the actual `hdr.type` value seen vs expected — that will tell us whether it's a message corruption or a protocol-version mismatch.

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
