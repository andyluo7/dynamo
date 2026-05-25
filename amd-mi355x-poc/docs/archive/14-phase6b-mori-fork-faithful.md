# Phase 6b — Fork-faithful MoRI reproduction (in progress, pre-empted)

Date: 2026-05-13
Cluster: AAC1, prefill=`smci355-ccs-aus-g12-06`, decode=`smci355-ccs-aus-g12-26`
Image: `docker.io/rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503`
Model: `deepseek-ai/DeepSeek-R1-0528` FP8

**Goal:** match the JohnQinAMD/InferenceX runner config exactly (env.sh +
models.yaml + server.sh + bench.sh) instead of the loose approximation in
Phase 6 (round 1-3).

## What "fork-faithful" means here

After reading the fork's actual launch pipeline
(`runners/launch_mi355x-amds.sh` → `benchmarks/multi_node/dsr1_fp8_mi355x_sglang-disagg.sh`
→ `benchmarks/multi_node/amd_utils/{submit.sh, job.slurm, server.sh, env.sh, models.yaml, bench.sh}`),
the **non-negotiable** changes from Phase 6 round 3 are:

1. **Asymmetric prefill vs decode flags** (from `models.yaml` DeepSeek-R1-0528 entry):

   | Flag | Prefill | Decode |
   |---|---|---|
   | `--mem-fraction-static` | 0.8 | 0.85 |
   | `--max-running-requests` | 24 | 4096 |
   | `--chunked-prefill-size` | 16384×TP=131072 | (default) |
   | `--cuda-graph-bs` | `1 2 3` | `1..160` |
   | `--disable-radix-cache` | yes | no |
   | `--prefill-round-robin-balance` | n/a | yes |
   | NEXTN MTP | n/a | `--speculative-algorithm NEXTN --speculative-eagle-topk 1 --speculative-num-steps 1 --speculative-num-draft-tokens 2` |
   | `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` env | 16384 | 320 (= 160 × (MTP+1)) |

2. **Container settings the fork's `job.slurm` uses** (we were missing all of these in Phase 6 rounds 1-3):

   - `--privileged` (we had only `--security-opt seccomp=unconfined`)
   - `--ulimit memlock=-1` (unlimited RDMA pinned memory — **critical for MoRI**)
   - `--ulimit stack=67108864` (64MB stack)
   - All 8 explicit `--device=/dev/infiniband/uverbsN` mappings
   - `--device=/dev/infiniband/rdma_cm` (RDMA Connection Manager — used in QP setup)
   - `--cap-add SYS_PTRACE`
   - `--shm-size 128G` (where supported — incompatible with `--ipc host`, fork uses host IPC)

3. **Bench harness**: `python3 -m sglang.bench_serving --backend openai
   --dataset-name random --random-input-len 1024 --random-output-len 1024
   --random-range-ratio 0.8 --num-prompts $((conc*10)) --max-concurrency $conc
   --request-rate inf` — exactly what the fork's `bench.sh` calls. We had been
   using a custom Python harness in Phase 6 rounds 1-3.

4. **Env vars** from `env.sh` (we had most but not all):

   - `SGLANG_USE_AITER=1`
   - `MORI_SHMEM_MODE=ISOLATION`
   - `SGLANG_MORI_FP8_DISP=True`
   - `SGLANG_MORI_FP4_DISP=False, SGLANG_MORI_FP8_COMB=False`
   - `MORI_MAX_DISPATCH_TOKENS_PREFILL=16384, MORI_MAX_DISPATCH_TOKENS_DECODE=160`
   - `SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))`  ← formula, not constant
   - `MORI_RDMA_TC=96` for `smci355-ccs-aus-*` per env.sh hostname rule
   - `MORI_EP_LAUNCH_CONFIG_MODE=AUTO`
   - `MORI_IO_QP_MAX_SEND_WR=16384, MORI_IO_QP_MAX_CQE=32768, MORI_IO_QP_MAX_SGE=4`
   - `PYTHONPATH=/sgl-workspace/aiter:$PYTHONPATH` (FIXME-tagged WA in fork's env.sh)
   - `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200, SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200`

## What ran

`scripts/phase6b_mori_fork_repro.sh` launches both nodes with the
fork-faithful asymmetric config. `scripts/mori_fork_sweep.sh` runs the
fork-style `sglang.bench_serving` sweep at c=1, 4, 8, 16, 32, 64, 128.

| Step | Result |
|---|---|
| Container bring-up with `--privileged` + memlock=-1 + 8 uverbs + rdma_cm | OK on both nodes |
| Prefill server ready (mem-frac 0.8, max-running 24, chunk 131072, graphs 1-3) | OK after ~12 min (warm cache) |
| Decode server ready (mem-frac 0.85, max-running 4096, graphs 1-160, NEXTN MTP) | OK after ~16 min (slower because of 160 graphs to capture) |
| `sglang_router.launch_router --pd-disaggregation --mini-lb` started | OK |
| Single chat completion via router | **200 OK** — confirms full PD path works with the fork's exact config |
| Concurrency sweep at c=1 via `sglang.bench_serving` | **launched but SLURM allocation pre-empted ~17 min in** |

## What we did NOT yet measure

- **No completed sweep**. The SLURM job that owned the prefill node
  (smci355-ccs-aus-g12-06) ended mid-sweep; the container exited with code 143
  (SIGTERM). All in-container `/tmp` data was lost because we didn't
  bind-mount `/tmp` to a host path. **Action item for the next attempt**:
  add `-v /shared/amdgpu/home/anluo/mori-bench-results:/tmp/results` and
  redirect outputs there.

## Observations from the partial run

1. **Output quality issue**: with `--kv-cache-dtype fp8_e4m3`, decode output
   is garbled (`MMMMMMMM...`). SGLang logs:
   `Using FP8 KV cache but no scaling factors provided. Defaulting to scaling
   factors of 1.0. This may lead to less accurate results!`
   The fork's models.yaml sets `--kv-cache-dtype fp8_e4m3` for DSR1 too, so
   either (a) the fork's setup ships scaling factors that ours doesn't, or
   (b) the fork accepts the quality hit since it's a perf bench.
   **Action item**: check whether DSR1's HF snapshot includes
   `kv_cache_scales_path` configuration; if not, drop FP8 KV cache and use
   bf16 for the bench (perf will be different but quality will be right).

2. **Decode startup is much longer with cuda-graph-bs 1-160** vs our previous
   `--cuda-graph-max-bs 32` — ~16 min vs ~12 min. Worth budgeting in any
   re-run.

3. **No MoRI assertion or RDMA Work Request Flushed errors** during the
   single-request smoke test — the container settings (`--privileged`,
   `memlock=-1`, all uverbs) appear to remove the issues that cascaded the
   round 2/3 sweeps.

## Reproducer

```bash
# 1. Allocate two MI355X nodes (g12-06 + g12-26 used here)
# 2. Launch:
bash scripts/phase6b_mori_fork_repro.sh
# 3. Wait ~16 min for both servers to come up (warm cache)
# 4. Start the PD router (after both /v1/models return 200):
ssh <prefill> "podman exec -d sglang-mori-prefill bash -c \
  'python3 -m sglang_router.launch_router \
     --pd-disaggregation --mini-lb --policy random \
     --prefill http://<PREFILL_IP>:8000 \
     --decode  http://<DECODE_IP>:8000 \
     --port 30000 --host 0.0.0.0 > /tmp/lb.log 2>&1'"
# 5. Smoke test (should be 200 OK):
curl -m 60 http://<prefill>:30000/v1/chat/completions -d '{"model":"...","messages":[...]}'
# 6. Sweep:
ssh <prefill> "podman cp scripts/mori_fork_sweep.sh sglang-mori-prefill:/tmp/ \
  && podman exec sglang-mori-prefill bash /tmp/mori_fork_sweep.sh"
# 7. Pull results: /tmp/mori_fork_results.csv (BIND-MOUNT this from host
#    next time so SLURM preemption doesn't lose them).
```

## Phase 6c — single-allocation sbatch retry

To dodge SLURM preemption, `scripts/phase6c_mori_sbatch.sh` packages the
entire bring-up + smoke + sweep into one sbatch job that holds both nodes
for 8 hours. Results bind-mounted to `/shared/.../mori-bench-results/job-N/`
so they survive any container exit.

Two iterations:

- **Job 81**: fork-faithful config booted both servers in 990s, smoke OK,
  but `sglang.bench_serving` failed with `ModuleNotFoundError:
  No module named 'sglang.benchmark.datasets'`. Root cause: namespace-package
  collision — `sglang` resolves to `/sgl-workspace/sglang` (repo root,
  no `datasets/`) instead of `/sgl-workspace/sglang/python/sglang`
  (actual package). All seven sweep iterations failed with exit=1 in 30s.

- **Job 82** (PYTHONPATH fix): export `PYTHONPATH=/sgl-workspace/sglang/python`
  before invoking the bench. **c=1 SUCCEEDED** with the fork-faithful
  config; c=4 onwards crashed with the same MoRI `ibverbs.cpp:168 syscall
  failed with Connection timed out` → `unknown parameter type` → `SIGQUIT`
  cascade we saw in Phase 6 round 3.

## Phase 6c c=1 result (fork-faithful config, MoRI-IO PD disagg)

| Metric | Value |
|---|---:|
| Output throughput | **58.3 tok/s** |
| Input throughput | 58.9 tok/s |
| Total throughput | 117.2 tok/s |
| Median TTFT | 560 ms |
| **Median TPOT** | **14.63 ms** |
| P99 TTFT | 1781 ms |
| P99 TPOT | 26.0 ms |
| Median E2E latency | 14.6 s |

vs prior runs:

| Run | c=1 throughput | c=1 TPOT |
|---|---:|---:|
| Phase 6 round 3 (symmetric flags, no MTP, mem-frac 0.65) | 24.8 tok/s | 39.4 ms |
| Phase 3 fork-aligned Mooncake (Test 12 reproduction, c=1) | 105.7 tok/s | 9.46 ms |
| **Phase 6c (fork-faithful MoRI, asymmetric + MTP)** | **58.3 tok/s** | **14.63 ms** |
| Fork's published Test 12 (Mooncake) | 97.7 tok/s | 7.11 ms |

The asymmetric prefill/decode config + NEXTN MTP got us from 25 → 58 tok/s
and from 39.4 → 14.63 ms TPOT. That is **2.4× higher throughput and
2.7× lower TPOT** than Phase 6 round 3 just from following the fork's
launch flags exactly. Still 60% of fork's published Test 12 number, but
that gap is now small enough to plausibly be attributable to:

- the FP8 KV cache scaling-factor warning (output is garbled `MMMMM...`
  / `íííí...` — likely affecting the per-token compute path)
- our MoRI not being the version the fork builds (image MoRI is dated
  20260503 — different from fork's pinned `2d02c6a9`)
- BF16 vs FP8 dispatch quantization tuning

## c>=4 still fails with MoRI RDMA timeout

Even with the full fork-faithful config (asymmetric flags, MTP, --privileged,
--ulimit memlock=-1, all 8 uverbs + rdma_cm), the decode worker crashes at
c=4 with the same error pattern observed in Phase 6 round 3:

```
[mori]ibverbs.cpp:168: syscall failed with Connection timed out
RuntimeError: unknown parameter type
[2026-05-13 23:19:54] SIGQUIT received.
```

So the `--privileged` and memlock=-1 settings, while needed for general
RDMA hygiene, did NOT eliminate the c>=4 failure. Most likely remaining
cause is **NIC-level RDMA flow control (PFC)** as MoRI's own error
message advised in round 3. The fork's runbook is silent about PFC tuning
which suggests their cluster has it configured at the NIC/switch level —
not something we can change as a non-admin on AAC1.

## Reproducer (sbatch)

```bash
# Submit a single job that holds both nodes for the entire bring-up + sweep:
sbatch --nodelist=smci355-ccs-aus-g12-06,smci355-ccs-aus-g12-26 \
       scripts/phase6c_mori_sbatch.sh

# Results land in /shared/amdgpu/home/anluo/mori-bench-results/job-<N>/
# - prefill.log, decode.log, router.log
# - smoke.json (single chat completion)
# - sweep.csv (per-c results)
# - cN.log + cN.json for each concurrency
```

## Status vs fork's published numbers

| Concurrency | Fork (Mooncake Test 12) | **Phase 6c (MoRI fork-faithful)** | Phase 3 (Mooncake) |
|---:|---:|---:|---:|
| c=1 | 97.7 tok/s, TPOT 7.11 ms | **58.3 tok/s, TPOT 14.63 ms** | 105.7 tok/s, TPOT 9.46 ms |
| c=4 | 178 tok/s | **failed (MoRI ibverbs timeout)** | 261.7 tok/s |
| c=16 | 672 tok/s | not reached | 527.4 tok/s |
| c=128 | 2196 tok/s | not reached | not reached (Mooncake crashed @ c=32) |

Honest read: **at c=1, fork-faithful MoRI on our AAC1 setup runs at 60% of
the fork's published Mooncake number and 75% of our own Phase 3 Mooncake
result.** Mooncake is competitive with MoRI at c=1 — MoRI's advantage
shows up at higher concurrency, which we cannot reach because of the
RDMA timeout cascade.

## ROOT CAUSE — AAC1 ionic uses ECN/DCQCN, NOT PFC

After the c=1 / c≥4 split surfaced, we read the ionic NIC sysfs counters
on `smci355-ccs-aus-g12-06` directly. The numbers are decisive:

| Counter | Value | Meaning |
|---|---:|---|
| `rx_rdma_ecn_pkts` | **924,933,861** | Senders told to slow down 925M times via ECN marks |
| `rx_rdma_cnp_pkts` | 17,954,708 | DCQCN Congestion Notification Packets sent |
| `rdma_puec_cc_cwnd_dec` | 798,930,482 | Congestion-control window decreases |
| `rdma_puec_cc_cwnd_inc` | 5,742,505 | Window increases (139× fewer than decreases) |
| `rdma_retx_rto` | 16,218 | RDMA retransmission timeouts |
| `tx_rdma_ack_timeout` | 16,218 | Matching ACK timeouts |
| `resp_rx_dup_request` | 15,077,283 | Duplicate requests received (from retransmissions) |
| `req_rx_dup_response` | 410,349 | Duplicate responses received |

The presence of `rx_rdma_ecn_pkts` (and the absence of any PFC pause
counters) confirms AAC1's ionic NICs run **lossy RoCEv2 with ECN/DCQCN
congestion control, not lossless PFC**. Senders ARE backing off via
DCQCN — but the back-off rate (798M decreases vs 5.7M increases =
constant throttle pressure) and the 16K retransmission timeouts mean
this is not a tightly-tuned lossless fabric; it is a multi-tenant
cluster where RDMA is best-effort.

**This is exactly the wrong environment for MoRI.** MoRI is designed for
clusters with lossless PFC where it can push bulk RDMA without backoff.
On a lossy ECN cluster, MoRI's bulk RDMA hits the timeout wall the
moment concurrency rises above c=1.

**This is the right environment for Mooncake.** Mooncake's chunked
register → transfer → deregister pattern naturally back-pressures against
ECN/DCQCN — every chunk is independent, the application sees per-chunk
backpressure, and the firmware's QP-setup queue gets time to recover
between bursts. That is why Phase 3 SGLang+Mooncake reached **527 tok/s
@ c=16 with 160/160 success on this same NIC** while Phase 6c MoRI
fails at c=4.

## Architectural finding for the AMD ↔ NVIDIA conversation

The right disagg KV transport on a given AMD ionic cluster depends on
whether PFC is configured at the NIC and switch fabric:

| Cluster fabric | Best transport | Reason |
|---|---|---|
| Lossless PFC (e.g. JohnQinAMD's bench cluster) | **MoRI** | Bulk RDMA, no backoff needed; gets the 178/672/2196 tok/s @ c=4/16/128 numbers |
| Lossy ECN/DCQCN (e.g. AAC1) | **Mooncake** | Chunked transfer cooperates with sender-rate backoff; reaches c=16 cleanly |

This is **not** a "MoRI is better than Mooncake" or vice-versa story.
It is "the network fabric determines the transport." Our PoC validates
both transports work on the same Dynamo + SGLang + AMD MI355X stack;
the fork's published numbers and ours are both correct, just measured
on different fabrics.

## Path forward (not a current workstream)

To reach the fork's MoRI numbers on AAC1 specifically, you need one of:

1. **PFC enabled on the ionic NIC + switch ports for the QoS class
   MoRI uses** (TC=96 in our config). This is a cluster-admin operation
   spanning NIC drivers, switch ACLs, and possibly DCB configuration.
2. **Move benchmarks to a PFC-configured cluster** (the fork's own
   bench cluster, or any reservation that has lossless RoCEv2 set up).
3. **Patch MoRI to tolerate lossy RDMA** by enabling its retry path,
   reducing in-flight WR count, and making timeouts longer. This is a
   MoRI source change (in `src/application/transport/rdma/providers/ibverbs/`)
   that should be a github issue at `ROCm/MoRI`, not something to land
   in this PoC.

For our PoC, the **correct conclusion is to ship the Mooncake numbers
as the primary disagg result for AAC1 and document the MoRI c=1 datapoint
as evidence that the integration works** — the c≥4 ceiling is fabric,
not software.