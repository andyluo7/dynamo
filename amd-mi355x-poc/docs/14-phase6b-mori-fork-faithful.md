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

## Status vs fork's published numbers

Still **not measured**. The Phase 6 round 1-3 doc (doc 13) has the only
numbers we have so far (c=1 = 24.8 tok/s, c=4 = 96.1 tok/s) and those were
NOT with the fork-faithful config. Re-running the sweep with the
fork-faithful config + bind-mounted results dir is the next step.