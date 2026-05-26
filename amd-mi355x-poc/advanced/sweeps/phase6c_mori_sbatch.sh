#!/bin/bash
#SBATCH --job-name=mori-bench
#SBATCH --partition=256C8G1H_MI355X_Ubuntu22
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=8:00:00
#SBATCH --output=/shared/amdgpu/home/anluo/mori-bench-results/slurm-%j.out
#SBATCH --error=/shared/amdgpu/home/anluo/mori-bench-results/slurm-%j.err
#
# Phase 6c: single-allocation sbatch that holds BOTH MoRI nodes for the full
# bring-up + sweep duration (no SLURM preemption mid-sweep). Equivalent of
# the fork's job.slurm but stripped down to AAC1 + podman + bind-mounted
# results.
#
# Submit with: sbatch --nodelist=smci355-ccs-aus-g12-06,smci355-ccs-aus-g12-26 phase6c_mori_sbatch.sh

set -e
RESULTS_DIR=/shared/amdgpu/home/anluo/mori-bench-results
mkdir -p $RESULTS_DIR/job-$SLURM_JOB_ID

echo "[$(date)] === Phase 6c sbatch: jobid=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST ==="

# Pick nodes from allocation
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
PREFILL_NODE=${NODES[0]}
DECODE_NODE=${NODES[1]}
PREFILL_IP=$(srun --nodes=1 --ntasks=1 --nodelist=$PREFILL_NODE bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7}'")
DECODE_IP=$(srun  --nodes=1 --ntasks=1 --nodelist=$DECODE_NODE  bash -c "ip route get 1.1.1.1 | awk '/src/ {print \$7}'")
echo "[$(date)] PREFILL=$PREFILL_NODE ($PREFILL_IP)  DECODE=$DECODE_NODE ($DECODE_IP)"

IMAGE=docker.io/rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503
HF_CACHE=/shared/amdgpu/home/anluo/.cache/huggingface
MODEL=deepseek-ai/DeepSeek-R1-0528
TP=8
DECODE_MTP_SIZE=1

# === inside-container script (role-aware, fork-faithful asymmetric flags) ===
cat > $RESULTS_DIR/job-$SLURM_JOB_ID/inside_p6c.sh <<'INNER'
#!/bin/bash
set -e
ROLE=$1
echo "[$(date)] role=$ROLE model=$MODEL TP=$TP MTP=$DECODE_MTP_SIZE"

export PYTHONDONTWRITEBYTECODE=1
export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export NCCL_IB_HCA=$IBDEVICES
export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -1)
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME

export SGLANG_USE_AITER=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200
export MORI_SHMEM_MODE=ISOLATION
export SGLANG_MORI_FP8_DISP=True
export SGLANG_MORI_FP4_DISP=False
export SGLANG_MORI_FP8_COMB=False
export MORI_MAX_DISPATCH_TOKENS_PREFILL=16384
export MORI_MAX_DISPATCH_TOKENS_DECODE=160
export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export MORI_IO_QP_MAX_SEND_WR=16384
export MORI_IO_QP_MAX_CQE=32768
export MORI_IO_QP_MAX_SGE=4
export MORI_APP_LOG_LEVEL=INFO
export MORI_RDMA_TC=96
export PYTHONPATH=/sgl-workspace/aiter:${PYTHONPATH:-}
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

BASE_FLAGS="--decode-log-interval 1000 --log-level warning --watchdog-timeout 3600
            --ep-dispatch-algorithm fake --load-balance-method round_robin
            --kv-cache-dtype fp8_e4m3 --attention-backend aiter
            --disaggregation-transfer-backend mori"
DP_FLAGS="--moe-a2a-backend mori --deepep-mode normal --enable-dp-attention
          --moe-dense-tp-size 1 --enable-dp-lm-head"

if [[ "$ROLE" == "prefill" ]]; then
    export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$MORI_MAX_DISPATCH_TOKENS_PREFILL
    PREFILL_CHUNK=$((MORI_MAX_DISPATCH_TOKENS_PREFILL * TP))
    python3 -m sglang.launch_server \
        --model-path $MODEL --tp-size $TP --ep-size $TP --dp-size $TP \
        $BASE_FLAGS $DP_FLAGS \
        --mem-fraction-static 0.8 \
        --max-running-requests 24 \
        --chunked-prefill-size $PREFILL_CHUNK \
        --cuda-graph-bs 1 2 3 \
        --disable-radix-cache \
        --disaggregation-mode prefill \
        --disaggregation-ib-device $IBDEVICES \
        --trust-remote-code \
        --host 0.0.0.0 --port 8000 --log-level-http warning 2>&1 | tee /results/prefill.log &
    S_PID=$!
elif [[ "$ROLE" == "decode" ]]; then
    export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$((MORI_MAX_DISPATCH_TOKENS_DECODE * (DECODE_MTP_SIZE + 1)))
    MTP_FLAGS="--speculative-algorithm NEXTN --speculative-eagle-topk 1
               --speculative-num-steps $DECODE_MTP_SIZE
               --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"
    DECODE_CUDA_GRAPH_BS=$(seq -s ' ' 1 160)
    python3 -m sglang.launch_server \
        --model-path $MODEL --tp-size $TP --ep-size $TP --dp-size $TP \
        $BASE_FLAGS $DP_FLAGS $MTP_FLAGS \
        --mem-fraction-static 0.85 \
        --max-running-requests 4096 \
        --cuda-graph-bs $DECODE_CUDA_GRAPH_BS \
        --prefill-round-robin-balance \
        --disaggregation-mode decode \
        --disaggregation-ib-device $IBDEVICES \
        --trust-remote-code \
        --host 0.0.0.0 --port 8000 --log-level-http warning 2>&1 | tee /results/decode.log &
    S_PID=$!
fi
sleep 7200
INNER
chmod +x $RESULTS_DIR/job-$SLURM_JOB_ID/inside_p6c.sh

# === podman flags (fork's job.slurm container settings) ===
PFLAGS="--device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --device /dev/infiniband/rdma_cm \
  --device /dev/infiniband/uverbs0 --device /dev/infiniband/uverbs1 \
  --device /dev/infiniband/uverbs2 --device /dev/infiniband/uverbs3 \
  --device /dev/infiniband/uverbs4 --device /dev/infiniband/uverbs5 \
  --device /dev/infiniband/uverbs6 --device /dev/infiniband/uverbs7 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host --ipc host \
  --group-add keep-groups --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  -v $HF_CACHE:/root/.cache/huggingface \
  -v $RESULTS_DIR/job-$SLURM_JOB_ID:/results \
  -v $RESULTS_DIR/job-$SLURM_JOB_ID/inside_p6c.sh:/tmp/inside_p6c.sh:ro \
  -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:ro \
  -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
  -e MODEL=$MODEL -e TP=$TP -e DECODE_MTP_SIZE=$DECODE_MTP_SIZE"

# Launch both containers in background via srun (one per node)
echo "[$(date)] launching PREFILL + DECODE containers"
srun --nodes=1 --ntasks=1 --nodelist=$PREFILL_NODE bash -c \
    "podman rm -f sglang-mori-p 2>/dev/null; \
     podman run -d --replace --name sglang-mori-p $PFLAGS --entrypoint bash $IMAGE /tmp/inside_p6c.sh prefill" &
srun --nodes=1 --ntasks=1 --nodelist=$DECODE_NODE bash -c \
    "podman rm -f sglang-mori-d 2>/dev/null; \
     podman run -d --replace --name sglang-mori-d $PFLAGS --entrypoint bash $IMAGE /tmp/inside_p6c.sh decode" &
wait

# Wait until both /v1/models endpoints answer
echo "[$(date)] waiting for both servers to be ready (up to 30 min)"
for i in $(seq 1 180); do
    sleep 10
    P_OK=$(srun --nodes=1 --ntasks=1 --nodelist=$PREFILL_NODE curl -sf -m 3 http://$PREFILL_IP:8000/v1/models 2>/dev/null | grep -c '"id"' || true)
    D_OK=$(srun --nodes=1 --ntasks=1 --nodelist=$DECODE_NODE curl -sf -m 3 http://$DECODE_IP:8000/v1/models 2>/dev/null | grep -c '"id"' || true)
    if [[ "$P_OK" -ge 1 && "$D_OK" -ge 1 ]]; then
        echo "[$(date)] BOTH READY @ ${i}0s"; break
    fi
    (( i % 6 == 0 )) && echo "[$(date)] readiness wait ${i}0s P_OK=$P_OK D_OK=$D_OK"
done

# Start router on prefill node
echo "[$(date)] starting PD router on prefill node :30000"
srun --nodes=1 --ntasks=1 --nodelist=$PREFILL_NODE bash -c \
  "podman exec -d sglang-mori-p bash -c 'nohup python3 -m sglang_router.launch_router \
     --pd-disaggregation --mini-lb --policy random \
     --prefill http://$PREFILL_IP:8000 --decode http://$DECODE_IP:8000 \
     --port 30000 --host 0.0.0.0 > /results/router.log 2>&1 &'"
sleep 8

# Smoke test (curl from prefill bare node, write to host bind-mounted dir)
echo "[$(date)] smoke test through router"
srun --nodes=1 --ntasks=1 --nodelist=$PREFILL_NODE bash -c "curl -sf -m 60 \
  http://$PREFILL_IP:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":12,\"temperature\":0}'" \
  > $RESULTS_DIR/job-$SLURM_JOB_ID/smoke.json 2>&1 || true
echo "smoke result:"; head -2 $RESULTS_DIR/job-$SLURM_JOB_ID/smoke.json

# Sweep: results go to BIND-MOUNTED /results
echo "[$(date)] starting sweep (results -> $RESULTS_DIR/job-$SLURM_JOB_ID/)"
srun --nodes=1 --ntasks=1 --nodelist=$PREFILL_NODE bash -c \
  "podman exec sglang-mori-p bash -c '
   set -uo pipefail
   export PYTHONPATH=/sgl-workspace/sglang/python:\${PYTHONPATH:-}
   RES=/results/sweep.csv
   echo \"conc,n_prompts,output_throughput,input_throughput,total_throughput,median_ttft,median_tpot,median_e2e_latency,p99_ttft,p99_tpot\" > \$RES
   for c in 1 4 8 16 32 64 128; do
       n=\$((c * 10))
       LOG=/results/c\${c}.log
       echo \"=== c=\$c ===\"
       timeout 1800 python3 -m sglang.bench_serving \
           --backend sglang \
           --base-url http://$PREFILL_IP:30000 \
           --dataset-name random --random-input-len 1024 --random-output-len 1024 \
           --random-range-ratio 0.8 \
           --num-prompts \$n --max-concurrency \$c --request-rate inf \
           --output-file /results/c\${c}.json > \$LOG 2>&1
       echo \"exit=\$?\"
       o=\$(grep -oP '\''Output token throughput \\(tok/s\\):\\s+\\K[0-9.]+'\'' \$LOG | head -1)
       i=\$(grep -oP '\''Input token throughput \\(tok/s\\):\\s+\\K[0-9.]+'\'' \$LOG | head -1)
       t=\$(grep -oP '\''Total token throughput \\(tok/s\\):\\s+\\K[0-9.]+'\'' \$LOG | head -1)
       mtt=\$(grep -oP '\''Median TTFT \\(ms\\):\\s+\\K[0-9.]+'\'' \$LOG | head -1)
       mtp=\$(grep -oP '\''Median TPOT \\(ms\\):\\s+\\K[0-9.]+'\'' \$LOG | head -1)
       me=\$(grep -oP '\''Median E2E Latency \\(ms\\):\\s+\\K[0-9.]+'\'' \$LOG | head -1)
       p9t=\$(grep -oP '\''P99 TTFT \\(ms\\):\\s+\\K[0-9.]+'\'' \$LOG | head -1)
       p9p=\$(grep -oP '\''P99 TPOT \\(ms\\):\\s+\\K[0-9.]+'\'' \$LOG | head -1)
       echo \"\$c,\$n,\${o:-na},\${i:-na},\${t:-na},\${mtt:-na},\${mtp:-na},\${me:-na},\${p9t:-na},\${p9p:-na}\" >> \$RES
   done
   echo === FINAL CSV ===
   cat \$RES
  '" 2>&1 | tee $RESULTS_DIR/job-$SLURM_JOB_ID/sweep_orchestrator.log

echo "[$(date)] done. results in $RESULTS_DIR/job-$SLURM_JOB_ID/"

# Cleanup containers
srun --nodes=1 --ntasks=1 --nodelist=$PREFILL_NODE podman rm -f sglang-mori-p 2>&1 | tail -1 || true
srun --nodes=1 --ntasks=1 --nodelist=$DECODE_NODE  podman rm -f sglang-mori-d 2>&1 | tail -1 || true
