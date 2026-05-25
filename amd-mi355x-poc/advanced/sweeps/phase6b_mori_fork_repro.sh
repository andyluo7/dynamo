#!/bin/bash
# Phase 6b: Faithful reproduction of the fork's MoRI benchmark.
# Applies the JohnQinAMD/InferenceX runner config exactly:
#   - container: --privileged + --ulimit memlock=-1 + all uverbs + rdma_cm
#   - asymmetric prefill/decode flags from models.yaml DeepSeek-R1-0528 entry
#   - MTP/NEXTN speculative decoding (decode side)
#   - sglang.bench_serving as the harness (--backend openai, ISL/OSL/conc/etc.)
#   - sglang_router.launch_router as the PD load balancer
set -e

PREFILL_NODE=smci355-ccs-aus-g12-06
DECODE_NODE=smci355-ccs-aus-g12-26
PREFILL_IP=10.194.30.23
DECODE_IP=10.194.30.28
IMAGE=docker.io/rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503
HF_CACHE=/shared/amdgpu/home/anluo/.cache/huggingface
MODEL=deepseek-ai/DeepSeek-R1-0528
TP=8
DECODE_MTP_SIZE=1   # NEXTN draft for DSR1

echo "[$(date)] === Phase 6b: SGLang+MoRI fork-faithful reproduction ($MODEL TP=$TP) ==="

for n in $PREFILL_NODE $DECODE_NODE; do
    ssh -o StrictHostKeyChecking=no $n "podman rm -f sglang-mori-prefill sglang-mori-decode 2>&1 | tail -1; pkill -9 -f 'sglang' 2>&1; sleep 5" || true
done

# Inside-container script - role-aware (from server.sh, simplified to skip SLURM-isms)
cat > /tmp/inside_p6b_mori.sh <<'INNER'
#!/bin/bash
set -e
ROLE=$1
echo "[$(date)] role=$ROLE model=$MODEL TP=$TP MTP=$DECODE_MTP_SIZE"

# === env.sh equivalent ===
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

# Per-role dispatch limits from env.sh
export MORI_MAX_DISPATCH_TOKENS_PREFILL=16384
export MORI_MAX_DISPATCH_TOKENS_DECODE=160
export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))

export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export MORI_IO_QP_MAX_SEND_WR=16384
export MORI_IO_QP_MAX_CQE=32768
export MORI_IO_QP_MAX_SGE=4
export MORI_APP_LOG_LEVEL=INFO
export MORI_RDMA_TC=96   # smci355-ccs-aus-* per env.sh hostname rule
export PYTHONPATH=/sgl-workspace/aiter:${PYTHONPATH:-}

# HF offline
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "iface=$NCCL_SOCKET_IFNAME ibv_devinfo: $(ibv_devinfo 2>&1 | grep hca_id | wc -l)"

# === models.yaml DeepSeek-R1-0528 (DP variant) flags ===
BASE_FLAGS="--decode-log-interval 1000 --log-level warning --watchdog-timeout 3600
            --ep-dispatch-algorithm fake --load-balance-method round_robin
            --kv-cache-dtype fp8_e4m3 --attention-backend aiter
            --disaggregation-transfer-backend mori"
DP_FLAGS="--moe-a2a-backend mori --deepep-mode normal --enable-dp-attention
          --moe-dense-tp-size 1 --enable-dp-lm-head"

if [[ "$ROLE" == "prefill" ]]; then
    # Prefill (DP): mem-frac 0.8, max-running 24, chunked 16384*8=131072,
    #               cuda-graph-bs "1 2 3", --disable-radix-cache
    export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$MORI_MAX_DISPATCH_TOKENS_PREFILL
    PREFILL_CHUNK=$((MORI_MAX_DISPATCH_TOKENS_PREFILL * TP))
    echo "[$(date)] launching PREFILL on :8000"
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
        --host 0.0.0.0 --port 8000 \
        --log-level-http warning \
        > /tmp/sglang.log 2>&1 &
    S_PID=$!
elif [[ "$ROLE" == "decode" ]]; then
    # Decode (DP+MTP): mem-frac 0.85, max-running 4096, cuda-graph-bs 1-160,
    #                   --prefill-round-robin-balance, NEXTN MTP
    export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$((MORI_MAX_DISPATCH_TOKENS_DECODE * (DECODE_MTP_SIZE + 1)))
    MTP_FLAGS="--speculative-algorithm NEXTN --speculative-eagle-topk 1
               --speculative-num-steps $DECODE_MTP_SIZE
               --speculative-num-draft-tokens $((DECODE_MTP_SIZE + 1))"
    DECODE_CUDA_GRAPH_BS=$(seq -s ' ' 1 160)
    echo "[$(date)] launching DECODE on :8000 (MTP=$DECODE_MTP_SIZE)"
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
        --host 0.0.0.0 --port 8000 \
        --log-level-http warning \
        > /tmp/sglang.log 2>&1 &
    S_PID=$!
fi

echo "sglang PID=$S_PID; waiting up to 50 min"
for i in $(seq 1 300); do
    sleep 10
    if ! kill -0 $S_PID 2>/dev/null; then
        echo "[$(date)] sglang died ~${i}0s; tail:"; tail -60 /tmp/sglang.log
        echo "STATUS=died"; sleep 7200; exit 1
    fi
    BODY=$(curl -sf http://localhost:8000/v1/models 2>/dev/null || true)
    if echo "$BODY" | grep -q '"id"'; then
        echo "[$(date)] $ROLE ready @ ${i}0s"; break
    fi
    (( i % 6 == 0 )) && tail -2 /tmp/sglang.log
done

echo "[$(date)] $ROLE ready; sleeping 7200"
sleep 7200
INNER
chmod +x /tmp/inside_p6b_mori.sh
scp -o StrictHostKeyChecking=no /tmp/inside_p6b_mori.sh ${PREFILL_NODE}:/tmp/inside_p6b_mori.sh
scp -o StrictHostKeyChecking=no /tmp/inside_p6b_mori.sh ${DECODE_NODE}:/tmp/inside_p6b_mori.sh

# === podman flags equivalent to fork's docker run (job.slurm L346-412) ===
# Critical ones: --privileged, --ulimit memlock=-1, all uverbs, rdma_cm, shm-size 128G
PODMAN_FLAGS="--device /dev/kfd --device /dev/dri --device /dev/infiniband \
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
  -v /tmp/inside_p6b_mori.sh:/tmp/inside_p6b_mori.sh:ro \
  -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:ro \
  -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
  -e MODEL=$MODEL -e TP=$TP -e DECODE_MTP_SIZE=$DECODE_MTP_SIZE"

echo "[$(date)] launching PREFILL on $PREFILL_NODE"
ssh -o StrictHostKeyChecking=no $PREFILL_NODE "podman run -d --replace --name sglang-mori-prefill $PODMAN_FLAGS --entrypoint bash $IMAGE /tmp/inside_p6b_mori.sh prefill"

echo "[$(date)] launching DECODE on $DECODE_NODE"
ssh -o StrictHostKeyChecking=no $DECODE_NODE "podman run -d --replace --name sglang-mori-decode $PODMAN_FLAGS --entrypoint bash $IMAGE /tmp/inside_p6b_mori.sh decode"

echo "containers up"
