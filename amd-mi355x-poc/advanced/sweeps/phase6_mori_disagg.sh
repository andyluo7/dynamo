#!/bin/bash
# Phase 6: SGLang 1P1D disagg with MoRI-IO transport (DSR1 TP=8 EP=8 DP=8 + DP-Attn).
# Targets fork's published MoRI numbers: 178 tok/s @ c=4, 672 @ c=16, 2196 @ c=128.
# Bypasses dynamo wrapper to match fork's actual stack.
set -e

PREFILL_NODE=smci355-ccs-aus-g12-06
DECODE_NODE=smci355-ccs-aus-g12-26
PREFILL_IP=10.194.30.23
DECODE_IP=10.194.30.28  # g12-26
IMAGE=docker.io/rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503
HF_CACHE=/shared/amdgpu/home/anluo/.cache/huggingface
MODEL=deepseek-ai/DeepSeek-R1-0528
TP=8
BOOTSTRAP_PORT=30001

echo "[$(date)] === Phase 6: SGLang 1P1D + MoRI-IO ($MODEL TP=$TP EP=$TP DP=$TP + DP-Attn) ==="

for n in $PREFILL_NODE $DECODE_NODE; do
    ssh -o StrictHostKeyChecking=no $n "podman rm -f sglang-mori-prefill sglang-mori-decode sglang-mori 2>&1 | tail -1; pkill -9 -f 'sglang.launch' 2>&1; pkill -9 -f 'sglang.srt' 2>&1; sleep 5" || true
done

cat > /tmp/inside_p6_disagg.sh <<'INNER'
#!/bin/bash
set -e
ROLE=$1
echo "[$(date)] role=$ROLE model=$MODEL TP=$TP"

# === MoRI env (fork's amd_utils/env.sh) ===
export SGLANG_USE_AITER=1
export MORI_SHMEM_MODE=ISOLATION
export SGLANG_MORI_DISPATCH_DTYPE=bf16
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=4096
export SGLANG_MORI_FP8_DISP=True
export SGLANG_MORI_FP8_COMB=False
# Round 3: pull back QP/workers to fix c=8 OOM on decode
export SGLANG_MORI_QP_PER_TRANSFER=1
export SGLANG_MORI_NUM_WORKERS=2
export SGLANG_MORI_POST_BATCH_SIZE=-1
export MORI_MAX_DISPATCH_TOKENS_PREFILL=16384
export MORI_MAX_DISPATCH_TOKENS_DECODE=160
export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=320
export MORI_RDMA_TC=104  # try 104 (was 96; assertion fired on subsequent req)
export MORI_APP_LOG_LEVEL=INFO
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export MORI_IO_QP_MAX_SEND_WR=16384
export MORI_IO_QP_MAX_CQE=32768
export MORI_IO_QP_MAX_SGE=4

# Disagg timeouts
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

# Network: ionic NICs
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
# Auto-detect socket interface (host network in container; eth0 typically)
export NCCL_SOCKET_IFNAME=$(ip -o -4 route show to default | awk '{print $5}' | head -1)
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME

# HF offline
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "iface=$NCCL_SOCKET_IFNAME ibv_devinfo: $(ibv_devinfo 2>&1 | grep hca_id | wc -l)"

# Common SGLang flags
COMMON="--model-path $MODEL \
    --tp-size $TP --ep-size $TP --dp-size $TP \
    --enable-dp-attention \
    --moe-a2a-backend mori \
    --enable-two-batch-overlap \
    --trust-remote-code \
    --load-balance-method round_robin \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --mem-fraction-static 0.65 \
    --chunked-prefill-size 32768 \
    --max-running-requests 128 \
    --context-length 12288 \
    --attention-backend aiter \
    --cuda-graph-max-bs 32 \
    --kv-cache-dtype fp8_e4m3 \
    --disaggregation-transfer-backend mori \
    --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
    --disaggregation-ib-device ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 \
    --host 0.0.0.0"

if [[ "$ROLE" == "prefill" ]]; then
    echo "[$(date)] launching PREFILL on :8001"
    python3 -m sglang.launch_server $COMMON \
        --disaggregation-mode prefill \
        --port 8001 \
        > /tmp/sglang.log 2>&1 &
    S_PID=$!
elif [[ "$ROLE" == "decode" ]]; then
    echo "[$(date)] launching DECODE on :8002"
    python3 -m sglang.launch_server $COMMON \
        --disaggregation-mode decode \
        --port 8002 \
        > /tmp/sglang.log 2>&1 &
    S_PID=$!
fi

echo "sglang PID=$S_PID; wait up to 50 min"
PORT=$([[ "$ROLE" == "prefill" ]] && echo 8001 || echo 8002)
for i in $(seq 1 300); do
    sleep 10
    if ! kill -0 $S_PID 2>/dev/null; then
        echo "[$(date)] sglang died ~${i}0s; tail:"; tail -60 /tmp/sglang.log
        echo "STATUS=died"; sleep 7200; exit 1
    fi
    BODY=$(curl -sf http://localhost:$PORT/v1/models 2>/dev/null || true)
    if echo "$BODY" | grep -q '"id"'; then
        echo "[$(date)] $ROLE ready @ ${i}0s"; break
    fi
    (( i % 6 == 0 )) && tail -2 /tmp/sglang.log
done

echo "[$(date)] $ROLE ready; sleeping 7200"
sleep 7200
INNER
chmod +x /tmp/inside_p6_disagg.sh
scp -o StrictHostKeyChecking=no /tmp/inside_p6_disagg.sh ${PREFILL_NODE}:/tmp/inside_p6_disagg.sh
scp -o StrictHostKeyChecking=no /tmp/inside_p6_disagg.sh ${DECODE_NODE}:/tmp/inside_p6_disagg.sh

COMMON_FLAGS="--device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add keep-groups --security-opt seccomp=unconfined \
  --network=host --ipc=host \
  -v $HF_CACHE:/root/.cache/huggingface \
  -v /tmp/inside_p6_disagg.sh:/tmp/inside_p6_disagg.sh:ro \
  -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:ro \
  -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
  -e MODEL=$MODEL -e TP=$TP -e BOOTSTRAP_PORT=$BOOTSTRAP_PORT \
  -e PREFILL_IP=$PREFILL_IP -e DECODE_IP=$DECODE_IP"

echo "[$(date)] launching PREFILL on $PREFILL_NODE"
ssh -o StrictHostKeyChecking=no $PREFILL_NODE "podman run -d --replace --name sglang-mori-prefill $COMMON_FLAGS --entrypoint bash $IMAGE /tmp/inside_p6_disagg.sh prefill"

echo "[$(date)] launching DECODE on $DECODE_NODE"
ssh -o StrictHostKeyChecking=no $DECODE_NODE "podman run -d --replace --name sglang-mori-decode $COMMON_FLAGS --entrypoint bash $IMAGE /tmp/inside_p6_disagg.sh decode"

echo "containers up"
