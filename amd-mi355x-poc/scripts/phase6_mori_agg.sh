#!/bin/bash
# Phase 6 smoke: SGLang + MoRI-EP single-node DSR1 (TP=8 EP=8 DP=8 + DP-Attn).
# Validates MoRI-EP setup before moving to 2-node disagg with MoRI-IO.
set -e

NODE=smci355-ccs-aus-g12-06
IMAGE=docker.io/rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503
HF_CACHE=/shared/amdgpu/home/anluo/.cache/huggingface
MODEL=deepseek-ai/DeepSeek-R1-0528

echo "[$(date)] === Phase 6 SMOKE: SGLang+MoRI-EP single-node ($MODEL TP=8/EP=8/DP=8) ==="

ssh -o StrictHostKeyChecking=no $NODE "podman rm -f sglang-mori 2>&1 | tail -1; pkill -9 -f 'sglang.srt' 2>&1; pkill -9 -f 'sglang.launch' 2>&1; sleep 5" || true

cat > /tmp/inside_p6_mori.sh <<'INNER'
#!/bin/bash
set -e
echo "[$(date)] inside container, model=$MODEL"

# MoRI env (from fork's amd_utils/env.sh)
export SGLANG_USE_AITER=1
export MORI_SHMEM_MODE=ISOLATION
export SGLANG_MORI_DISPATCH_DTYPE=bf16
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=4096
export MORI_RDMA_TC=96   # smci355-ccs-aus-* nodes per fork's runbook
export MORI_APP_LOG_LEVEL=INFO
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export MORI_IO_QP_MAX_SEND_WR=16384
export MORI_IO_QP_MAX_CQE=32768
export MORI_IO_QP_MAX_SGE=4

# NCCL ionic config (single-node so less critical but harmless)
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7

# HF offline
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[$(date)] launching sglang.launch_server"
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --tp-size 8 \
    --ep-size 8 \
    --dp-size 8 \
    --enable-dp-attention \
    --moe-a2a-backend mori \
    --trust-remote-code \
    --load-balance-method round_robin \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --mem-fraction-static 0.72 \
    --chunked-prefill-size 32768 \
    --max-running-requests 128 \
    --context-length 12288 \
    --attention-backend aiter \
    --cuda-graph-max-bs 32 \
    --host 0.0.0.0 \
    --port 8000 \
    > /tmp/sglang.log 2>&1 &
S_PID=$!
echo "sglang PID=$S_PID"

echo "[$(date)] waiting up to 50 min for readiness (DSR1 cold cache + MoRI JIT)"
for i in $(seq 1 300); do
    sleep 10
    if ! kill -0 $S_PID 2>/dev/null; then
        echo "[$(date)] sglang died ~${i}0s; tail:"
        tail -50 /tmp/sglang.log
        echo "STATUS=died"; sleep 7200; exit 1
    fi
    BODY=$(curl -sf http://localhost:8000/v1/models 2>/dev/null || true)
    if echo "$BODY" | grep -q '"id"'; then
        echo "[$(date)] sglang ready @ ${i}0s"
        # smoke test: one chat completion
        echo "[$(date)] smoke chat completion:"
        curl -sf http://localhost:8000/v1/chat/completions \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}],\"max_tokens\":8,\"temperature\":0}" \
            2>&1 | head -3
        break
    fi
    (( i % 6 == 0 )) && tail -2 /tmp/sglang.log
done

echo "[$(date)] smoke complete; sleeping 7200"
sleep 7200
INNER
chmod +x /tmp/inside_p6_mori.sh
scp -o StrictHostKeyChecking=no /tmp/inside_p6_mori.sh ${NODE}:/tmp/inside_p6_mori.sh

echo "[$(date)] launching SGLang+MoRI container on $NODE"
ssh -o StrictHostKeyChecking=no $NODE "podman run -d --replace --name sglang-mori \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add keep-groups --security-opt seccomp=unconfined \
  --network=host --ipc=host \
  -v $HF_CACHE:/root/.cache/huggingface \
  -v /tmp/inside_p6_mori.sh:/tmp/inside_p6_mori.sh:ro \
  -e MODEL=$MODEL \
  --entrypoint bash $IMAGE /tmp/inside_p6_mori.sh"

echo "container up"
