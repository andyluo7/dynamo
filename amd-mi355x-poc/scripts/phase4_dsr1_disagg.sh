#!/bin/bash
# Phase 4 cross-validation: vLLM 1P1D disagg with DSR1 + RIXL/UCX-ROCm.
# Same stack as phase4_m25_disagg.sh but with DeepSeek-R1-0528 FP8 (671B) and TP=8.
# Tests whether RIXL handles DSR1 (61 layers, big KV per token) past c=16
# where SGLang+Mooncake crashed on the same model + same hardware.
set -e

PREFILL_NODE=smci355-ccs-aus-g12-06
DECODE_NODE=smci355-ccs-aus-g12-30
PREFILL_IP=10.194.30.23
IMAGE=localhost/dynamo-vllm-rixl:latest
HF_CACHE=/shared/amdgpu/home/anluo/.cache/huggingface
MODEL=deepseek-ai/DeepSeek-R1-0528
TP=8

echo "[$(date)] === Phase 4-DSR1: vLLM 1P1D disagg ($MODEL TP=$TP, RIXL+UCX-ROCm) ==="

# Cleanup: assumes external pre-cleanup (we just had M2.5 running)
ssh -o StrictHostKeyChecking=no $PREFILL_NODE "podman rm -f dynamo-vllm-prefill 2>&1 | tail -1; pkill -9 -f 'VLLM::Worker' 2>&1; pkill -9 -f 'vllm.v1' 2>&1; sleep 5" || true
ssh -o StrictHostKeyChecking=no $DECODE_NODE "podman rm -f dynamo-vllm-decode 2>&1 | tail -1; pkill -9 -f 'VLLM::Worker' 2>&1; pkill -9 -f 'vllm.v1' 2>&1; sleep 5" || true

cat > /tmp/inside_p4_dsr1.sh <<'INNER'
#!/bin/bash
set -e
ROLE=$1
echo "[$(date)] role=$ROLE model=$MODEL TP=$TP"

SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")

# apt update + install for rdma-core etc.
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq --no-install-recommends rdma-core ibverbs-providers iproute2 gcc 2>&1 | tail -1

# nixl→rixl shim (image already has it but just in case)
mkdir -p $SITE/nixl
echo "from rixl import *"           > $SITE/nixl/__init__.py
echo "from rixl._api import *"      > $SITE/nixl/_api.py
echo "from rixl._bindings import *" > $SITE/nixl/_bindings.py

pip install --quiet --break-system-packages --no-deps ai-dynamo==1.1.1 ai-dynamo-runtime==1.1.1 \
    blake3 kubernetes msgpack msgspec prometheus-client pyzmq 2>&1 | tail -1

# LD_PRELOAD interposer (strips IBV_ACCESS_REMOTE_ATOMIC for ionic compat)
cat > /tmp/ibv_ionic_compat.c <<'CEOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define IBV_ACCESS_REMOTE_ATOMIC 0x8
void *ibv_reg_mr(void *pd, void *addr, size_t length, int access) {
    typedef void *(*fn_t)(void *, void *, size_t, int);
    static fn_t real_fn;
    if (__builtin_expect(!real_fn, 0)) { real_fn = (fn_t)dlsym(RTLD_NEXT, "ibv_reg_mr"); if (!real_fn) abort(); }
    access &= ~IBV_ACCESS_REMOTE_ATOMIC;
    return real_fn(pd, addr, length, access);
}
void *ibv_reg_mr_iova2(void *pd, void *addr, size_t length, uint64_t iova, int access) {
    typedef void *(*fn_t)(void *, void *, size_t, uint64_t, int);
    static fn_t real_fn;
    if (__builtin_expect(!real_fn, 0)) { real_fn = (fn_t)dlsym(RTLD_NEXT, "ibv_reg_mr_iova2"); if (!real_fn) abort(); }
    access &= ~IBV_ACCESS_REMOTE_ATOMIC;
    return real_fn(pd, addr, length, iova, access);
}
void *ibv_reg_dmabuf_mr(void *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access) {
    typedef void *(*fn_t)(void *, uint64_t, size_t, uint64_t, int, int);
    static fn_t real_fn;
    if (__builtin_expect(!real_fn, 0)) { real_fn = (fn_t)dlsym(RTLD_NEXT, "ibv_reg_dmabuf_mr"); if (!real_fn) return NULL; }
    access &= ~IBV_ACCESS_REMOTE_ATOMIC;
    return real_fn(pd, offset, length, iova, fd, access);
}
CEOF
gcc -shared -fPIC -O2 -o /tmp/ibv_ionic_compat.so /tmp/ibv_ionic_compat.c -ldl
export LD_PRELOAD="/tmp/ibv_ionic_compat.so${LD_PRELOAD:+:$LD_PRELOAD}"
echo "ibv_devinfo count: $(ibv_devinfo 2>&1 | grep hca_id | wc -l)"

MY_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
export VLLM_NIXL_SIDE_CHANNEL_HOST=$MY_IP
echo "mgmt IP=$MY_IP"

if [[ "$ROLE" == "prefill" ]]; then
    echo "[$(date)] launching FRONTEND on :8000"
    python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin > /tmp/frontend.log 2>&1 &
    sleep 3
    echo "[$(date)] launching vLLM PREFILL worker (TP=$TP, HIP graphs ON)"
    export VLLM_NIXL_SIDE_CHANNEL_PORT=20097
    python3 -m dynamo.vllm \
        --model $MODEL --tensor-parallel-size $TP \
        --max-model-len 4096 --max-num-seqs 4 \
        --gpu-memory-utilization 0.65 --trust-remote-code \
        --disaggregation-mode prefill \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
        > /tmp/worker.log 2>&1 &
    W_PID=$!
elif [[ "$ROLE" == "decode" ]]; then
    echo "[$(date)] launching vLLM DECODE worker (TP=$TP, HIP graphs ON)"
    export VLLM_NIXL_SIDE_CHANNEL_PORT=20098
    python3 -m dynamo.vllm \
        --model $MODEL --tensor-parallel-size $TP \
        --max-model-len 4096 --max-num-seqs 4 \
        --gpu-memory-utilization 0.65 --trust-remote-code \
        --disaggregation-mode decode \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
        > /tmp/worker.log 2>&1 &
    W_PID=$!
fi

echo "worker PID=$W_PID"
echo "[$(date)] waiting up to 35 min for readiness (DSR1 = ~5 min NFS load + HIP graphs)"
for i in $(seq 1 210); do
    sleep 10
    if ! kill -0 $W_PID 2>/dev/null; then
        echo "[$(date)] worker died ~${i}0s; tail:"; tail -50 /tmp/worker.log
        echo "STATUS=worker-died"; sleep 7200; exit 1
    fi
    if [[ "$ROLE" == "prefill" ]]; then
        BODY=$(curl -sf http://localhost:8000/v1/models 2>/dev/null || true)
        N=$(echo "$BODY" | python3 -c 'import sys,json;print(len(json.load(sys.stdin).get("data",[])))' 2>/dev/null || echo 0)
        if [[ "$N" -ge 1 ]]; then echo "[$(date)] prefill ready @ ${i}0s"; break; fi
    fi
    (( i % 6 == 0 )) && tail -2 /tmp/worker.log
done

echo "[$(date)] $ROLE setup complete; sleeping 7200"
sleep 7200
INNER
chmod +x /tmp/inside_p4_dsr1.sh
scp -o StrictHostKeyChecking=no /tmp/inside_p4_dsr1.sh ${PREFILL_NODE}:/tmp/inside_p4_dsr1.sh
scp -o StrictHostKeyChecking=no /tmp/inside_p4_dsr1.sh ${DECODE_NODE}:/tmp/inside_p4_dsr1.sh

ssh -o StrictHostKeyChecking=no $PREFILL_NODE "podman ps --format '{{.Names}}' | grep -q '^dynamo-etcd$' || podman run -d --replace --name dynamo-etcd --network=host quay.io/coreos/etcd:v3.5.21 etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://${PREFILL_IP}:2379 >/dev/null
podman ps --format '{{.Names}}' | grep -q '^dynamo-nats$' || podman run -d --replace --name dynamo-nats --network=host docker.io/library/nats:2.10.28 -p 4222 -js >/dev/null
podman exec dynamo-etcd etcdctl --endpoints=http://localhost:2379 del --prefix dynamo/ 2>&1 | tail -1"

# Note: NO HIP_VISIBLE_DEVICES set — vLLM uses all 8 GPUs for TP=8
COMMON_FLAGS="--device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add keep-groups --security-opt seccomp=unconfined \
  --network=host --ipc=host \
  -v $HF_CACHE:/root/.cache/huggingface \
  -v /tmp/inside_p4_dsr1.sh:/tmp/inside_p4_dsr1.sh:ro \
  -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:ro \
  -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
  -e HF_HOME=/root/.cache/huggingface \
  -e MODEL=$MODEL -e TP=$TP \
  -e ETCD_ENDPOINTS=http://${PREFILL_IP}:2379 \
  -e NATS_SERVER=nats://${PREFILL_IP}:4222 \
  -e UCX_TLS=rc_v,tcp,rocm,rocm_copy,rocm_ipc,self,sm \
  -e UCX_MEMTYPE_CACHE=y \
  -e UCX_LOG_LEVEL=warn \
  -e VLLM_ROCM_USE_AITER=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1"

echo "[$(date)] launching PREFILL on $PREFILL_NODE"
ssh -o StrictHostKeyChecking=no $PREFILL_NODE "podman run -d --replace --name dynamo-vllm-prefill $COMMON_FLAGS --entrypoint bash $IMAGE /tmp/inside_p4_dsr1.sh prefill"

echo "[$(date)] launching DECODE on $DECODE_NODE"
ssh -o StrictHostKeyChecking=no $DECODE_NODE "podman run -d --replace --name dynamo-vllm-decode $COMMON_FLAGS --entrypoint bash $IMAGE /tmp/inside_p4_dsr1.sh decode"

echo "containers up"
