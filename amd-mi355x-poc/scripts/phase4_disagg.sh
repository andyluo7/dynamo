#!/bin/bash
# Phase 4: vLLM 1P1D disaggregated serving with NixlConnector (real RIXL).
# Run from login node.
set -e

PREFILL_NODE=smci355-ccs-aus-g12-22
DECODE_NODE=smci355-ccs-aus-g12-26
PREFILL_IP=10.194.30.27
IMAGE=localhost/dynamo-vllm-rixl:latest
HF_CACHE=/shared/amdgpu/home/anluo/.cache/huggingface  # has Qwen3-0.6B and DSR1; M2.5 is in inferencex-agentic-test/hf-cache (not used in Phase 4 yet)
MODEL=Qwen/Qwen3-0.6B   # match fork's Phase 4 reference test (de-risk; M2.5 escalation later)
TP=1

echo "[$(date)] === Phase 4: vLLM 1P1D disagg ($MODEL) ==="

for n in $PREFILL_NODE $DECODE_NODE; do
    ssh -o StrictHostKeyChecking=no $n "podman rm -f dynamo-vllm-prefill dynamo-vllm-decode dynamo-vllm-perf 2>&1 | tail -3" || true
done

# Save image from prefill (where it was built) into a tar; load on decode node.
# Skip if decode already has the image.
if ssh -o StrictHostKeyChecking=no $DECODE_NODE "podman images --format '{{.Repository}}' | grep -q '^localhost/dynamo-vllm-rixl$'"; then
    echo "[$(date)] image already on decode node; skipping copy"
else
    echo "[$(date)] copying image to decode node..."
    ssh -o StrictHostKeyChecking=no $PREFILL_NODE "rm -f /tmp/dynamo-vllm-rixl.tar; podman save -o /tmp/dynamo-vllm-rixl.tar localhost/dynamo-vllm-rixl:latest"
    scp -o StrictHostKeyChecking=no $PREFILL_NODE:/tmp/dynamo-vllm-rixl.tar /tmp/dynamo-vllm-rixl.tar
    scp -o StrictHostKeyChecking=no /tmp/dynamo-vllm-rixl.tar $DECODE_NODE:/tmp/
    ssh -o StrictHostKeyChecking=no $DECODE_NODE "podman load -i /tmp/dynamo-vllm-rixl.tar"
fi

# Inside-container script
cat > /tmp/inside_p4.sh <<'INNER'
#!/bin/bash
set -e
ROLE=$1
PEER_IP=$2  # for prefill: empty/0; for decode: prefill ip (info only)
echo "[$(date)] role=$ROLE PEER=$PEER_IP"

SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")

# Make ibverbs see ionic: install rdma-core + ibverbs-providers + iproute2
echo "[$(date)] apt update + install (idempotent)"
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq --no-install-recommends rdma-core ibverbs-providers iproute2 gcc 2>&1 | tail -1

# Build LD_PRELOAD interposer that strips IBV_ACCESS_REMOTE_ATOMIC from ibv_reg_mr.
# Required for ionic NICs (they reject access=0xf with EINVAL; UCX hardcodes
# UCP_FEATURE_AMO32|AMO64 which sets that bit; RIXL never uses atomic ops so
# stripping is safe). Source extracted from JohnQinAMD fork's nixl_rocm_staging.py.
cat > /tmp/ibv_ionic_compat.c <<'CEOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#define IBV_ACCESS_REMOTE_ATOMIC 0x8
void *ibv_reg_mr(void *pd, void *addr, size_t length, int access) {
    typedef void *(*fn_t)(void *, void *, size_t, int);
    static fn_t real_fn;
    if (__builtin_expect(!real_fn, 0)) {
        real_fn = (fn_t)dlsym(RTLD_NEXT, "ibv_reg_mr");
        if (!real_fn) abort();
    }
    int orig = access; access &= ~IBV_ACCESS_REMOTE_ATOMIC;
    void *r = real_fn(pd, addr, length, access);
    if (!r) fprintf(stderr, "[ibv_compat] ibv_reg_mr len=%zu access=%x->%x failed\n", length, orig, access);
    return r;
}
void *ibv_reg_mr_iova2(void *pd, void *addr, size_t length,
                        uint64_t iova, int access) {
    typedef void *(*fn_t)(void *, void *, size_t, uint64_t, int);
    static fn_t real_fn;
    if (__builtin_expect(!real_fn, 0)) {
        real_fn = (fn_t)dlsym(RTLD_NEXT, "ibv_reg_mr_iova2");
        if (!real_fn) abort();
    }
    int orig = access; access &= ~IBV_ACCESS_REMOTE_ATOMIC;
    void *r = real_fn(pd, addr, length, iova, access);
    if (!r) fprintf(stderr, "[ibv_compat] ibv_reg_mr_iova2 len=%zu access=%x->%x failed\n", length, orig, access);
    return r;
}
/* dmabuf-based MR registration (used for GPU VRAM via libdrm). Strip atomic too. */
void *ibv_reg_dmabuf_mr(void *pd, uint64_t offset, size_t length, uint64_t iova,
                        int fd, int access) {
    typedef void *(*fn_t)(void *, uint64_t, size_t, uint64_t, int, int);
    static fn_t real_fn;
    if (__builtin_expect(!real_fn, 0)) {
        real_fn = (fn_t)dlsym(RTLD_NEXT, "ibv_reg_dmabuf_mr");
        if (!real_fn) {
            fprintf(stderr, "[ibv_compat] ibv_reg_dmabuf_mr not in libibverbs\n");
            return NULL;
        }
    }
    int orig = access; access &= ~IBV_ACCESS_REMOTE_ATOMIC;
    void *r = real_fn(pd, offset, length, iova, fd, access);
    if (!r) fprintf(stderr, "[ibv_compat] ibv_reg_dmabuf_mr len=%zu fd=%d access=%x->%x failed\n", length, fd, orig, access);
    else    fprintf(stderr, "[ibv_compat] ibv_reg_dmabuf_mr len=%zu fd=%d access=%x->%x OK\n", length, fd, orig, access);
    return r;
}
CEOF
gcc -shared -fPIC -O2 -o /tmp/ibv_ionic_compat.so /tmp/ibv_ionic_compat.c -ldl
export LD_PRELOAD="/tmp/ibv_ionic_compat.so${LD_PRELOAD:+:$LD_PRELOAD}"
echo "[$(date)] LD_PRELOAD=$LD_PRELOAD"

# nixl→rixl Python shim (Dockerfile didn't include this in current build)
mkdir -p $SITE/nixl
echo "from rixl import *" > $SITE/nixl/__init__.py
echo "from rixl._api import *" > $SITE/nixl/_api.py
echo "from rixl._bindings import *" > $SITE/nixl/_bindings.py

# install ai-dynamo
pip install --quiet --break-system-packages --no-deps ai-dynamo==1.1.1 ai-dynamo-runtime==1.1.1 \
    blake3 kubernetes msgpack msgspec prometheus-client pyzmq 2>&1 | tail -1

echo "ibv_devinfo count: $(ibv_devinfo 2>&1 | grep hca_id | wc -l)"
python3 -c "import nixl; print('nixl OK', getattr(nixl, '__file__', '?'))" 2>&1 | tail -3

MY_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
echo "mgmt IP=$MY_IP"
export VLLM_NIXL_SIDE_CHANNEL_HOST=$MY_IP

if [[ "$ROLE" == "prefill" ]]; then
    echo "[$(date)] launching FRONTEND"
    python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin > /tmp/frontend.log 2>&1 &
    sleep 3

    echo "[$(date)] launching vLLM PREFILL worker (TP=$TP) with NixlConnector"
    export VLLM_NIXL_SIDE_CHANNEL_PORT=20097
    python3 -m dynamo.vllm \
        --model $MODEL \
        --tensor-parallel-size $TP \
        --max-model-len 4096 \
        --max-num-seqs 4 \
        --enforce-eager \
        --trust-remote-code \
        --disaggregation-mode prefill \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
        > /tmp/worker.log 2>&1 &
    W_PID=$!
elif [[ "$ROLE" == "decode" ]]; then
    echo "[$(date)] launching vLLM DECODE worker (TP=$TP) with NixlConnector"
    export VLLM_NIXL_SIDE_CHANNEL_PORT=20098
    python3 -m dynamo.vllm \
        --model $MODEL \
        --tensor-parallel-size $TP \
        --max-model-len 4096 \
        --max-num-seqs 4 \
        --enforce-eager \
        --trust-remote-code \
        --disaggregation-mode decode \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
        > /tmp/worker.log 2>&1 &
    W_PID=$!
fi

echo "worker PID=$W_PID"
echo "[$(date)] waiting up to 25 min for readiness"
for i in $(seq 1 150); do
    sleep 10
    if ! kill -0 $W_PID 2>/dev/null; then
        echo "[$(date)] worker died after ~${i}0s"
        tail -100 /tmp/worker.log
        echo "STATUS=worker-died"; sleep 7200; exit 1
    fi
    if [[ "$ROLE" == "prefill" ]]; then
        BODY=$(curl -sf http://localhost:8000/v1/models 2>/dev/null || true)
        N=$(echo "$BODY" | python3 -c 'import sys,json;print(len(json.load(sys.stdin).get("data",[])))' 2>/dev/null || echo 0)
        if [[ "$N" -ge 1 ]]; then echo "[$(date)] prefill ready @ ${i}0s"; break; fi
    fi
    (( i % 6 == 0 )) && tail -3 /tmp/worker.log
done

echo "[$(date)] $ROLE setup complete; sleeping 7200"
sleep 7200
INNER
chmod +x /tmp/inside_p4.sh
scp -o StrictHostKeyChecking=no /tmp/inside_p4.sh $PREFILL_NODE:/tmp/inside_p4.sh
scp -o StrictHostKeyChecking=no /tmp/inside_p4.sh $DECODE_NODE:/tmp/inside_p4.sh

# etcd+nats already running on prefill node from Phase 3
ssh -o StrictHostKeyChecking=no $PREFILL_NODE "podman ps --format '{{.Names}}' | grep -E 'dynamo-(etcd|nats)' || echo 'WARN: etcd/nats missing'"

COMMON_FLAGS="--device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add keep-groups --security-opt seccomp=unconfined \
  --network=host --ipc=host \
  -v $HF_CACHE:/root/.cache/huggingface \
  -v /tmp/inside_p4.sh:/tmp/inside_p4.sh:ro \
  -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:ro \
  -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
  -e HF_HOME=/root/.cache/huggingface \
  -e MODEL=$MODEL -e TP=$TP \
  -e HIP_VISIBLE_DEVICES=4 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e ETCD_ENDPOINTS=http://${PREFILL_IP}:2379 \
  -e NATS_SERVER=nats://${PREFILL_IP}:4222 \
  -e UCX_TLS=rc_v,tcp,rocm,rocm_copy,rocm_ipc,self,sm \
  -e UCX_MEMTYPE_CACHE=y \
  -e UCX_LOG_LEVEL=warn \
  -e VLLM_ROCM_USE_AITER=1"

echo "[$(date)] launching PREFILL on $PREFILL_NODE"
ssh -o StrictHostKeyChecking=no $PREFILL_NODE "podman run -d --replace --name dynamo-vllm-prefill $COMMON_FLAGS --entrypoint bash $IMAGE /tmp/inside_p4.sh prefill"

echo "[$(date)] launching DECODE on $DECODE_NODE"
ssh -o StrictHostKeyChecking=no $DECODE_NODE "podman run -d --replace --name dynamo-vllm-decode $COMMON_FLAGS --entrypoint bash $IMAGE /tmp/inside_p4.sh decode $PREFILL_IP"

echo "containers up; logs: ssh \$NODE podman logs -f dynamo-vllm-{prefill,decode}"
