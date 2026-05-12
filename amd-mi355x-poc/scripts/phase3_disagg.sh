#!/bin/bash
# Phase 3 PoC: 2-node SGLang disaggregated serving (1 prefill + 1 decode) with Mooncake.
# Run THIS SCRIPT FROM THE LOGIN NODE. It orchestrates work on g12-22 + g12-26.
set -e

PREFILL_NODE=smci355-ccs-aus-g12-22
DECODE_NODE=smci355-ccs-aus-g12-26
PREFILL_IP=10.194.30.27   # g12-22 mgmt IP
IMAGE=docker.io/rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503
HF_CACHE=/shared/amdgpu/home/anluo/.cache/huggingface
PATCHES=/shared/amdgpu/home/anluo/dynamo-poc/fork-patches
MODEL=Qwen/Qwen3-0.6B
TP=1   # for Qwen3-0.6B

echo "[$(date)] === Phase 3: SGLang 1P1D disagg ($MODEL) ==="

# Teardown old containers everywhere
for n in $PREFILL_NODE $DECODE_NODE; do
    ssh -o StrictHostKeyChecking=no $n "podman rm -f dynamo-sglang-poc dynamo-vllm-perf dynamo-prefill dynamo-decode dynamo-frontend 2>&1 | tail -3" || true
done

# Inside-container script (same on both nodes)
cat > /tmp/inside_p3.sh <<'INNER'
#!/bin/bash
set -e
ROLE=$1   # prefill | decode | frontend
echo "[$(date)] container starting role=$ROLE"

SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")

# nixl stub
mkdir -p $SITE/nixl
cat > $SITE/nixl/__init__.py <<'PYEOF'
from . import _api, _bindings
PYEOF
cat > $SITE/nixl/_api.py <<'PYEOF'
class nixl_agent:
    def __init__(self, *a, **k): raise RuntimeError("nixl stubbed")
class nixl_agent_config:
    def __init__(self, *a, **k): pass
class nixl_xfer_handle: pass
class nixl_reg_dlist: pass
class nixl_xfer_dlist: pass
PYEOF
echo "" > $SITE/nixl/_bindings.py

# typing.Self compat (Python 3.10)
cat > $SITE/zzz_typing_self_compat.pth <<'PTHEOF'
import typing, typing_extensions; (not hasattr(typing, "Self")) and setattr(typing, "Self", typing_extensions.Self)
PTHEOF

pip install --quiet --no-deps ai-dynamo==1.1.1 ai-dynamo-runtime==1.1.1 \
    blake3 kubernetes msgpack msgspec prometheus-client pyzmq uvloop typing_extensions 2>&1 | tail -1

# sgl.Engine compat (sed-patch publisher.py)
PUB=$SITE/dynamo/sglang/publisher.py
if grep -q "^import sglang as sgl$" $PUB && ! grep -q "_SglEngine" $PUB; then
    sed -i 's|^import sglang as sgl$|import sglang as sgl\nfrom sglang.srt.entrypoints.engine import Engine as _SglEngine\nsgl.Engine = _SglEngine  # AMD-PoC compat|' $PUB
fi

# Install fork's mooncake_rocm_staging.py + dependencies. These are in
# /shared/amdgpu/home/anluo/dynamo-poc/fork-patches mounted at /opt/fork-patches.
echo "[$(date)] installing fork mooncake/rocm staging patches"
cp /opt/fork-patches/mooncake_rocm_staging.py $SITE/dynamo/sglang/mooncake_rocm_staging.py
cp /opt/fork-patches/rocm_dram_staging_common.py $SITE/dynamo/sglang/rocm_dram_staging_common.py
# Auto-activate via .pth at python startup (per fork's Dockerfile.rocm-sglang pattern)
cat > $SITE/zzz_dynamo_rocm_autofix.pth <<'PTHEOF'
import os; os.environ.setdefault("SGLANG_MOONCAKE_ROCM_STAGING", "1");
PTHEOF
cat > $SITE/zzz_dynamo_mooncake_autofix.pth <<'PTHEOF'
try:
 import dynamo.sglang.mooncake_rocm_staging as _m  # noqa: F401
except Exception as _e:
 import sys; print(f"[mooncake_rocm_staging activation FAILED: {_e}]", file=sys.stderr)
PTHEOF

# libionic ABI fix: replace container's libionic.so.1 with the host's matching version.
# The host libionic was bind-mounted at /host-libionic.
HOST_LIB=/host-libionic
if [[ -f $HOST_LIB ]]; then
    cp -fL $HOST_LIB /usr/lib/x86_64-linux-gnu/libionic.so.1
    echo "[$(date)] libionic ABI fix applied"
fi
echo "ibv_devinfo count:"
ibv_devinfo 2>&1 | grep hca_id | wc -l

# Verify staging activation
python3 -c "import dynamo.sglang.mooncake_rocm_staging; print('mooncake_rocm_staging imported OK')" 2>&1 | tail -3

if [[ "$ROLE" == "prefill" ]]; then
    echo "[$(date)] launching FRONTEND"
    python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin > /tmp/frontend.log 2>&1 &
    sleep 3
    echo "[$(date)] launching PREFILL worker (TP=$TP)"
    python3 -m dynamo.sglang \
        --model-path $MODEL \
        --tp-size $TP \
        --trust-remote-code \
        --host 0.0.0.0 \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend mooncake \
        > /tmp/worker.log 2>&1 &
    W_PID=$!
elif [[ "$ROLE" == "decode" ]]; then
    echo "[$(date)] launching DECODE worker (TP=$TP)"
    python3 -m dynamo.sglang \
        --model-path $MODEL \
        --tp-size $TP \
        --trust-remote-code \
        --host 0.0.0.0 \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend mooncake \
        > /tmp/worker.log 2>&1 &
    W_PID=$!
fi

echo "worker PID=$W_PID"
echo "[$(date)] waiting up to 20 min for readiness"
for i in $(seq 1 120); do
    sleep 10
    if ! kill -0 $W_PID 2>/dev/null; then
        echo "[$(date)] worker died after ~${i}0s; tail of worker.log:"; tail -50 /tmp/worker.log
        echo "STATUS=worker-died"
        sleep 7200
        exit 1
    fi
    if [[ "$ROLE" == "prefill" ]]; then
        BODY=$(curl -sf http://localhost:8000/v1/models 2>/dev/null || true)
        N=$(echo "$BODY" | python3 -c 'import sys,json;print(len(json.load(sys.stdin).get("data",[])))' 2>/dev/null || echo 0)
        if [[ "$N" -ge 1 ]]; then
            echo "[$(date)] prefill ready @ ${i}0s; /v1/models returns $N model(s)"
            break
        fi
    fi
    (( i % 6 == 0 )) && echo "  [$(date)] still waiting ~${i}0s; worker.log tail:" && tail -3 /tmp/worker.log
done

echo "[$(date)] $ROLE setup complete; sleeping 7200 to keep alive"
sleep 7200
INNER
chmod +x /tmp/inside_p3.sh
scp -o StrictHostKeyChecking=no /tmp/inside_p3.sh ${PREFILL_NODE}:/tmp/inside_p3.sh
scp -o StrictHostKeyChecking=no /tmp/inside_p3.sh ${DECODE_NODE}:/tmp/inside_p3.sh

# Common podman flags
COMMON_FLAGS="--device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add keep-groups --security-opt seccomp=unconfined \
  --network=host --ipc=host \
  -v $HF_CACHE:/root/.cache/huggingface \
  -v $PATCHES:/opt/fork-patches:ro \
  -v /tmp/inside_p3.sh:/tmp/inside_p3.sh:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:/host-libionic:ro \
  -e HF_HOME=/root/.cache/huggingface \
  -e ETCD_ENDPOINTS=http://${PREFILL_IP}:2379 \
  -e NATS_SERVER=nats://${PREFILL_IP}:4222 \
  -e SGLANG_MOONCAKE_ROCM_STAGING=1 \
  -e MC_MAX_SGE=2 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e MODEL=$MODEL \
  -e TP=$TP"

# Reuse etcd+nats on prefill node
ssh -o StrictHostKeyChecking=no $PREFILL_NODE "podman ps --format '{{.Names}}' | grep -q '^dynamo-etcd$' || podman run -d --replace --name dynamo-etcd --network=host quay.io/coreos/etcd:v3.5.21 etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://${PREFILL_IP}:2379 >/dev/null
podman ps --format '{{.Names}}' | grep -q '^dynamo-nats$' || podman run -d --replace --name dynamo-nats --network=host docker.io/library/nats:2.10.28 -p 4222 -js >/dev/null
sleep 2"

echo "[$(date)] launching PREFILL container on $PREFILL_NODE"
ssh -o StrictHostKeyChecking=no $PREFILL_NODE "podman run -d --replace --name dynamo-prefill $COMMON_FLAGS --entrypoint bash $IMAGE /tmp/inside_p3.sh prefill"

echo "[$(date)] launching DECODE container on $DECODE_NODE"
ssh -o StrictHostKeyChecking=no $DECODE_NODE "podman run -d --replace --name dynamo-decode $COMMON_FLAGS --entrypoint bash $IMAGE /tmp/inside_p3.sh decode"

echo "[$(date)] containers launched; tailing logs..."
echo "  prefill: ssh $PREFILL_NODE 'podman logs -f dynamo-prefill'"
echo "  decode:  ssh $DECODE_NODE 'podman logs -f dynamo-decode'"
