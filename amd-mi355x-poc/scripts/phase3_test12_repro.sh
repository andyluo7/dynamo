#!/bin/bash
# Reproduce fork's Test 12: DSR1 SGLang 1P1D disagg + Mooncake @ c=1, ISL≈1024, OSL=1024.
# Targets: 97.7 tok/s @ c=1, TPOT 7.11 ms.
# vs our prior PoC measurement: 17.2 tok/s @ c=1.
#
# Aligns with the fork's:
#   - scripts/benchmark/models.yaml DeepSeek-R1 entry (substituting --disaggregation-transfer-backend mooncake for mori)
#   - scripts/benchmark/env.sh env vars
#   - scripts/benchmark/bench.sh defaults (ISL=1024 OSL=1024 num_prompts=conc*10 warmups=conc*2)
set -e

PREFILL_NODE=smci355-ccs-aus-g12-06
DECODE_NODE=smci355-ccs-aus-g12-22   # new decode (was g12-26 — that holder expired)
PREFILL_IP=10.194.30.23
IMAGE=docker.io/rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503
HF_CACHE=/shared/amdgpu/home/anluo/.cache/huggingface
PATCHES=/shared/amdgpu/home/anluo/dynamo-poc/fork-patches
MODEL=deepseek-ai/DeepSeek-R1-0528
TP=8

echo "[$(date)] === Phase 3 Test 12 reproduction: $MODEL TP=$TP Mooncake ==="

# Cleanup is assumed done externally (caller verified GPUs clean + containers gone).
echo "[$(date)] skipping in-script cleanup; assumed prior state is clean"

# Inside-container setup script
cat > /tmp/inside_test12.sh <<'INNER'
#!/bin/bash
set -e
ROLE=$1
echo "[$(date)] container starting role=$ROLE model=$MODEL TP=$TP"

SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")

# nixl stub
mkdir -p $SITE/nixl
cat > $SITE/nixl/__init__.py <<'PYEOF'
from . import _api, _bindings
PYEOF
cat > $SITE/nixl/_api.py <<'PYEOF'
class nixl_agent:
    def __init__(self, *a, **k):
        raise RuntimeError("nixl stubbed")
class nixl_agent_config:
    def __init__(self, *a, **k): pass
class nixl_xfer_handle: pass
class nixl_reg_dlist: pass
class nixl_xfer_dlist: pass
PYEOF
echo "" > $SITE/nixl/_bindings.py

# typing.Self compat
cat > $SITE/zzz_typing_self_compat.pth <<'PTHEOF'
import typing, typing_extensions; (not hasattr(typing, "Self")) and setattr(typing, "Self", typing_extensions.Self)
PTHEOF

pip install --quiet --no-deps ai-dynamo==1.1.1 ai-dynamo-runtime==1.1.1 \
    blake3 kubernetes msgpack msgspec prometheus-client pyzmq uvloop typing_extensions 2>&1 | tail -1

# sgl.Engine compat sed-patch (in-place; idempotent)
PUB=$SITE/dynamo/sglang/publisher.py
if grep -q "^import sglang as sgl$" $PUB && ! grep -q "_SglEngine" $PUB; then
    sed -i 's|^import sglang as sgl$|import sglang as sgl\nfrom sglang.srt.entrypoints.engine import Engine as _SglEngine\nsgl.Engine = _SglEngine|' $PUB
fi

# Install fork's mooncake_rocm_staging.py
cp /opt/fork-patches/mooncake_rocm_staging.py    $SITE/dynamo/sglang/mooncake_rocm_staging.py
cp /opt/fork-patches/rocm_dram_staging_common.py $SITE/dynamo/sglang/rocm_dram_staging_common.py
cat > $SITE/zzz_dynamo_rocm_autofix.pth <<'PTHEOF'
import os; os.environ.setdefault("SGLANG_MOONCAKE_ROCM_STAGING", "1");
PTHEOF
cat > $SITE/zzz_dynamo_mooncake_autofix.pth <<'PTHEOF'
try:
 import dynamo.sglang.mooncake_rocm_staging as _m  # noqa: F401
except Exception as _e:
 import sys; print(f"[mooncake_rocm_staging activation FAILED: {_e}]", file=sys.stderr)
PTHEOF

# libionic ABI fix
HOST_LIB=/host-libionic
[[ -f $HOST_LIB ]] && cp -fL $HOST_LIB /usr/lib/x86_64-linux-gnu/libionic.so.1
echo "ibv_devinfo count: $(ibv_devinfo 2>&1 | grep hca_id | wc -l)"

python3 -c "import dynamo.sglang.mooncake_rocm_staging; print('mooncake_rocm_staging OK')" 2>&1 | tail -2

# === Fork-aligned launch flags (from scripts/benchmark/models.yaml DeepSeek-R1 entry,
# substituting --disaggregation-transfer-backend mori → mooncake for our PoC) ===
BASE_FLAGS="--decode-log-interval 1000 --log-level warning --watchdog-timeout 3600 \
            --ep-dispatch-algorithm fake --load-balance-method round_robin \
            --kv-cache-dtype fp8_e4m3 --attention-backend aiter \
            --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 30001"

if [[ "$ROLE" == "prefill" ]]; then
    echo "[$(date)] launching FRONTEND on :8000 (Dynamo round-robin, per fork InferenceX bench)"
    python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin > /tmp/frontend.log 2>&1 &
    sleep 3
    echo "[$(date)] launching PREFILL worker (TP=$TP, fork-aligned flags)"
    python3 -m dynamo.sglang \
        --model-path $MODEL \
        --tp-size $TP \
        --trust-remote-code \
        --host 0.0.0.0 \
        --disaggregation-mode prefill \
        --mem-fraction-static 0.8 \
        --disable-radix-cache \
        --max-running-requests 128 \
        --chunked-prefill-size 262144 \
        --cuda-graph-bs $(seq -s ' ' 1 128) \
        $BASE_FLAGS \
        > /tmp/worker.log 2>&1 &
    W_PID=$!
elif [[ "$ROLE" == "decode" ]]; then
    echo "[$(date)] launching DECODE worker (TP=$TP, fork-aligned flags)"
    python3 -m dynamo.sglang \
        --model-path $MODEL \
        --tp-size $TP \
        --trust-remote-code \
        --host 0.0.0.0 \
        --disaggregation-mode decode \
        --mem-fraction-static 0.85 \
        --prefill-round-robin-balance \
        --max-running-requests 128 \
        --chunked-prefill-size 262144 \
        --cuda-graph-bs $(seq -s ' ' 1 128) \
        $BASE_FLAGS \
        > /tmp/worker.log 2>&1 &
    W_PID=$!
fi

echo "worker PID=$W_PID"
echo "[$(date)] waiting up to 25 min for readiness (~60s NFS load + ~3min HIP graphs)"
for i in $(seq 1 150); do
    sleep 10
    if ! kill -0 $W_PID 2>/dev/null; then
        echo "[$(date)] worker died after ~${i}0s; tail of worker.log:"
        tail -50 /tmp/worker.log
        echo "STATUS=worker-died"; sleep 7200; exit 1
    fi
    if [[ "$ROLE" == "prefill" ]]; then
        BODY=$(curl -sf http://localhost:8000/v1/models 2>/dev/null || true)
        N=$(echo "$BODY" | python3 -c 'import sys,json;print(len(json.load(sys.stdin).get("data",[])))' 2>/dev/null || echo 0)
        if [[ "$N" -ge 1 ]]; then
            echo "[$(date)] prefill ready @ ${i}0s; /v1/models:"
            echo "$BODY"
            break
        fi
    fi
    (( i % 6 == 0 )) && echo "  [$(date)] still waiting ~${i}0s; tail:" && tail -2 /tmp/worker.log
done

echo "[$(date)] $ROLE setup complete; sleeping 7200"
sleep 7200
INNER
chmod +x /tmp/inside_test12.sh
scp -o StrictHostKeyChecking=no /tmp/inside_test12.sh ${PREFILL_NODE}:/tmp/inside_test12.sh
scp -o StrictHostKeyChecking=no /tmp/inside_test12.sh ${DECODE_NODE}:/tmp/inside_test12.sh

# === Container env vars (from fork's scripts/benchmark/env.sh) ===
COMMON_FLAGS="--device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add keep-groups --security-opt seccomp=unconfined \
  --network=host --ipc=host \
  -v $HF_CACHE:/root/.cache/huggingface \
  -v $PATCHES:/opt/fork-patches:ro \
  -v /tmp/inside_test12.sh:/tmp/inside_test12.sh:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:/host-libionic:ro \
  -e HF_HOME=/root/.cache/huggingface \
  -e MODEL=$MODEL -e TP=$TP \
  -e ETCD_ENDPOINTS=http://${PREFILL_IP}:2379 \
  -e NATS_SERVER=nats://${PREFILL_IP}:4222 \
  -e SGLANG_MOONCAKE_ROCM_STAGING=1 \
  -e MC_MAX_SGE=2 \
  -e SGLANG_USE_AITER=1 \
  -e RCCL_MSCCL_ENABLE=0 \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200 \
  -e SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200 \
  -e SGLANG_AITER_MLA_PERSIST=False"

ssh -o StrictHostKeyChecking=no $PREFILL_NODE "podman ps --format '{{.Names}}' | grep -q '^dynamo-etcd$' || podman run -d --replace --name dynamo-etcd --network=host quay.io/coreos/etcd:v3.5.21 etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://${PREFILL_IP}:2379 >/dev/null
podman ps --format '{{.Names}}' | grep -q '^dynamo-nats$' || podman run -d --replace --name dynamo-nats --network=host docker.io/library/nats:2.10.28 -p 4222 -js >/dev/null
podman exec dynamo-etcd etcdctl --endpoints=http://localhost:2379 del --prefix dynamo/ 2>&1 | tail -1"

echo "[$(date)] launching PREFILL on $PREFILL_NODE"
ssh -o StrictHostKeyChecking=no $PREFILL_NODE "podman run -d --replace --name dynamo-prefill $COMMON_FLAGS --entrypoint bash $IMAGE /tmp/inside_test12.sh prefill"

echo "[$(date)] launching DECODE on $DECODE_NODE"
ssh -o StrictHostKeyChecking=no $DECODE_NODE "podman run -d --replace --name dynamo-decode $COMMON_FLAGS --entrypoint bash $IMAGE /tmp/inside_test12.sh decode"

echo "containers up; tail logs: ssh \$NODE podman logs -f dynamo-{prefill,decode}"
