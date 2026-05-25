#!/bin/bash
# Phase 2 PoC: stand up etcd+NATS+dynamo.frontend+dynamo.vllm with MiniMax-M2.5 on one MI355X node.
set -e

CONTAINER_NAME=dynamo-vllm-poc
NATS_NAME=dynamo-nats
ETCD_NAME=dynamo-etcd
HF_CACHE=/shared/amdgpu/home/anluo/inferencex-agentic-test/hf-cache
MODEL=MiniMaxAI/MiniMax-M2.5
LAUNCH_LOG=/shared/amdgpu/home/anluo/dynamo-poc/phase2-launch.log

# Cleanup any prior runs
podman rm -f $CONTAINER_NAME $NATS_NAME $ETCD_NAME 2>/dev/null || true

# Get host management IP (NOT hostname -I per fork runbook — that returns ionic IP)
MY_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
echo "[$(date)] host mgmt IP: $MY_IP"

# Start etcd (rootless podman, host network)
echo "[$(date)] starting etcd..."
podman run -d --replace --name $ETCD_NAME --network=host \
    quay.io/coreos/etcd:v3.5.21 etcd \
    --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls http://${MY_IP}:2379 >/dev/null
sleep 2
podman exec $ETCD_NAME etcdctl --endpoints=http://localhost:2379 endpoint health 2>&1 | head -3

# Start NATS with JetStream
echo "[$(date)] starting nats..."
podman run -d --replace --name $NATS_NAME --network=host \
    docker.io/library/nats:2.10.28 -p 4222 -js >/dev/null
sleep 2
podman logs $NATS_NAME 2>&1 | tail -3

# Run the dynamo+vllm container
# Expose ports 8000 (frontend), 8081 (worker system), 30000+ (vllm internal)
# Mount HF cache, the dyn_v5 setup script, and a launcher we generate inline.
cat > /tmp/inside_container.sh <<'INNER'
#!/bin/bash
set -e
echo "[$(date)] container started"
echo "ROCm devices:"
ls /dev/kfd /dev/dri/renderD* 2>&1 | head -3
echo

echo "[$(date)] creating nixl stub..."
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
mkdir -p $SITE/nixl
cat > $SITE/nixl/__init__.py <<'PYEOF'
"""NIXL stub for AMD agg-only PoC."""
from . import _api, _bindings
PYEOF
cat > $SITE/nixl/_api.py <<'PYEOF'
class nixl_agent:
    def __init__(self, *a, **k):
        raise RuntimeError("nixl is stubbed (AMD agg PoC)")
class nixl_agent_config:
    def __init__(self, *a, **k): pass
class nixl_xfer_handle: pass
class nixl_reg_dlist: pass
class nixl_xfer_dlist: pass
PYEOF
echo "" > $SITE/nixl/_bindings.py

echo "[$(date)] installing ai-dynamo..."
pip install --quiet --no-deps ai-dynamo==1.1.1 ai-dynamo-runtime==1.1.1 \
    blake3 kubernetes msgpack msgspec prometheus-client pyzmq 2>&1 | tail -2

echo "[$(date)] starting dynamo.frontend..."
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin > /tmp/frontend.log 2>&1 &
FE_PID=$!
sleep 5
echo "frontend PID=$FE_PID"
tail -5 /tmp/frontend.log || true

echo "[$(date)] starting dynamo.vllm with $MODEL (this can take 5-15 min for model load)..."
# tp=8 to use all 8 MI355X GPUs and have plenty of KV cache
python3 -m dynamo.vllm \
    --model $MODEL \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --enforce-eager \
    --trust-remote-code \
    > /tmp/worker.log 2>&1 &
W_PID=$!
echo "worker PID=$W_PID"

# Poll worker readiness — wait for /v1/models data to be non-empty (worker registered with frontend)
echo "[$(date)] waiting up to 30 min for worker registration in /v1/models..."
READY=0
for i in $(seq 1 180); do
    BODY=$(curl -sf http://localhost:8000/v1/models 2>/dev/null || true)
    if [[ -n "$BODY" ]]; then
        # Use python to parse data length robustly
        N=$(echo "$BODY" | python3 -c 'import sys,json; print(len(json.load(sys.stdin).get("data",[])))' 2>/dev/null || echo 0)
        if [[ "$N" -ge 1 ]]; then
            echo "[$(date)] worker registered after ~${i}*10s. /v1/models:"
            echo "$BODY"
            READY=1
            break
        fi
    fi
    sleep 10
    if (( i % 6 == 0 )); then
        echo "  [$(date)] still waiting (~${i}0s); worker log tail:"
        tail -3 /tmp/worker.log 2>&1
    fi
    # If the worker died, bail
    if ! kill -0 $W_PID 2>/dev/null; then
        echo "  [$(date)] worker PID $W_PID died! tail of worker.log:"
        tail -50 /tmp/worker.log
        echo "STATUS=worker-died"
        # keep container alive for inspection
        sleep 7200
        exit 1
    fi
done

if [[ "$READY" != "1" ]]; then
    echo "[$(date)] timed out waiting for worker registration; tail:"
    tail -50 /tmp/worker.log
    echo "STATUS=timeout"
    sleep 7200
    exit 1
fi

echo "[$(date)] sending chat completion..."
curl -sS -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello, please reply with one short sentence.\"}],\"max_tokens\":48}" \
    2>&1 | tee /tmp/response.json
echo
echo "[$(date)] STATUS=success"
echo "[$(date)] container will stay alive for 2 hours for further testing; podman exec dynamo-vllm-poc bash to enter"
sleep 7200
INNER
chmod +x /tmp/inside_container.sh

echo "[$(date)] launching dynamo container..."
podman run -d --replace --name $CONTAINER_NAME \
    --device /dev/kfd --device /dev/dri \
    --group-add keep-groups --security-opt seccomp=unconfined \
    --network=host --ipc=host \
    -v $HF_CACHE:/root/.cache/huggingface \
    -v /tmp/inside_container.sh:/tmp/run.sh:ro \
    -e HF_HOME=/root/.cache/huggingface \
    -e MODEL=$MODEL \
    -e ETCD_ENDPOINTS=http://${MY_IP}:2379 \
    -e NATS_SERVER=nats://${MY_IP}:4222 \
    -e VLLM_ROCM_USE_AITER=1 \
    --entrypoint bash docker.io/rocm/vllm-dev:nightly /tmp/run.sh

echo "[$(date)] container started; tailing logs to $LAUNCH_LOG"
podman logs -f $CONTAINER_NAME > $LAUNCH_LOG 2>&1 &
TAIL_PID=$!
echo "log tail PID=$TAIL_PID; container=$CONTAINER_NAME"
echo "use:  podman logs -f $CONTAINER_NAME    OR    tail -f $LAUNCH_LOG"
