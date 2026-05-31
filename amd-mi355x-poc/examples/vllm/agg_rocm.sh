#!/bin/bash
# Phase 2.5: vLLM agg perf with HIP graphs enabled. Same stack as phase2_e2e.sh
# but drop --enforce-eager and --max-num-seqs higher to let HIP graphs amortize.
set -e

# Tunables (override via env): DYNAMO_VERSION, HF_CACHE, MODEL, LAUNCH_LOG, IMAGE
CONTAINER_NAME="${CONTAINER_NAME:-dynamo-vllm-perf}"
NATS_NAME="${NATS_NAME:-dynamo-nats}"
ETCD_NAME="${ETCD_NAME:-dynamo-etcd}"
IMAGE="${IMAGE:-docker.io/rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
MODEL="${MODEL:-MiniMaxAI/MiniMax-M2.5}"
LAUNCH_LOG="${LAUNCH_LOG:-/tmp/dynamo-vllm-agg.log}"
DYNAMO_VERSION="${DYNAMO_VERSION:-1.3.0}"

# Tear down prior eager run; keep etcd+nats from the previous run if running.
podman rm -f $CONTAINER_NAME 2>/dev/null || true
podman rm -f dynamo-vllm-poc 2>/dev/null || true

MY_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
echo "[$(date)] mgmt IP $MY_IP"

# Reuse etcd+nats if up, else (re)start them
if ! podman ps --format '{{.Names}}' | grep -q "^${ETCD_NAME}$"; then
    podman run -d --replace --name $ETCD_NAME --network=host \
        quay.io/coreos/etcd:v3.5.21 etcd \
        --listen-client-urls http://0.0.0.0:2379 \
        --advertise-client-urls http://${MY_IP}:2379 >/dev/null
    sleep 2
fi
if ! podman ps --format '{{.Names}}' | grep -q "^${NATS_NAME}$"; then
    podman run -d --replace --name $NATS_NAME --network=host \
        docker.io/library/nats:2.10.28 -p 4222 -js >/dev/null
    sleep 2
fi
echo "[$(date)] etcd+nats up"

cat > /tmp/inside_perf.sh <<'INNER'
#!/bin/bash
set -e
DYNAMO_VERSION="${DYNAMO_VERSION:-1.3.0}"
echo "[$(date)] container started (perf run)"

# ai-dynamo with PR #9929 ships the nixl lazy-import proxy and
# typing_extensions.Self, so no inline stubs or .pth shim are needed.
# Pin DYNAMO_VERSION to any release containing #9929 (>= 1.3.0).
pip install --quiet --no-deps "ai-dynamo==${DYNAMO_VERSION}" "ai-dynamo-runtime==${DYNAMO_VERSION}" \
    blake3 kubernetes msgpack msgspec prometheus-client pyzmq 2>&1 | tail -1

echo "[$(date)] starting frontend"
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin > /tmp/frontend.log 2>&1 &
sleep 5

echo "[$(date)] starting worker (NO --enforce-eager, max-num-seqs 64 for graph amortization)"
# Note: vLLM ROCm warns about aiter+gfx950 issues. If HIP graph capture fails,
# rerun this script with EXTRA_FLAGS="--enforce-eager" exported.
python3 -m dynamo.vllm \
    --model $MODEL \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --trust-remote-code \
    > /tmp/worker.log 2>&1 &
W_PID=$!
echo "worker PID=$W_PID"

echo "[$(date)] waiting up to 35 min for readiness (model load + HIP graph capture)"
READY=0
for i in $(seq 1 210); do
    BODY=$(curl -sf http://localhost:8000/v1/models 2>/dev/null || true)
    if [[ -n "$BODY" ]]; then
        N=$(echo "$BODY" | python3 -c 'import sys,json; print(len(json.load(sys.stdin).get("data",[])))' 2>/dev/null || echo 0)
        if [[ "$N" -ge 1 ]]; then
            echo "[$(date)] worker ready in ~${i}*10s"
            READY=1; break
        fi
    fi
    sleep 10
    if (( i % 6 == 0 )); then
        echo "  [$(date)] still waiting (~${i}0s):"; tail -2 /tmp/worker.log
    fi
    if ! kill -0 $W_PID 2>/dev/null; then
        echo "  [$(date)] worker died! tail:"; tail -50 /tmp/worker.log
        echo "STATUS=worker-died"; sleep 7200; exit 1
    fi
done
[[ "$READY" != "1" ]] && { echo "STATUS=timeout"; tail -50 /tmp/worker.log; sleep 7200; exit 1; }

echo
echo "[$(date)] === concurrency sweep (3 warmup + 5 timed each) ==="
python3 - <<'PYBENCH'
import requests, time, concurrent.futures, statistics
URL = "http://localhost:8000/v1/chat/completions"
MODEL = "MiniMaxAI/MiniMax-M2.5"
PROMPT = "Write a single-paragraph story about a friendly robot exploring a coral reef."

def send(i, max_tok=128):
    t0 = time.time()
    r = requests.post(URL, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": f"{PROMPT} (request {i})"}],
        "max_tokens": max_tok, "temperature": 0.7,
    }, timeout=120)
    dt = time.time() - t0
    if r.status_code != 200:
        return {"ok": False, "ms": dt*1000, "err": r.text[:200]}
    j = r.json()
    usage = j.get("usage", {})
    return {"ok": True, "ms": dt*1000,
            "in": usage.get("prompt_tokens"),
            "out": usage.get("completion_tokens")}

# warmup
print("warmup...")
for i in range(3):
    r = send(i, max_tok=32)
    print(f"  warmup {i}: ok={r['ok']} ms={r['ms']:.0f}")

print(f"\n{'conc':>4} {'N':>3} {'P50_ms':>8} {'P95_ms':>8} {'tps_total':>10} {'in_avg':>7} {'out_avg':>7} {'ok':>4}")
for conc in [1, 4, 8]:
    N = max(conc * 3, 8)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(conc) as pool:
        results = list(pool.map(send, range(N)))
    wall = time.time() - t0
    ok = [r for r in results if r["ok"]]
    if not ok:
        print(f"{conc:>4} {N:>3} {'FAIL':>8} {'-':>8} {'-':>10} {'-':>7} {'-':>7} {len(ok):>4}/{N}")
        if results:
            print("    first error:", results[0].get("err"))
        continue
    times = sorted(r["ms"] for r in ok)
    p50 = times[len(times)//2]
    p95 = times[int(0.95 * len(times))]
    out_total = sum(r["out"] or 0 for r in ok)
    in_avg = statistics.mean(r["in"] for r in ok if r["in"])
    out_avg = statistics.mean(r["out"] for r in ok if r["out"])
    print(f"{conc:>4} {N:>3} {p50:>8.0f} {p95:>8.0f} {out_total/wall:>10.1f} {in_avg:>7.1f} {out_avg:>7.1f} {len(ok):>4}/{N}")
PYBENCH

echo
echo "[$(date)] STATUS=success"
echo "[$(date)] container will stay alive 2h for further testing"
sleep 7200
INNER
chmod +x /tmp/inside_perf.sh

echo "[$(date)] launching perf container"
podman run -d --replace --name $CONTAINER_NAME \
    --device /dev/kfd --device /dev/dri \
    --group-add keep-groups --security-opt seccomp=unconfined \
    --network=host --ipc=host \
    -v $HF_CACHE:/root/.cache/huggingface \
    -v /tmp/inside_perf.sh:/tmp/run.sh:ro \
    -e HF_HOME=/root/.cache/huggingface \
    -e MODEL=$MODEL \
    -e ETCD_ENDPOINTS=http://${MY_IP}:2379 \
    -e NATS_SERVER=nats://${MY_IP}:4222 \
    -e VLLM_ROCM_USE_AITER=1 \
    --entrypoint bash docker.io/rocm/vllm-dev:nightly /tmp/run.sh

echo "[$(date)] container=$CONTAINER_NAME; stream logs to $LAUNCH_LOG"
nohup podman logs -f $CONTAINER_NAME > $LAUNCH_LOG 2>&1 &
