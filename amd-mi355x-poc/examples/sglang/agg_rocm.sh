#!/bin/bash
# Phase 1 PoC: Dynamo + SGLang + DeepSeek-R1-0528 FP8 single-node aggregated.
set -e

# Tunables (override via env): DYNAMO_VERSION, HF_CACHE, LAUNCH_LOG, IMAGE
CONTAINER_NAME="${CONTAINER_NAME:-dynamo-sglang-poc}"
IMAGE="${IMAGE:-docker.io/rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
DSR1_LOCAL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"
LAUNCH_LOG="${LAUNCH_LOG:-/tmp/dynamo-sglang-agg.log}"
DYNAMO_VERSION="${DYNAMO_VERSION:-1.3.0}"

podman rm -f $CONTAINER_NAME 2>/dev/null || true
MY_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
echo "[$(date)] mgmt IP $MY_IP"

# Reuse etcd+nats from Phase 2 if up
podman ps --format '{{.Names}}' | grep -q '^dynamo-etcd$' || \
    podman run -d --replace --name dynamo-etcd --network=host \
      quay.io/coreos/etcd:v3.5.21 etcd \
      --listen-client-urls http://0.0.0.0:2379 \
      --advertise-client-urls http://${MY_IP}:2379 >/dev/null
podman ps --format '{{.Names}}' | grep -q '^dynamo-nats$' || \
    podman run -d --replace --name dynamo-nats --network=host \
      docker.io/library/nats:2.10.28 -p 4222 -js >/dev/null
sleep 2

cat > /tmp/inside_p1.sh <<'INNER'
#!/bin/bash
set -e
DYNAMO_VERSION="${DYNAMO_VERSION:-1.3.0}"
echo "[$(date)] container started"
echo "ROCm: $(ls /opt/rocm 2>&1 | head -3 | tr '\n' ' ')"
echo "Python: $(python3 --version)"
echo "SGLang: $(pip show sglang 2>&1 | grep -E '^(Name|Version)')"

# ai-dynamo with PR #9929 ships the nixl lazy-import proxy and
# typing_extensions.Self, so no inline stubs or .pth shim are needed.
# Pin DYNAMO_VERSION to any release containing #9929 (>= 1.3.0).
echo "[$(date)] installing ai-dynamo (no-deps)"
pip install --quiet --no-deps "ai-dynamo==${DYNAMO_VERSION}" "ai-dynamo-runtime==${DYNAMO_VERSION}" \
    blake3 kubernetes msgpack msgspec prometheus-client pyzmq uvloop typing_extensions 2>&1 | tail -2

# rocm/sgl-dev's sglang doesn't re-export Engine at top level (only via
# sglang.srt.entrypoints.engine). Patch dynamo's references in-place — this
# IS a candidate upstream patch (use the modern import path).
PUB=$SITE/dynamo/sglang/publisher.py
if grep -q "import sglang as sgl" $PUB; then
    # Add the Engine re-export inline (idempotent)
    sed -i 's|^import sglang as sgl$|import sglang as sgl\nfrom sglang.srt.entrypoints.engine import Engine as _SglEngine\nsgl.Engine = _SglEngine  # AMD-PoC compat shim|' $PUB
    echo "[$(date)] patched $PUB"
fi
# Same shim for any other dynamo file that uses sgl.Engine
grep -rl 'sgl\.Engine' $SITE/dynamo/ 2>/dev/null | while read f; do
    grep -q "_SglEngine = " "$f" || true
done

echo "[$(date)] sanity import"
python3 -c "import sglang; print('sglang import OK')" 2>&1
python3 -c "import dynamo.frontend; import dynamo.sglang; print('dynamo modules OK')" 2>&1 | tail -10

echo "[$(date)] dynamo.sglang --help | head"
python3 -m dynamo.sglang --help 2>&1 | head -25

echo "[$(date)] starting frontend"
python3 -m dynamo.frontend --http-port 8000 --router-mode round-robin > /tmp/frontend.log 2>&1 &
sleep 5

echo "[$(date)] starting dynamo.sglang worker (DSR1, TP=8)"
# DSR1-0528 FP8 = ~640 GB → needs all 8 MI355X (288 GB each = 2.3 TB total)
python3 -m dynamo.sglang \
    --model-path $DSR1_LOCAL \
    --tp-size 8 \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    > /tmp/worker.log 2>&1 &
W_PID=$!
echo "worker PID=$W_PID"

echo "[$(date)] waiting up to 35 min for readiness"
READY=0
for i in $(seq 1 210); do
    BODY=$(curl -sf http://localhost:8000/v1/models 2>/dev/null || true)
    if [[ -n "$BODY" ]]; then
        N=$(echo "$BODY" | python3 -c 'import sys,json; print(len(json.load(sys.stdin).get("data",[])))' 2>/dev/null || echo 0)
        if [[ "$N" -ge 1 ]]; then
            echo "[$(date)] worker ready in ~${i}*10s. /v1/models:"
            echo "$BODY"
            READY=1; break
        fi
    fi
    sleep 10
    if (( i % 6 == 0 )); then
        echo "  [$(date)] still waiting (~${i}0s):"; tail -3 /tmp/worker.log
    fi
    if ! kill -0 $W_PID 2>/dev/null; then
        echo "  [$(date)] worker died! tail:"; tail -50 /tmp/worker.log
        echo "STATUS=worker-died"; sleep 7200; exit 1
    fi
done
[[ "$READY" != "1" ]] && { echo "STATUS=timeout"; tail -100 /tmp/worker.log; sleep 7200; exit 1; }

echo
echo "[$(date)] === smoke chat completion ==="
curl -sS -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-ai/DeepSeek-R1-0528","messages":[{"role":"user","content":"Hello, give a short reply."}],"max_tokens":48}' \
    | tee /tmp/response.json
echo

echo "[$(date)] === concurrency sweep (3 warmup + N timed) ==="
python3 - <<'PYBENCH'
import requests, time, concurrent.futures, statistics
URL = "http://localhost:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-R1-0528"
PROMPT = "Write a single-paragraph story about a friendly robot exploring a coral reef."

def send(i, max_tok=128):
    t0 = time.time()
    r = requests.post(URL, json={
        "model": MODEL,
        "messages": [{"role":"user","content": f"{PROMPT} (req {i})"}],
        "max_tokens": max_tok, "temperature": 0.7,
    }, timeout=180)
    dt = time.time() - t0
    if r.status_code != 200:
        return {"ok": False, "ms": dt*1000, "err": r.text[:200]}
    j = r.json()
    u = j.get("usage", {})
    return {"ok": True, "ms": dt*1000, "in": u.get("prompt_tokens"), "out": u.get("completion_tokens")}

print("warmup...")
for i in range(3):
    r = send(i, max_tok=32)
    print(f"  warmup {i}: ok={r['ok']} ms={r['ms']:.0f}")

print(f"\n{'conc':>4} {'N':>3} {'P50_ms':>8} {'P95_ms':>8} {'tps_total':>10} {'in':>5} {'out':>5} {'ok':>5}")
for conc in [1, 4, 8]:
    N = max(conc * 3, 8)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(conc) as pool:
        results = list(pool.map(send, range(N)))
    wall = time.time() - t0
    ok = [r for r in results if r["ok"]]
    if not ok:
        print(f"{conc:>4} {N:>3} {'FAIL':>8}")
        if results: print("    err:", results[0].get("err"))
        continue
    times = sorted(r["ms"] for r in ok)
    p50 = times[len(times)//2]; p95 = times[int(0.95*len(times))]
    out_total = sum(r["out"] or 0 for r in ok)
    in_avg = statistics.mean(r["in"] for r in ok if r["in"])
    out_avg = statistics.mean(r["out"] for r in ok if r["out"])
    print(f"{conc:>4} {N:>3} {p50:>8.0f} {p95:>8.0f} {out_total/wall:>10.1f} {in_avg:>5.0f} {out_avg:>5.0f} {len(ok):>3}/{N}")
PYBENCH

echo
echo "[$(date)] STATUS=success"
sleep 7200
INNER
chmod +x /tmp/inside_p1.sh

echo "[$(date)] launching $CONTAINER_NAME"
podman run -d --replace --name $CONTAINER_NAME \
    --device /dev/kfd --device /dev/dri \
    --group-add keep-groups --security-opt seccomp=unconfined \
    --network=host --ipc=host \
    -v $HF_CACHE:/root/.cache/huggingface \
    -v /tmp/inside_p1.sh:/tmp/run.sh:ro \
    -e HF_HOME=/root/.cache/huggingface \
    -e DSR1_LOCAL=$DSR1_LOCAL \
    -e ETCD_ENDPOINTS=http://${MY_IP}:2379 \
    -e NATS_SERVER=nats://${MY_IP}:4222 \
    --entrypoint bash $IMAGE /tmp/run.sh

echo "[$(date)] streaming logs to $LAUNCH_LOG"
nohup podman logs -f $CONTAINER_NAME > $LAUNCH_LOG 2>&1 &
