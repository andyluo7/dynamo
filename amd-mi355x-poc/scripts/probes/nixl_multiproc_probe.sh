#!/bin/bash
# Spawn N concurrent Python processes inside the dynamo-vllm-rixl container,
# each running nixl_pytorch_probe.py to register MR_MB MiB on all ionic devs.
# Mirrors vLLM TP=N multiprocessing-worker startup pattern.
set -uo pipefail
NPROC=${1:-8}
MR_MB=${2:-2638}
N_REGIONS=${3:-1}

echo "=== spawning $NPROC processes, each registering ${N_REGIONS} x ${MR_MB} MiB via NIXL ==="
pids=()
for i in $(seq 0 $((NPROC-1))); do
    HIP_VISIBLE_DEVICES=$i python3 /tmp/nixl_pytorch_probe.py $MR_MB $N_REGIONS \
        > /tmp/nixl_proc_$i.log 2>&1 &
    pids+=($!)
done
echo "PIDs: ${pids[@]}"

ok=0; fail=0
for p in "${pids[@]}"; do
    wait $p
    rc=$?
    if [[ $rc -eq 0 ]]; then ok=$((ok+1)); else fail=$((fail+1)); fi
done

echo ""
echo "=== summary: $ok ok, $fail fail ==="
for i in $(seq 0 $((NPROC-1))); do
    last=$(grep -E 'SUCCESS|FAIL|Error|EngineCore' /tmp/nixl_proc_$i.log | tail -1)
    echo "proc $i: ${last:-<no result line>}"
done

if [[ $fail -gt 0 ]]; then
    echo ""
    echo "=== first failing proc full log ==="
    for i in $(seq 0 $((NPROC-1))); do
        if grep -q -E 'FAIL|Error' /tmp/nixl_proc_$i.log; then
            cat /tmp/nixl_proc_$i.log
            break
        fi
    done
fi
