#!/bin/bash
# Multi-process ionic registration test: spawn N workers, each tries to
# register 1 large MR on ionic_0. Mirrors vLLM TP=8 cross-process pattern.
set -uo pipefail
NPROC=${1:-8}
SIZE_MB=${2:-2638}   # ~vLLM's 2.57 GiB per-region size for DSR1 TP=8
PROBE=${3:-/tmp/ionic_concurrent_probe}

echo "=== spawning $NPROC processes, each registering 1 x ${SIZE_MB} MiB on ionic_0 ==="
pids=()
for i in $(seq 0 $((NPROC-1))); do
    LD_PRELOAD=/tmp/ibv_ionic_compat.so $PROBE $SIZE_MB 1 0 > /tmp/proc_$i.log 2>&1 &
    pids+=($!)
done
echo "PIDs: ${pids[@]}"
for p in "${pids[@]}"; do wait $p; echo "pid $p exit=$?"; done
echo "=== summary ==="
for i in $(seq 0 $((NPROC-1))); do
    last=$(tail -3 /tmp/proc_$i.log | grep -E 'reg OK|reg.*FAIL|summary' | head -1)
    echo "proc $i: $last"
done
