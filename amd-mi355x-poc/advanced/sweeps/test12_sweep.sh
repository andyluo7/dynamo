#!/bin/bash
# Concurrency sweep using test12_bench.py against the current DSR1 disagg setup.
# Prints a single summary table at the end.

set -uo pipefail
RESULTS=/tmp/test12_sweep_results.csv
echo "conc,n_prompts,P50_ms,P95_ms,TPOT_P50_ms,TTFT_P50_ms,output_tok_per_req,output_tok_aggregate,total_tok_aggregate,success" > $RESULTS

for c in 1 4 8 16 32 64; do
    n=$((c * 10))
    w=$((c * 2))
    echo "" >&2
    echo "=== c=$c, num_prompts=$n, warmup=$w ===" >&2
    out=$(python3 /tmp/test12_bench.py --isl 1024 --osl 1024 --conc $c --num-prompts $n --warmup $w --ignore-eos 2>&1)
    echo "$out" >&2 | tail -20

    # Extract metrics from the output
    p50=$(echo "$out" | grep -oP 'total P50=\K[0-9]+(?=ms)' | head -1)
    p95=$(echo "$out" | grep -oP 'total P50=[0-9]+ms P95=\K[0-9]+(?=ms)' | head -1)
    tpot=$(echo "$out" | grep -oP 'TPOT  P50=\K[0-9.]+(?=ms)' | head -1)
    ttft=$(echo "$out" | grep -oP 'TTFT  P50=\K[0-9]+(?=ms)' | head -1)
    perreq=$(echo "$out" | grep -oP 'output tok/s \(per request avg\):\s+\K[0-9.]+' | head -1)
    agg=$(echo "$out" | grep -oP 'output tok/s \(aggregate\):\s+\K[0-9.]+' | head -1)
    total=$(echo "$out" | grep -oP 'total tok/s \(in\+out aggregate\):\s+\K[0-9.]+' | head -1)
    okline=$(echo "$out" | grep -oP 'results \(\K[0-9]+/[0-9]+' | head -1)

    echo "$c,$n,${p50:-na},${p95:-na},${tpot:-na},${ttft:-na},${perreq:-na},${agg:-na},${total:-na},${okline:-na}" >> $RESULTS
done

echo "" >&2
echo "=========== SWEEP SUMMARY ===========" >&2
column -t -s, $RESULTS
