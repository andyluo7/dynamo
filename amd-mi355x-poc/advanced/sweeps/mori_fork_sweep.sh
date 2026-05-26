#!/bin/bash
# Faithful reproduction of fork's bench.sh:
#   sglang.bench_serving --backend openai --input-len 1024 --output-len 1024
#                        --random-range-ratio 0.8 --num-prompts $((conc*10))
#                        --max-concurrency $conc --result-filename FILE
set -uo pipefail
RESULTS=/tmp/mori_fork_results.csv
echo "conc,n_prompts,output_throughput,input_throughput,total_throughput,median_ttft,median_tpot,median_e2e_latency,p99_ttft,p99_tpot" > $RESULTS

for c in 1 4 8 16 32 64 128; do
    n=$((c * 10))
    OUT="/tmp/mori_fork_c${c}.json"
    LOG="/tmp/mori_fork_c${c}.log"
    echo "" >&2
    echo "=== c=$c n=$n ===" >&2

    timeout 1800 python3 -m sglang.bench_serving \
        --backend openai --model deepseek-ai/DeepSeek-R1-0528 \
        --base-url http://localhost:30000 \
        --dataset-name random --random-input-len 1024 --random-output-len 1024 \
        --random-range-ratio 0.8 \
        --num-prompts $n --max-concurrency $c \
        --request-rate inf \
        --output-file $OUT \
        > $LOG 2>&1
    rc=$?
    echo "exit=$rc" >&2
    tail -25 $LOG >&2

    out_tps=$(grep -oP 'Output token throughput \(tok/s\):\s+\K[0-9.]+' $LOG | head -1)
    in_tps=$(grep -oP 'Input token throughput \(tok/s\):\s+\K[0-9.]+' $LOG | head -1)
    tot_tps=$(grep -oP 'Total token throughput \(tok/s\):\s+\K[0-9.]+' $LOG | head -1)
    med_ttft=$(grep -oP 'Median TTFT \(ms\):\s+\K[0-9.]+' $LOG | head -1)
    med_tpot=$(grep -oP 'Median TPOT \(ms\):\s+\K[0-9.]+' $LOG | head -1)
    med_e2e=$(grep -oP 'Median E2E Latency \(ms\):\s+\K[0-9.]+' $LOG | head -1)
    p99_ttft=$(grep -oP 'P99 TTFT \(ms\):\s+\K[0-9.]+' $LOG | head -1)
    p99_tpot=$(grep -oP 'P99 TPOT \(ms\):\s+\K[0-9.]+' $LOG | head -1)

    echo "$c,$n,${out_tps:-na},${in_tps:-na},${tot_tps:-na},${med_ttft:-na},${med_tpot:-na},${med_e2e:-na},${p99_ttft:-na},${p99_tpot:-na}" >> $RESULTS
done

echo "" >&2
echo "=========== SWEEP SUMMARY ===========" >&2
cat $RESULTS
