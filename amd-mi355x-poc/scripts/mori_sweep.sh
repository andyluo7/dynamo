#!/bin/bash
# Concurrency sweep for SGLang + MoRI-IO disagg DSR1.
# Targets fork's published: 178 @c=4, 672 @c=16, 2196 @c=128.
set -uo pipefail
RESULTS=/tmp/mori_sweep_results.csv
echo "conc,n_prompts,P50_ms,P95_ms,TPOT_P50_ms,TTFT_P50_ms,output_tok_per_req,output_tok_aggregate,total_tok_aggregate,success" > $RESULTS

cat > /tmp/mori_bench.py <<'PYEOF'
import argparse, json, random, statistics, time
import requests, concurrent.futures as cf

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-R1-0528"
WORDS = ("Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
         "incididunt ut labore et dolore magna aliqua").split()

def make_prompt(target_tokens, ratio=0.8, seed=0):
    random.seed(seed)
    n = int(target_tokens * random.uniform(ratio, 1.0))
    nw = max(8, int(n / 1.3))
    return " ".join(random.choices(WORDS, k=nw))

def send(i, isl, osl, ratio, ignore_eos=True):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": make_prompt(isl, ratio, i)}],
            "max_tokens": osl, "temperature": 0.0, "stream": True}
    if ignore_eos:
        body["ignore_eos"] = True; body["min_tokens"] = osl
    t0 = time.perf_counter()
    r = requests.post(URL, json=body, timeout=600, stream=True)
    if r.status_code != 200:
        return {"ok": False, "ms": (time.perf_counter()-t0)*1000, "err": r.text[:200]}
    first_t = last_t = None; n_chunks = 0
    for line in r.iter_lines():
        if not line: continue
        line = line.decode() if isinstance(line, bytes) else line
        if not line.startswith("data: "): continue
        data = line[6:].strip()
        if data == "[DONE]": break
        try:
            j = json.loads(data)
            d = j.get("choices",[{}])[0].get("delta",{}).get("content")
            if d is not None:
                if first_t is None: first_t = time.perf_counter()
                last_t = time.perf_counter(); n_chunks += 1
        except: continue
    total_ms = (time.perf_counter()-t0)*1000
    ttft = (first_t-t0)*1000 if first_t else None
    decode = (last_t-first_t)*1000 if first_t and last_t else None
    tpot = decode/max(n_chunks-1,1) if decode else None
    return {"ok": True, "total_ms": total_ms, "ttft_ms": ttft,
            "out_tokens": n_chunks, "tpot_ms": tpot}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isl", type=int, default=1024)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    print(f"=== MoRI sweep: ISL={args.isl} OSL={args.osl} c={args.conc} ===")
    for i in range(args.warmup):
        r = send(i, args.isl, args.osl, 0.8)
        print(f"  warmup {i}: ok={r.get('ok')} ttft={r.get('ttft_ms',0):.0f}ms total={r.get('total_ms',0):.0f}ms")
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(args.conc) as p:
        results = list(p.map(lambda i: send(i+args.warmup, args.isl, args.osl, 0.8),
                             range(args.num_prompts)))
    wall = time.perf_counter() - t0
    ok = [r for r in results if r.get("ok")]
    if not ok:
        print("ALL FAILED"); print("err:", results[0].get("err") if results else ""); return
    out_total = sum(r["out_tokens"] for r in ok)
    in_total  = args.num_prompts * args.isl
    ttft = sorted(r["ttft_ms"] for r in ok if r["ttft_ms"])
    tpot = sorted(r["tpot_ms"] for r in ok if r["tpot_ms"])
    total = sorted(r["total_ms"] for r in ok)
    print(f"\nresults ({len(ok)}/{args.num_prompts} ok, wall={wall:.1f}s):")
    print(f"  output tokens total: {out_total} ({statistics.mean(r['out_tokens'] for r in ok):.0f} avg)")
    print(f"  TTFT  P50={ttft[len(ttft)//2]:.0f}ms  P95={ttft[int(0.95*len(ttft))]:.0f}ms")
    print(f"  TPOT  P50={tpot[len(tpot)//2]:.2f}ms P95={tpot[int(0.95*len(tpot))]:.2f}ms")
    print(f"  total P50={total[len(total)//2]:.0f}ms P95={total[int(0.95*len(total))]:.0f}ms")
    print(f"  output tok/s (per request avg): {1000.0/statistics.mean(tpot):.1f}")
    print(f"  output tok/s (aggregate):       {out_total/wall:.1f}")
    print(f"  total tok/s (in+out aggregate): {(in_total+out_total)/wall:.1f}")

main()
PYEOF

for c in 1 4 8 16 32 64 128; do
    n=$((c * 10))
    w=$((c < 4 ? 2 : c * 2))
    echo ""  >&2
    echo "=== c=$c n=$n warmup=$w ===" >&2
    out=$(timeout 1800 python3 /tmp/mori_bench.py --isl 1024 --osl 1024 --conc $c --num-prompts $n --warmup $w 2>&1)
    echo "$out" | tail -10 >&2
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
