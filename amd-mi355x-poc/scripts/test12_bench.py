"""Test 12 reproduction bench: DSR1 disagg, ISL≈1024, OSL=1024, c=1, 2 warmup + 10 timed.

Matches the fork's `scripts/benchmark/bench.sh` parameters as closely as possible
without depending on the (broken) sgl-dev `sglang.bench_serving` import.

Reports: TTFT, TPOT, ITL, throughput (tokens/sec aggregate).
"""
import argparse, json, random, statistics, time
import requests
import concurrent.futures as cf

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-R1-0528"

# Generate a long prompt of ~ISL tokens via random word filler.
WORDS = ("Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
         "incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis "
         "nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat "
         "duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore "
         "eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt "
         "in culpa qui officia deserunt mollit anim id est laborum at vero eos et "
         "accusamus et iusto odio dignissimos ducimus qui blanditiis praesentiums").split()


def make_prompt(target_tokens: int, ratio: float = 0.8, seed: int = 0) -> str:
    """Create a prompt of approximately target_tokens (* random factor in [ratio, 1.0])."""
    random.seed(seed)
    n = int(target_tokens * random.uniform(ratio, 1.0))
    # Roughly 1 word ≈ 1.3 tokens for English; generate ceil(n / 1.3) words.
    nw = max(8, int(n / 1.3))
    return " ".join(random.choices(WORDS, k=nw))


def send(i: int, isl: int, osl: int, ratio: float, stream: bool = True,
         ignore_eos: bool = False) -> dict:
    prompt = make_prompt(isl, ratio=ratio, seed=i)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": osl,
        "temperature": 0.0,
        "stream": stream,
    }
    if ignore_eos:
        # SGLang/vLLM extension: forces decode to run for full max_tokens.
        # Matches fork's bench.sh `--ignore-eos` behavior.
        body["ignore_eos"] = True
        body["min_tokens"] = osl
    t0 = time.perf_counter()
    if stream:
        r = requests.post(URL, json=body, timeout=600, stream=True)
        if r.status_code != 200:
            return {"ok": False, "ms": (time.perf_counter() - t0) * 1000, "err": r.text[:200]}
        first_token_t = None
        last_token_t = None
        n_chunks = 0
        out_text_parts = []
        for line in r.iter_lines():
            if not line: continue
            line = line.decode() if isinstance(line, bytes) else line
            if not line.startswith("data: "): continue
            data = line[len("data: "):].strip()
            if data == "[DONE]": break
            try:
                j = json.loads(data)
                delta = j.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta is not None:
                    if first_token_t is None: first_token_t = time.perf_counter()
                    last_token_t = time.perf_counter()
                    n_chunks += 1
                    out_text_parts.append(delta)
            except Exception:
                continue
        total_ms = (time.perf_counter() - t0) * 1000
        ttft_ms = (first_token_t - t0) * 1000 if first_token_t else None
        decode_ms = (last_token_t - first_token_t) * 1000 if first_token_t and last_token_t else None
        out_tokens = n_chunks  # one token per chunk in OAI streaming
        tpot_ms = decode_ms / max(out_tokens - 1, 1) if decode_ms else None
        return {"ok": True, "total_ms": total_ms, "ttft_ms": ttft_ms,
                "decode_ms": decode_ms, "out_tokens": out_tokens, "tpot_ms": tpot_ms}
    else:
        r = requests.post(URL, json=body, timeout=600)
        return {"ok": r.status_code == 200, "total_ms": (time.perf_counter() - t0) * 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isl", type=int, default=1024)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--ignore-eos", action="store_true",
                    help="force decode to run for full --osl (matches fork's bench.sh --ignore-eos)")
    args = ap.parse_args()

    print(f"=== Test 12 reproduction: DSR1 disagg, ISL={args.isl}, OSL={args.osl}, c={args.conc} ===")
    print(f"Warmup: {args.warmup} requests")
    for i in range(args.warmup):
        r = send(i, args.isl, args.osl, args.ratio, ignore_eos=args.ignore_eos)
        print(f"  warmup {i}: ok={r.get('ok')} ttft={r.get('ttft_ms', 0):.0f}ms total={r.get('total_ms', 0):.0f}ms out={r.get('out_tokens', 0)}")

    print(f"\nTimed: {args.num_prompts} requests at c={args.conc}")
    t_wall = time.perf_counter()
    with cf.ThreadPoolExecutor(args.conc) as pool:
        results = list(pool.map(lambda i: send(i + args.warmup, args.isl, args.osl, args.ratio,
                                              ignore_eos=args.ignore_eos),
                                range(args.num_prompts)))
    wall = time.perf_counter() - t_wall

    ok = [r for r in results if r.get("ok")]
    if not ok:
        print("ALL FAILED")
        for r in results[:3]: print("  err:", r.get("err"))
        return

    out_total = sum(r["out_tokens"] for r in ok)
    in_total = args.num_prompts * args.isl  # approximate
    ttft = sorted(r["ttft_ms"] for r in ok if r["ttft_ms"])
    tpot = sorted(r["tpot_ms"] for r in ok if r["tpot_ms"])
    total = sorted(r["total_ms"] for r in ok)
    print(f"\nresults ({len(ok)}/{args.num_prompts} ok, wall={wall:.1f}s):")
    print(f"  output tokens total: {out_total} ({statistics.mean(r['out_tokens'] for r in ok):.0f} avg)")
    print(f"  TTFT  P50={ttft[len(ttft)//2]:.0f}ms  P95={ttft[int(0.95*len(ttft))]:.0f}ms  mean={statistics.mean(ttft):.0f}ms")
    print(f"  TPOT  P50={tpot[len(tpot)//2]:.2f}ms P95={tpot[int(0.95*len(tpot))]:.2f}ms mean={statistics.mean(tpot):.2f}ms")
    print(f"  total P50={total[len(total)//2]:.0f}ms P95={total[int(0.95*len(total))]:.0f}ms")
    print(f"\nThroughput:")
    print(f"  output tok/s (per request avg): {1000.0 / statistics.mean(tpot):.1f}")
    print(f"  output tok/s (aggregate):       {out_total / wall:.1f}")
    print(f"  total tok/s (in+out aggregate): {(in_total + out_total) / wall:.1f}")
    print(f"\n=== Compare to fork Test 12: 97.7 tok/s @ c=1, TPOT 7.11 ms ===")


if __name__ == "__main__":
    main()
