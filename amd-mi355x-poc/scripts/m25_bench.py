import requests, time, concurrent.futures, statistics
URL="http://localhost:8000/v1/chat/completions"
MODEL="MiniMaxAI/MiniMax-M2.5"
def send(i, n=64):
    t0=time.time()
    r=requests.post(URL, json={"model":MODEL,"messages":[{"role":"user","content":f"Tell me a brief fact ({i}) in 1 sentence."}],"max_tokens":n,"temperature":0.7}, timeout=180)
    dt=time.time()-t0
    if r.status_code!=200: return {"ok":False,"ms":dt*1000,"err":r.text[:200]}
    j=r.json(); u=j.get("usage",{})
    return {"ok":True,"ms":dt*1000,"in":u.get("prompt_tokens"),"out":u.get("completion_tokens")}
print("warmup...")
for i in range(3):
    r=send(i,32); print(f"  warmup {i}: ok={r['ok']} ms={r['ms']:.0f}")
print()
print(f"{'conc':>4} {'N':>3} {'P50_ms':>8} {'P95_ms':>8} {'tps':>8} {'out_avg':>7} {'ok':>5}")
for conc in [1,4,8]:
    N=max(conc*3,8)
    t0=time.time()
    with concurrent.futures.ThreadPoolExecutor(conc) as p:
        results=list(p.map(send, range(N)))
    wall=time.time()-t0
    ok=[r for r in results if r["ok"]]
    if not ok: print(f"{conc:>4} {N:>3} FAIL"); print("err:", results[0].get("err")); continue
    times=sorted(r["ms"] for r in ok); p50=times[len(times)//2]; p95=times[int(0.95*len(times))]
    out_total=sum(r["out"] or 0 for r in ok); out_avg=statistics.mean(r["out"] for r in ok if r["out"])
    print(f"{conc:>4} {N:>3} {p50:>8.0f} {p95:>8.0f} {out_total/wall:>8.1f} {out_avg:>7.1f} {len(ok):>3}/{N}")
