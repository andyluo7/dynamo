# Docs

Curated docs for current users; `archive/` holds the original phase-by-phase
development reports (still useful for deep-dives).

## Curated

- [`architecture.md`](architecture.md) — how Dynamo + the AMD-specific
  pieces fit together (RIXL/UCX, Mooncake, ionic NICs).
- [`findings.md`](findings.md) — full results table per backend + topology,
  vs-fork comparison, side-by-side transport behavior.
- [`network-debug.md`](network-debug.md) — root-cause notes on ionic NIC
  RoCEv2 behavior (ECN/DCQCN vs PFC) and what it means for MoRI/Mooncake
  transport choice.

## Archive — original phase-by-phase reports

These were written incrementally as the PoC progressed. Read them if you
want the full dev journal; not necessary for running the examples.

| File | Phase | Topic |
|---|---|---|
| [00-poc-plan.md](archive/00-poc-plan.md) | — | original plan + scope decisions |
| [01-phase0-inventory.md](archive/01-phase0-inventory.md) | 0 | AAC1 hardware inventory + tooling |
| [02-phase1-sglang-agg.md](archive/02-phase1-sglang-agg.md) | 1 | SGLang + DSR1 single-node |
| [03-phase2-vllm-design.md](archive/03-phase2-vllm-design.md) | 2 | vLLM phase design rationale |
| [04-phase2-vllm-agg.md](archive/04-phase2-vllm-agg.md) | 2 | vLLM + MiniMax-M2.5 single-node |
| [05-phase3-sglang-disagg.md](archive/05-phase3-sglang-disagg.md) | 3 | SGLang + Mooncake 2-node disagg |
| [06-phase4-vllm-disagg.md](archive/06-phase4-vllm-disagg.md) | 4 | vLLM + RIXL/UCX 2-node disagg |
| [07-phase5-final-report.md](archive/07-phase5-final-report.md) | 5 | consolidated final report + PR breakdown |
| [08-phase34-escalation-results.md](archive/08-phase34-escalation-results.md) | 3+4 | escalation runs at production scale |
| [09-test12-reproduction.md](archive/09-test12-reproduction.md) | 3 | reproducing the JohnQinAMD fork's Test 12 (+8.2%) |
| [10-dsr1-concurrency-sweep.md](archive/10-dsr1-concurrency-sweep.md) | 3 | DSR1 sweep; hits Mooncake ionic ceiling |
| [10-dsr1-sweep-results.csv](archive/10-dsr1-sweep-results.csv) | 3 | raw per-c CSV |
| [11-m25-vllm-rixl-sweep.md](archive/11-m25-vllm-rixl-sweep.md) | 4 | M2.5 sweep; RIXL has no transport ceiling |
| [11-m25-sweep-results.csv](archive/11-m25-sweep-results.csv) | 4 | raw per-c CSV |
| [12-dsr1-vllm-rixl-cross-validation.md](archive/12-dsr1-vllm-rixl-cross-validation.md) | 4 | DSR1+RIXL crash root-cause (11-hypothesis elimination matrix) |
| [13-phase6-mori-progress.md](archive/13-phase6-mori-progress.md) | 6 | MoRI investigation progress |
| [14-phase6b-mori-fork-faithful.md](archive/14-phase6b-mori-fork-faithful.md) | 6b | definitive ECN/DCQCN root cause for MoRI c≥4 wall |
