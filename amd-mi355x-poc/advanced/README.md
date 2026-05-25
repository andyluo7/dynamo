# Advanced — reproducing the PoC sweeps

The scripts here ran the original PoC matrix. They're preserved verbatim
so anyone can reproduce the numbers in [`../docs/findings.md`](../docs/findings.md)
and the side-by-side transport comparisons.

If you just want a single working example, use [`../examples/`](../examples/)
instead.

## Layout

```
advanced/
├── benchmarks/         ← single-c benchmark drivers used as building blocks
│   ├── dsr1_bench.py
│   ├── m25_bench.py
│   ├── phase3_bench.py
│   ├── phase4_bench.py
│   └── test12_bench.py  ← streaming bench matching the JohnQinAMD fork's bench.sh
├── sweeps/             ← full concurrency-sweep wrappers + alt-topology variants
│   ├── m25_sweep.sh                  ← MiniMax-M2.5 sweep across c
│   ├── test12_sweep.sh               ← DSR1 sweep across c (matches fork's Test 12)
│   ├── mori_sweep.sh                 ← MoRI agg sweep
│   ├── mori_fork_sweep.sh            ← MoRI fork-faithful sweep
│   ├── phase2_e2e.sh                 ← vLLM agg eager (smoke variant)
│   ├── phase3_disagg.sh              ← SGLang+Mooncake disagg with Qwen3-0.6B
│   ├── phase3_dsr1_disagg.sh         ← SGLang+Mooncake disagg DSR1 conservative
│   ├── phase4_disagg.sh              ← vLLM+RIXL disagg with Qwen3-0.6B
│   ├── phase4_dsr1_disagg.sh         ← vLLM+RIXL disagg DSR1 (currently broken; see docs/findings.md ³)
│   ├── phase6_mori_{agg,disagg}.sh   ← MoRI agg / disagg
│   ├── phase6b_mori_fork_repro.sh    ← MoRI fork-faithful reproduction
│   └── phase6c_mori_sbatch.sh        ← MoRI sbatch wrapper
└── debug-probes/       ← standalone RDMA + MR probes used for DSR1 cross-validation
    ├── ionic_mr_probe.c                ← DRAM MR size sweep on ionic_0 (4 GiB OK)
    ├── ionic_rocm_mr_probe.c           ← ROCm/VRAM MR size sweep (8 GiB OK)
    ├── ionic_concurrent_probe.cpp      ← repeat-registration loop (32×2 GiB OK)
    ├── ionic_multiproc_probe.sh        ← N parallel libibverbs procs (8×2.58 GiB OK)
    ├── ionic_atomic_probe.cpp          ← access-flag combinations (0xf OK on ionic)
    ├── nixl_pytorch_probe.py           ← NIXL+PyTorch via RIXL (mimics vLLM call shape)
    ├── nixl_multiproc_probe.sh         ← N parallel NIXL+PyTorch procs (matches vLLM TP=8)
    ├── nixl_after_weights_probe.py     ← weight-pressure + KV register order (post-load OK)
    └── rixl_probe.py                   ← direct RIXL register-memory probe (Phase 4 debug)
```

## When to use which

| Want to … | Use … |
|---|---|
| Run one working example | [`../examples/`](../examples/) |
| Reproduce a single concurrency point from `findings.md` | `benchmarks/<model>_bench.py` directly |
| Reproduce a full sweep | `sweeps/<model>_sweep.sh` |
| Investigate an RDMA / MR / multiprocess issue | `debug-probes/` |

The debug-probes were what let us eliminate every transport-layer
hypothesis for the DSR1 vLLM+RIXL startup failure. See
[`../docs/archive/12-dsr1-vllm-rixl-cross-validation.md`](../docs/archive/12-dsr1-vllm-rixl-cross-validation.md)
for the 11-hypothesis elimination matrix.
