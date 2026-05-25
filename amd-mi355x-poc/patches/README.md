# Patches used by AMD examples

Two pieces, applied at runtime inside the container — neither modifies any
source file in upstream `ai-dynamo/dynamo`.

> The `nixl_stub/` and `zzz_typing_self_compat.pth` shims that earlier
> versions of this PoC shipped are **retired**: they're no longer needed
> on `ai-dynamo/dynamo:main` (the `typing.Self` issue is fixed in
> [#9929](https://github.com/ai-dynamo/dynamo/pull/9929); the unconditional
> `nixl._api` import is made lazy in the same PR). If you're running
> against an older Dynamo that still has those issues, fetch the shims
> from this directory's git history.

## `ibv_ionic_compat.c` — LD_PRELOAD interposer for AMD Pensando ionic NICs

**Used by**: `examples/vllm/disagg_rocm.sh` (any RIXL-on-ionic deployment).

Strips `IBV_ACCESS_REMOTE_ATOMIC` (0x8) from the access flags passed to
`ibv_reg_mr`, `ibv_reg_mr_iova2`, and `ibv_reg_dmabuf_mr` before forwarding
to the real libibverbs implementations.

Pensando ionic NICs reject the REMOTE_ATOMIC bit with `EINVAL`; UCX
hardcodes the bit; RIXL never uses RDMA atomic ops, so stripping it is
safe. Without this shim — specifically without the `ibv_reg_dmabuf_mr`
wrapper, since UCX uses dmabuf-based MR registration for GPU VRAM —
the vLLM disagg path fails at `register_kv_caches → register_memory(VRAM)`
with `nixlBackendError: NIXL_ERR_BACKEND`.

Build + install:

```bash
gcc -shared -fPIC -O2 -o ibv_ionic_compat.so ibv_ionic_compat.c -ldl
export LD_PRELOAD=$(pwd)/ibv_ionic_compat.so
```

The example scripts in `../examples/vllm/` do this automatically inside
the container at startup.

Origin: extracted and extended from the JohnQinAMD fork's
`components/src/dynamo/sglang/nixl_rocm_staging.py` (`_IBV_INTERPOSER_SRC`).
The fork's version wrapped only `ibv_reg_mr` and `ibv_reg_mr_iova2`; the
`ibv_reg_dmabuf_mr` wrapper here is added because GPU VRAM goes through
the dmabuf path.

**Upstream path**: this belongs in either a small native helper alongside
the AMD Dockerfile, or — better — a vendor-aware UCX patch in `ROCm/ucx`.
Tracked as a follow-up.

## `fork-patches/` — Mooncake ROCm DRAM staging adapter (640 + 352 LoC)

**Used by**: `examples/sglang/disagg_rocm.sh` (SGLang + Mooncake disagg).

Files copied as-is from
[`JohnQinAMD/dynamo` branch `amd-dynamo`](https://github.com/JohnQinAMD/dynamo/tree/amd-dynamo/components/src/dynamo/sglang/)
(Apache 2.0, NVIDIA copyright):

| File | LoC | Purpose |
|---|---|---|
| `mooncake_rocm_staging.py` | 640 | DRAM staging wrapper around Mooncake's Python API |
| `rocm_dram_staging_common.py` | 352 | Shared helpers (mmap+mlock buffers, chunked MR registration, slab→GPU copy) |
| `nixl_rocm_staging.py` | 1225 | SGLang-specific monkey-patches; reference only, **NOT used by the polished examples** |
| `nixl_dram_staging.py` | 188 | reference only |

What the staging adapter does (the two files we actually use):

- KV buffer mirroring in host DRAM (mmap + mlock; ionic rejects
  `hipHostMalloc`-allocated regions for MR registration)
- Chunked MR registration ≤190 MB to fit ionic's per-device MR limit
  (~250 MB)
- Pre-registered receive slab (~200 MB)
- Subnet-aware ionic device selection for cross-node transfers
- Slab → GPU direct copy via `hipMemcpyAsync` (avoids a 7.6 GB/s memmove
  on the device-to-host path)

**Upstream path**: proposed as a new `dynamo.sglang.transports.mooncake_rocm`
submodule (opt-in via `SGLANG_MOONCAKE_ROCM_STAGING=1`). RFC will be filed
at `sgl-project/sglang` — if absorbed upstream, this directory disappears.
Tracked as P1a in the AMD-side roadmap.
