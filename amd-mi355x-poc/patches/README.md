# Runtime patches used by the AMD MI355X PoC

All of these are runtime patches applied **inside the container** at startup
by the launch scripts in `../scripts/`. None of them modify any source file
in the upstream `dynamo` tree.

## `nixl_stub/` — Minimal `nixl` Python package (4 files, ~60 LoC)

Used in: Phase 1 (SGLang agg), Phase 2 (vLLM agg), Phase 3 (SGLang disagg).

`dynamo.nixl_connect.__init__.py` line 42 does `import nixl._api as nixl_api`
unconditionally. The PyPI `nixl` package only ships CUDA wheels (`nixl-cu12`),
which won't install on AMD. This 4-file stub satisfies the import without
providing real RDMA — sufficient for any path that doesn't actually invoke
`nixl_agent()` (i.e. all aggregated serving and Mooncake-based disagg).

For Phase 4 (vLLM disagg) we replace this stub with a real `nixl→rixl` shim
that re-exports from the RIXL build (see `../container/Dockerfile.rocm-vllm-rixl`).

**Upstream candidate:** make `dynamo.nixl_connect.import nixl._api` lazy
(try/except, fall back to a no-op stub) so AMD installs need no shim. ~5 LoC
patch to upstream — the smallest landable PR from this PoC.

## `ibv_ionic_compat.c` — LD_PRELOAD interposer (~60 LoC)

Used in: Phase 4 (vLLM disagg).

Strips `IBV_ACCESS_REMOTE_ATOMIC` (0x8) from the access flags passed to
`ibv_reg_mr`, `ibv_reg_mr_iova2`, **and `ibv_reg_dmabuf_mr`** before
forwarding to the real libibverbs implementations. Pensando ionic NICs
reject the REMOTE_ATOMIC bit with EINVAL; UCX hardcodes the bit; RIXL
never uses RDMA atomic ops, so stripping it is safe.

Extracted and extended from the JohnQinAMD fork's
`components/src/dynamo/sglang/nixl_rocm_staging.py` (`_IBV_INTERPOSER_SRC`).
The fork's version wraps only `ibv_reg_mr` and `ibv_reg_mr_iova2`. We added
`ibv_reg_dmabuf_mr` because UCX uses dmabuf-based MR registration for GPU
VRAM, and that path was not covered. Without this third wrapper, the vLLM
disagg path fails at `register_kv_caches → register_memory(VRAM descs)` with
`nixlBackendError: NIXL_ERR_BACKEND`.

**Upstream candidate:** ship as a small native helper alongside the
`Dockerfile.rocm-vllm`, or document the LD_PRELOAD workaround until AMD's
vendor-aware UCX patch lands in `ROCm/ucx`.

Build:
```bash
gcc -shared -fPIC -O2 -o ibv_ionic_compat.so ibv_ionic_compat.c -ldl
export LD_PRELOAD=$(pwd)/ibv_ionic_compat.so
```

## `zzz_typing_self_compat.pth` — Python 3.10 compat shim (1 line)

Used in: Phase 1, Phase 3, Phase 4 (any container with Python 3.10).

`ai-dynamo` 1.1.1 does `from typing import Self` (Python 3.11+ only). The
`rocm/sgl-dev` containers ship Python 3.10. This `.pth` file is processed by
`site.py` at Python startup, BEFORE any user imports — adds `typing.Self`
from `typing_extensions` if missing.

**Upstream candidate:** ai-dynamo metadata should either bump
`python_requires>=3.11` or use `typing_extensions.Self` directly so 3.10
containers work. ~1 LoC patch.

Install:
```bash
SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')
cp zzz_typing_self_compat.pth $SITE/
pip install typing_extensions
```

## `fork-patches/` — Files copied as-is from the JohnQinAMD fork

Used in: Phase 3 (SGLang+Mooncake disagg).

| File | LoC | Used in |
|---|---|---|
| `mooncake_rocm_staging.py` | 640 | Phase 3 |
| `rocm_dram_staging_common.py` | 352 | Phase 3 |
| `nixl_rocm_staging.py` | 1225 | reference / Phase 4 study (SGLang-specific monkey-patches; NOT used in Phase 4 PASS path) |
| `nixl_dram_staging.py` | 188 | reference |

Source: `https://github.com/JohnQinAMD/dynamo/tree/amd-dynamo/components/src/dynamo/sglang/`
(Apache 2.0, NVIDIA copyright, included unmodified).

These provide AMD-specific Mooncake adaptation:
- KV buffer mirroring in host DRAM (mmap+mlock; ionic rejects `hipHostMalloc`)
- Chunked MR registration ≤190 MB to fit ionic's per-device MR limit (~250 MB)
- Pre-registered receive slab (~200 MB)
- Subnet-aware ionic device selection (cross-node)
- Slab → GPU direct copy via `hipMemcpyAsync` (avoids 7.6 GB/s memmove)

**Upstream candidate:** new submodule `dynamo.sglang.transports.mooncake_rocm`
opt-in via `SGLANG_MOONCAKE_ROCM_STAGING=1`. Largest body of "real work"
from the AMD-Dynamo project, ~1000 LoC.
