"""nixl Python stub for AMD aggregated-only PoC.

The PyPI `nixl` package only ships CUDA wheels (`nixl-cu12`). On AMD,
`dynamo.nixl_connect.__init__` does `import nixl._api as nixl_api`
unconditionally at import time. This stub satisfies that import without
actually providing RDMA functionality. Sufficient for SGLang/vLLM
single-node aggregated serving where KV transfer is never invoked.

For disaggregated serving on AMD, REPLACE this stub with real RIXL
(see ../../container/Dockerfile.rocm-vllm-rixl which builds RIXL from
https://github.com/ROCm/RIXL and provides `from rixl import *` shim).

Install:
    SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')
    mkdir -p $SITE/nixl
    cp __init__.py _api.py _bindings.py $SITE/nixl/
"""
from . import _api, _bindings  # noqa: F401
