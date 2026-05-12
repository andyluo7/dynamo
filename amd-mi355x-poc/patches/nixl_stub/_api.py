"""nixl._api stub.

Agg path doesn't exercise these symbols. They're defined to satisfy the
import-time `from nixl._api import ...` attribute lookups in
`dynamo.nixl_connect`. Any actual call into nixl_agent() will RuntimeError
with a clear message pointing at the disaggregated-serving requirement.
"""


class nixl_agent:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "nixl is stubbed (AMD aggregated-only PoC). "
            "For disaggregated serving on AMD MI355X+ionic, install RIXL "
            "(https://github.com/ROCm/RIXL) and replace this stub package "
            "with the `from rixl import *` shim — see "
            "amd-mi355x-poc/container/Dockerfile.rocm-vllm-rixl"
        )


class nixl_agent_config:
    def __init__(self, *args, **kwargs):
        pass


class nixl_xfer_handle:
    pass


class nixl_reg_dlist:
    pass


class nixl_xfer_dlist:
    pass
