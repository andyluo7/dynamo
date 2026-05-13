#!/usr/bin/env python3
"""
nixl_pytorch_probe: mirror vLLM NixlConnector's actual register_memory call
path WITHOUT loading a model.

Allocates VRAM via torch.cuda (= ROCm/HIP under the hood on AMD), builds NIXL
descriptors, and calls nixl_wrapper.register_memory(...) on a list of regions
sized to match DSR1 TP=8 (~2.5 GiB per region).

Tests two hypotheses:
  - allocator: does PyTorch-allocated memory behave differently than hipMalloc?
  - multi-device: does failure depend on UCX_NET_DEVICES (1 vs 9 ionic devs)?

Run:
  UCX_NET_DEVICES=ionic_0:1 python3 nixl_pytorch_probe.py 2638 1
  UCX_NET_DEVICES=all       python3 nixl_pytorch_probe.py 2638 1
"""
import os
import sys

import torch

print(f"torch={torch.__version__} cuda_avail={torch.cuda.is_available()} dev={torch.cuda.device_count()}")
print(f"UCX_NET_DEVICES={os.environ.get('UCX_NET_DEVICES','(unset)')}")
print(f"UCX_TLS={os.environ.get('UCX_TLS','(unset)')}")

mb_per_region = int(sys.argv[1]) if len(sys.argv) > 1 else 2638
n_regions     = int(sys.argv[2]) if len(sys.argv) > 2 else 1
bytes_per     = mb_per_region * 1024 * 1024

# Allocate one big tensor per region on cuda:0 (= HIP device 0 on ROCm)
torch.cuda.set_device(0)
tensors = []
for i in range(n_regions):
    t = torch.zeros(bytes_per // 2, dtype=torch.bfloat16, device='cuda:0')
    tensors.append(t)
    print(f"  allocated region {i}: addr=0x{t.data_ptr():x} bytes={bytes_per} ({mb_per_region} MiB)")
torch.cuda.synchronize()

print("\nimporting nixl ...")
from nixl._api import nixl_agent, nixl_agent_config
print("creating nixl agent (mimicking vLLM: num_threads=4) ...")
n_threads = int(os.environ.get("PROBE_NUM_THREADS", "4"))
config = nixl_agent_config(num_threads=n_threads, capture_telemetry=True)
print(f"  config: num_threads={n_threads} capture_telemetry=True")
import uuid
agent = nixl_agent(str(uuid.uuid4()), config)
print(f"  backends available: {agent.get_plugin_list()}")
print(f"  agent.backends keys: {list(agent.backends.keys())}")
print(f"  agent.backends['UCX']: {agent.backends.get('UCX')}")
backend_handle = agent.backends.get("UCX")
print(f"  using UCX backend handle: {backend_handle}")

# Build descs in same shape vLLM uses: list of (addr, size, device_id, "")
caches_data = [(t.data_ptr(), bytes_per, 0, "") for t in tensors]
print(f"\nbuilding reg descs for {len(caches_data)} regions of {mb_per_region} MiB each ...")
descs = agent.get_reg_descs(caches_data, "VRAM")

print("\ncalling register_memory ...")
try:
    # Try the same call shape vLLM uses (positional descs, keyword backends as list of strings)
    agent.register_memory(descs, backends=["UCX"])
    print(f"  SUCCESS: registered {n_regions} x {mb_per_region} MiB on {os.environ.get('UCX_NET_DEVICES','default')} devices")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    sys.exit(1)

print("\nderegistering ...")
agent.deregister_memory(descs)
print("done")
