#!/usr/bin/env python3
"""Simulate vLLM's order: allocate model-weight-sized tensor first, do model
ops on it, THEN register a KV-pool region via NIXL.  Tests whether VRAM state
after weight loading interacts with ionic registration.
"""
import os, sys, time
import torch

WEIGHT_GB = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
KV_MB     = int(sys.argv[2])   if len(sys.argv) > 2 else 2638

print(f"weight_gb={WEIGHT_GB} kv_mb={KV_MB}", flush=True)

torch.cuda.set_device(0)
free0, _ = torch.cuda.mem_get_info()
print(f"free VRAM at start: {free0/1024**3:.2f} GiB", flush=True)

# Allocate "weights" - many small tensors that get scattered, then concatenated
# (mimics safetensors load behavior in real workers)
chunks = []
chunk_bytes = 256 * 1024 * 1024  # 256 MiB chunks
n_chunks = int(WEIGHT_GB * 1024 / 256)
for i in range(n_chunks):
    t = torch.randn(chunk_bytes // 2, dtype=torch.bfloat16, device='cuda:0')
    chunks.append(t)
torch.cuda.synchronize()
free1, _ = torch.cuda.mem_get_info()
print(f"free VRAM after {WEIGHT_GB} GiB 'weights' ({n_chunks} chunks): {free1/1024**3:.2f} GiB", flush=True)

# Do some compute on weights (mimic prefill warmup)
for c in chunks[:4]:
    _ = (c * 2.0).sum()
torch.cuda.synchronize()

# Now allocate the KV-pool tensor
print(f"allocating KV pool {KV_MB} MiB ...", flush=True)
kv = torch.zeros(KV_MB * 1024 * 1024 // 2, dtype=torch.bfloat16, device='cuda:0')
torch.cuda.synchronize()
free2, _ = torch.cuda.mem_get_info()
print(f"free VRAM after KV alloc: {free2/1024**3:.2f} GiB; KV addr=0x{kv.data_ptr():x}", flush=True)

# Register via NIXL
from nixl._api import nixl_agent
agent = nixl_agent("after_weights_probe")
backend_handle = agent.backends.get("UCX")
print(f"agent ready (backend={backend_handle}); calling register_memory ...", flush=True)

caches_data = [(kv.data_ptr(), KV_MB * 1024 * 1024, 0, "")]
descs = agent.get_reg_descs(caches_data, "VRAM")
try:
    agent.register_memory(descs, backends=["UCX"])
    print(f"  SUCCESS: registered {KV_MB} MiB after {WEIGHT_GB} GiB weights")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    sys.exit(1)
agent.deregister_memory(descs)
print("done")
