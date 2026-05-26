"""Direct probe of what RIXL/nixl can register on this hardware."""
import os, ctypes, traceback
print("=== env ===")
for k in ['UCX_TLS', 'LD_PRELOAD', 'NIXL_PREFIX', 'HIP_VISIBLE_DEVICES']:
    print(f"  {k}={os.environ.get(k, '<unset>')}")
print()

import nixl
print("=== nixl module ===")
print(f"  file: {nixl.__file__}")
print(f"  attrs: {[a for a in dir(nixl) if not a.startswith('_')][:20]}")
print()

print("=== create agent ===")
try:
    agent = nixl.nixl_agent("probe-agent")
    print("  agent OK:", agent)
except Exception as e:
    traceback.print_exc()
    raise

print()
print("=== plugins/backends ===")
try:
    print("  plugins:", agent.get_plugin_list())
except Exception as e:
    print("  get_plugin_list err:", e)
try:
    print("  backends:", agent.get_backend_list())
except Exception as e:
    print("  get_backend_list err:", e)
print()

print("=== try DRAM register (1 MB malloc) ===")
buf = ctypes.create_string_buffer(1024*1024)
addr = ctypes.addressof(buf)
print(f"  addr={hex(addr)}")
try:
    desc = nixl.nixl_reg_dlist("DRAM", [(addr, len(buf), 0, "")])
    print(f"  desc={desc}")
    handle = agent.register_memory(desc)
    print("  DRAM register OK:", handle)
    agent.deregister_memory(handle)
    print("  DRAM deregister OK")
except Exception as e:
    print("  DRAM register FAIL:", type(e).__name__, e)
    traceback.print_exc()
print()

print("=== try VRAM register (1 MB torch.cuda) ===")
try:
    import torch
    t = torch.zeros(256*1024, dtype=torch.float32, device='cuda')  # 1 MB
    print(f"  tensor on device {t.device}, ptr={hex(t.data_ptr())}, bytes={t.numel()*4}")
    desc = nixl.nixl_reg_dlist("VRAM", [(t.data_ptr(), t.numel()*4, 0, "")])
    handle = agent.register_memory(desc)
    print("  VRAM register OK:", handle)
    agent.deregister_memory(handle)
    print("  VRAM deregister OK")
except Exception as e:
    print("  VRAM register FAIL:", type(e).__name__, e)
    traceback.print_exc()
