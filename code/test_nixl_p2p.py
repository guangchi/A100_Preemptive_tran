#!/usr/bin/env python3
"""Legacy single-process probe. Use test_nixl_nvlink.py to verify the link."""
import os
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '6,7')
os.environ['UCX_TLS'] = 'cuda_copy,cuda_ipc,tcp'
os.environ['UCX_NET_DEVICES'] = 'lo'
os.environ['UCX_LOG_LEVEL'] = 'info'

import torch
import nixl_cu13 as nixl

# 创建两个 agent 模拟 P 和 D
cfg = nixl.nixl_agent_config(backends=['UCX'])
agent_p = nixl.nixl_agent('P_side', cfg)
agent_d = nixl.nixl_agent('D_side', cfg)

print("Both agents created successfully")
print(f"P plugins: {agent_p.get_plugin_list()}")

# 在 GPU6 上创建源 buffer
torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES=6,7 -> 0 is GPU6
size = 128 * 1024 * 1024  # 128 MB
src = torch.randn(size // 4, device='cuda:0', dtype=torch.float32)
torch.cuda.synchronize()
print(f"Source buffer on GPU6: {size/1e6:.0f} MB")

# 在 GPU7 上创建目标 buffer
dst = torch.empty(size // 4, device='cuda:1', dtype=torch.float32)
torch.cuda.synchronize()
print(f"Dest buffer on GPU7: {size/1e6:.0f} MB")

# 获取 VRAM 类型
ucx_backend = agent_p.backends['UCX']
print(f"UCX backend handle created")

# 注册内存
# nixl agent 用 register_mem 来注册 buffer
import nixl_cu13.nixl_bind as nixlBind
src_addr = src.data_ptr()
dst_addr = dst.data_ptr()
print(f"src ptr: {hex(src_addr)}, dst ptr: {hex(dst_addr)}")

# 创建 descriptor list
dlist = agent_p.get_dlist(nixlBind.VRAM_SEG)
dlist.add_descr(nixlBind.nixlBlob(src_addr, size, 0, 0), 0)
print("Source descriptor list created")

# 注册内存到 UCX backend
agent_p.register_mem(dlist)
print("Source memory registered to P agent")

dlist_d = agent_d.get_dlist(nixlBind.VRAM_SEG)
dlist_d.add_descr(nixlBind.nixlBlob(dst_addr, size, 0, 0), 0)
agent_d.register_mem(dlist_d)
print("Dest memory registered to D agent")

print("\n=== Transfer Test ===")
# 填充源数据
src.fill_(3.14)
torch.cuda.synchronize()
dst.fill_(0.0)
torch.cuda.synchronize()

# 传输 - P 写到 D
t0 = time.perf_counter()
# 这里需要建立 agent 间连接，单进程内模拟
# 实际 P/D 分离是两个进程，通过 agent name 通信
print("Attempting transfer from P to D...")

# 尝试 make/get prepped list 并 transfer
src_prepped = agent_p.prep_xfer_dlist(ucx_backend, dlist)
print("Source prepped for transfer")

# 注意：实际传输需要两个 agent 互相知道对方
# 在 vLLM P/D 中通过 metadata exchange 完成
# 这里只验证 backend 是否支持 NVLink 路径
print("\nBackend initialized with cuda_copy,cuda_ipc,tcp")
print("VRAM_SEG supported, GPU memory registration works")
print(f"Total setup time: {time.perf_counter()-t0:.3f}s")

