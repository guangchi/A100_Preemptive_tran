# P/D 分离 NVLink 传输实验方案

> **执行状态更新（2026-11-15）**：已完成物理 GPU6/7 的 NVLink 单项实验：baseline、sort_only、preempt_only 各三轮、25/50 ms 两场景，并补测 stream_fifo 三轮。24 次正式运行共 360 条请求及等量预热全部通过输出与协议审计；C8、单活动 WRITE、连续 offset、唯一 final 均已验证。各次正式负载 NVLink TX/RX 约 5.817500 GB，与填充 KV payload 匹配。排序改善 short/medium TTFT，代价是 long TTFT 增加；抢占相对流式 FIFO 的 short TTFT 改善约 3%，仍慢于整请求 baseline。完整数据及解释见[本轮报告](../results/nvlink-isolated-ablation-20261115/REPORT.md)及技术报告第 18 节。
>
> **当前配置优先于下文历史计划**：NIXL backend 仍为 `UCX`，独立 P/D 进程配合 `UCX_TLS=cuda,tcp`、`UCX_NET_DEVICES=lo`；不切换到名为 CUDA_IPC 的 backend。窗口 5 ms、批间隔 25/50 ms、输出 64 tokens、C8/单槽、writer 轮询 0.1 ms；开关为 `PD_LOCAL_SORT` 和 `PD_PRIORITY_PREEMPTION`。sort_only 与 preempt_only 均保留流式基础设施，不能视为 stock baseline 仅增加一个开关。FIFO 为主实验结束后的补测，存在时间混杂；本轮无同时开启组及同窗口 FCFS，不能估计完整因子交互。workload SHA256 为 `ad9d732ba5a55de1ac9e4c682a8171df38f0513de74c1d23251c2cbc2746e94a`。下文“尚未生效”、估算带宽及旧参数属于历史设计，不是本轮状态；历史实验未记录 transport，仍不能追溯确认。双进程微基准见[带宽测试记录](../results/nixl-bandwidth-20261115/RESULTS.md)。

修订日期：2026-11-13。本文是在原 UCX TCP 方案（Plan.txt）基础上，针对
A100-SXM4-80GB NVLink 环境的完整实验重设计，包含 baseline 和新完整方案的全部
变更说明。先不执行实验，仅记录方案设计。

本文记录的所有变更均以"与 Plan.txt 中已完成实验的差异"为基准，原方案实验结论
仍保留在 Plan.txt 中，不被本文覆盖。新实验结果将单独归档。

---

## 1. 硬件与环境

### 1.1 实验平台

| 项目 | 配置 |
|------|------|
| 机器 | 单机 8× NVIDIA A100-SXM4-80GB（每卡 80 GB HBM2e） |
| 使用卡 | GPU6（P，Prefill）、GPU7（D，Decode） |
| 物理 PCI 地址 | GPU6: `00000000:C9:00.0`；GPU7: `00000000:CF:00.0` |
| GPU 间互联 | NV12（12 条 NVLink 3 代），理论双向峰值 **600 GB/s** |
| PyTorch P2P 单向参考 | **275.8 GB/s**（GPU6→GPU7，1024 MiB×10，NVLink 计数匹配；不代表 NIXL） |
| P2P 访问 | `torch.cuda.can_device_access_peer(6, 7)` = True（已验证） |
| NUMA 拓扑 | GPU6/GPU7 同属 NUMA node 1，CPU affinity 32-63, 96-127 |
| OS / 驱动 | Linux 5.15，Driver 595.112.07，CUDA 13.0 |
| Python / Torch | Python 3.11，PyTorch 2.14.0+cu130 |
| vLLM | **0.33.0**（site-packages，含自研修改；原方案为 0.32.0） |
| NIXL | nixl-cu13 1.5.2 |
| 工作目录 | `/data01/home/zjj/zjj/code`（原方案为 `/home/ubuntu/zjj/code`） |
| conda 环境 | `/data01/home/zjj/miniconda3/envs/cc`（原方案为 `/home/ubuntu/miniconda3/envs/cc`） |

### 1.2 与原方案的关键差异

| 对比维度 | 原方案（Plan.txt） | 新方案 |
|----------|-------------------|--------|
| GPU 型号 | RTX 4090 × 2 | A100-SXM4-80GB × 2（取 GPU6/7） |
| GPU 显存 | 24 GB / 卡 | **80 GB / 卡** |
| BF16 算力（参考） | ~165 TFLOPS | **~312 TFLOPS**（约 1.9 倍） |
| KV cache 容量 | ~21 GB（0.90×24GB） | **~72 GB**（0.90×80GB），约 3.4 倍 |
| KV 传输路径 | UCX TCP + cuda_copy (loopback) | **NIXL + CUDA P2P（NVLink）** |
| 传输带宽量级 | ~10 GB/s（TCP staging 瓶颈，历史参考） | PyTorch P2P参考275.3 GB/s；早期单进程NIXL测得0.43–0.45 GB/s并回退TCP，双进程NIXL NVLink复测见12.5节 |
| TCP 分段参数 | UCX_TCP_TX/RX_SEG_SIZE=128k | **不再需要**，改用 P2P backend |
| GPU 编号 | P=GPU0, D=GPU1 | **P=GPU6, D=GPU7** |
| vLLM 版本 | 0.32.0 | **0.33.0** |
| CUDA 版本 | 11.8（nvcc）/ CUDA runtime | **13.0** |
| 工作目录 | `/home/ubuntu/zjj/code` | **`/data01/home/zjj/zjj/code`** |
| conda 环境路径 | `/home/ubuntu/miniconda3/envs/cc` | **`/data01/home/zjj/miniconda3/envs/cc`** |
| 显存使用比例 | 0.90 | 0.90（维持） |

---

## 2. 传输路径变更：从 UCX TCP 到 NVLink P2P

### 2.1 原路径的问题

原方案使用 `UCX_NET_DEVICES=lo` + `TCP+cuda_copy`，每次 WRITE 经过：

```
GPU(P) → [DtoH staging] → Host Memory → [TCP loopback] → Host Memory → [HtoD] → GPU(D)
```

这导致：
- 实测 8KB 分段时 `transfer()` 单次约 42 ms
- 换 128KB 分段后降至约 10 ms，但仍是主要瓶颈
- 大量 DtoH copy 被 profiler 记录为 169216 次 × 8KB 操作

### 2.2 新路径

A100 GPU6↔GPU7 通过 NVLink 直连，支持 `cudaMemcpyPeer` / GPUDirect P2P，
数据路径变为：

```
GPU6(P) → [NVLink P2P DMA，不经过 Host Memory] → GPU7(D)
```

NIXL 配置上，将 backend 从 `UCX` 切换到支持 CUDA P2P 的路径（`CUDA_IPC` 或
`P2P` backend）。具体实现见第 5 节启动脚本变更。

这条路径的关键特点：
- **零 Host Memory 拷贝**：不经过 DtoH / HtoD staging，消除了原方案最大的开销
- **DMA 驱动**：传输由 GPU DMA 引擎执行，CPU 只负责提交描述符，不参与数据搬运
- **延迟极低**：1.22 GB KV 预计 ~4.5 ms，原 TCP 路径需要 ~120 ms（约 27× 差距）

### 2.3 对实验设计的影响

纯 PyTorch P2P 参考带宽约275 GB/s；早期单进程 NIXL UCX 测量仅约0.45 GB/s且没有 NVLink 计数增量，原因是 CUDA IPC 被 same-process 条件排除。双进程 NIXL NVLink microbenchmark 已达到约77–201 GB/s（见12.5节），但仍不能直接推断模型 KV workload 的端到端延迟或瓶颈。原设计中的以下判断仍是待验证假设：

- 原方案中"UCX staging 是主要瓶颈"这一诊断结论不再适用
- C8 分块大小对 staging 开销的影响消失或大幅缩小
- 实验将更纯粹地体现**调度策略**（P 局部排序、传输仲裁）对 TTFT 的影响
- 计算排队、KV 供给时机成为更主导的因素

估算：10000-token long 的 KV 约 1.22 GB，NVLink 下传输耗时约 **4.5 ms**
（原 TCP 下约 120+ ms）。这意味着 KV 传输几乎不再成为延迟来源，实验结果
将更直接反映调度层面的优化效果。

---

## 3. 实验方案总体设计

延续原方案三方案对比框架，全部方案在新硬件/新传输路径下重跑。

### 3.1 三方案定义（保持不变）

| 项目 | baseline | 旧方案 | 新完整方案 |
|------|----------|--------|-----------|
| 传输路径 | NVLink P2P（整请求） | NVLink P2P（流式） | NVLink P2P（流式） |
| 整请求发送 | 是（prompt 全部完成后） | 否 | 否 |
| batch 提前发送 | 否 | 是 | 是 |
| short/medium 按 batch 聚合 | 否 | 是 | 是 |
| long C8 分块 | 否 | 是 | 是 |
| short 传输插队 + 三项保护 | 否 | 是 | 是 |
| **P 局部排序** | 关 | 关 | **开** |
| D 调度改动 | 无 | 无 | 无 |
| UCX TCP 分段 | **不适用** | **不适用** | **不适用** |

> baseline 同样使用 NVLink P2P 路径整请求发送，不回退到 TCP。
> 这样三方案的传输硬件条件完全一致，差异只在调度策略。

### 3.2 固定负载（与原方案相同）

继续使用原 15 请求固定负载文件：

```
workload.sweep-interleaved-8s3m4l.jsonl
SHA256: ad9d732ba5a55de1ac9e4c682a8171df38f0513de74c1d23251c2cbc2746e94a
```

| 编号 | 类别 | 输入 tokens | seed | 发送批次 |
|------|------|------------|------|---------|
| 00 | short | 128 | 1500 | 0 |
| 01 | long | 10000 | 1501 | 0 |
| 02 | short | 128 | 1502 | 0 |
| 03 | medium | 1107 | 1503 | 0 |
| 04 | short | 128 | 1504 | 0 |
| 05 | long | 10000 | 1505 | 1 |
| 06 | short | 128 | 1506 | 1 |
| 07 | long | 10000 | 1507 | 1 |
| 08 | short | 128 | 1508 | 1 |
| 09 | medium | 1107 | 1509 | 1 |
| 10 | short | 128 | 1510 | 2 |
| 11 | long | 10000 | 1511 | 2 |
| 12 | short | 128 | 1512 | 2 |
| 13 | medium | 1107 | 1513 | 2 |
| 14 | short | 128 | 1514 | 2 |

合计 44345 输入 tokens，8 short + 3 medium + 4 long，每条输出 64 tokens。

### 3.3 到达模式（与原方案相同）

分三批到达，测试两种批间隔场景：

| 场景 | 第1批起点 | 第2批起点 | 第3批起点 |
|------|----------|----------|----------|
| 密集场景（25 ms 间隔） | 0 ms | 25 ms | 50 ms |
| 稀疏场景（50 ms 间隔） | 0 ms | 50 ms | 100 ms |

每批包含原序的 5 条请求（00-04、05-09、10-14），批内并发提交，间隔 0 ms。

### 3.4 重复次数与轮换顺序

每场景每方案 **3 次正式重复**，共 2 场景 × 3 方案 × 3 轮 = **18 次正式运行**，
270 条正式请求。每轮轮换三方案顺序：

| 轮次 | 方案运行顺序 | 场景顺序 |
|------|------------|---------|
| 1 | baseline → 旧 → 新 | 25 ms → 50 ms |
| 2 | 旧 → 新 → baseline | 50 ms → 25 ms |
| 3 | 新 → baseline → 旧 | 25 ms → 50 ms |

每个方案启动前进行 **1 次预热**（两种场景各一次），共 270 条预热请求，
总计 540 条。

---

## 4. 参数配置变更清单

### 4.1 必须修改的参数

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| `CUDA_VISIBLE_DEVICES` | `0,1` 或 `GPU0,GPU1` | **`6,7`** | 使用 GPU6/7 |
| `UCX_TCP_TX_SEG_SIZE` | 128k | **删除/不设置** | 不使用 UCX TCP |
| `UCX_TCP_RX_SEG_SIZE` | 128k | **删除/不设置** | 不使用 UCX TCP |
| `UCX_NET_DEVICES` | `lo` | **删除/不设置** | 不使用 UCX |
| NIXL backend | `UCX` (默认) | **`CUDA_IPC` 或 P2P backend** | NVLink 直传 |
| `gpu_memory_utilization` | 0.90 | **0.90**（维持）| 80 GB 显存充裕 |

### 4.2 保持不变的参数

| 参数 | 值 | 说明 |
|------|----|----|
| `PD_LOCAL_SORT` | 新方案=1，旧/baseline=0 | 排序开关不变 |
| `PD_LOCAL_SORT_WINDOW_MS` | **5** | 本轮实验固定值 |
| `PD_STREAM_BY_BATCH` | 1 | 流式发送开关 |
| `PD_AGGREGATE_SHORT_MEDIUM` | 1 | 短中聚合 |
| `PD_PRIORITY_PREEMPTION` | 1 | 传输仲裁 |
| `CHUNK_SIZE_BLOCKS` | 8 | long C8 分块 |
| `LONG_REQUEST_PROTECT_PROGRESS` | 0.85 | 进度保护阈值 |
| `LONG_REQUEST_MAX_PREEMPTIONS` | 3 | 次数保护阈值 |
| `LONG_REQUEST_MAX_PREEMPT_WAIT_MS` | 50 | 等待保护阈值 |
| `PD_KV_SOURCE_FENCE` | event | CUDA event 可见性 |
| `PD_STREAM_MAX_IN_FLIGHT` | 1 | 单槽 WRITE |
| `VLLM_BATCH_INVARIANT` | 1 | 数值确定性 |
| `ENABLE_PREFIX_CACHING` | false | 关闭前缀缓存 |
| `PD_MEASURE_EXECUTION` | 1 | 执行轨迹采集 |
| vLLM 计算 token 预算 | 2048 | chunked prefill 上限 |
| 模型 | Llama-3.1-8B-Instruct BF16 | 同模型 |
| 输出长度 | 64 tokens | 同输出长度 |
| `temperature` | 0 | 确定性采样 |
| 逐请求 seed | 1500-1514 | 同 seed |

### 4.3 需要重新评估的参数

以下参数在原方案中与 UCX TCP staging 瓶颈或旧硬件条件强相关，NVLink 下需重新验证：

**`CHUNK_SIZE_BLOCKS=8`（C8）**

原方案 C8 是为了在 TCP 下平衡应用层 chunk 数与 staging 开销。
NVLink 下单次传输耗时仅约 4-5 ms，C8 的分块粒度是否仍是最优需要重新考量。
候选做法：先保持 C8 与原方案对齐，记录传输时间，再根据结果决定是否做 chunk
大小消融实验（例如 C4 / C8 / C16 对比）。

**`LONG_REQUEST_MAX_PREEMPT_WAIT_MS=50`**

原方案 50 ms 等待保护在 TCP 下实际从未触发。NVLink 下 KV 传输更快，
long 被 short 插队的等待也会缩短，该阈值可能更难触发。保持默认 50 ms，
观察新环境下是否触发。

**`PD_STREAM_MAX_IN_FLIGHT=1`**

原方案单槽主要因 TCP staging 时 CPU 是热点。NVLink P2P 传输为 DMA 驱动，
CPU 参与更少，多槽并发的潜力可能更大。初版实验保持 1，作为后续探索项。

**vLLM 0.33.0 接口兼容性（重要风险）**

原方案自研修改涉及 `nixl/preemptive_transfer.py`、`nixl/push_scheduler.py`、
`nixl/push_worker.py`、`v1/worker/gpu/kv_connector.py` 等文件，
均基于 vLLM 0.32.0 的内部接口。0.33.0 的 nixl connector 目录结构已有变化
（新增了 `base_scheduler.py`、`base_worker.py`、`pull_scheduler.py`、`pull_worker.py`
等文件），自研 patch 能否直接应用需要逐文件核查。

正式实验前必须完成：
- 检查 0.33.0 中 `push_scheduler.py` / `push_worker.py` 的接口是否与原 patch 兼容
- 检查 `v1/worker/gpu/kv_connector.py` 中 post-forward CUDA event 挂点是否仍然存在
- 检查 `pd_local_sort_scheduler.py` 继承的 `AsyncScheduler` 类签名是否变化
- 若接口不兼容，需先完成 patch 迁移再运行实验

**`gpu_memory_utilization=0.90` 对 KV cache pool 的影响**

原方案 24 GB × 0.90 ≈ 21.6 GB，新方案 80 GB × 0.90 ≈ 72 GB。KV cache
pool 大了约 3.4 倍，单机能同时持有的 KV blocks 大幅增加。对本实验的直接影响：
- 15 条请求（总 44345 tokens）的 KV 完全在 GPU 上，不会有 offload 或驱逐
- P 侧 send lease 持有的 blocks 不会与其他请求竞争池资源，原方案的结论可沿用
- 无需为显存容量担心，0.90 比例可维持不变

---

## 5. 启动脚本变更说明

### 5.1 GPU 编号变更

所有涉及 GPU 物理编号的地方从 `0,1` 改为 `6,7`：

```bash
# 原方案
export CUDA_VISIBLE_DEVICES=0,1
# P 实例：物理 GPU0
# D 实例：物理 GPU1

# 新方案
export CUDA_VISIBLE_DEVICES=6,7
# P 实例：物理 GPU6（在 CUDA_VISIBLE_DEVICES 中序号为 0）
# D 实例：物理 GPU7（在 CUDA_VISIBLE_DEVICES 中序号为 1）
```

注意：vLLM 内部看到的设备编号从 0 开始（由 `CUDA_VISIBLE_DEVICES` 重映射），
launcher 中的 `--device` 参数填 `0`（P）和 `1`（D），不填物理编号 6/7。

### 5.2 工作目录和 Python 路径变更

所有脚本、路径引用和 `PYTHONPATH` 中的 `/home/ubuntu` 前缀需替换为
`/data01/home/zjj`：

```bash
# 原方案
PYTHONPATH=/home/ubuntu/zjj/code:$PYTHONPATH
/home/ubuntu/miniconda3/envs/cc/bin/python run_pd_local_sort_comparison.py

# 新方案
PYTHONPATH=/data01/home/zjj/zjj/code:$PYTHONPATH
/data01/home/zjj/miniconda3/envs/cc/bin/python run_pd_local_sort_comparison.py
```

需要检查的文件：`run_pd_pair.sh`、`run_pd_router.py`、各 `run_pd_*.py` 脚本
中所有硬编码的绝对路径（模型路径、结果目录、参考输出路径等）。

### 5.3 NIXL 传输 backend 变更

从 `run_pd_pair.sh` 中移除 UCX 相关环境变量，增加 P2P 相关配置：

```bash
# 删除以下 UCX 配置
# export UCX_NET_DEVICES=lo
# export UCX_TCP_TX_SEG_SIZE=128k
# export UCX_TCP_RX_SEG_SIZE=128k

# 新增 P2P / CUDA IPC 配置
# 方案 A：通过 NIXL 环境变量指定 backend（视 NIXL 1.5.2 实际接口）
export NIXL_BACKEND=CUDA_IPC    # 或 P2P，以实际可用 plugin 为准

# 方案 B：若 NIXL 通过 vllm kv-transfer 配置文件指定 backend
# 在 kv_transfer_config.json 中设置 "backend": "CUDA_IPC"

# 方案 C：若 NIXL 自动探测并优先选择 P2P（当 P2P 可用时）
# 只需不设置 UCX_NET_DEVICES，NIXL 应自动选用 CUDA IPC/P2P 路径
```

> **待确认**：NIXL 1.5.2 的实际 P2P backend 名称和激活方式。
> 在正式实验前需通过 `nixl_agent.get_plugin_list()` 确认可用 plugin，
> 或查阅 NIXL 1.5.2 文档，记录实际使用的 backend 名称和版本信息。
> 详见第 12 节（NVLink 配置专项）。

### 5.4 端口配置

端口配置不受 GPU 编号影响，保持原方案端口不变（P: 8100，D: 8200，Router: 8000）。
启动前必须确认物理 GPU6/7 可见且没有计算进程占用；设备查询失败时脚本拒绝启动：

```bash
# 检查目标 GPU 是否空闲
nvidia-smi -i 6,7 --query-compute-apps=pid,used_memory --format=csv,noheader
```

---

## 6. 测量体系与正确性标准

### 6.1 测量指标（与原方案一致）

保留原方案全部测量指标：

- `ttft_ms`：客户端提交至收到首个非空文本 SSE 事件
- `e2e_ms` / `latency_ms`：客户端提交至完整响应结束
- `workload_makespan`：首条实际提交至最后响应结束
- 类别平均 TTFT：short / medium / long 分类均值
- 逐条 long TTFT/E2E：必须保留个体数据，不用均值掩盖退化
- P/D 执行时间线：batch 级执行耗时、首次调度、首次 execute_model
- chunk 生命周期：accepted / dispatch / start / complete / D receive

### 6.2 新增测量项

NVLink 路径下新增以下观测点：

| 新增指标 | 目的 |
|---------|------|
| **单次 NIXL WRITE 耗时**（start→complete） | 验证 P2P 传输时间是否确实降至 ~5 ms 量级 |
| **传输占 TTFT 的比例** | 量化 NVLink 下传输不再是瓶颈 |
| **P 计算 vs 传输重叠比例** | 对比原 TCP 方案的重叠比例变化 |
| **D 等待 KV 的时间**（KV 完整 - 首次调度） | NVLink 下该值应大幅降低 |
| **实际 P2P backend 确认** | 记录 NIXL 实际使用的 transport 名称 |

### 6.3 正确性验证（与原方案一致）

1. 预期请求数、HTTP 状态、输入长度、输出长度正确
2. token IDs 与无传输直接参考比对（需重新生成 A100 上的参考输出）
3. chunk 序号连续，block 覆盖无缺失/重叠，final/summary 正确
4. D 首次模型提交在所需 KV 收齐之后
5. P/D 资源在安全边界清理，失败后后续请求可继续

> **注意**：需要在 A100 + `VLLM_BATCH_INVARIANT=1` + `temperature=0` 条件下
> 重新生成参考输出。原 RTX 4090 上的参考 token IDs 不能直接复用（不同 GPU
> 架构的浮点规约顺序可能不同，参考 Plan.txt §10.4 关于 batch invariant 的说明）。

---

## 7. 预期结果与假设

### 7.1 核心假设

1. **NVLink 传输耗时可忽略**：10000-token long 的约 1.22 GB KV 在 NVLink 下
   传输仅需约 4.5 ms，相比原 TCP 的 ~120 ms 缩短约 27 倍。传输不再是
   端到端延迟的主要来源。

2. **调度效应更纯粹**：原方案中 UCX staging 掩盖了部分调度收益，NVLink 下
   P 局部排序和传输仲裁的效果将更直接体现在 TTFT 上。

3. **A100 计算更快，baseline makespan 本身缩短**：A100 BF16 算力约为
   RTX 4090 的 1.9 倍，P 的 prefill 计算和 D 的 decode 都会更快。预计
   原方案 baseline makespan ~6 s，A100 上可能降至 ~3-4 s 量级。
   三方案之间的绝对差值会缩小，但相对百分比差异可能维持或扩大。

4. **总 makespan 变化**：由于传输时间大幅缩短，整批 makespan 主要由 P 计算
   时间和 D decode 时间决定。三方案的 makespan 差异预计比原方案更小（原方案
   约 8-10% 的差异中有一部分来自传输时间差）。

5. **short TTFT 改善仍显著**：P 局部排序的收益来自计算预算重分配，与传输
   路径无关。新完整方案的 short/medium TTFT 改善预计仍然显著。

6. **long 受抢占代价变小**：原方案 long 被插队时的等待时间包含传输等待，
   NVLink 下传输快，插队造成的绝对时间损失应更小。

### 7.2 可能需要重新评估的原结论

| 原结论 | 在 NVLink 下是否仍成立 | 说明 |
|--------|----------------------|------|
| "传输是主要瓶颈" | **否** | NVLink 下传输耗时 ~5 ms |
| "UCX 128KB 分段是关键优化" | **不适用** | 该优化针对 TCP staging |
| "提前发送贡献最大" | **需重新验证** | NVLink 下传输快，提前发送的隐藏收益缩小 |
| "short TTFT 下降 30-40%" | **预计仍成立** | 排序收益来自计算层，不依赖传输路径 |
| "makespan 降低 8-10%" | **预计偏小** | 传输时间减少，baseline 本身也快 |
| "传输插队等待保护未触发" | **预计仍未触发** | 传输更快，插队等待更短 |
| "baseline makespan ~6 s" | **否** | A100 计算更快，预计降至 ~3-4 s 量级 |
| "DtoH staging 占 0.258 s" | **不适用** | P2P 路径无 DtoH staging |

---

## 8. 消融实验方案

保留原方案的三模块消融，新增 NVLink 专属对照：

### 8.1 保留的消融（同原方案）

与原方案相同的五配置消融，固定 100 ms 逐条到达：

| 配置 | 目的 |
|------|------|
| 整请求 baseline | 对照基准 |
| 旧完整方案（full） | 无排序的流式+仲裁 |
| 关闭提前发送（no_stream） | 隔离流式发送贡献 |
| 关闭聚合，短中也 C8 | 隔离聚合贡献 |
| 关闭插队（FIFO） | 隔离仲裁贡献 |

### 8.2 新增对照（NVLink 专属）

| 新增配置 | 目的 |
|---------|------|
| **C4 vs C8 vs C16 分块大小对比** | 验证 NVLink 下分块粒度对性能是否仍有影响 |
| **单槽 vs 双槽 WRITE（in_flight=1 vs 2）** | NVLink DMA 下多槽并发的潜力 |

> 原初版方案中曾有"新完整方案 NVLink vs 原 TCP 旧方案"的跨平台对比项，
> 已删除。该对比同时混入了 GPU 型号差异（A100 vs RTX 4090）、vLLM 版本差异
> （0.33.0 vs 0.32.0）和传输路径差异，无法单独归因，不作为正式实验结论。
> 如需在论文中提及传输路径升级的收益，只能作为定性说明，
> 并明确标注无法排除硬件算力差异的影响。

---

## 9. 实验执行检查清单

实验前需要完成以下确认（实际执行时逐项核对）：

### 9.1 环境准备

- [ ] 确认 GPU6、GPU7 空闲（无其他进程占用）
- [ ] 确认 `CUDA_VISIBLE_DEVICES=6,7` 时 P2P 可用（已验证：True）
- [ ] **确认 vLLM 0.33.0 自研 patch 兼容性**（`push_scheduler`、`push_worker`、
      `preemptive_transfer`、`kv_connector` 中的 event 挂点，逐文件与 0.33.0 接口比对）
- [ ] **更新脚本中所有 `/home/ubuntu` 路径为 `/data01/home/zjj`**
- [ ] 确认 NIXL 实际可用的 P2P / CUDA IPC plugin 名称（见第 12 节）
- [ ] 修改 `run_pd_pair.sh`：GPU 编号改为 6/7，删除 UCX 参数，设置 P2P backend
- [ ] 在 A100 上重新生成无传输参考输出（solo/mixed 直接推理，`VLLM_BATCH_INVARIANT=1`）
- [ ] 确认端口 8000/8100/8200 未被占用

### 9.2 正式实验前

- [ ] 预热运行验证：1 次 3 方案×2 场景，确认 15 请求全部成功
- [ ] 检查 token IDs 与新参考输出匹配
- [ ] 确认 chunk 生命周期日志中传输 start→complete 时间符合 NVLink 量级
- [ ] 记录实际 NIXL backend 名称到 environment.json

### 9.3 实验后

- [ ] `collect_pd_trace.py --require-complete --require-execution` 通过
- [ ] 逐条 long 的 TTFT/E2E 记录完整（不只看均值）
- [ ] 两卡显存回到约初始值（无泄漏）
- [ ] 保存 environment.json（含 GPU 型号、NVLink 配置、NIXL backend 版本）

---

## 10. 结果目录结构

```
results/
└── nvlink-local-sort-comparison-YYYYMMDD/
    ├── REPORT.md              # 主报告
    ├── runs.csv               # 三轮 makespan、分类均值、抢占次数
    ├── requests.csv           # 逐请求 TTFT/E2E
    ├── audit.json             # 窗口、汇总、计数、校验
    ├── environment.json       # GPU 型号、NIXL backend、参数快照
    ├── workload.jsonl         # 固定负载副本（含 SHA256 核对）
    ├── source-runtime/        # 本次实验 vLLM 修改文件快照
    ├── SHA256SUMS             # 关键文件哈希
    └── runs/
        ├── baseline_25ms_run{1,2,3}/
        ├── baseline_50ms_run{1,2,3}/
        ├── old_25ms_run{1,2,3}/
        ├── old_50ms_run{1,2,3}/
        ├── new_25ms_run{1,2,3}/
        └── new_50ms_run{1,2,3}/
```

---

## 11. 论文写作影响

NVLink 实验结果对论文的影响：

1. **不再需要解释 UCX 128KB 分段优化**，这简化了系统描述。

2. **实验平台描述变更**：单机 8× A100-SXM4，取 GPU6/7，NVLink 双向 ~600 GB/s。
   这是更有说服力的硬件平台。

3. **传输部分的叙述重点变化**：从"如何克服 TCP staging 瓶颈"转变为
   "在低延迟传输路径下，调度策略的纯粹效果"。

4. **新的正面论据**：在传输不是瓶颈的条件下，P 局部排序的 TTFT 收益完全
   来自计算调度层，论证更干净，贡献归因更清晰。

5. **需要更新的限制说明**：原 Plan.txt §18.2 的"拓扑和通信路径单一（TCP/lo）"
   限制可以去掉，但仍需说明这是单机环境而非跨机 RDMA。

---

## 12. NVLink P2P 配置专项

本节专门记录从 UCX TCP 切换到 NVLink P2P 的配置确认步骤，待实验前完成。

### 12.1 NIXL plugin 探测结果（已确认）

在本机运行探测后，确认 NIXL 1.5.2 可用 plugin 为：

```
['AZURE_BLOB', 'GDS', 'GDS_MT', 'GPUNETIO', 'GUSLI', 'INFINIA', 'LIBFABRIC', 'OBJ', 'POSIX', 'UCX']
```

支持 VRAM_SEG（GPU 显存）的只有：
- **UCX**：DRAM_SEG + VRAM_SEG ✓（唯一可用的 GPU 传输 backend）
- GDS/GDS_MT：VRAM_SEG，但这是 GPUDirect Storage（读写 NVMe），不是 GPU 间传输

**结论：没有独立的 CUDA_IPC 或 P2P plugin，传输路径只能走 UCX。**
NIXL UCX 可以识别 VRAM，但这不代表已经建立 CUDA IPC/NVLink 数据路径；需以实际 UCX transport 日志及 NVLink 计数器验证。

### 12.2 UCX cuda_ipc transport 确认

NIXL 自带 UCX 提供 CUDA memory support。理想情况下，同主机 GPU 间的 CUDA IPC 路径可使用 NVLink DMA；当前 GPU6/7 检查没有观察到此路径。

NIXL 在拒绝 `UCX_TLS=tcp` 或 `UCX_TLS=cuda_ipc` 时提示应使用 `cuda_copy` 提供 GPU 基础支持，或使用 `cuda` 启用包含 NVLink 的 CUDA transport 集合。成功初始化 agent 仅证明 backend/memory type 初始化成功，不证明具体传输 transport。

初始化检查命令：

```bash
export UCX_TLS=cuda,tcp
export UCX_NET_DEVICES=lo
python3 -c "
import nixl_cu13 as nixl
cfg = nixl.nixl_agent_config(backends=['UCX'])
a = nixl.nixl_agent('test', cfg)
print('OK, UCX mem types:', a.get_plugin_mem_types('UCX'))
"
# 输出: OK, UCX mem types: ['DRAM_SEG', 'VRAM_SEG']
```

### 12.3 vLLM 0.33.0 的 UCX 配置方式（已审计）

查看 `base_worker.py` 确认：
- 正常路径（无 sleep_mode）：UCX backend 由 `nixl_agent_config(num_threads=4)` 初始化，
  **不限制 TLS**，由 `UCX_TLS` 环境变量控制
- sleep_mode 路径：UCX 被设置 `TLS=^cuda_ipc`（排除 cuda_ipc），与本实验无关

设置 `UCX_TLS` 只选择候选 transport。当前实际检查显示 `UCX_TLS=cuda,tcp` 下 intra-node RMA lane 仍为 `tcp/lo`，且 UCX 报告 GDAKI 不支持；不能据配置断言 vLLM 使用 NVLink。

### 12.4 推荐 transport 候选配置

```bash
# NIXL 建议的 CUDA transport 候选配置；实际使用 NVLink 还需硬件计数验证
export CUDA_VISIBLE_DEVICES=6,7
export UCX_TLS=cuda,tcp
export UCX_NET_DEVICES=lo          # loopback 用于控制面 TCP
# 删除原方案的 UCX_TCP_TX/RX_SEG_SIZE 分段参数（不再需要）
```

UCX 配置候选项包含 `cuda_ipc`、`cuda_copy` 和 `tcp`。单进程双 agent 测量因 `same process` 排除 CUDA IPC；改为双进程后，UCX intra-node lane 日志包含 `device(cuda_ipc/cuda)`，NVLink TX/RX 计数增量与传输 payload 相符，确认 NIXL GPU 数据走 NVLink。纯 PyTorch P2P 测量仍作为设备路径参考。

### 12.5 传输带宽测量

```
torch P2P (GPU6 -> GPU7, 10次复制，十进制GB/s):
   16 MiB:  166.6 GB/s
   64 MiB:  236.6 GB/s
  256 MiB:  267.5 GB/s
  512 MiB:  272.3 GB/s
 1024 MiB:  275.3 GB/s
```

2026-11-15 在 GPU6/7 上直接测量 NIXL UCX，结果与 PyTorch P2P 参考显著不同：

| 单次 WRITE 大小 | 10次总耗时 | NIXL有效吞吐 | GPU6 NVLink TX / GPU7 RX增量 | 校验 |
|---:|---:|---:|---:|---|
| 16 MiB | 0.375 s | 0.45 GB/s | 0 / 0 GB | 通过 |
| 64 MiB | 1.565 s | 0.43 GB/s | 0 / 0 GB | 通过 |
| 256 MiB | 5.978 s | 0.45 GB/s | 0 / 0 GB | 通过 |
| 512 MiB | 12.022 s | 0.45 GB/s | 0 / 0 GB | 通过 |
| 1024 MiB | 23.982 s | 0.45 GB/s | 0 / 0 GB | 通过 |

以上 NIXL 吞吐包含每次 transfer handle 创建、提交、CPU 轮询和释放；不含 agent 创建、内存注册及一次预热。源和目标数据全元素校验通过。NIXL 在 `UCX_TLS=tcp` 时拒绝 GPU memory support；`cuda_copy,tcp` 下实测有效吞吐约0.44–0.45 GB/s且 NVLink 计数为零。`cuda,tcp` info 日志报告 intra-node RMA 使用 `tcp/lo`。日志中的 `GDAKI not supported` / `PeerMappingOverride=1` 诊断对应 mlx5/GDAKI（网卡侧 GPUDirect RDMA）能力探测，不足以证明 CUDA IPC/NVLink 被该驱动参数阻止。`/proc/driver/nvidia/params` 的 RegistryDwords 为空，但不能据此认定这是 GPU 间 NVLink 路径不可用的原因。

后续 UCX DEBUG 日志进一步定位到本带宽脚本的结构问题：它在同一 Python 进程中创建 P、D 两个 agent。UCX 明确记录 `cuda_ipc/cuda cannot reach ... same process`，并为该 endpoint 选择 `rma_am(tcp/lo)`；因此这次约0.45 GB/s 的 NIXL 数据传输是该单进程测试的 TCP/软件回退结果，不是 NVLink 带宽。vLLM 正式实验的 P、D 是独立进程，不能把这个 same-process 诊断直接套用到旧实验。旧实验没有保存足以确认逐 WRITE 实际 transport 的 UCX lane 日志或 NVLink 计数，因此其传输路径仍未证实。应先用双进程 NIXL microbenchmark 验证 CUDA IPC lane 与 NVLink 计数，再决定是否需要修改驱动参数或 UCX 配置。

双进程复测（2026-11-15）已确认 CUDA IPC/NVLink 可用。P、D 各自在独立进程和 GPU context 下运行，`UCX_TLS=cuda,tcp`、`UCX_NET_DEVICES=lo`；UCX intra-node 配置包含 `device(cuda_ipc/cuda)`，NVLink TX/RX 与每尺寸 10 次 WRITE 的 payload 总量吻合，校验通过。具体数据见[带宽测试记录](../results/nixl-bandwidth-20261115/RESULTS.md)。此前约0.45 GB/s 是单进程双 agent 测试触发 same-process 回退，不代表 NIXL 的 NVLink 吞吐；约275 GB/s 是 PyTorch P2P 参考值，不代表 NIXL。旧 vLLM 实验仍未采集实际传输计数，不能追溯确认是否走 NVLink；启动器默认 UCX info 日志以便后续实验保留 lane 证据。

### 12.6 若 cuda_ipc 不可用的回退方案

`cuda_copy,tcp` 是 host-staging/TCP 对照候选配置。PCIe 理论值约32 GB/s不等于本机 NIXL 实测；本次有效吞吐约0.45 GB/s，需先查明 UCX lane 与软件开销后才能归因。配置示例：

```bash
# GPU host staging + TCP
export UCX_TLS=cuda_copy,tcp
```

**无论哪种路径，都在 `environment.json` 中明确记录 `UCX_TLS` 实际值。**

---

## 附录：与原方案 Plan.txt 的章节对应关系

| 本文章节 | 原 Plan.txt 对应章节 | 变更程度 |
|---------|-------------------|---------|
| §1 硬件与环境 | §11.1 主要平台 | **重写**（GPU型号/编号/传输路径/vLLM版本全变） |
| §2 传输路径变更 | §10.7 UCX 根因诊断 | **新增**（原无此对比节） |
| §3 实验方案总体设计 | §11.3-11.5 | 微调（GPU编号、删UCX参数） |
| §4 参数配置变更 | §17.3 关键参数 | **部分重写**（UCX参数删除，新增兼容性风险） |
| §5 启动脚本变更 | §17.1-17.2 源码/工具 | **部分重写**（新增路径变更节） |
| §6 测量体系 | §12 测量体系 | 继承+新增NVLink测量项 |
| §7 预期结果与假设 | §15 为什么有效 | **新增** |
| §8 消融实验 | §13.5 三模块消融 | 继承+新增NVLink消融，删除跨硬件对比 |
| §9 实验执行检查清单 | 无 | **新增** |
| §10 结果目录结构 | §17.5 证据索引 | 更新目录名 |
| §11 论文写作影响 | §19 可用/应避免表述 | **新增** |
| §12 NVLink P2P 配置专项 | 无 | **新增** |
