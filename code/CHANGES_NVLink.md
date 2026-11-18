# 从 Plan.txt 到 NVLink 方案的变更摘要

> **当前执行状态（2026-11-15）**：GPU6/7 的 NVLink 单项复测已完成：baseline / sort_only / preempt_only 主实验 18 次正式运行、270 请求，补充 stream_fifo 6 次、90 请求，及等量预热均通过输出与协议审计。每次正式负载 NVLink TX/RX 约 5.817500 GB，与填充 KV payload 匹配。排序降低 short TTFT 48%–63%，long 增加 11%–13%；抢占相对 baseline 的 short TTFT 增加约 6%，相对流式 FIFO 降低约 3%。详见[本轮完整报告](../results/nvlink-isolated-ablation-20261115/REPORT.md)及技术报告第 18 节。FIFO 在主实验后补测，未交错；无同轮同时开启组，不能估计完整因子交互。
>
> **历史摘要适用范围**：下文 250/500 ms、20 ms 窗口、独立 CUDA_IPC backend 及纯带宽推算均为旧计划。当前为 25/50 ms、5 ms、64 输出 tokens、C8/单槽、writer 轮询 0.1 ms；backend 为 `UCX`，独立 P/D 进程使用 `UCX_TLS=cuda,tcp`。开关为 `PD_LOCAL_SORT` / `PD_PRIORITY_PREEMPTION`，启动器支持 sort_only、preempt_only、stream_fifo；两个单项仍保留流式设施。早期单进程约 0.45 GB/s 是 same-process TCP 回退，双进程微基准约 77–201 GB/s；历史模型实验未采集硬件计数，不能追溯确认 transport。带宽证据见[带宽测试记录](../results/nixl-bandwidth-20261115/RESULTS.md)。

**生成日期**：2026-11-13  
**主文档**：`Experiment_Plan_NVLink.md`（633 行）

---

## 关键变更总览

### 硬件平台
- **GPU**：RTX 4090 × 2 → **A100-SXM4-80GB × 2**（使用 GPU6/GPU7）
- **显存**：24 GB → **80 GB**（KV cache pool 3.4× 扩大）
- **算力**：~165 TFLOPS BF16 → **~312 TFLOPS**（1.9× 提升）
- **传输**：原设计从 UCX TCP loopback 转向 GPU P2P；纯 PyTorch P2P参考约275 GB/s。早期单进程 NIXL 测试未观察到 NVLink 流量并因 same-process 回退 TCP；后续双进程测试已确认 CUDA IPC/NVLink，数据见带宽测试记录

### 软件环境
- **vLLM**：0.32.0 → **0.33.0**（需验证 patch 兼容性）
- **CUDA**：11.8 → **13.0**
- **工作目录**：`/home/ubuntu` → **`/data01/home/zjj`**

### 传输路径
- **原路径**：GPU(P) → DtoH staging → Host TCP → HtoD → GPU(D)
- **新路径**：GPU6(P) → **NVLink P2P DMA（零 Host copy）** → GPU7(D)
- **延迟估算**：10000-token long KV 从 ~120 ms → **~4.5 ms**（27× 加速）

---

## 实验方案不变部分

以下内容与原 Plan.txt 保持一致，确保可比性：

✓ 固定 15 请求负载（8 short + 3 medium + 4 long，workload SHA256 不变）  
✓ 三方案定义（baseline / 旧方案 / 新完整方案）  
✓ 两种到达场景（250 ms / 500 ms 批间隔）  
✓ 3×3 轮换重复（18 次正式运行，270 条请求）  
✓ 模型、context、输出长度、seed、temperature 等全部参数  
✓ P 局部排序窗口（20 ms）、C8 分块、三项保护阈值（85%/3/50ms）  
✓ 测量指标定义（TTFT、E2E、makespan、逐条 long）

---

## 必须修改的配置

### 环境变量
```bash
# GPU 编号
CUDA_VISIBLE_DEVICES=6,7  # 原 0,1

# 删除 UCX TCP 参数
# UCX_NET_DEVICES=lo
# UCX_TCP_TX_SEG_SIZE=128k
# UCX_TCP_RX_SEG_SIZE=128k

# 新增 P2P backend（待确认实际名称）
export NIXL_DEFAULT_BACKEND=CUDA_IPC  # 或 P2P
```

### 脚本路径
所有硬编码的 `/home/ubuntu` 前缀改为 `/data01/home/zjj`：
- 工作目录、模型路径、结果目录、参考输出路径
- `PYTHONPATH`、conda 环境路径

---

## 新增风险与待办

### ⚠️ 高优先级
1. **vLLM 0.33.0 兼容性**：逐文件核查自研 patch 是否适用新接口
   - `push_scheduler.py` / `push_worker.py`
   - `v1/worker/gpu/kv_connector.py` 中的 event 挂点
   - `pd_local_sort_scheduler.py` 的 `AsyncScheduler` 继承
   
2. **NIXL P2P plugin 确认**：运行 `nixl_agent.get_plugin_list()` 确认实际名称
   - 若不支持 P2P：回退到 UCX CUDA_IPC / shared memory / TCP 128k
   - 记录实际 backend 到 `environment.json`

3. **A100 参考输出生成**：重新跑 solo/mixed 直接推理生成新参考 token IDs
   - 原 RTX 4090 参考不可复用（浮点规约顺序可能不同）
   - 使用 `VLLM_BATCH_INVARIANT=1` 确保确定性

### 📋 实验前检查清单

- [ ] GPU6/7 空闲，P2P 可用（已验证 `torch.cuda.can_device_access_peer`）
- [ ] vLLM 0.33.0 patch 兼容性核查
- [ ] 脚本路径全部更新（`/home/ubuntu` → `/data01/home/zjj`）
- [ ] NIXL P2P backend 确认并配置（见 §12）
- [ ] A100 参考输出生成
- [ ] 预热运行验证（3 方案 × 2 场景，确认传输耗时 ~5 ms 量级）

---

## 预期结果变化

### 更快的 baseline
- A100 计算更快 → baseline makespan 预计从 ~6 s 降至 **~3-4 s**
- 原假设“传输几乎瞬时”不能按纯 PyTorch P2P 带宽推定：双进程 NIXL 实测约77–201 GB/s（含 NIXL API/轮询开销），单进程约0.45 GB/s 是 CUDA IPC 不可达时的 TCP 回退

### 调度效应更纯粹
- 原方案中 TCP staging 掩盖了部分调度收益
- NVLink 下 P 局部排序的 TTFT 改善将更直接体现
- short TTFT 改善预计仍显著（30-40%），但 makespan 差异可能更小（<8%）

### 新增对照实验
- C4 vs C8 vs C16：验证 NVLink 下分块粒度是否仍重要
- 单槽 vs 双槽：P2P DMA 的并发潜力

### 原结论中失效的部分
- ❌ "传输是主要瓶颈"
- ❌ "UCX 128KB 分段优化"
- ❌ "DtoH staging 占 0.258 s"
- ⚠️ "提前发送贡献最大"（需重新验证，隐藏收益缩小）

---

## 论文写作影响

### 正面
1. **平台更有说服力**：A100 + NVLink 是更标准的高性能配置
2. **贡献归因更清晰**：传输不是瓶颈，排序收益完全来自计算调度层
3. **系统描述简化**：不再需要解释 TCP staging 和 128KB 分段优化

### 需要更新的部分
1. **实验平台**：单机 8× A100-SXM4，GPU6/7，NVLink ~600 GB/s 双向
2. **传输叙述重点**：从"克服 staging 瓶颈"→"低延迟路径下的调度效果"
3. **限制说明**：去掉"TCP/lo 单一路径"，但保留"单机非跨机 RDMA"

---

## 文档结构

```
Experiment_Plan_NVLink.md (633行)
├── §1  硬件与环境（重写）
├── §2  传输路径变更（新增）
├── §3  实验方案总体设计（微调）
├── §4  参数配置变更清单（部分重写）
├── §5  启动脚本变更说明（部分重写 + 新增路径节）
├── §6  测量体系与正确性标准（继承 + 新增 NVLink 测量项）
├── §7  预期结果与假设（新增）
├── §8  消融实验方案（继承 + 新增 NVLink 消融）
├── §9  实验执行检查清单（新增）
├── §10 结果目录结构（更新目录名）
├── §11 论文写作影响（新增）
├── §12 NVLink P2P 配置专项（新增，详细说明 backend 配置与回退）
└── 附录 与原方案章节对应关系
```

---

## 相关文件

- **原方案文档**：`Plan.txt`（89 KB，1128 行）
- **新方案文档**：`Experiment_Plan_NVLink.md`（本文档，19 KB，633 行）
- **固定负载**：`workload.sweep-interleaved-8s3m4l.jsonl`（SHA256 不变）

---

**下一步**：在 0.33.0 上重新实现三大优化（流式提前发送、传输仲裁、P 局部排序）。
原方案代码不在新环境，且 0.33.0 push connector 是全新架构（base_scheduler / base_worker / push_scheduler / push_worker 分层），与 0.32.0 不兼容，需从头实现。

## 实现路径（0.33.0 新架构）

### 0.33.0 baseline 行为
- `push_scheduler.request_finished` → prompt 全部完成后把所有 blocks 一次性打包
- `push_worker` 的 `nixl-push-writer` thread → 收到 finished_blocks 后匹配 D registration，发起 NIXL WRITE
- 这正好是三方案中的 **baseline（整请求传输）** 行为，不需要修改

### 需要新增的三个模块

**1. stream_by_batch（按 batch 提前发送）**
- 在 `base_worker.build_connector_meta` 返回时，携带当前 batch 已就绪的 KV frontier
- 在 `push_scheduler` 追踪每个请求的 `computed_tokens` frontier，每 batch 更新
- push_writer 收到 partial blocks 时立即发起 WRITE，不等 request_finished

**2. preemptive_transfer（传输仲裁 + 三项保护）**
- 在 push_writer 的传输队列中，实现 short 请求优先级逻辑
- 按 prompt 长度分类（short <256, medium <=8192, long >8192）
- 实现 85%/3次/50ms 三项保护，防止 long 请求饥饿

**3. pd_local_sort_scheduler（P 侧局部排序）**
- 继承 vLLM 的 AsyncScheduler，在 on_new_request 时收集 20ms 窗口内的请求
- 窗口到期后按 prompt token 数升序排列，再追加到原 FCFS 队列

## 2026-11-14 流式协议修复

修复活动请求与增量块表、并发 Router 提前注册、offset WRITE、post-forward CUDA event、partial/final 通知、实际单槽限制和长请求完成进度。完整修复说明和范围见 [Streaming_Repair_20261114.md](Streaming_Repair_20261114.md)。修复后的25/50ms并发 smoke 共30/30请求逐 token ID 一致，686个 WRITE/DONE 轨迹通过；历史正式归档保留原样，修复后的正式实验独立归档。

正式修复后对比已完成：18/18运行、270/270正式及270/270预热请求逐token一致；360条流式请求的8236个WRITE/DONE、C8/单槽和唯一final通知审计通过。当前old/new makespan仍高于baseline（约29%–32%），未获得整体完成时间收益，数据与边界见 [RESULTS.md](../results/nvlink-streamfix-v3-20261114/RESULTS.md)。仅GPU6/7，结束后已释放；历史数据保留。

## 2026-11-14 writer完成推进优化

writer独占WRITE handle完成推进；final ID恰好一次交接给主线程释放请求lease；wake响应与0.1ms回退轮询避免逐chunk等模型步。13项CPU回归通过，18/18正式、270/270正式及270/270预热逐token一致，8239 WRITE/DONE审计通过。new相对v3 makespan缩短约22%；相对本轮baseline整体完成时间增加不足1%，short/medium TTFT改善35%–48%，long TTFT增加约9%。详细记录见 [Writer_Progress_Optimization_20261114.md](Writer_Progress_Optimization_20261114.md)。仅GPU6/7，结束后已释放，历史归档保留。

## 2026-11-14 v5边界复核与重新实验

修复局部排序取消接口及窗口顺序/共享deadline日志、Router流式转发与失败/取消task清理、256-token分类边界、partial失败标记清理；非流式TTFT记为null。新增实际token/SSE/DONE/时间戳独立核验，强化单槽执行顺序审计。22项CPU回归、18/18正式、270正式+270预热请求及8236 WRITE/DONE全部通过。问题与解决方案、参数、性能权衡和剩余边界已写入技术报告第15节，完整结果见 [RESULTS.md](../results/nvlink-review-v5-20261114/RESULTS.md)。仅GPU6/7，已释放。


## 2026-11-16 入窗诊断及GPU2/3补齐（已完成）

24/24正式运行完成，前12次GPU5/6、后12次GPU2/3，第三轮四组均在2/3。360正式+360预热请求输出校验、540条流式请求6739 WRITE/DONE审计、48次NVLink计数及32项CPU检查通过。原阶段12/24报告与源数据已保留。

精确时间线确认EngineCore执行步期间输入队列积累，集中出队后形成跨客户端批次窗口；5ms从admit开始。窗口成员相同的五组排序/组合对照中组合short均更快，同卡第三轮25/50ms分别快58.15/57.37ms。25ms均值排序占优主要由第一轮1+14对1+4+10的分组差异驱动。候选日志还确认受保护long可以越过已可传short；不能假设开启抢占后short只快不慢。详见[完整诊断报告](../results/nvlink-admission-trace-gpu5-6-20261116-150826/REPORT.md)及技术报告第21节。
