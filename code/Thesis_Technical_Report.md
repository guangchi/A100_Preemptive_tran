# P/D 分离预填充 LLM 推理优化——完整技术报告

**作者**：zjj  
**日期**：2026-11-16（最后更新）  
**硬件平台**：单机 8× A100-SXM4-80GB，使用 GPU6（Prefill）/ GPU7（Decode），NVLink 互联  
**软件栈**：vLLM 0.33.0 · NIXL 1.5.2 · UCX（双进程 microbenchmark 已验证 CUDA IPC/NVLink）· Python 3.11

> **阅读说明**：本文档按时间保留项目架构、修复与实验记录；当前实测状态以第 20 节为准。第 1–17 节中的实现限制与待办应结合其日期阅读，旧实验未记录 NVLink 硬件计数，不能由本次复测反推历史传输路径。

---

> **最新状态（2026-11-16）**：C8、C16、C32 三档块大小下的NVLink消融已完成。最新C16四组为baseline、只开排序、只开抢占和排序+抢占，各三轮；24次正式运行360请求及等量预热通过输出与协议审计，48次负载GPU6 TX/GPU7 RX计数与KV payload匹配。三档只开抢占的短请求TTFT均约2.72–2.78秒，调整块大小没有明显改善；流式WRITE数随块从C8到C16到C32下降8231→6739→3950（每档540条流式请求含预热）。C16完整数据见第20节及[C16报告](../results/nvlink-c16-ablation-20261116/REPORT.md)，C32见第19节，C8见第18节。跨日三轮比较不能作为严格因果估计，所有配置均为单槽，尚未测试多槽。

## 目录

1. [项目背景与动机](#1-项目背景与动机)
2. [总体架构设计](#2-总体架构设计)
3. [三项核心优化](#3-三项核心优化)
4. [NVLink 传输配置](#4-nvlink-传输配置)
5. [工程实现细节](#5-工程实现细节)
6. [困难问题与解决方案](#6-困难问题与解决方案)
7. [实验设计](#7-实验设计)
8. [实验结果与分析](#8-实验结果与分析)
9. [文件清单](#9-文件清单)
10. [结论](#10-结论)
11. [当前状态与待续工作](#11-当前状态与待续工作)（历史记录与最新索引）
18. [NVLink 单项开关实验与 baseline 对比](#18-nvlink-单项开关实验与-baseline-对比2026-11-15)
19. [C32 四组 NVLink 消融实验](#19-c32-四组-nvlink-消融实验2026-11-16) ← **最新实测**
20. [C16 四组 NVLink 消融实验](#20-c16-四组-nvlink-消融实验2026-11-16) ← **最新实测**

---

## 1. 项目背景与动机

### 1.1 P/D 分离推理的核心思想

大语言模型推理由两个计算阶段组成：

- **Prefill（预填充）**：输入 prompt 的所有 token 并行前向计算，生成对应的 KV cache，计算密集。
- **Decode（解码）**：每次自回归地生成一个新 token，访存密集。

两阶段的算力/带宽特征截然不同，混在同一 GPU 上会互相干扰，导致 GPU 利用率下降。**P/D 分离**（Prefill/Decode disaggregation）将两个阶段分配到不同 GPU：

```
用户请求
   │
   ▼
Router
   ├─► P-GPU（prefill）: 计算所有 prompt KV cache
   │        │  KV transfer（NVLink P2P）
   └─► D-GPU（decode）: 接收 KV cache 后自回归生成
```

### 1.2 从 TCP 到 NVLink

原始设计（Plan.txt）使用 UCX TCP loopback 传输 KV cache，存在 staging 瓶颈：

```
原路径：GPU(P) → DtoH staging → Host TCP ~10 GB/s → HtoD → GPU(D)
硬件参考：GPU6 ──── PyTorch NVLink P2P ~276 GB/s ────► GPU7
```

A100 通过 NVLink 3.0 互联。GPU6/7 的 PyTorch P2P 参考测量达到约275 GB/s；早期单进程 NIXL UCX 检查因 UCX 排除 CUDA IPC 而回退到 tcp/lo，测得约0.45 GB/s。改为双进程后，NIXL UCX NVLink microbenchmark 测得约77–201 GB/s。microbenchmark 不等同于模型 KV workload，因此仍需端到端数据判断传输是否构成系统瓶颈。

### 1.3 研究目标

在 NVLink P/D 分离基础上，研究三项调度优化：

1. **stream_by_batch**：按计算批次提前发送 KV，不等 prefill 全部完成
2. **preemptive_transfer**：短请求优先占用传输通道，附三项长请求保护
3. **P 侧局部排序**：5 ms 窗口内按 prompt 长度升序排列，减少调度队头阻塞

---

## 2. 总体架构设计

### 2.1 系统组件

```
┌─────────────────────────────────────────────────────────────────┐
│                         单机 A100 × 8                           │
│                                                                  │
│   ┌──────────────────────────┐  NVLink P2P  ┌─────────────────┐ │
│   │  P-GPU (GPU6, port 8100) │─────────────►│ D-GPU (GPU7,    │ │
│   │  vLLM Prefill Server     │  NIXL WRITE  │ port 8200)      │ │
│   │  VLLM_NIXL_SC_PORT=5600  │◄─────────────│ VLLM_NIXL_SC_   │ │
│   │  NixlPushConnector       │  ZMQ 5600/   │ PORT=5601       │ │
│   │  (kv_producer)           │  5601        │ (kv_consumer)   │ │
│   └──────────────┬───────────┘              └────────┬────────┘ │
│                  │ HTTP /v1/completions               │          │
│                  │ X-Request-Id 共享头                │          │
│            ┌─────▼───────────────────────────────────▼─────┐    │
│            │            Router (port 8000)                  │    │
│            │  P 请求（do_remote_decode=True）→              │    │
│            │  D 请求（do_remote_prefill=True）              │    │
│            └───────────────────┬───────────────────────────┘    │
└──────────────────────────────── │ ──────────────────────────────┘
                                  │ HTTP /v1/chat/completions
                               客户端（run_pd_load.py）
```

### 2.2 请求生命周期

```
1. 客户端 → Router：发送 prompt_token_ids + max_tokens
2. Router → P（/v1/completions）：
      prompt: [token_ids]              # list[int] 直接作为 token IDs
      max_tokens: 1
      kv_transfer_params: {do_remote_decode: True, remote_port: 5601}
      X-Request-Id: <shared_id>        # 关键：与 D 共享
3. P 完成 prefill → 返回 kv_transfer_params（含 remote_request_id, remote_engine_id, remote_block_ids）
4. Router → D（/v1/completions）：
      prompt: [token_ids]
      kv_transfer_params: {do_remote_prefill: True, remote_request_id: ..., remote_block_ids: ...}
      X-Request-Id: <same shared_id>   # 关键：P/D 请求 ID 必须一致
5. D 等待 P 推送 KV（PUSH_REG 匹配），收到后开始 decode
6. D 返回生成文本 → Router 附加 pd_metrics → 客户端
```

### 2.3 vLLM 0.33.0 Push Connector 架构

vLLM 0.33.0 的 KV 传输采用 Push 模型，分三层：

```
NixlPushConnector（工厂层）
├── NixlPushConnectorScheduler（调度层）
│   └── build_connector_meta() — 每个调度步骤打包待发 KV 块 ID
│   └── request_finished() — 请求完成时标记全量块
└── NixlPushConnectorWorker（Worker 层）
    └── push_writer 线程 — 匹配 PUSH_REG/WRITE，执行 NIXL WRITE
```

P 端发出 NIXL WRITE，D 端通过 ZMQ 侧信道（端口 5600/5601）发送 PUSH_REG 注册接收地址，两端通过 `get_base_request_id` 匹配请求 ID 完成 KV 交接。

---

## 3. 三项核心优化

### 3.1 stream_by_batch — 按批次提前发送

> 此节描述设计目标和代码框架。本次审计发现活动请求追踪和D提前注册路径不完整，不能视为当前已生效功能，详见第12节。

**原理**：标准 NixlPushConnector 在 prefill 全部完成后一次性发送全量 KV 块。stream_by_batch 在每个计算批次（forward step）结束时立即发送已就绪的 KV 增量，D 端在 P 仍计算时就可以接收早期层的 KV。

**实现**（`pd_preemptive_scheduler.py`）：

```python
class PreemptivePushScheduler(NixlPushConnectorScheduler):
    _partial_cursor: dict[ReqId, int]  # 每请求已发块数

    def build_connector_meta(self, scheduler_output):
        emeta = super().build_connector_meta(scheduler_output)
        if _STREAM_BY_BATCH:
            self._build_partial_push_meta(emeta, scheduler_output)
        return emeta

    def _build_partial_push_meta(self, meta, scheduler_output):
        # 计算当前 frontier（已计算块数）
        frontier_blocks = num_computed // block_size
        cursor = self._partial_cursor.get(req_id, 0)
        delta = frontier_blocks - cursor  # 本批次新就绪块
        if delta > 0:
            # 短/中请求：发送全部 delta（一次性追上）
            # 长请求：每次最多发送 C8（8块 = 128 tokens）防止独占通道
            slice_end = cursor + min(delta, _CHUNK_SIZE_BLOCKS)
            meta.push_partial_blocks[req_id] = block_slice(cursor, slice_end)
            self._partial_cursor[req_id] = slice_end
```

**关键参数**：`CHUNK_SIZE_BLOCKS=8`（128 tokens per chunk，可调）

### 3.2 preemptive_transfer — 传输优先级仲裁

**原理**：在 push_writer 传输队列中，短请求可以抢占长请求的传输通道，减少短请求 TTFT。为防止长请求饥饿，设置三重保护条件。

**分类阈值**（`pd_preemptive_worker.py`）：

```python
_SHORT_BLOCKS = 16   # ≤ 256 tokens → short
_LONG_BLOCKS  = 512  # ≤ 8192 tokens → medium；> 8192 → long
```

**三重保护条件**（满足任一即转为受保护、不可被抢占）：

| 保护条件 | 参数 | 默认值 | 含义 |
|---------|------|-------|------|
| P1 进度保护 | `LONG_REQUEST_PROTECT_PROGRESS` | 0.85 | 已发送 ≥ 85% 总块 |
| P2 抢占次数 | `LONG_REQUEST_MAX_PREEMPTIONS` | 3 | 已被短请求抢占 ≥ 3 次 |
| P3 等待时间 | `LONG_REQUEST_MAX_PREEMPT_WAIT_MS` | 50 ms | 累计等待 ≥ 50 ms |

```python
def _select_next_item(self) -> TransferItem | None:
    # 1. 受保护的长请求最优先（不可再被抢占）
    for item in protected_longs:
        return item
    # 2. 短/中请求可以抢占未保护的长请求
    if has_short_or_medium and has_unprotected_long:
        bypass_long.record_bypass_start()
        return short_or_medium_item
    # 3. 否则按 FIFO
    return queue[0]
```

### 3.3 P 侧局部排序 — 5ms 窗口调度

**原理**：P 侧调度器（AsyncScheduler）在接收新请求时，不立即放入 FCFS 等待队列，而是先放入一个 5 ms 时间窗口缓冲区。窗口到期后，将窗口内所有请求按 prompt 长度**升序**排列后整批推入 FCFS 队列。短请求先执行，其 KV 更快计算完成并发送给 D，D 更早开始 decode，TTFT 下降。

**两项特殊规则**：
- **单请求窗口立即放行**：若引擎空闲且窗口中只有 1 个请求，立刻放行不等 5ms（单请求无排序价值，等待只增加延迟）
- **调度循环轮询**：排序不用独立 timer 线程，而是在 vLLM 每次调用 `schedule()` 时检查窗口是否过期（寄生于 vLLM 自身的调度节拍，约每 2ms 一次）

**实现**（`pd_local_sort_scheduler.py`）：

```python
class LocalWindowBuffer:
    window_ms: float = 5.0   # 默认 5ms，可通过 PD_LOCAL_SORT_WINDOW_MS 覆盖
    _lock: threading.Lock    # 线程安全（scheduler/worker 两个线程访问）

    def admit(self, request):
        now = time.monotonic()
        with self._lock:
            if self._windows and now < self._windows[-1][0]:
                self._windows[-1][1].append((request, now))  # 追加到当前窗口
            else:
                deadline = now + self.window_ms / 1000.0
                self._windows.append((deadline, [(request, now)]))  # 开新窗口

    def pop_singleton_if(self, should_release) -> list:
        """引擎空闲时立即放行单请求窗口，无需等 deadline"""
        with self._lock:
            for idx, (_, entries) in enumerate(self._windows):
                if len(entries) == 1 and should_release(entries[0][0]):
                    return [self._windows.pop(idx)[1][0]]
        return []

class LocalSortMixin:
    def schedule(self, *args, **kwargs):
        self._release_idle_singletons()  # 单请求空闲放行
        self._drain_expired_windows()    # 到期窗口排序放行
        return super().schedule(*args, **kwargs)

    def _release_idle_singletons(self):
        # 仅当 running 队列和 waiting 队列都为空时才放行
        if len(self.running) > 0 or len(self.waiting) > 0:
            return
        released = self._window_buffer.pop_singleton_if(lambda _: True)
        for request, _ in released:
            self.waiting.add_request(request)

    def _drain_expired_windows(self):
        for window_entries in self._window_buffer.expired_windows():
            window_entries.sort(key=lambda x: (_prompt_tokens(x[0]), x[1]))
            for request, _ in window_entries:
                self.waiting.add_request(request)

class LocalSortAsyncScheduler(LocalSortMixin, AsyncScheduler):
    pass  # 生产用类，通过 --scheduler-cls 传入
```

**激活方式**：P 节点启动时添加 `--scheduler-cls pd_local_sort_scheduler.LocalSortAsyncScheduler`

---

## 4. NVLink 传输配置

### 4.1 NIXL 后端选择

NIXL 1.5.2 支持多种传输后端，仅 UCX 后端支持 `VRAM_SEG`（GPU 显存段），因此 NVLink P2P 必须通过 UCX 实现：

```
NIXL 后端
├── UCX            ← 唯一支持 VRAM_SEG 的后端，本项目使用
│   └── 传输层（UCX_TLS）
│       ├── cuda_ipc  → NVLink P2P DMA（VRAM→VRAM，零 host 拷贝）✓
│       ├── cuda_copy → GPU↔Host staging（备用）
│       └── tcp       → 控制平面（地址交换、通知）
├── GDR Copy       → GPUDirect RDMA（需 HCA，不适用）
└── 其他...
```

### 4.2 环境变量配置

```bash
# NVLink P2P（核心配置）
export UCX_TLS="cuda_copy,cuda_ipc,tcp"
export UCX_NET_DEVICES="lo"      # TCP 控制平面走 loopback
export UCX_LOG_LEVEL="warn"

# ZMQ 侧信道端口（P/D 必须不同，否则 bind 冲突）
VLLM_NIXL_SIDE_CHANNEL_PORT=5600  # P 节点
VLLM_NIXL_SIDE_CHANNEL_PORT=5601  # D 节点

# GPU 分配
CUDA_VISIBLE_DEVICES=6  # P
CUDA_VISIBLE_DEVICES=7  # D

# 数值确定性（实验可复现）
export VLLM_BATCH_INVARIANT="1"
```

### 4.3 vLLM 参数（P/D 对称配置）

```bash
--model /data01/home/zjj/zjj/models/Llama-3.1-8B-Instruct
--tensor-parallel-size 1
--max-model-len 32768
--gpu-memory-utilization 0.90
--block-size 16
--max-num-seqs 64
--disable-prefix-caching
--kv-transfer-config '{"kv_connector":"NixlPushConnector","kv_role":"kv_producer"}'  # P
--kv-transfer-config '{"kv_connector":"NixlPushConnector","kv_role":"kv_consumer"}'  # D
```

自定义 connector 通过 `kv_connector_module_path` 字段加载外部模块：

```json
{
  "kv_connector": "PreemptiveNixlPushConnector",
  "kv_connector_module_path": "pd_preemptive_connector",
  "kv_role": "kv_producer"
}
```

---

## 5. 工程实现细节

### 5.1 Router 请求 ID 共享机制

Router 是整个 P/D 协议的关键协调者。核心代码（`run_pd_router.py`）：

```python
async def handle_completion(request):
    # 生成共享 ID，同时发给 P 和 D
    shared_id = uuid.uuid4().hex[:16]
    shared_headers = {"X-Request-Id": shared_id}

    # P 请求：/v1/completions，prompt 为 token IDs 列表
    p_body = {
        "prompt": prompt_token_ids,     # list[int] → vLLM 识别为 raw token IDs
        "max_tokens": 1,
        "kv_transfer_params": {"do_remote_decode": True, "remote_port": 5601}
    }
    p_data = await _post_json(session, f"{P_URL}/v1/completions",
                               p_body, headers=shared_headers)
    p_kv = p_data["kv_transfer_params"]   # 包含 remote_request_id, remote_block_ids

    # D 请求：携带 P 返回的 KV 位置信息
    d_body = {
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "kv_transfer_params": {
            "do_remote_prefill": True,
            "remote_request_id": p_kv["remote_request_id"],
            "remote_block_ids":  p_kv["remote_block_ids"],
            "remote_engine_id":  p_kv["remote_engine_id"],
        }
    }
    d_data = await _post_json(session, f"{D_URL}/v1/completions",
                               d_body, headers=shared_headers)
```

### 5.2 自定义 Connector 工厂（PreemptiveNixlPushConnector）

```python
class PreemptiveNixlPushConnector(NixlPushConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        # 跳过 NixlPushConnector.__init__，直接调用基类基类
        NixlBaseConnector.__init__(self, vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = PreemptivePushScheduler(...)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = PreemptivePushWorker(...)
```

### 5.3 扩展的 Metadata 结构

```python
class PreemptiveNixlMetadata(NixlConnectorMetadata):
    """在标准 NixlConnectorMetadata 基础上增加流式传输字段。"""
    push_partial_blocks: dict[ReqId, BlockIds] = {}  # 本批次增量块
    push_total_blocks:   dict[ReqId, int]      = {}  # 请求总块数（用于 D 端校验）
```

### 5.4 实验编排脚本

`run_pd_local_sort_comparison.py` 自动化运行 3 × 3 × 2 实验：

```
对每轮（3 轮，方案顺序随机旋转）:
  对每个方案（baseline / old / new）:
    ./run_pd_pair.sh <scheme>          # 启动 P/D，含 35s NIXL 初始化等待
    python run_pd_router.py &          # 启动 router
    对每个场景（25ms / 50ms 批间隔）:
      Warmup：发送一次负载，丢弃结果
      Formal：发送一次负载，记录 CSV
    ./run_pd_pair.sh stop              # 停止 P/D/Router
```

方案顺序每轮旋转（`['baseline','old','new'] → ['old','new','baseline'] → ['new','baseline','old']`）可消除启动顺序带来的系统偏差。

---

## 6. 困难问题与解决方案

### 问题 1：ZMQ 侧信道端口冲突

**现象**：P 和 D 在同一台机器上，`VLLM_NIXL_SIDE_CHANNEL_PORT` 默认值相同，两个进程都尝试 bind 同一端口，启动失败。

**根因**：vLLM 0.33.0 的 NIXL 侧信道默认端口是固定常量，没有按角色区分。

**解决方案**：
```bash
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 python -m vllm ... (P)
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 python -m vllm ... (D)
```
两者互相知道对方端口：P 在请求中告知 `remote_port=5601`，D 读取该字段注册接收端。

---

### 问题 2：P/D 请求 ID 不匹配（核心协议 Bug）

**现象**：P 完成 prefill，D 收到请求后长时间卡在 `WAITING_FOR_REMOTE_KVS` 状态，始终超时。

**根因**：P 和 D 各自独立生成 UUID 作为 request_id。vLLM 的 `get_base_request_id` 函数截掉随机后缀后，P 和 D 的 base ID 完全不同，PUSH_REG 匹配失败，KV 永远无法交接。

```python
# vLLM 内部：request_id = "cmpl-{X-Request-Id}-{random_8hex}"
# get_base_request_id 截掉 "-{8hex}" 后缀
# 若 X-Request-Id 不同，base_id 不同 → PUSH_REG 永不匹配
```

**解决方案**：Router 生成一个 `shared_id`，在发给 P 和 D 的 HTTP 请求中都加上相同的 `X-Request-Id: {shared_id}` 头。vLLM 读取该头作为 request_id 的中段，两端 base ID 相同，PUSH_REG 匹配成功。

---

### 问题 3：prompt 格式与 endpoint 不匹配

**现象**：所有请求的输出完全相同（`"It seems like you've entered a placeholder"`），与参考输出不符。

**根因**：`/v1/chat/completions` endpoint 忽略 `prompt_token_ids` 字段，始终把请求当作空 prompt 处理，模型每次生成相同的默认回复。

**解决方案**：改用 `/v1/completions` endpoint，传入 `"prompt": [token_id_list]`。vLLM 检测到 prompt 为 `list[int]` 时直接作为 raw token IDs 处理，绕过 tokenizer，确保 prompt 内容正确。

---

### 问题 4：vLLM generate() API 变更

**现象**：生成参考输出时 `TypeError: got unexpected keyword argument 'prompt_token_ids'`。

**根因**：vLLM 0.33.0 修改了离线推理 API，不再接受关键字参数 `prompt_token_ids`，改用 `PromptType` dict。

**解决方案**：
```python
# 旧（0.32.x）：
outputs = llm.generate(prompt_token_ids=token_ids, ...)
# 新（0.33.0）：
outputs = llm.generate(prompts={"prompt_token_ids": token_ids}, ...)
```

---

### 问题 5：set -e 与 grep 管道误报错误

**现象**：脚本在检查 GPU 空闲时用 `grep -v "^$" | wc -l` 检测非空行数，当结果为空（GPU 空闲）时 `grep` 返回非零退出码，`set -e` 触发脚本退出。

**解决方案**：
```bash
# 旧（有问题）：
BUSY=$(nvidia-smi ... | grep -v "^$" | wc -l)
# 新（正确）：
BUSY=$(nvidia-smi ... | { grep -v "^$" || true; } | wc -l)
```

---

### 问题 6：自定义 Connector 继承链冲突

**现象**：`PreemptiveNixlPushConnector` 继承 `NixlPushConnector`，但 `NixlPushConnector.__init__` 会创建默认的 scheduler 和 worker，之后再赋值自定义实例无效（某些属性被覆盖）。

**解决方案**：在 `PreemptiveNixlPushConnector.__init__` 中跳过父类初始化，直接调用祖父类 `NixlBaseConnector.__init__`，然后按角色赋值自定义 scheduler/worker：

```python
class PreemptiveNixlPushConnector(NixlPushConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        NixlBaseConnector.__init__(self, vllm_config, role, kv_cache_config)  # 跳过父类
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = PreemptivePushScheduler(...)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = PreemptivePushWorker(...)
```

---

### 问题 7：NIXL EngineCore 初始化时序问题

**现象**：P/D HTTP 健康检查返回 200 后立即发送请求，仍然偶发 KV 匹配超时。

**根因**：HTTP server ready 是 FastAPI 层面的就绪，vLLM 的 NIXL EngineCore（包括 NIXL agent 注册、侧信道 handshake）在 HTTP ready 之后约 30s 才完成。

**解决方案**：在 HTTP 双端健康检查通过后额外等待 35s，确保 NIXL 初始化完毕后再发送请求：
```bash
echo "[wait] Extra 35s for NIXL EngineCore initialization..."
sleep 35
```

---

## 7. 实验设计

### 7.1 固定负载（workload）

文件：`workload.sweep-interleaved-8s3m4l.jsonl`（SHA256: `ad9d732b...46e94a`）

| 类别 | 数量 | Prompt 长度 | max_tokens | 说明 |
|------|------|------------|-----------|------|
| short | 8 | 128 tokens | 64 | 短请求，prefill 快，KV 小 |
| medium | 3 | 1107 tokens | 64 | 中等请求 |
| long | 4 | 10000 tokens | 64 | 长请求，prefill 慢，KV 大 |
| **合计** | **15** | — | — | 15 个请求 / 轮 |

请求按 short/long/short/medium 交错排列，模拟真实混合负载。

> **注意**：max_tokens 经历了 8 → 128 → **64** 的变化过程。64 token 的 decode 耗时约 2s，E2E 与 TTFT 之间有清晰间隔，便于同时观察调度优化（TTFT）和吞吐（E2E）的效果。

**TTFT 测量方式**：通过 SSE 流式（`stream: True`）测量，客户端记录首个 `data:` 载荷到达的时刻。E2E 是完整响应到达时刻。两者从同一个 `t0`（router 收到请求）起算。

### 7.2 两种到达场景

| 场景 | 批次间隔 | 含义 |
|------|---------|------|
| 25ms | 0 / 25 / 50 ms | 密集到达，调度压力较大 |
| 50ms | 0 / 50 / 100 ms | 相对稀疏到达 |

每个场景 15 个请求分 3 批（每批 5 个），按绝对时间偏移发送。

### 7.3 三方案对比

| 方案 | Connector | stream_by_batch | preemptive | local_sort |
|------|-----------|----------------|------------|------------|
| baseline | NixlPushConnector | ✗ | ✗ | ✗ |
| old | PreemptiveNixlPushConnector | ✓ | ✓ | ✗ |
| new | PreemptiveNixlPushConnector | ✓ | ✓ | ✓ |

### 7.4 实验规模

- 3 轮 × 3 方案 × 2 场景 = **18 次正式运行**
- 每轮 warmup（丢弃），每次运行 15 请求
- 共 270 条请求记录，每轮方案顺序轮换消除顺序偏差
- 参考输出：单 GPU 离线推理 `VLLM_BATCH_INVARIANT=1` 生成，用于正确性校验

---

## 8. 实验结果与分析

> **注意**：下列数据来自旧的 **250/500 ms 到达间隔实验**（max_tokens=8，stream=False，TTFT=E2E），实验方案和代码在此后有重大修复（见第11节）。采用 max_tokens=64、25/50 ms 间隔的新一轮测量已完成，见第12节；完整流式算法仍需修复验收。
>
> 下列数据仅供参考，不用于论文最终结论。

> TTFT = Time to First Token（从 router 收到请求到收到第一个 token 的时延）  
> E2E = End-to-End（从 router 收到请求到全部 token 返回的总时延）  
> v1 实验 `max_tokens=8`，decode 耗时极短，TTFT ≈ E2E（非流式测量）。

### 8.1 场景一：250ms 批间隔（batches at 0 / 250 / 500 ms）

#### TTFT（ms，均值 ± 标准差）

| 方案 | Makespan | Short | Medium | Long | **Overall** |
|------|----------|-------|--------|------|-------------|
| baseline | 4.452 s | 2533 ± 12 | 2986 ± 8 | 2747 ± 8 | 2681 ± 5 |
| old | 4.459 s | 2493 ± 29 | 2994 ± 12 | 2758 ± 31 | 2664 ± 11 |
| **new** | **4.436 s** | **1773 ± 24** | **1854 ± 24** | 2866 ± 9 | **2081 ± 8** |

#### E2E（ms，均值 ± 标准差）

| 方案 | Makespan | Short | Medium | Long | **Overall** |
|------|----------|-------|--------|------|-------------|
| baseline | 4.452 s | 2533 ± 12 | 2986 ± 8 | 2747 ± 8 | 2681 ± 5 |
| old | 4.459 s | 2493 ± 29 | 2994 ± 12 | 2758 ± 31 | 2664 ± 11 |
| **new** | **4.436 s** | **1773 ± 24** | **1854 ± 24** | 2866 ± 9 | **2081 ± 8** |

### 8.2 场景二：500ms 批间隔（batches at 0 / 500 / 1000 ms）

#### TTFT（ms，均值 ± 标准差）

| 方案 | Makespan | Short | Medium | Long | **Overall** |
|------|----------|-------|--------|------|-------------|
| baseline | 4.454 s | 2274 ± 28 | 2732 ± 9 | 2493 ± 6 | 2424 ± 12 |
| old | 4.488 s | 2249 ± 48 | 2764 ± 44 | 2541 ± 74 | 2430 ± 18 |
| **new** | **4.427 s** | **1527 ± 15** | **1630 ± 30** | 2578 ± 27 | **1828 ± 15** |

#### E2E（ms，均值 ± 标准差）

| 方案 | Makespan | Short | Medium | Long | **Overall** |
|------|----------|-------|--------|------|-------------|
| baseline | 4.454 s | 2274 ± 28 | 2732 ± 9 | 2493 ± 6 | 2424 ± 12 |
| old | 4.488 s | 2249 ± 48 | 2764 ± 44 | 2541 ± 74 | 2430 ± 18 |
| **new** | **4.427 s** | **1527 ± 15** | **1630 ± 30** | 2578 ± 27 | **1828 ± 15** |

### 8.3 相对改善（new vs baseline）

| 指标 | 250ms 场景 | 500ms 场景 |
|------|-----------|-----------|
| Short TTFT 改善 | **−30%** (−760 ms) | **−33%** (−747 ms) |
| Medium TTFT 改善 | **−38%** (−1132 ms) | **−40%** (−1102 ms) |
| Long TTFT 变化 | +4% (+119 ms) | +3% (+85 ms) |
| **Overall TTFT 改善** | **−22%** (−600 ms) | **−25%** (−596 ms) |
| Makespan 变化 | −0.4% | −0.6% |

### 8.4 个别长请求 TTFT（ms，3 轮均值）

| 请求 | Prompt | baseline (250ms) | old | new | Δ(new−baseline) |
|------|--------|-----------------|-----|-----|-----------------|
| 01-long-1501 | 10000 | 1523.3 | 1519.1 | 1660.8 | +137.5 |
| 05-long-1505 | 10000 | 2341.5 | 2379.6 | 2484.3 | +142.8 |
| 07-long-1507 | 10000 | 3234.7 | 3228.7 | 3383.5 | +148.8 |
| 11-long-1511 | 10000 | 3889.2 | 3904.8 | 3933.5 | +44.3 |

| 请求 | Prompt | baseline (500ms) | old | new | Δ(new−baseline) |
|------|--------|-----------------|-----|-----|-----------------|
| 01-long-1501 | 10000 | 1521.9 | 1569.9 | 1615.6 | +93.7 |
| 05-long-1505 | 10000 | 2089.9 | 2130.0 | 2183.4 | +93.5 |
| 07-long-1507 | 10000 | 2981.7 | 3031.3 | 3088.3 | +106.6 |
| 11-long-1511 | 10000 | 3377.7 | 3431.8 | 3424.7 | +47.0 |

越晚到达的长请求 regression 越小（`11-long` 仅 +44ms），因为后批到达时短/中请求的调度窗口已过，不再参与排序竞争。

### 8.5 结论

1. **local sort（new）效果显著**：Overall TTFT 下降 22–25%，Short TTFT 下降 30–33%，Medium TTFT 下降 38–40%。该历史实验的短窗口排序使短请求更早获得 GPU 计算资源和 NVLink 传输通道；新一轮固定使用 5 ms 窗口。

2. **stream_by_batch + preemptive 单独贡献极小（old ≈ baseline）**：NVLink 传输仅需 ~4.5ms，不是瓶颈，streaming 的隐藏收益被掩盖。若使用更大模型或更高并发，streaming 效果将更显著。

3. **总吞吐量三方案持平**：new 方案 makespan 变化在 0.6% 以内，零吞吐损失。

4. **long 请求小幅 regression 是预期 trade-off**：增加约 44–149ms（基线 1500–3900ms 的 1–4%），可通过调整 `PD_LOCAL_SORT_WINDOW_MS` 控制权衡。

5. **历史成功计数**：270/270 HTTP请求成功；输出哈希多样不能证明与参考输出一致，因此该旧记录不构成KV正确性验收。本次v2采用精确token IDs逐条校验，见第12节。

---

## 9. 文件清单

```
/data01/home/zjj/zjj/code/
├── 核心实现
│   ├── pd_local_sort_scheduler.py     # 局部排序调度器（268 行）
│   ├── pd_preemptive_scheduler.py     # 流式 KV 调度器（~200 行）
│   ├── pd_preemptive_worker.py        # 传输仲裁 Worker（~430 行）
│   ├── pd_preemptive_connector.py     # Connector 工厂（~60 行）
│   ├── pd_streaming_metadata.py       # 扩展 Metadata（~30 行）
│   ├── run_pd_router.py               # P/D 路由 HTTP 服务（184 行）
│   └── run_pd_pair.sh                 # P/D 启动脚本（252 行）
│
├── 实验工具
│   ├── run_pd_load.py                 # 负载发生器（265 行）
│   ├── run_pd_local_sort_comparison.py # 自动化对比实验（~300 行）
│   ├── analyze_pd_local_sort_comparison.py # 结果分析（~250 行）
│   ├── generate_reference_outputs.py  # 参考输出生成（125 行）
│   ├── generate_workload.py           # 负载生成器
│   └── workload.sweep-interleaved-8s3m4l.jsonl  # 15 请求固定负载
│
├── 测试脚本
│   ├── test_e2e_smoke.py              # 端到端冒烟测试
│   ├── test_nixl_nvlink.py            # NVLink P2P 验证
│   └── test_nixl_p2p.py              # NIXL P2P 基础测试
│
└── 文档
    ├── Experiment_Plan_NVLink.md      # 完整实验方案（633 行）
    ├── CHANGES_NVLink.md              # Plan.txt → NVLink 变更摘要
    └── Thesis_Technical_Report.md     # 本文档

/data01/home/zjj/zjj/results/
├── reference_outputs_a100.csv         # A100 参考输出（15 行）
└── nvlink-local-sort-comparison-20261113/
    ├── runs/                          # 18 次正式运行 CSV（每个 15 行）
    │   ├── baseline_250ms_run{1,2,3}/requests.csv
    │   ├── baseline_500ms_run{1,2,3}/requests.csv
    │   ├── old_250ms_run{1,2,3}/requests.csv
    │   ├── old_500ms_run{1,2,3}/requests.csv
    │   ├── new_250ms_run{1,2,3}/requests.csv
    │   └── new_500ms_run{1,2,3}/requests.csv
    ├── logs_round{1,2,3}_{baseline,old,new}/
    │   ├── p_*.log / d_*.log          # vLLM 进程日志
    │   ├── warmup_*.csv               # warmup 数据（已丢弃）
    │   └── pd_pair_pids.txt
    ├── logs/experiment_main.log       # 实验主进程日志
    └── analysis.json                  # 汇总分析结果
```

---

## 10. 结论

本节保留 v2 阶段结论；后续完整传输路径验收见第 13–17 节，NVLink单项和C32实验分别见第18、19节。

2026-11-14已完成当前v2实现的三轮测量，正式270条和预热270条请求均精确匹配A100参考输出。新方案的P局部排序日志及延迟表现见第12节；该收益同时包含收集窗口和排序行为，尚未通过同窗口FCFS消融单独分离。

实现审计发现提前发送、有效C8分块和单WRITE限制尚未完整生效，现有结果不能支持“完整三项优化已验证”或“old无收益是因为NVLink传输太快”的因果结论。后续应先修复并验收完整传输路径，再做模块消融和性能归因。

---

## 11. 当前状态与待续工作

**2026-11-16 更新**：第 11.1–11.3 节为 v2 历史记录。C8单项和C32四组NVLink复测分别见第18、19节。C32仍固定单槽；多槽和多个C值尚无同轮因子对照，不能区分单槽串行与块大小贡献。本轮不扩展故障恢复、多 TP/PP 或更多负载。

### 11.1 代码版本状态（2026-11-14）

2026-11-14 已恢复 GPU 访问，并完成 25/50ms、64输出token、5ms窗口的三轮对比；详细数据和实现审计见第12节。仅使用物理 GPU6/7。下列 v1→v2 修复记录保留为历史，不代表完整流式/C8算法已经验收。

#### 已修复的关键 Bug（v1 → v2）

| Bug | 现象 | 修复方案 | 文件 |
|-----|------|---------|------|
| `logging` 树隔离 | 自定义模块 INFO 日志全部丢失 | `getLogger("vllm." + __name__)` | 三个 pd_*.py |
| short 不入 `_transfer_queue` | 传输抢占从不发生 | `_finished_blocks_inbox` 内容 re-route 进 `_transfer_queue` | `pd_preemptive_worker.py` |
| long 每次 dispatch 整个队列 | short 无法插队 | long 每次只发 1 chunk 后退出循环 | `pd_preemptive_worker.py` |
| 窗口单请求等待 5ms | 孤立请求无意义延迟 | 引擎空闲时单请求立即放行 | `pd_local_sort_scheduler.py` |
| router 流式支持缺失 | TTFT = E2E（无法区分） | `_post_stream()` + SSE 首字节计时 | `run_pd_router.py` |
| workload max_tokens=8 | decode 耗时< 1ms，E2E≈TTFT | 改为 64 | workload JSONL + 相关脚本 |

#### 修复后的抢占验证（smoke_fix_v4）

在 `smoke_fix_v4` 冒烟测试中确认：
- 抢占真实发生：**2 次 bypass 事件**被记录在 `p_new.log`
- w2 窗口的 2 个 short 请求抢占了 w1 窗口的 long（队首）
- 传输后端：avg xfer time=93ms，15/15 成功
- 流式 TTFT 测量正常：short TTFT ~270ms vs E2E ~5400ms（差值 = 64 token decode）

#### 抢占次数少的原因

实验只有 15 请求，4 个 long，窗口切分后每窗口通常只有 1 个 long。C8 传输约 100ms，而每窗口 short 总计 2-3 个，完成时机窗口短，所以只有 2 次抢占。在更高并发（比如 50+ 请求同时到达）的场景下，抢占会更频繁。

### 11.2 参考输出（v2）

已重新生成（`results/reference_outputs_a100.csv`，max_tokens=64，15 条，全部 64 output tokens）。SHA256 验证通过，04-short 和 12-short 输出相同（temperature=0 贪心退化，不影响按 request_id 的逐条校验）。

### 11.3 25/50 ms 实验复现命令（已执行）

运行命令：
```bash
cd /data01/home/zjj/zjj/code

# 新建结果目录（日期自动）
RESULT_DIR=/data01/home/zjj/zjj/results/nvlink-local-sort-v2-25_50ms-$(date +%Y%m%d)

/data01/home/zjj/miniconda3/envs/cc/bin/python run_pd_local_sort_comparison.py \
    --result-dir $RESULT_DIR \
    --workload workload.sweep-interleaved-8s3m4l.jsonl \
    --rounds 3 \
    --warmup \
    > $RESULT_DIR/../logs/experiment_v2.log 2>&1 &

tail -f $RESULT_DIR/../logs/experiment_v2.log
```

分析命令（实验结束后）：
```bash
/data01/home/zjj/miniconda3/envs/cc/bin/python analyze_pd_local_sort_comparison.py \
    --result-dir $RESULT_DIR
```

正式运行需确认执行环境能访问空闲的物理 GPU 6/7。运行器会在启动服务前检查 GPU 状态和 workload 中所有请求的 `max_tokens=64`；P/D 启动脚本也会在 GPU 查询失败时拒绝运行。预计运行时间约 16 分钟（18 次正式运行，另有 warmup）。

### 11.4 历史分析计划（实际结论以第12节为准）

1. **TTFT 对比**：new vs baseline，重点看 short/medium TTFT 下降幅度（预期仍 30-40%）
2. **E2E 对比**：new vs baseline，E2E = TTFT + decode time（64 tokens），三方案 E2E 差距
3. **old vs baseline**：preemptive 在 max_tokens=64 下是否有改善（v1 中几乎无差）
4. **抢占日志**：从 `p_new.log` 统计 "Preemption: short req" 次数，与 v1(2 次) 对比
5. **正确性校验**：与 `reference_outputs_a100.csv` 比对 token hash

### 11.5 实验运行的重要环境细节

- P/D 启动后需等 **35s** NIXL EngineCore 初始化（HTTP 健康后还不能发请求）
- GPU6/7 空闲检查：`nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader`
- 停止服务：`bash run_pd_pair.sh stop`
- Router 由实验运行器持有进程句柄并在每次服务运行结束后退出；无需按名称批量终止进程。
- 本轮日志：各 `logs_round{1,2,3}_{baseline,old,new}/` 下的 P/D、Router、预热日志，以及结果目录根下的 `experiment.log`。
- Python 路径：`/data01/home/zjj/miniconda3/envs/cc/bin/python`
- 自定义模块需在 PYTHONPATH：`export PYTHONPATH=/data01/home/zjj/zjj/code:$PYTHONPATH`（`run_pd_pair.sh` 已自动设置）


## 12. 2026-11-14 v2 正式实验执行记录

状态：**已完成**。18/18 次正式运行、270/270 条正式请求通过精确输出 token IDs 校验；18 次预热共270条请求也全部通过。结果目录：`/data01/home/zjj/zjj/results/nvlink-local-sort-v2-25_50ms-20261114-041044`。

完整数据和分析：[REPORT.md](../results/nvlink-local-sort-v2-25_50ms-20261114-041044/REPORT.md)、[audit.json](../results/nvlink-local-sort-v2-25_50ms-20261114-041044/audit.json)、[逐请求数据](../results/nvlink-local-sort-v2-25_50ms-20261114-041044/requests.csv)、[逐轮数据](../results/nvlink-local-sort-v2-25_50ms-20261114-041044/runs.csv)、[逐条long](../results/nvlink-local-sort-v2-25_50ms-20261114-041044/long_requests.csv)。

**GPU 范围**：仅物理 GPU6（P）和 GPU7（D），服务分别绑定对应 UUID，其他实验子进程继承 `CUDA_VISIBLE_DEVICES=6,7`。启动前、轮次间清理和最终释放均检查两张卡；NVML PID→GPU UUID 采样未发现本实验进程访问其他卡。GPU0–5 的其他业务未被停止或修改。

**本轮参数**：Llama-3.1-8B-Instruct BF16；TP=1；context32768；gpu_memory_utilization=0.90；block16；max_num_seqs64；计算预算2048（显式传参）；prefix cache关闭；VLLM_BATCH_INVARIANT=1；UCX_TLS=cuda_copy,cuda_ipc,tcp；UCX_NET_DEVICES=lo。固定15请求：8 short×128、3 medium×1107、4 long×10000，合计44345输入tokens，每条max_tokens64、temperature0、seed1500–1514。new窗口5ms；C8、单槽和85%/3次/50ms保护仅作为已配置参数记录，不能假定全部生效（见下文）。

**设计**：三方案×两场景×三轮，三批起点分别为[0,25,50]ms、[0,50,100]ms；每批5条并发；方案/场景顺序轮换，每个正式运行前预热一次。

**采集修正**：TTFT从客户端实际提交计时，到首次解析出非空文本SSE；E2E到流完整结束；makespan从首条实际提交到最后一条结束。保存真实输出token IDs、完整文本和SSE时间戳，核验DONE、finish_reason、usage、输入/输出数量以及逐请求参考一致性。旧的文本哈希和SSE事件计数已不再当作token ID和token数量。

| 批间隔 | 方案 | makespan(s) | short TTFT(ms) | medium TTFT(ms) | long TTFT(ms) | overall E2E(ms) |
|---|---|---:|---:|---:|---:|---:|
| 25ms | baseline | 6.96±0.02 | 2623.78±53.03 | 3071.33±33.43 | 2914.55±33.79 | 5423.74±27.31 |
| 25ms | old | 6.98±0.03 | 2657.94±2.09 | 3101.04±10.98 | 2926.86±13.07 | 5446.87±13.70 |
| 25ms | new | 7.04±0.02 | 1327.20±700.82 | 1654.37±908.66 | 3039.52±66.27 | 4342.00±639.69 |
| 50ms | baseline | 6.92±0.03 | 2595.40±24.30 | 3056.27±20.37 | 2851.07±15.05 | 5390.34±34.56 |
| 50ms | old | 6.94±0.04 | 2574.33±19.31 | 3032.80±9.02 | 2857.04±14.72 | 5377.57±27.15 |
| 50ms | new | 7.01±0.01 | 1296.53±461.93 | 1519.54±364.92 | 2984.81±14.97 | 4380.08±332.13 |

表中为三轮均值±样本标准差。

new 的 short/medium TTFT 重复间波动明显，特别是25ms场景；应结合原始逐轮数据和窗口日志解读三轮均值。这里没有通过同窗口FCFS消融单独隔离收集等待与长度排序的贡献。

25ms场景，new相对baseline：short TTFT -49.42%；medium TTFT -46.13%；long TTFT +4.29%；makespan +1.10%；overall E2E -19.94%。

50ms场景，new相对baseline：short TTFT -50.04%；medium TTFT -50.28%；long TTFT +4.69%；makespan +1.29%；overall E2E -18.74%。

**解释边界及下一步**：

- 提前发送未形成有效路径：归档版 PreemptivePushScheduler 通过 self.requests 查找活动请求，但本类初始化及父类均未创建该属性；因此活动 prefill 请求被跳过。
- Router 在收到 P 完成响应后才向 D 提交请求，P prefill 期间不存在 D 提前注册接收地址的路径。
- CHUNK_SIZE_BLOCKS=8 和 PD_STREAM_MAX_IN_FLIGHT=1 是配置值，尚不能证明 C8 提前发送和单 WRITE 并发限制生效；归档 Worker 的 max-in-flight 仅被读取及打印。
- 早期单进程 NIXL UCX 测量因 same-process 限制排除 CUDA IPC，回退至 tcp/lo，约0.45 GB/s；双进程 microbenchmark 已确认 CUDA IPC/NVLink，吞吐约77–201 GB/s。该 microbenchmark 仍包含每次 transfer handle 创建、提交、轮询和释放，不代表 vLLM KV workload 的端到端有效带宽，不能用纯 P2P 参考值估算 NIXL 传输耗时。
- 插队日志表示软件队列 bypass 决策，汇总次数包含预热，不能据此证明在活动 C8 DMA 块之间抢占。
- 5ms 是收集窗口配置，释放由调度循环检查触发，不能保证实际仅等待5ms；具体等待分布见 local_sort_observations_including_warmup。

P局部排序日志的实际最大等待（含预热）为179.12ms；日志内窗口均按输入长度非降序排列。

**后续任务**：先修复活动请求追踪、D提前注册、按offset增量WRITE、最终完成通知/源KV可见性及lease、真实C8分块和并发限制，并补齐逐chunk执行轨迹，再重新验收完整流式算法。当前结果可用于描述当前实现以及P局部排序的表现，不能作为完整三项优化均已实现的性能结论。

实验环境允许其他GPU业务继续运行，因此没有隔离共享CPU、内存和互联资源；三轮重复和轮换不能替代更大样本或跨负载验证。

复现归档见结果目录的 `environment.json`、`source-runtime/`、`source-vllm/`、`source-analysis/`、`SHA256SUMS`、`gpu_telemetry.jsonl`、`gpu_contexts.jsonl`。

## 13. 2026-11-14 流式协议修复

实现审计后已修复活动请求表、每步增量 block table、D 提前注册、带 offset 的目标块切片、post-forward CUDA event、partial/final 完成通知、每请求 WRITE 顺序、真实单槽限制及按完成块数保护进度。详细原因、适用范围与验收命令见 [Streaming_Repair_20261114.md](Streaming_Repair_20261114.md)。

修复后两场景 GPU 并发 smoke 共30/30请求、每条64 tokens，逐 token ID 与参考完全一致；686 个 WRITE/DONE 轨迹检查 offset 连续、唯一 final、long 每块≤8 blocks、最大活动 WRITE=1 均通过。记录见 `results/smoke-protocol-v2-20261114/streaming_audit.json`。所有计算仅使用物理 GPU6/7。新正式三轮对比已完成，独立归档在 `results/nvlink-streamfix-v3-20261114/`。

### 13.1 正式实验最终验收

2026-11-14 UTC 已完成三方案×两场景×三轮：18/18次正式运行、270/270条正式请求及18次预热共270条请求全部成功，每条64个输出 token IDs 与参考完全一致。正式数据位于 [RESULTS.md](../results/nvlink-streamfix-v3-20261114/RESULTS.md)，逐请求与逐轮数据见同目录 `requests.csv`、`runs.csv`。

[streaming_audit.json](../results/nvlink-streamfix-v3-20261114/streaming_audit.json) 审计通过：old/new含预热共360条流式请求、8236个实际WRITE及对应DONE；offset连续且无缺口，long每块≤8 blocks，最大活动WRITE=1，360条请求各有唯一D final通知，无writer错误或lease过期。CPU回归10/10通过。运行前后7个核心运行文件与快照哈希均一致。

| 批间隔 | 方案 | makespan(s) | short TTFT(ms) | medium TTFT(ms) | long TTFT(ms) | overall E2E(ms) |
|---|---|---:|---:|---:|---:|---:|
| 25ms | baseline | 6.95±0.01 | 2646.51±14.63 | 3088.28±6.63 | 2913.66±10.62 | 5449.16±19.10 |
| 25ms | old | 8.99±0.27 | 4570.46±205.40 | 5568.36±331.80 | 5678.79±223.83 | 7603.84±209.36 |
| 25ms | new | 9.01±0.31 | 3390.96±83.09 | 4555.13±283.32 | 5719.65±251.45 | 6669.23±70.49 |
| 50ms | baseline | 6.94±0.01 | 2572.74±4.80 | 3053.30±1.36 | 2878.17±5.52 | 5397.17±12.05 |
| 50ms | old | 9.15±0.35 | 4593.31±141.51 | 5708.04±356.32 | 5732.67±235.77 | 7664.08±207.91 |
| 50ms | new | 8.93±0.19 | 3342.53±313.29 | 4336.23±499.05 | 5567.35±114.77 | 6620.20±326.63 |

三轮均值±样本标准差。与baseline相比，old/new的makespan分别增加29.30%/29.58%（25ms）和31.77%/28.70%（50ms）。new相对old改善了short/medium TTFT，但仍慢于baseline；本次验收证明协议正确性和实际C8/单槽生效，没有证明流式方案的性能收益。不能把历史串行Router数据与本次三方案共用的并发Router数据直接合并。

当前存在逐chunk提交、完成回报和1ms轮询开销；尚未消融其与排序等待、传输仲裁的各自贡献，不能定量归因。后续性能实验应先测chunk等待/WRITE/DONE各阶段耗时，再比较事件驱动完成处理与轮询、不同chunk大小，并在相同Router与负载下保留逐token验收。本次不以未验证的性能改动替换已验收的代码。

实验仅绑定物理GPU6(P)/GPU7(D) UUID；监控207次采样、338条两卡上下文记录，无采样错误。该监控只查询两张卡，部分EngineCore环境mask不可读，不能声称逐进程全设备审计。最终两卡均0 MiB、0%利用率、无计算进程，8000/8100/8200端口无监听，证据见结果目录 `gpu_final_state.json` 和 `gpu_monitor_summary.json`。其他GPU业务未被停止或修改。

历史归档和失败尝试保留，完整源代码、分析脚本与文档快照随结果保存。

## 14. writer完成推进性能优化

原有模型主线程逐chunk完成推进和final占用单槽的延迟已优化。writer独占WRITE handle检查与回收，通过锁保护的集合将final恰好一次交给主线程沿父类流程释放请求KV；wake可中断等待，回退轮询0.1ms。13项CPU回归及全部正式验收通过。详细设计、边界及数据见 [Writer_Progress_Optimization_20261114.md](Writer_Progress_Optimization_20261114.md)。

### 14.1 正式实验结果

2026-11-14 UTC 完成18/18次正式运行，270/270条正式及270/270条预热请求均与参考逐token一致。13项CPU回归通过。old/new含预热共360条流式请求、8239次WRITE/DONE审计无错误，long每块≤8 blocks、最大活动WRITE=1、每请求唯一D final通知。正式期间8个核心运行文件与快照哈希一致。

| 批间隔 | 方案 | makespan(s) | short TTFT(ms) | medium TTFT(ms) | long TTFT(ms) | overall E2E(ms) |
|---|---|---:|---:|---:|---:|---:|
| 25ms | baseline | 6.93±0.04 | 2631.24±27.95 | 3079.02±12.36 | 2909.86±18.53 | 5430.42±48.69 |
| 25ms | old | 6.98±0.05 | 2775.72±42.87 | 3238.71±67.37 | 3024.38±38.38 | 5598.63±48.32 |
| 25ms | new | 6.98±0.03 | 1437.41±607.11 | 1588.84±500.51 | 3172.98±42.96 | 4402.40±506.82 |
| 50ms | baseline | 6.92±0.01 | 2580.24±29.76 | 3034.90±15.57 | 2863.71±15.30 | 5394.62±29.91 |
| 50ms | old | 6.96±0.03 | 2730.14±23.57 | 3206.73±31.52 | 2976.71±14.59 | 5553.11±23.95 |
| 50ms | new | 6.97±0.08 | 1686.06±365.32 | 1853.07±250.20 | 3115.90±75.59 | 4644.09±278.22 |

相对上一版v3，old的makespan下降22.39%/23.86%，new下降22.58%/21.95%（25ms/50ms），baseline仅变化-0.31%/-0.24%。前述跨归档变化没有单独消融完成推进、wake和轮询间隔，不能归因到单一修改。

本轮new相对本轮baseline：25ms short/medium TTFT下降45.37%/48.40%，overall E2E下降18.93%，makespan增加0.64%，long TTFT增加9.04%；50ms分别下降34.65%/38.94%、13.91%，makespan增加0.70%，long TTFT增加8.81%。短中请求延迟得到改善，但未获得整体完成时间收益，long首token存在代价；三轮new波动明显，详见表中样本标准差，不宣称统计显著性。

阶段日志（包含预热）中old/new final observed_ms均值为1.617/1.716ms，p95为3.283/3.439ms；相较partial-only探测的final均值77.251ms已有明显改善。partial均值仍为3.711/3.741ms，long块submit_ms均值2.564/2.634ms；这些均为软件观测，包含Python/CPU调度与轮询，不是纯DMA时间。原有主线程逐chunk等待和final占槽的主要退化已解决，剩余C8提交开销与长请求TTFT仍可优化。本轮保留C8/单槽，未用增大chunk或并发来替代协议要求。

采样470次、744条属于实验进程的GPU上下文均仅落在物理GPU6/7；无监控错误。按2秒采样，不能排除采样间短暂上下文。结束后两卡0 MiB、0%利用率、无计算进程，8000/8100/8200无监听，见 `gpu_final_state.json`。其他GPU业务未被停止或修改。

完整数据：[RESULTS.md](../results/nvlink-writer-progress-v4-20261114/RESULTS.md)、[优化前后对比](../results/nvlink-writer-progress-v4-20261114/OPTIMIZATION_COMPARISON.md)、[传输审计](../results/nvlink-writer-progress-v4-20261114/streaming_audit.json)、`writer_timings.json`、`requests.csv`及`runs.csv`。旧结果、探测及中断记录独立保留。

## 15. 2026-11-14 边界复核、修复与重新验收

本次在v4完成推进优化后继续复核调度器、writer、Router及实验数据验收。历史章节和归档保留；修复后新实验独立保存在 `results/nvlink-review-v5-20261114/`。

数据更新约定：后续每次正式实验完成并通过独立审计后，除更新结果目录的数据文档，还必须在本技术报告追加对应版本的参数、三轮结果表、相对baseline变化、正确性验收和剩余问题。历史实验表保留版本与归档链接，不用新数据覆盖旧结论。当前最新已完成数据为改进技术的独立重跑，完整结果见17节；v6首次验收见16.3，v5历史数据见15.3。

### 15.1 发现的问题与解决方案

| 问题 | 原因及影响 | 解决方案与验证 |
|---|---|---|
| 全量取消不兼容vLLM 0.33 | `finish_requests(None, status)`尝试`list(None)`；覆写也丢掉父类返回的aborted Requests | None展开为当前request IDs，保留str/迭代器支持，返回父类结果；删除空窗口，补全量和生成器取消回归 |
| 单请求可能越过旧窗口 | schedule先扫描并提前释放任意singleton，较新singleton可能越过已到期的多请求窗口 | 先释放到期窗口；提前释放仅允许队首singleton；保持窗口间FCFS，并记录实际共享deadline |
| 窗口deadline日志不准确 | 每条请求用自己的admitted_at+5ms计算deadline，同窗口成员实际共用首条deadline | 使用窗口内最早admission+实际window_ms，保留实际wait_ms；5ms仍是最小收集窗口，不保证引擎恰好5ms释放 |
| Router流式转发被P响应阻塞 | 已收到D首批SSE，仍await P HTTP完成再转发，客户端TTFT可能包含无关P回包延迟 | D SSE边读取边转发，P/D task独立推进；两个task完成后结束HTTP流。真实本地HTTP回归证明P未完成时首条SSE已可读取 |
| 后端异常时另一侧任务未完整回收 | gather默认遇异常不取消兄弟task；流式入口也只取消P且不await，可能让失败请求持续等待 | FIRST_EXCEPTION及时处理任一侧失败，finally取消并await两个task；响应头已发则用SSE error，未发则HTTP502；客户端断连启用handler cancellation，回归验证双向失败和外层取消 |
| 非流式TTFT被误记为E2E | 非流式只有完整响应时刻，无法得到首token时间 | 非流式pd_metrics.ttft_ms改为null，仅报告可测E2E；正式TTFT仍从客户端首次非空文本SSE计时 |
| 256-token边界分类错误 | 仅按ceil(prompt/16)块数分short/medium，256 tokens也被分为short | chunk metadata携带实际prompt_tokens，按short<256、medium≤8192分类；241/255/256/8192/8193边界回归。固定负载128/1107/10000不受边界改动影响 |
| partial失败标记可能长期滞留 | 父类只消费请求级final IDs，内部synthetic partial failure IDs不属于请求lease | partial失败保留在writer请求状态，进行中DMA handle继续保留；释放后删除内部失败ID，不交给父类请求完成集合；补失败后handle释放且无错误成功回报回归 |
| 数据审计过度信任成功标记 | CSV success/reference_match为采集器自报，审计未独立读取reference和SSE；也未按日志时间顺序验证单槽 | 新增独立validate_pd_run，核对请求集合、实际token IDs、hash、SSE tokens/text/usage/finish、TTFT/E2E/makespan；新采集保存DONE。按日志顺序验证WRITE后必须先DONE才有下一WRITE，要求active=1；正式目录集合完整且流式请求/通知数均360才发布汇总 |
| CSV文本读取可能改变换行 | 默认universal newline会把模型原始文本中的CR归一化，妨碍SSE/CSV逐文本比较 | 验证和汇总CSV使用newline=''，保留文本原始换行；旧v4归档使用新独立验证在内存重审通过，不改动历史文件 |

### 15.2 验收与适用范围

CPU命令：`CUDA_VISIBLE_DEVICES='' PYTHONPATH=code /data01/home/zjj/miniconda3/envs/cc/bin/python -m unittest discover -s code -p 'test_pd_*.py'`。包括真实本地HTTP流式转发、双方失败/外层取消、窗口顺序/取消、长度边界、scheduler/worker lifecycle和篡改token IDs检测。修复后正式实验参数沿用v4：三方案×25/50ms×三轮，每次正式前预热；模型BF16、TP=PP=1、block16、计算预算2048、C8、单槽、5ms排序窗口、0.1ms writer轮询、64输出不变。

本次Router转发与窗口顺序有变化，因此新数据独立归档，并在本轮三个方案中使用同一Router。跨v4/v5差异不能全部归给一个模块。改动不扩展至多TP/PP、混合KV group、prefix cache或已发partial后的GPU内存重分配；真实GPU故障/传输失败后的自动远端重试仍未实现，CPU失败回归不等同故障注入GPU验收。

状态：**已完成**。22/22项CPU回归通过，三轮正式实验、独立数据审计和GPU释放检查均已完成，详见15.3。

### 15.3 修复后正式重跑结果

2026-11-14 UTC完成三方案×两场景×三轮：18/18次正式运行、270/270条正式请求及18次预热共270条请求均通过。36组CSV/SSE独立核验实际token IDs与参考完全一致，并校验请求集合、64输出、SSE结束标记、usage、finish、文本、首个非空文本TTFT、E2E及makespan时间戳；所有逐次核验记录为`*.validation.json`。22项CPU回归及故障/取消模拟全部通过，CPU测试未使用GPU。

| 批间隔 | 方案 | makespan(s) | short TTFT(ms) | medium TTFT(ms) | long TTFT(ms) | overall E2E(ms) |
|---|---|---:|---:|---:|---:|---:|
| 25ms | baseline | 6.94±0.05 | 2625.41±51.77 | 3088.32±23.03 | 2914.09±25.77 | 5435.09±57.45 |
| 25ms | old | 6.99±0.04 | 2781.61±30.00 | 3244.11±38.31 | 3024.96±18.61 | 5601.01±30.90 |
| 25ms | new | 6.98±0.01 | 1114.55±342.13 | 1405.43±341.35 | 3161.84±30.85 | 4178.87±324.67 |
| 50ms | baseline | 6.94±0.02 | 2586.88±22.57 | 3039.84±8.99 | 2872.13±13.98 | 5408.37±29.65 |
| 50ms | old | 6.98±0.01 | 2729.22±14.49 | 3183.61±8.39 | 2978.92±14.36 | 5549.67±12.39 |
| 50ms | new | 6.98±0.02 | 1765.50±425.03 | 1956.91±370.44 | 3100.47±21.91 | 4726.91±353.88 |

表为三轮均值±样本标准差。25ms场景，new相对本轮baseline：short TTFT -57.55%，medium TTFT -54.49%，long TTFT +8.50%，overall E2E -23.11%，makespan +0.51%；50ms依次为-31.75%、-35.62%、+7.95%、-12.60%、+0.55%。new短中请求仍有轮次波动，不以三轮均值宣称统计显著性。old无排序的TTFT仍高于baseline，不能把“协议正确且可提前发送”解释为该负载上的普遍性能收益。

相对v4，本轮三个方案makespan均值变化仅+0.04%～+0.24%，主要性能行为保持。Router转发、窗口和边界修复未单独消融；两轮实验时间及共享CPU/内存/互联争用不同，不能把short TTFT跨归档变化全归因于单一修复。历史v4用新审计只读复查36组数据全部通过，复查结果另存本轮`previous_independent_reaudit.json`，旧文件未覆盖。

**传输验收**：old/new含预热360条流式请求、8236次WRITE/DONE；日志按时间顺序确认单槽任一WRITE之前上一WRITE已经DONE，active恒为1；offset连续无重复/缺口、long每块≤8 blocks、每请求唯一final，360个D final通知与P请求base IDs一致。P/D日志未发现writer错误、传输失败、未识别通知或lease过期。正式期间10个运行及验证文件与快照哈希一致，见`runtime_source_consistency.json`。

**窗口验收与剩余代价**：new含预热共27个正常释放窗口、168条窗口请求及12条idle singleton提前释放。正常窗口内输入长度非降序、rank连续、共享deadline、释放未早于deadline；无分析错误。正常窗口请求实际等待均值77.72ms、最大177.38ms。5ms是接纳收集窗口而非5ms延迟上限；请求接纳后要等引擎下次调度才能释放。此次修复保证窗口间顺序及日志准确，没有消除模型执行期间的调度等待，不更改论文算法为另一种调度策略。

长请求平均TTFT代价约8%，逐条long的三轮均值与配对baseline变化见`long_requests.csv`。三项保护作用于传输chunk边界，不保证整体TTFT/E2E SLA，也不能消除P计算排序带来的长请求等待。尚未覆盖的远端故障自动重试、多TP/PP及内存重分配见15.2；本负载验收通过不等同这些路径已获支持。

**GPU范围与清理**：P/D按UUID仅绑定物理GPU6/7，472次采样、743条实验进程GPU上下文均只出现这两张卡，监控无错误。采样间隔2秒，证据不排除采样间极短上下文。09:10:12 UTC结束检查两卡均0 MiB、0%利用率、无计算进程，8000/8100/8200无监听，见`gpu_final_state.json`；其他GPU业务未被停止或修改。

数据与文档：[本轮RESULTS.md](../results/nvlink-review-v5-20261114/RESULTS.md)、[v4/v5对比](../results/nvlink-review-v5-20261114/OPTIMIZATION_COMPARISON.md)、[独立输出与传输审计](../results/nvlink-review-v5-20261114/streaming_audit.json)、[窗口分析](../results/nvlink-review-v5-20261114/review_analysis.json)、[逐请求数据](../results/nvlink-review-v5-20261114/requests.csv)、[逐轮数据](../results/nvlink-review-v5-20261114/runs.csv)、[逐条long](../results/nvlink-review-v5-20261114/long_requests.csv)。归档含环境、负载、参考、运行源码/vLLM接口/最终分析源码、原始SSE与P/D/Router日志、SOURCE_SHA256SUMS和最终SHA256SUMS。

### 15.4 仍未解决或尚未验收的问题

当前固定负载的输出与协议审计通过，不能据此声称所有输入、取消、故障和并行配置均不存在问题。下表区分已实测的性能问题与尚未实现/验收的能力；不把后者描述为正式实验已经触发的故障。

| 项目 | 当前证据与影响 | 下一步工作及验收标准 |
|---|---|---|
| 实际窗口等待远大于5ms（v6已优化） | v5历史均值77.72ms、最大177.38ms；v6实际提交均值5.144ms、最大8.656ms | 独立deadline线程与线程安全FCFS队列已完成三轮验收，见16.3；提交后的计算等待单独报告，不保证GPU5ms开算 |
| long首token代价仍约8% | new相对baseline的long TTFT增加8.50%/7.95%；三项传输保护不保证整体TTFT SLA | 分离P计算等待、KV就绪、传输排队与D计算等待，先定位代价来源；修改保护或排序时保留逐条long配对数据，同时检查short/medium收益 |
| C8单槽尚未带来整体完成时间收益 | old/new makespan仍比baseline高约0.5%～0.6%；逐块提交存在CPU开销，old各类TTFT也高于baseline | 在C8/单槽语义下定位descriptor准备、native提交与轮询开销；保持相同Router/参数做对照。当前不能宣称流式传输本身使该负载整体加速 |
| 真实故障恢复尚不完整 | 已有CPU失败/取消回归，但真实GPU WRITE失败、超时及传输中断连后的P/D联合回收与自动重试尚未GPU验收；自动远端重试未实现 | 先做有界故障注入，验证失败请求不会报告成功、在途DMA结束前不复用KV块、两端最终回收；之后再实现幂等重试 |
| 内存抢占和扩展配置未支持/未验收 | 已发partial后P端重新分配KV需要重启远端请求；当前限制TP=PP=1、单device KV group、prefix cache关闭；host buffer等未验收 | 为重新分配设计请求代次和远端失效协议；逐项扩展支持并补GPU验收。在此之前维持当前实验支持范围 |
| 机制证据和统计覆盖仍不足 | 无逐WRITE物理transport/重叠时间线；仅一种15请求负载、两种间隔、每场景三轮，共享CPU/内存/互联未隔离；new TTFT存在轮次波动 | 分别补充传输/计算时间线、同窗口FCFS及仲裁消融、更多轮次和负载。现有数据不能独立量化排序、窗口、仲裁贡献，也不能证明每个WRITE走NVLink或结果具统计显著性 |

优先顺序：先定位窗口等待与long代价，再优化逐块提交开销；真实故障恢复属于可靠性扩展，应单独验收。任何新改动继续仅用物理GPU6/7计算，保留历史数据，并将新正式结果同步追加到本报告。

## 16. 5ms窗口独立到期提交优化（v6）

### 16.1 问题与实现

v5正常窗口请求实际接纳等待平均77.7158ms、最大177.38ms。原因是仅在`Scheduler.schedule()`中摘除到期窗口，引擎在模型future等待期间不会执行这段逻辑，5ms到期后仍要等待下一次调度。此次只优化本地窗口的提交与回收；本轮正式结果独立保存在`results/nvlink-deadline-v6-20261114/`，探测另存`results/nvlink-deadline-probe-20261114/`。

新增独立deadline线程，使用Condition等待最早窗口的原始monotonic deadline；到期时立即摘除窗口，按(prompt tokens, admitted_at)稳定排序，将整窗原子追加到线程安全的实际FCFS waiting队列。线程只追加请求，Request状态、KV分配及GPU调度仍由engine线程处理。等待队列支持快照迭代，避免并发追加导致deque迭代失败；不在完整schedule或模型执行期间持有Condition。保留跨窗FCFS与idle singleton提前释放策略。

父类add_request在入窗后才登记request与connector，因此将完整注册过程与到期发布同步；取消、全量取消与到期摘除也使用同一Condition，删除空窗口及每请求提交时间记录，避免取消后复活。取消首个成员不会改变该窗原始deadline。shutdown唤醒并join线程，后台异常在engine后续访问时显式抛出。

测量口径：`wait_ms`为engine入窗至实际waiting队列追加完成；`late_ms`为追加完成减共享deadline；`ready_queue_ms`为队列发布至首次包含该请求的schedule完成观测，不是GPU kernel开始时刻。窗口收集从engine入窗计时，不是HTTP到达时间。普通Python/OS定时线程不提供硬实时保证；是否改善以GPU负载下测得的lateness为准，不将移到普通等待队列后的等待隐藏为已消失。

### 16.2 验证与参数

CPU回归30/30通过，包含无任何schedule调用时到期发布、取消/到期并发不复活、发布前完成注册、首成员取消保留deadline、快照迭代及线程关闭。GPU探测与正式对照仅绑定物理GPU6(P)/GPU7(D) UUID，三方案×两场景×三轮及每次正式前预热；模型、15请求负载、64输出、5ms窗口、C8、单槽、0.1ms writer轮询及其余参数沿用15.2。到期统计包含预热，独立输出与传输审计沿用v5标准。

状态：**已完成**。三轮正式实验、独立输出/协议/窗口审计和GPU释放检查均已通过，详见16.3。


### 16.3 正式实验结果与窗口验收

2026-11-14T09:48:41.123632+00:00 完成三方案×两场景×三轮：18/18次正式运行、270/270条正式请求与270/270条预热请求全部逐token一致；36组CSV/SSE独立审计通过。30/30项CPU回归通过，其中取消竞争测试让到期线程与取消线程在实际deadline后竞争同一Condition，验证两种执行顺序均不复活请求。运行期间11个核心执行文件和vLLM接口快照哈希一致。

| 指标（含预热，正常释放请求加权） | v5 | v6 |
|---|---:|---:|
| 正常窗口数/请求数 | 27/168 | 24/169 |
| 入窗到提交平均(ms) | 77.716 | 5.144 |
| 入窗到提交p95(ms) | 176.650 | 8.401 |
| 入窗到提交最大(ms) | 177.380 | 8.656 |
| 超过deadline平均(ms) | 72.726 | 0.417 |
| 超过deadline p95(ms) | 172.000 | 3.656 |
| 超过deadline最大(ms) | 173.000 | 3.656 |

正常窗口23/24由独立timer释放、1由engine路径释放；另有11条idle singleton提前释放，共180条new请求均恰好一次提交并记录首次schedule。正常窗口内rank连续、长度稳定排序、共享deadline和发布时间、未提前释放，跨窗口时间有序。到期后窗口结构立即摘除，取消后删除空窗与时间记录；30项CPU回归及GPU日志审计均无重复/遗漏。窗口平均等待相对v5降低93.38%。

v5到期延迟根据原日志毫秒精度时间戳重算，v6为微秒时间戳、0.001ms耗时记录；p95为nearest rank。窗口成员与数量随真实engine入窗时刻变化，属于两次相同负载实验的分布比较，不能视为逐窗口配对。

**提交后的计算等待仍存在**：全部180条new请求（含singleton与预热）在实际waiting队列中至首次schedule完成观测等待平均700.364ms，p95 2657.201ms，最大2707.071ms。到期发布已独立于下一次schedule，但模型当前执行、请求前序与2048-token计算预算仍决定何时获得计算机会；没有测量GPU kernel开始时间，不能承诺5ms开始计算或硬实时deadline。

| 批间隔 | 方案 | makespan(s) | short TTFT(ms) | medium TTFT(ms) | long TTFT(ms) | overall E2E(ms) |
|---|---|---:|---:|---:|---:|---:|
| 25ms | baseline | 6.90±0.04 | 2622.12±6.29 | 3064.43±13.20 | 2894.69±15.32 | 5401.83±21.96 |
| 25ms | old | 6.98±0.01 | 2773.73±11.10 | 3250.19±4.16 | 3024.84±16.89 | 5599.54±4.48 |
| 25ms | new | 6.99±0.02 | 1463.48±625.88 | 1598.27±497.84 | 3181.94±33.94 | 4422.98±519.17 |
| 50ms | baseline | 6.91±0.02 | 2607.35±22.76 | 3032.36±15.16 | 2863.40±15.37 | 5411.10±35.81 |
| 50ms | old | 6.97±0.01 | 2734.91±7.70 | 3212.54±8.19 | 2976.11±5.81 | 5559.32±4.47 |
| 50ms | new | 6.95±0.03 | 1257.75±27.25 | 1516.12±2.02 | 3098.00±19.49 | 4309.55±17.30 |

表为三轮均值±样本标准差。new相对本轮baseline：

- 25ms：makespan +1.32%；short TTFT -44.19%；medium TTFT -47.84%；long TTFT +9.92%；overall E2E -18.12%。
- 50ms：makespan +0.50%；short TTFT -51.76%；medium TTFT -50.00%；long TTFT +8.19%；overall E2E -20.36%。

及时提交窗口不等于整体完成时间收益；本轮仍按实测数据报告long代价及short/medium收益。三轮不足以证明统计显著性，共享CPU、内存与互联条件未隔离，不能将跨归档TTFT变化全部归因于单个机制。

**传输与输出验收**：old/new含预热360条流式请求、8234次实际WRITE及对应DONE，按日志顺序验证C8、单槽、offset连续与每请求唯一final；无writer错误、lease过期或未识别通知。

**GPU范围与清理**：465次采样、738条实验进程上下文均仅位于物理GPU6/7，监控无错误；2秒采样不能排除采样间极短上下文。2026-11-14T09:48:41.123632+00:00最终检查两卡均0 MiB、0%利用率且无计算进程，8000/8100/8200无监听。其他GPU业务未被停止或修改。

停服日志另存shutdown_log_audit.json：逐服务检查主动shutdown前没有ERROR；部分vLLM默认abort/timeout=0停服后出现EngineDeadError与semaphore回收提示，保留原始日志，不能将其描述为运行中请求失败。GPU与端口最终回收由独立查询确认。

### 16.4 结论与剩余范围

已消除窗口释放依赖下一次schedule造成的77–177ms滞留，固定负载下到期后平均0.417ms、最大3.656ms完成提交；到期结构、取消和线程生命周期已覆盖。本结果是实测时限改善，普通Python/OS仍可能受GIL、注册过程与CPU调度影响，不能宣称任意负载下零误差硬实时。

本次单独优化窗口，不改变长请求计算排序、C8/单槽或三项保护策略。15.4中long首token代价、逐块CPU开销、真实故障恢复、内存重分配、多TP/PP与统计覆盖事项仍保留；不以窗口验收宣称这些能力已完成。

完整数据：[RESULTS.md](../results/nvlink-deadline-v6-20261114/RESULTS.md)、[窗口前后对比](../results/nvlink-deadline-v6-20261114/OPTIMIZATION_COMPARISON.md)、[deadline分析](../results/nvlink-deadline-v6-20261114/deadline_analysis.json)、[输出与传输审计](../results/nvlink-deadline-v6-20261114/streaming_audit.json)、[逐请求](../results/nvlink-deadline-v6-20261114/requests.csv)、[逐轮](../results/nvlink-deadline-v6-20261114/runs.csv)、[逐条long](../results/nvlink-deadline-v6-20261114/long_requests.csv)。原始日志、SSE、负载、参考、冻结运行源码、最终测试/分析源码、GPU证据与SHA256SUMS同目录归档。

## 17. 窗口改进后独立重跑实验（2026-11-14）

### 17.1 配置、测量口径与验收

本次按照改进后的技术重新采集数据，独立归档`results/nvlink-deadline-rerun-20261114-144441/`，不覆盖v5/v6历史实验，也不与旧轮次合并。启动UTC见environment.json，结束后验收UTC为2026-11-14T15:03:48.068197+00:00。11个核心运行文件和vLLM接口与v6及启动快照一致；本次未新增算法修改，属于相同实现的独立重复测量。

方案：baseline为整请求KV传输；old为流式C8/单槽与保护、无局部排序；new在old上启用5ms固定窗口、独立deadline线程和线程安全FCFS队列。三方案按轮次轮换顺序，两场景每次正式前预热，共18次正式及18次预热。

参数：Llama-3.1-8B-Instruct BF16，TP=PP=1，context32768，memory0.90，block16，max_num_seqs64，计算预算2048，prefix cache关闭，VLLM_BATCH_INVARIANT=1。每次15请求：8×128、3×1107、4×10000输入；每请求64输出，temperature0，seed1500–1514。批起点[0,25,50]/[0,50,100]ms；new窗口5ms；old/new C8、单槽，长请求保护85%/3次/50ms，writer轮询0.1ms。仅物理GPU6(P)/GPU7(D)，按UUID绑定。

TTFT从客户端发出请求到首个非空文本SSE，E2E到完整响应结束，窗口及计算排队耗时均保留，不扣减；makespan为最早发送至最后结束。正式性能表先按每轮请求类别求平均，再报告三轮均值±样本标准差。窗口wait从engine入窗到实际waiting队列原子发布完成，late为发布完成减共享deadline；ready_queue从发布到首次schedule完成观测，不是GPU kernel开始。窗口统计包含预热、按请求加权，p95采用nearest rank。

验收：30/30项CPU回归通过；18/18次正式运行，270/270条正式及270/270条预热请求的实际输出token IDs与参考完全一致，36组CSV/SSE独立校验请求集合、64输出、文本/usage/finish/DONE及TTFT/E2E/makespan；old/new共360条流式请求、8233次WRITE及对应DONE审计通过，offset连续、long每块≤8 blocks、活动WRITE=1且每请求唯一D final。

### 17.2 本次正式数据

| 批间隔 | 方案 | makespan(s) | short TTFT(ms) | medium TTFT(ms) | long TTFT(ms) | overall E2E(ms) |
|---|---|---:|---:|---:|---:|---:|
| 25ms | baseline | 6.97±0.02 | 2645.43±19.90 | 3099.23±21.33 | 2936.56±25.15 | 5452.85±6.52 |
| 25ms | old | 7.01±0.01 | 2796.40±14.27 | 3259.42±14.42 | 3041.08±19.24 | 5617.14±12.60 |
| 25ms | new | 7.00±0.02 | 1107.48±646.97 | 1297.75±520.07 | 3209.77±30.93 | 4125.96±538.77 |
| 50ms | baseline | 6.96±0.00 | 2626.17±12.49 | 3066.16±5.86 | 2888.57±5.86 | 5435.13±22.23 |
| 50ms | old | 7.00±0.02 | 2767.75±19.65 | 3225.48±6.33 | 3011.40±25.75 | 5584.29±13.84 |
| 50ms | new | 6.95±0.06 | 1726.78±460.64 | 2168.67±589.28 | 3088.61±68.08 | 4694.99±369.16 |

补充各类别E2E及整体TTFT（三轮均值±样本标准差）：

| 批间隔 | 方案 | short E2E(ms) | medium E2E(ms) | long E2E(ms) | overall TTFT(ms) |
|---|---|---:|---:|---:|---:|
| 25ms | baseline | 5270.22±7.99 | 5771.83±9.28 | 5578.87±11.36 | 2813.83±21.00 |
| 25ms | old | 5446.92±11.76 | 5952.39±13.89 | 5706.15±13.47 | 2954.25±15.57 |
| 25ms | new | 3433.91±784.06 | 3687.24±627.49 | 5839.10±24.93 | 1706.15±442.38 |
| 50ms | baseline | 5264.35±26.59 | 5749.37±16.18 | 5541.00±18.17 | 2784.14±7.89 |
| 50ms | old | 5415.85±15.37 | 5917.00±4.70 | 5671.63±18.49 | 2924.27±18.31 |
| 50ms | new | 4208.13±493.31 | 4696.11±613.45 | 5667.87±67.15 | 2178.31±348.79 |

new相对本轮baseline（负值表示时间缩短）：

- 25ms：makespan +0.54%；short TTFT -58.14%；medium TTFT -58.13%；long TTFT +9.30%；overall E2E -24.33%。
- 50ms：makespan -0.13%；short TTFT -34.25%；medium TTFT -29.27%；long TTFT +6.93%；overall E2E -13.62%。

上述是同轮对照的描述统计；仅三轮且共享CPU、内存和互联资源未隔离，不能据此宣称统计显著性或将跨归档波动归因于新修改。本次与v6使用相同代码，较v6的变化属于独立重跑差异，不代表又一次技术优化。

### 17.3 窗口提交及计算等待

| 指标（含预热） | v5优化前 | v6首次验收 | 本次独立重跑 |
|---|---:|---:|---:|
| 正常窗口数/请求数 | 27/168 | 24/169 | 30/170 |
| 入窗至提交均值(ms) | 77.716 | 5.144 | 5.680 |
| 入窗至提交最大(ms) | 177.380 | 8.656 | 10.589 |
| 超过deadline均值(ms) | — | 0.417 | 0.849 |
| 超过deadline p95(ms) | — | 3.656 | 5.589 |
| 超过deadline最大(ms) | — | 3.656 | 5.589 |

本次正常窗口30/30由timer发布、0由engine路径发布；另有10条idle singleton提前释放，共180条new请求均恰好一次发布并记录首次schedule。窗内稳定排序、rank连续、共享deadline及发布时间、跨窗有序且正常窗口未提前释放，deadline审计无错误。入窗至提交p95为10.338ms。相对优化前v5，平均窗口等待降低92.69%。

提交后的首次schedule观测等待仍为平均927.357ms、p95 2633.487ms、最大2721.862ms（180请求、含预热与singleton）。不能把及时发布解释为GPU立即开算；TTFT保持包含这段真实排队。5ms是窗口设定，Python/OS唤醒、GIL与注册/取消锁仍可能造成到期延迟；没有采集足以分解单次尾延迟来源的时间线，也不承诺硬实时。

### 17.4 运行范围、清理与数据索引

GPU监控466次采样、744条属于实验进程的上下文均仅出现GPU6/7，监控无错误；2秒采样不能排除采样间极短上下文。2026-11-14T15:03:48.068197+00:00检查两卡均0 MiB、0%利用率、无计算进程，8000/8100/8200无监听。其他GPU业务未被停止或修改。

主动shutdown之前的服务日志无ERROR；shutdown_log_audit.json保留停服后vLLM abort模式可能出现的EngineDeadError与semaphore回收提示。运行输出、传输正确性及最终GPU/端口释放分别验收。

当前重跑不扩展长请求SLA、真实远端故障自动重试、已发partial后的KV重分配、多TP/PP或统计覆盖范围，相关限制继续见15.4/16.4。

数据：[RESULTS.md](../results/nvlink-deadline-rerun-20261114-144441/RESULTS.md)、[REPEAT_COMPARISON.md](../results/nvlink-deadline-rerun-20261114-144441/REPEAT_COMPARISON.md)、[requests.csv](../results/nvlink-deadline-rerun-20261114-144441/requests.csv)、[runs.csv](../results/nvlink-deadline-rerun-20261114-144441/runs.csv)、[long_requests.csv](../results/nvlink-deadline-rerun-20261114-144441/long_requests.csv)、[deadline_analysis.json](../results/nvlink-deadline-rerun-20261114-144441/deadline_analysis.json)、[streaming_audit.json](../results/nvlink-deadline-rerun-20261114-144441/streaming_audit.json)、[writer_timings.json](../results/nvlink-deadline-rerun-20261114-144441/writer_timings.json)、[gpu_final_state.json](../results/nvlink-deadline-rerun-20261114-144441/gpu_final_state.json)。完整原始SSE、P/D/Router日志、冻结源码、环境、负载、参考与哈希清单同目录保存。

### 17.5 本轮baseline、窗口更新前v5及整体平均TTFT对比

此处的“窗口更新前”是v5的new方案：已有5ms局部排序，但只能在schedule时提交到期窗口；不是本轮old方案。本轮old表示完全不使用局部排序，两者不能混为一组。

TTFT包含窗口等待和真实计算排队；以下仅统计正式数据，不含预热。每方案每场景15请求×3轮=45条；整体TTFT按所有请求等权，即(8×short均值+3×medium均值+4×long均值)/15，不能直接对三个类别均值取平均。表中±为三轮均值的样本标准差，不是单请求分布标准差。

| 批间隔 | 对照组 | short TTFT(ms) | medium TTFT(ms) | long TTFT(ms) | 所有请求平均TTFT(ms) | makespan(s) | overall E2E(ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| 25ms | 本轮baseline | 2645.43±19.90 | 3099.23±21.33 | 2936.56±25.15 | 2813.83±21.00 | 6.97±0.02 | 5452.85±6.52 |
| 25ms | 本轮old（无排序） | 2796.40±14.27 | 3259.42±14.42 | 3041.08±19.24 | 2954.25±15.57 | 7.01±0.01 | 5617.14±12.60 |
| 25ms | v5 new（窗口更新前） | 1114.55±342.13 | 1405.43±341.35 | 3161.84±30.85 | 1718.67±243.51 | 6.98±0.01 | 4178.87±324.67 |
| 25ms | 本轮new（窗口更新后） | 1107.48±646.97 | 1297.75±520.07 | 3209.77±30.93 | 1706.15±442.38 | 7.00±0.02 | 4125.96±538.77 |
| 50ms | 本轮baseline | 2626.17±12.49 | 3066.16±5.86 | 2888.57±5.86 | 2784.14±7.89 | 6.96±0.00 | 5435.13±22.23 |
| 50ms | 本轮old（无排序） | 2767.75±19.65 | 3225.48±6.33 | 3011.40±25.75 | 2924.27±18.31 | 7.00±0.02 | 5584.29±13.84 |
| 50ms | v5 new（窗口更新前） | 1765.50±425.03 | 1956.91±370.44 | 3100.47±21.91 | 2159.78±296.38 | 6.98±0.02 | 4726.91±353.88 |
| 50ms | 本轮new（窗口更新后） | 1726.78±460.64 | 2168.67±589.28 | 3088.61±68.08 | 2178.31±348.79 | 6.95±0.06 | 4694.99±369.16 |

本轮new相对两种对照的变化，负值表示时间缩短：

| 批间隔 | 对照 | short TTFT | medium TTFT | long TTFT | 整体TTFT | makespan | overall E2E |
|---|---|---:|---:|---:|---:|---:|---:|
| 25ms | 本轮baseline | -58.14% | -58.13% | +9.30% | -39.37% | +0.54% | -24.33% |
| 25ms | v5 new（窗口更新前） | -0.63% | -7.66% | +1.52% | -0.73% | +0.36% | -1.27% |
| 50ms | 本轮baseline | -34.25% | -29.27% | +6.93% | -21.76% | -0.13% | -13.62% |
| 50ms | v5 new（窗口更新前） | -2.19% | +10.82% | -0.38% | +0.86% | -0.44% | -0.68% |

所有请求整体平均TTFT汇总（单位ms）：

| 范围 | 每组请求数 | 本轮baseline | v5窗口更新前 | 本轮窗口更新后 | 更新后相对baseline | 更新后相对v5 |
|---|---:|---:|---:|---:|---:|---:|
| 25ms场景 | 45 | 2813.83 | 1718.67 | 1706.15 | -39.37% | -0.73% |
| 50ms场景 | 45 | 2784.14 | 2159.78 | 2178.31 | -21.76% | +0.86% |
| 两场景合并 | 90 | 2798.98 | 1939.22 | 1942.23 | -30.61% | +0.15% |

两场景合并为每方案90条正式请求的等权平均，不把三个不同方案的270条请求混成一个用于比较的平均值，也不将预热加入正式平均。两场景请求数相同，因此合并均值也等于两场景均值的算术平均；不把合并统计解释为新增实验轮次。

窗口平均等待从v5的77.716ms降至本轮5.680ms，说明到期提交滞留明显减少；但本轮整体TTFT相对v5在25ms场景仅变化−0.73%、50ms场景变化+0.86%，两场景合并为+0.15%，没有显示出明确的整体TTFT收益。提交后仍有计算排队，窗口变快不会必然按相同比例降低TTFT。

本轮与baseline是同次实验的轮换对照，v5与本轮属于不同时间的历史对比。固定参数和负载一致，但共享资源、engine接纳时序及请求分组未隔离；三轮均值与较大波动不足以将跨归档差异归因于窗口线程。保留v5同轮baseline可查看历史15.3；本表明确使用本轮baseline作当前对照。

可复核数据：本轮BEFORE_AFTER_COMPARISON.md、before_after_comparison.csv及before_after_comparison.json；所有均值重新从各自runs/*/requests.csv读取，与已审计performance_summary.json一致，未改动原始请求/运行文件或历史v5/v6归档。

### 17.6 完整方案收益与机制贡献的解释边界

本轮两个场景合计每方案90条正式请求，完整new方案的平均TTFT为1942.23ms，相对baseline的2798.98ms下降30.61%。这表示全部类别合计的平均值改善，不表示每条请求均改善：long TTFT在25ms/50ms场景仍分别比同轮baseline增加9.30%/6.93%。

| 本轮方案 | 开启的机制 | 所有请求平均TTFT(ms) | 描述性对照 |
|---|---|---:|---|
| baseline | 整请求KV传输 | 2798.98 | 参考 |
| old | 流式C8/单槽、抢占仲裁及保护，无局部排序窗口 | 2939.26 | 比baseline增加5.01% |
| new | old机制加5ms窗口排序及独立到期接纳 | 1942.23 | 比old下降33.92%，比baseline下降30.61% |

现有对照更支持窗口排序及接纳机制是本负载TTFT收益的主要来源：启用流式抢占但无排序的old并未比baseline改善平均TTFT，而开启局部排序窗口的new明显更低。但这是本轮三组的描述性观察，不能进一步声称抢占传输没有任何贡献，或给出排序/抢占各自占比。

原因有三点：baseline到old同时改变整请求/流式传输及抢占仲裁，不能把二者的差值视为纯抢占效果；old到new加入窗口收集、排序和接纳机制，不能把差值视为纯排序效果；排序影响KV就绪及chunk到达时间，与传输抢占可能存在交互作用。禁止把三组差值强行归一化成两个百分比，因为这不能表示独立贡献。

要估计局部窗口排序机制与传输抢占的贡献，需要在相同流式C8/单槽、writer、保护、模型和负载下补齐“窗口排序机制开/关 × 抢占开/关”四组消融，分别报告主效应和交互作用，再明确贡献分配口径。若要将5ms窗口收集与排序本身分开，还需要保留相同5ms窗口及独立发布、仅把窗内排序替换为FCFS的额外对照。当前实验尚无这些数据，独立贡献比例记为未测定。

另外，窗口提交延迟优化的前后比较见17.5：平均窗口等待降低92.69%，但相对v5的整体TTFT几乎持平。这项“到期发布提速”不能与完整方案相对baseline的30.61%平均TTFT改善混为同一个收益。

## 18. NVLink 单项开关实验与 baseline 对比（2026-11-15）

### 18.1 设计与验收

本次独立归档 `results/nvlink-isolated-ablation-20261115/`，不覆盖历史实验。主实验于 15:18:27–15:36:39 UTC 运行，FIFO 对照于 15:38:37–15:44:42 UTC 补测。仅物理 GPU6（P）/GPU7（D），按 UUID 绑定、P/D 为独立进程。NIXL backend 为 UCX，`UCX_TLS=cuda,tcp`、`UCX_NET_DEVICES=lo`。模型 worker 未输出 endpoint lane；NVLink 结论来自正式 workload 的硬件计数，不把配置或此前微基准当作本轮 lane 的直接日志证据。

| 方案 | PD_LOCAL_SORT | PD_PRIORITY_PREEMPTION | 传输路径 |
|---|---:|---:|---|
| baseline | 0 | 0 | 标准 NixlPushConnector 整请求传输 |
| sort_only | 1 | 0 | 自定义流式 connector，C8/单槽，eligible FIFO |
| preempt_only | 0 | 1 | 同流式设施，chunk 边界短请求优先及长请求保护 |
| stream_fifo | 0 | 0 | 同流式设施，C8/单槽，eligible FIFO |

两个单项方案仍保留流式基础设施，因此相对 stock baseline 的差值包含流式成本。补充 FIFO 是为了在相同设施上观察开关差异；该组在主实验之后顺序补测，未交错，共享 CPU/内存及时间差仍可能影响结果。

固定 Llama-3.1-8B-Instruct BF16，TP=PP=1，context32768，memory0.90，block16，max_seqs64，计算预算2048，prefix cache 关闭，VLLM_BATCH_INVARIANT=1。15请求为8×128、3×1107、4×10000输入 tokens，各64输出，temperature0，seed1500–1514。每5条一批，批起点为0/25/50 ms或0/50/100 ms；窗口5 ms，长chunk≤8 blocks，max_active_WRITE=1，writer轮询0.1 ms，长请求保护85%/3次/50 ms。workload SHA256 为 `ad9d732ba5a55de1ac9e4c682a8171df38f0513de74c1d23251c2cbc2746e94a`。

主实验18次正式270请求，补测6次正式90请求，各有等量预热。全部720条请求输出token IDs、SSE、DONE、usage与参考逐条核验通过。流式审计覆盖主实验360条/8231 chunks、补测180条/4116 chunks（含预热），offset连续、C8上限、WRITE/DONE、单槽和唯一final全部通过；30项CPU回归通过。

48次负载（正式24、预热24）逐次核验GPU6 TX/GPU7 RX的12链路计数，每边约5.817500 GB，与填充KV payload 5,817,499,648 bytes匹配，误差仅KiB取整量级。payload按32 layers×K/V2×8 KV heads×128 head dim×BF16 2 bytes/token及16-token块对齐计算。原始计数和独立重算见 [nvlink_validation.json](../results/nvlink-isolated-ablation-20261115/nvlink_validation.json)。这支持本轮模型数据实际使用NVLink；旧实验路径仍无法追溯。

### 18.2 正式数据与 baseline 差异

TTFT从客户端提交至首个非空文本SSE；E2E至响应完成；makespan为最早实际提交至最后响应结束，均不扣窗口、轮询或排队。表中为三轮运行均值±样本标准差，不是单请求分布标准差。

| 批间隔 | 方案 | makespan(s) | short TTFT(ms) | medium TTFT(ms) | long TTFT(ms) | overall E2E(ms) |
|---|---|---:|---:|---:|---:|---:|
| 25ms | baseline（整请求） | 6.91±0.03 | 2618.50±22.28 | 3064.26±33.55 | 2884.69±30.43 | 5418.10±5.93 |
| 25ms | 只开窗口排序 | 7.01±0.04 | 967.04±323.34 | 1209.17±305.70 | 3245.93±79.60 | 4033.86±305.37 |
| 25ms | 只开抢占传输 | 6.99±0.01 | 2781.80±6.93 | 3251.00±25.79 | 3030.38±13.08 | 5604.28±7.48 |
| 25ms | 流式 FIFO（补测） | 7.02±0.02 | 2862.58±25.22 | 3267.17±26.13 | 3113.79±22.38 | 5674.71±23.58 |
| 50ms | baseline（整请求） | 6.91±0.03 | 2573.39±38.38 | 3034.34±6.55 | 2856.71±3.58 | 5372.75±58.53 |
| 50ms | 只开窗口排序 | 6.98±0.02 | 1333.70±32.85 | 1567.48±17.64 | 3170.05±27.80 | 4375.07±29.31 |
| 50ms | 只开抢占传输 | 6.97±0.02 | 2737.67±11.83 | 3215.42±19.66 | 2984.90±7.82 | 5562.84±12.73 |
| 50ms | 流式 FIFO（补测） | 6.99±0.02 | 2820.73±25.22 | 3228.26±19.26 | 3073.65±24.83 | 5624.96±33.97 |

相对本轮 baseline（负值为耗时下降）：

| 批间隔 | 方案 | makespan | short TTFT | medium TTFT | long TTFT | overall E2E |
|---|---|---:|---:|---:|---:|---:|
| 25ms | 只开窗口排序 | +1.46% | -63.07% | -60.54% | +12.52% | -25.55% |
| 25ms | 只开抢占传输 | +1.17% | +6.24% | +6.09% | +5.05% | +3.44% |
| 50ms | 只开窗口排序 | +0.93% | -48.17% | -48.34% | +10.97% | -18.57% |
| 50ms | 只开抢占传输 | +0.90% | +6.38% | +5.97% | +4.49% | +3.54% |

整体平均TTFT按请求等权，即(8×short+3×medium+4×long)/15；完整类别E2E和整体TTFT见 [performance_summary.json](../results/nvlink-isolated-ablation-20261115/performance_summary.json)。

### 18.3 差距原因、日志证据与限制

**窗口排序主要减少计算排队。** 25/50 ms下short TTFT相对baseline下降63.07%/48.17%，medium下降60.54%/48.34%，long增加12.52%/10.97%。Router的P响应时间同方向变化：25 ms下short由2273.78降至427.77 ms，long由2474.91升至2680.22 ms。P响应包括排队、prefill及HTTP处理，不能当纯GPU时间。短/中请求提前进入计算队列，长请求后移；总工作量不减，最后长请求和decode决定尾部，所以makespan仍增加1.46%/0.93%。

排序日志180条请求（含预热）中，正常窗口170条的发布等待平均5.072 ms、p95 5.704 ms、最大6.027 ms；发布后首次schedule完成观测等待平均626.1 ms、p95 2586.1 ms。发布时间不代表GPU开算。25 ms前两轮正式请求为1条singleton加14条同窗，第三轮为1条singleton加4/10条两个窗；50 ms三轮均为singleton加4/10。引擎接纳时序改变可重排范围，与25 ms排序short TTFT的323.34 ms轮间标准差一致；不能承诺每轮固定比例收益。

**抢占有效触发，但只调度已就绪KV。** 正式preempt_only共40条short-bypasses-long日志，sort_only/FIFO为0；一次选择可能绕过多个long并记录多条，不代表40次DMA中断。抢占发生于chunk边界，不中断已提交DMA或P端kernel，不能消除短请求生成KV之前的计算等待。

**流式软件成本覆盖了小幅传输仲裁收益。** 正式各流式方案均2058次WRITE；sort/preempt/FIFO的submit均值为3.689/3.679/3.432 ms，提交前至完成观测均值5.302/4.867/4.915 ms。后者包含前者，不可相加，也不是纯DMA时间。C8约16 MiB；单buffer微基准约77 GB/s、每次约0.22 ms不能直接替代分散描述符、线程与单槽推进的实际耗时。本轮不足以将Python/GIL、native、描述符或轮询各自占比精确拆开。

FIFO相对baseline的short TTFT增加9.32%/9.61%；打开抢占后相对FIFO下降2.82%/2.94%，但相对baseline仍增加6.24%/6.38%。结果与“当前抢占小幅收益不足以抵消流式开销”一致；FIFO未交错且仅三轮，不将差值作为严格独立因果或显著性估计。

本轮补齐相同流式设施上的00/10/01，未重跑同时开启的11，因而仍不能估计完整2×2交互或给排序/抢占分配百分比贡献。窗口开关同时包含收集、发布与排序；要分离收集与窗内排序，仍需相同5 ms窗口的FCFS对照。第17.6节的历史贡献边界据此更新，历史结果不与本轮合并。仅覆盖固定15请求负载、TP=PP=1，共享资源未完全隔离；不推断更多负载、故障恢复或长请求SLA。

### 18.4 归档与释放

本轮未采集连续GPU进程监控，不沿用历史章节的采样数量。服务按GPU UUID绑定；源码快照在主实验后、FIFO期间捕获，runner/launcher在主实验后扩展FIFO支持，未声称启动前后哈希一致。调度、writer、Router、负载及vLLM算法文件与当前快照一致的核对见snapshot_current_consistency.json；准确快照时点见source-runtime/SNAPSHOT_NOTES.md。

2026-11-15T15:55:26 UTC最终查询GPU6/7均0 MiB、0%利用率、无计算进程，8000/8100/8200无监听，见[gpu_final_state.json](../results/nvlink-isolated-ablation-20261115/gpu_final_state.json)。

完整证据：[REPORT.md](../results/nvlink-isolated-ablation-20261115/REPORT.md)、[逐轮runs.csv](../results/nvlink-isolated-ablation-20261115/runs.csv)、[逐请求requests.csv](../results/nvlink-isolated-ablation-20261115/requests.csv)、[主实验审计](../results/nvlink-isolated-ablation-20261115/streaming_audit.json)、[FIFO审计](../results/nvlink-isolated-ablation-20261115/fifo_control/streaming_audit.json)、[窗口分析](../results/nvlink-isolated-ablation-20261115/deadline_analysis.json)。原始SSE、服务日志、环境、负载、参考、运行/分析源码与SHA256SUMS同目录保存。

## 19. C32 四组 NVLink 消融实验（2026-11-16）

本轮将长请求块上限由8 blocks增至32 blocks（512 tokens，约64 MiB KV），在GPU6/7沿用相同workload、模型、25/50ms到达场景、writer单槽和0.1ms轮询，三轮轮换比较baseline、sort_only、preempt_only及排序+抢占同时开启的new。每组每场景均预热后正式运行。24次正式运行360请求及等量预热全部通过输出校验；540条流式请求/3950个WRITE协议审计通过，最大活动WRITE为1。正式及预热48次GPU6 TX/GPU7 RX计数均与5,817,499,648 bytes KV payload匹配。完整参数、逐请求指标、原始计数和审计见[C32报告](../results/nvlink-c32-ablation-20261116/REPORT.md)。

三轮均值±样本标准差（TTFT单位ms；makespan单位s）：

| 间隔 | 方案 | Short TTFT | Medium TTFT | Long TTFT | Makespan |
|---|---|---:|---:|---:|---:|
| 25ms | baseline | 2608.8±31.9 | 3075.6±17.4 | 2904.9±15.5 | 6.929±0.019 |
| 25ms | sort_only | 725.4±5.0 | 982.6±15.5 | 3230.1±3.5 | 6.973±0.010 |
| 25ms | preempt_only | 2783.4±36.6 | 3222.3±42.9 | 3037.2±37.5 | 6.979±0.034 |
| 25ms | new（两项开启） | 1591.6±410.6 | 1840.1±298.9 | 3142.1±37.0 | 6.944±0.008 |
| 50ms | baseline | 2580.3±38.6 | 3034.2±9.1 | 2867.7±11.1 | 6.907±0.026 |
| 50ms | sort_only | 1787.8±433.7 | 1911.1±355.6 | 3092.0±30.4 | 6.921±0.019 |
| 50ms | preempt_only | 2724.9±24.0 | 3172.7±22.7 | 2978.2±22.6 | 6.945±0.025 |
| 50ms | new（两项开启） | 1257.5±15.5 | 1538.4±16.7 | 3111.3±4.5 | 6.958±0.013 |

只开抢占的short TTFT仍比baseline高5.60%–6.69%，和C8的6.24%/6.38%同一量级；将长块增至32没有消除端到端差距。只开排序在25ms场景短/中TTFT分别下降72.19%/68.05%，在50ms场景下降30.71%/37.01%，且后者轮间波动较大。两项同时开启的短TTFT比baseline下降38.99%/51.27%，但没有稳定优于sort_only。传输仲裁发生在KV ready之后，改变C值会同时改变单次不可插队时长和WRITE次数；这些总体现象不能单独归因于某项内部开销。

C8与C32跨日运行，不是严格配对实验；日期间共享资源与engine接纳时序会影响比较。本轮固定单槽，没有测试多槽，因此不能判定单槽是主要瓶颈。下一步若需定位，可固定C32交错比较1/2/4槽，再独立扫描C8/C16/C32，并采集短请求KV-ready、首WRITE及槽位释放时间。三轮描述统计不代表显著性，也不支持超出当前负载的泛化结论。

## 20. C16 四组 NVLink 消融实验（2026-11-16）

本轮使用16-block长请求chunk（256 tokens、约32 MiB KV）、单槽WRITE、0.1ms writer poll，在GPU6/7轮换比较baseline、只开排序、只开抢占、排序+抢占，25/50ms到达场景各三轮。模型与workload同第18、19节。24次正式运行360请求及等量预热全部通过输出校验；540条流式请求、6739次WRITE协议审计通过，48次负载GPU6 TX/GPU7 RX计数均与5,817,499,648 bytes填充KV payload相符。完整逐请求数据、环境和计数见[C16完整报告](../results/nvlink-c16-ablation-20261116/REPORT.md)。

三轮均值±样本标准差（TTFT单位ms；makespan单位秒）：

| 间隔 | 方案 | Short TTFT | Medium TTFT | Long TTFT | Makespan |
|---|---|---:|---:|---:|---:|
| 25ms | baseline | 2624.5±18.1 | 3049.5±23.3 | 2880.4±22.9 | 6.914±0.021 |
| 25ms | sort_only | 1641.5±790.7 | 1995.9±925.4 | 3185.0±57.1 | 6.985±0.034 |
| 25ms | preempt_only | 2745.8±11.0 | 3215.5±10.3 | 2997.5±7.5 | 6.958±0.024 |
| 25ms | 排序+抢占 | 1646.0±287.7 | 1752.9±160.0 | 3150.9±17.1 | 6.947±0.028 |
| 50ms | baseline | 2605.2±8.5 | 3028.6±7.5 | 2853.9±9.0 | 6.926±0.024 |
| 50ms | sort_only | 1557.1±417.6 | 1747.6±319.8 | 3150.6±51.5 | 6.958±0.014 |
| 50ms | preempt_only | 2733.9±11.2 | 3198.5±23.6 | 2981.5±8.7 | 6.970±0.019 |
| 50ms | 排序+抢占 | 1265.6±20.4 | 1552.0±30.5 | 3111.0±20.6 | 6.975±0.032 |

只开抢占短请求TTFT较baseline高4.62%/4.94%，与C8、C32结果接近；跨三档的流式WRITE数随chunk增大而下降，但短请求TTFT没有呈现明显随块大小改善的趋势。排序仍大幅降低短/中TTFT并推迟long TTFT；联合组结果随场景而异。C8/C16/C32分别跨日运行、且都固定单槽，不能据此断言最优chunk或单槽是性能差距主因。若要识别槽位影响，需在同一C值和交错运行条件下比较不同`max_in_flight`，并采集每条短请求KV ready到首个WRITE及完成的时间线。


## 21. C16 入窗时序诊断与 GPU2/3 补齐（2026-11-16，已完成）

原计划24/24次正式运行全部完成：正式360请求、预热360请求全部逐token/SSE/DONE/usage校验通过。前12次使用GPU5(P)/6(D)，后12次按用户指示改用GPU2(P)/3(D)，未覆盖此前输出。第一轮全部及第二轮sort_only/preempt_only在5/6；第二轮new/baseline和完整第三轮在2/3。两对都是A100-SXM4-80GB、NV12，NUMA分别为1/0，因此三轮均值是混合卡组的描述统计，同时给出同卡第三轮四组对照。GPU6训练任务和GPU7原有服务均未操作，实验结束P/D均已退出。

C16、单槽WRITE、0.1ms poll、5ms窗口及其余负载/模型参数不变；增加可关闭的单调时间戳采集、回放ID、物理GPU配置和恢复元数据，未修改排序/抢占/保护算法或共享安装vLLM。完整结果与源代码版本见[诊断报告](../results/nvlink-admission-trace-gpu5-6-20261116-150826/REPORT.md)。

同在GPU2/3的第三轮，单位ms，单轮回放均值：

|间隔|方案|Short TTFT|Medium TTFT|Long TTFT|全部请求TTFT|
|---|---|---:|---:|---:|---:|
|25ms|baseline|2631.3|3092.3|2920.4|2800.6|
|25ms|只排序|1382.8|1609.8|3267.8|1930.8|
|25ms|只抢占|2883.4|3324.6|3084.3|3025.2|
|25ms|排序+抢占|1324.6|1627.3|3223.0|1891.4|
|50ms|baseline|2650.7|3098.7|2925.3|2813.5|
|50ms|只排序|1320.3|1547.3|3240.7|1877.8|
|50ms|只抢占|2804.3|3280.9|3057.9|2967.3|
|50ms|排序+抢占|1263.0|1531.6|3130.0|1814.5|

**5ms计时起点解释。** 它从调度器admit开始，并非客户端发送或P API接收开始。第一轮sort_only/25ms中14条请求客户端发送跨度51.366ms、P API入口跨度64.141ms、引擎入队跨度55.262ms，却在1.798ms内集中出队、1.756ms内进入同一窗口。首批long输入队列等64.396ms，其中前一模型执行步剩余64.244ms。窗口于相对首条发送109.965ms才开始，deadline114.965ms，发布115.236ms。因此不是窗口被延长到50ms，而是窗口开始之前已有等待；独立deadline线程无法发布尚未入窗的请求。全正式回放35个窗口中12个跨客户端批次，deadline迟到按请求加权均值0.715ms、最大5.501ms。

**25/50ms排序/组合优劣反转的主要原因。** 六组轮次×间隔对照中，五组排序/组合窗口成员和顺序相同，组合Short均更低35.0–62.0ms；唯一不同是第一轮25ms，排序为1+14、组合为1+4+10，排序领先509.9ms，足以拉低三轮25ms排序均值。同卡第三轮两场景均为1+4+10且成员相同，组合在25/50ms分别快58.15/57.37ms。因此“25ms只排序一定更好、50ms组合才有优势”不成立；主要是输入队列集中出队造成的入窗分组与计算排队位置变化。相同发送顺序/间隔并不固定异步EngineCore接纳分组，不能简单归因为25ms“拥堵”。第二轮两组来自不同卡组，其35ms差值不单独当作抢占净收益。

**组合收益发生在哪。** 同卡第三轮50ms，只排序/组合的Short首次观察到final chunk具备完整发送条件→WRITE提交分别75.64/32.88ms，组合缩短42.76ms。final事件记录→首次观察到CUDA event完成（查询还受前序chunk/槽位/注册条件限制）分别238.30/257.99ms，组合反而长19.69ms。最终TTFT差值还含发送→首次调度等变化。因此可确认可发送后的排队改善，但不能把端到端57.37ms全计成纯仲裁收益；观察到event完成也不是CUDA真实完成时刻。

**只抢占为什么仍慢。** 同卡第三轮25/50ms只抢占比baseline的Short高252.13/153.58ms，首次P调度→观察传输完成多148.11/117.04ms；只抢占的final可发送→提交仅3.11/7.83ms，更长的记录→首次观察event完成为221.38/220.94ms。baseline原生整请求与流式分块路径在保留final块、post-forward event、单槽writer与发布/完成推进上不同，不能当作在相同路径免费附加插队。未做额外stream_fifo或CUDA真实完成时间消融，所以不把baseline差值全归给某一项内部开销。

**保护规则的直接证据。** 当前仲裁先选受保护long，再考虑short越过long；不是无条件short优先。第三轮25/50ms，只抢占分别有8/16次决策在首个可发送候选为short时选择后面的受保护long；组合分别5/5次。保护会让已可传的short继续等待，故“开启后short只会更快”的前提不成立。这些是chunk边界选择，不是中断DMA次数。

最终360条正式时间线齐全有序；540条流式请求、6739 WRITE/DONE单槽协议审计通过；48次正式/预热负载的NVLink计数均匹配每次5,817,499,648bytes填充KV数据。32项相关CPU检查通过。采集有观察开销、没有有/无采集严格配对；三轮描述统计不作为显著性或普遍性能保证。

如果要使5ms反映真正P入口的窗口，需要在输入阶段记录单调到达时间/窗口边界，主循环接纳时保留该分组，并保持主循环对Request/KV状态的所有权。当前只解决已入窗请求按时发布，尚未固定入口分组；本轮诊断没有擅自修改这一算法。

历史WRITE分母更正：C8主实验8231次覆盖360条流式请求（sort_only/preempt_only），不是540条；C16/C32的6739/3950包含new，覆盖540条。共同两方案、同为360条时C8/C16/C32为8231/4492/2630次，不可把不同方案集合的总量按同分母比较。

## 22. P HTTP 入口固定窗口实现（2026-11-17，仅代码与 CPU 验证）

按用户确认的语义，将窗口起点从 Scheduler 接纳移动至 P HTTP 入口：首条请求开启固定 5ms 区间，保留此区间内实际到达的成员，到期后按实际 prompt token 数稳定排序，将整窗作为一条 vLLM UTILITY 消息提交。EngineCore 主循环先注册完全部成员，再进入下一次调度；普通 FCFS 保留接纳顺序，计算继续受原有 token/KV 预算与 chunked prefill 约束。

删除原调度器窗口缓冲、deadline 发布线程与 singleton 提前放行，避免引擎输入排队压缩窗口和首条抢先执行。入口窗不因引擎忙而合并；预处理迟到的成员仍属于原窗，后窗等待，取消或无效请求会清理。`PD_LOCAL_SORT` 与传输抢占开关仍独立，生产入口钩子不依赖 trace 开启，D 不启用窗口，未修改共享 vLLM 安装。

本次没有进行 GPU 推理或性能实验。第 21 节及此前 C8/C16/C32 数据描述旧实现，不能直接推断新入口窗口的 TTFT。实现边界、日志字段与 CPU 验证方法见 [P 入口窗口实现说明](Ingress_Window_20261117.md)。

## 23. GPU4/5 入口固定窗口四组重测（2026-11-17，已完成）

使用第22节修复后的入口窗口代码，在物理GPU4(P)/GPU5(D)完成四方案×25/50ms×3轮，共24次正式、24次预热；360条正式与360条预热均通过逐token/SSE/DONE/usage独立校验。C16、单槽WRITE、0.1ms poll、5ms入口窗口及其余模型/负载/保护参数沿用最近一轮。两卡A100-SXM4-80GB、NV12、NUMA1，UUID绑定；GPU监控仅观察到实验上下文位于4/5，结束后两卡均已释放。其他GPU任务未操作。

三轮TTFT均值±样本标准差，单位ms；全部均值按8×short、3×medium、4×long请求数加权：

|间隔|方案|Short|Medium|Long|全部|
|---|---|---|---|---|---|
|25ms|baseline|2633.4±12.0|3067.7±16.5|2899.2±19.8|2791.1±14.9|
|25ms|只窗口排序|2837.3±20.2|3272.3±7.4|3069.2±4.2|2986.1±9.5|
|25ms|只抢占|2793.1±33.0|3259.5±36.1|3041.6±30.8|2952.6±33.0|
|25ms|排序+抢占|2780.6±17.5|3257.3±10.8|2990.3±12.2|2931.9±14.5|
|50ms|baseline|2578.3±53.7|3026.2±15.0|2856.8±14.5|2742.1±35.0|
|50ms|只窗口排序|2778.7±76.9|3219.6±23.1|3022.1±25.7|2931.8±46.3|
|50ms|只抢占|2759.8±32.2|3228.5±35.0|3008.3±23.0|2919.8±30.3|
|50ms|排序+抢占|2770.5±8.6|3219.2±13.4|2959.2±6.3|2910.5±3.6|

**新窗口按预期工作，但实际并非5+5+5。** 正式排序请求形成89个窗口，固定收集边界与稳定排序、整窗注册先于首次调度均校验通过；4个窗口跨客户端batch，源于真实P入口到达接近，而不是引擎出队时重新合窗。客户端每批发送跨度约2ms，但P入口同批跨度均值约17–22ms，超过5ms，故常见2+3+1+2等分组。所有正式排序回放的首窗均只有00-short与01-long，后面的short不能跨窗超过该long。此前调度器大窗导致的强排序收益不再存在；历史C8/C16/C32不能直接视为新实现的重复。

**25/50ms的差异以计时起点、入窗成员和计算排队变化解释。** 50ms−25ms的Short TTFT分别为−55.15ms（baseline）、−58.57ms（只排序）、−33.31ms（只抢占）、−10.09ms（组合）。客户端Short平均发送偏移后移约25ms；以各次首条发送为统一原点，对应首文本时刻变化分别−30.08/−33.74/−8.27/+14.97ms。因此组合Short的TTFT低10ms并不表示整体首文本时刻更早，不能当作50ms强化抢占的证据。

直接窗口证据：第三轮只排序50ms，05-long与06-short入口间隔4.561ms，同窗排序后06先于05；25ms两者相隔10.768ms，分属前后窗。06-short首次调度等待由1663.68降至751.13ms，TTFT由2723.76降至1704.06ms，单条变化−1019.70ms，按8条short加权贡献−127.46ms，基本解释该轮Short总体−121.40ms。第二轮两间隔窗口完全相同，Short仅低10.83ms。因此不存在普遍的“50ms一定产生更好排序”。

阶段拆分中，50ms−25ms的Short发送→首次调度分别减少36.61/58.43/35.95/5.32ms，同方案首次调度→P端观察传输完成仅变化−2.17/−4.71/−0.46/−2.78ms，未见NVLink本身因间隔变大显著提速。baseline还在完成观察→首文本减少16.36ms，此段含D输出和P观察时机，不能当作纯网络时间。第三轮baseline/50ms某short首文本早于P端观察到完成，说明发送端轮询完成日志不是DMA实际完成时刻，不能当作D计算的严格先决时间点。

**baseline本轮仍最快。** 三种流式方案Short比同间隔baseline慢约5.6%–7.8%。组合使完整发送条件就绪→WRITE等待从只排序约50ms降至4ms，具有可观察的仲裁效果，但流式路径首次调度→观察传输完成约629–692ms，高于baseline约513–516ms，抵消了这部分收益。该阶段混合计算、发布、事件观察和writer推进，未做更多消融，不能把差值全归给单项开销。多数排序/组合窗口成员不同，只在相同窗口对照下讨论仲裁净作用的范围；三轮数据不作显著性或普遍优劣承诺。

流式540条含预热请求、6745对WRITE/DONE连续覆盖与单槽协议通过；48次负载NVLink P Tx/D Rx均与每次5,817,499,648bytes填充KV数据在0.1%内匹配。完整报告包含E2E、makespan、逐轮TTFT、窗口成员、阶段、原始回放与源码版本：[GPU4/5完整报告](../results/nvlink-ingress-window-gpu4-5-20261117-015530/REPORT.md)。本轮仅为设备可配监控和入口窗口分析工具增加支持，实验期间未修改窗口/抢占算法。


## 24. GPU4/5 先50ms后25ms顺序复测（2026-11-17，已完成）

围绕05/06入窗问题，只复测sort_only。三次全新服务，每次预热50→25ms后正式50→25→50→25ms，12次正式180条、6次预热90条。C16/单槽/0.1ms poll/固定5ms入口窗不变，仅增加可选PD_API_TIMING细分观测。六组先跑50ms的Short均更快，不支持“25ms只是第一次冷启动慢”这一单一解释；不是随机化设计，不能排除所有顺序影响。

TTFT均值±回放样本标准差(ms)：

|间隔|Short|Medium|Long|全部|
|---|---:|---:|---:|---:|
|50ms|2714.12±136.51|3120.36±204.85|3007.86±24.94|2873.70±109.66|
|25ms|2862.86±11.55|3277.11±8.45|3064.00±11.90|2999.35±9.24|

前三分之二正式对照05/06在两间隔均不同窗；第三次重启两组50ms的05 JSON+逐token校验仅1.229/1.256ms，入口间隔3.314/3.166ms，同窗；随后25ms变为4.512/9.407ms，入口间隔11.978/21.930ms，分窗。CPU时间几乎等于wall时间，直接证实入口同步CPU工作会挤占事件循环并影响5ms边界；已解析头的06还可能等待ASGI调度，故协议处理与应用入口不可混同。第三次重启首组06首次调度等待706.47/1671.16ms，TTFT1651.32/2722.02ms，说明分窗改变计算排队。其他窗口也变化，不能将整组收益全算给06。

Short总差50−25为−148.75ms，发送→首次P调度贡献约−116.74ms。改用首条发送共同原点仍差−123.84ms，故不只是TTFT起点。同步函数本身CPU耗时波动的更底层原因（核迁移、频率、SMT/缓存或解释器状态）本轮尚未唯一定位，不能擅自确认为某一项。新增采集存在开销，未做有/无采集严格对照；也不能从混合阶段推出NVLink纯带宽变化。

85个正式窗口、270条流式请求3375对WRITE/DONE、18次NVLink计数与所有输出校验通过；实验上下文仅4/5且已释放。完整逐组表、软件时间线、适用边界及原始数据见[先50后25完整报告](../results/nvlink-ingress-order-50first-gpu4-5-20261117-030213/REPORT.md)。

## 25. 入口先收集后预处理与重复校验消除（2026-11-17，仅代码与CPU验证）

针对第24节发现的入口阻塞，ASGI包装器现只登记到达与固定窗口，异步等待本窗5ms收集截止，再让本窗各HTTP请求进入FastAPI/vLLM预处理；全部有效成员准备好后继续稳定排序，以一条UTILITY原子提交。这样本窗首条长请求的JSON处理不再占用本窗收集时间。删除自定义入口的body缓冲重放、重复JSON解析和逐token类型扫描，直接复用FastAPI已校验的CompletionRequest，在render/generation之前以常数时间检查单prompt/n/best_of/beam限制，保留原有fanout防死锁约束。

这是批量组织与提交，JSON/Pydantic预处理仍逐请求执行，未实现CPU任务的多核并行。EngineCore共享Request/KV状态仍由主循环串行注册并保留整窗原子性。前窗预处理、HTTP协议和router仍可影响后窗真实ASGI进入时刻，不能保证客户端5条自动同窗；进一步完全隔离需独立接收/预处理架构。收集后才预处理也有延迟代价，本次不宣称TTFT一定降低，未运行任何GPU实验。实现与CPU检查说明见[入口窗口说明](Ingress_Window_20261117.md)。

## 26. 排序流程复查与五请求同窗验证（2026-11-17，仅CPU验证）

固定边界→窗内实际token数稳定排序→FCFS保序路径复查通过；整窗提交用于保序，未增加第二个5ms窗口。自定义入口JSON去重、避免逐token类型重扫的优化保留。本次额外消除router每请求重建后端ClientSession（改为应用生命周期keep-alive连接池），并把EngineCore对排序结果再sorted一次的校验改为线性相邻检查。原生模型/词表合法性检查保留；API输入到引擎调度对象的转换不是第二次分词。

6个CPU测试模块51项通过，并用真实HTTP、生产router/gate和真实FastAPI/CompletionRequest进行30条请求功能验证，模型/EngineCore/D输出为CPU桩、CUDA设备不可见。25ms三批实际为1+4、5、5，50ms三批为5、5、5；五批入口跨度3.354–3.684ms。首次拆窗的第二条P请求头到首窗开始后5.318ms才发送，首条预处理5.708ms才开始，因此该次拆窗已在上游产生。没有GPU推理性能或真实忙引擎分窗结论。

结论是当前代码可形成五请求窗口，但固定5ms语义不保证客户端每组五条必定同窗；若在P实际到达跨过5ms仍强制成批，则属于不同的显式批次协议。完整代码审查、排序成员、原始时间线和适用边界见[流程复查报告](../results/ingress-flow-review-20261117-040748/REPORT.md)。

## 27. GPU4/5 10ms入口窗口只排序重测（2026-11-17，已完成）

生产窗口默认改为10ms并支持环境变量覆盖。在第25/26节优化后的代码上使用GPU4(P)/5(D)、NVLink、只排序关闭抢占、C16、单槽0.1ms poll，同一15请求及64输出token负载。三次独立服务重启，预热50→25ms，正式50→25→50→25ms，共12次正式180条、6次预热90条。

6次正式TTFT均值±回放均值样本标准差(ms)：

|间隔|Short|Medium|Long|全部|
|---|---:|---:|---:|---:|
|25ms|2344.41±148.30|2477.74±239.63|3141.75±9.17|2583.70±121.31|
|50ms|2315.58±52.91|2414.21±29.41|3060.73±22.96|2534.01±32.54|

平均E2E分别5197.23/5154.25ms，makespan6976.85/6942.62ms。Short 50−25为−28.83ms；平均发送偏移后移25.34ms后，按每次首条发送共同原点计的首文本均值仅差−3.48ms。因此不能将TTFT差全部解释为计算或传输提速。软件阶段中首次调度前等待−24.26ms，final frontier→P观察完成−33.81ms，另有阶段抵消；这些包含计算/发布/轮询，不能直接推导NVLink纯速率。

实际25/50ms分别10/18、6/18个客户端批次完整同窗，严格整次5+5+5为1/6、0/6；出现1个跨客户端批次窗口。最后一组25ms为5+5+5，Short2061.77ms，比配对50ms快353.28ms；06首次P调度等待715.03/1624.15ms，TTFT1700.74/2608.51ms，源于25ms中05/06同窗且06先于05，其他窗口也变化。10ms并未保证每批五条同窗，也不存在50ms必优的结论。

全部输出、55个正式窗口固定10ms/排序/原子接纳、270条流式请求3370对WRITE/DONE、18次NVLink payload计数通过；实验期源码未变，实验上下文仅4/5且两卡已释放。51项运行前CPU检查通过。本轮无baseline/只抢占/组合，且与旧5ms相比含多项代码优化，不能视作仅窗长变化的严格性能消融。完整逐组结果和原始记录见[10ms GPU4/5报告](../results/nvlink-ingress-window10-gpu4-5-20261117-080249/REPORT.md)。

## 28. Router同步JSON成本优化（2026-11-17，仅CPU验证）

第27节实验后，将router由标准JSON解析与P/D各自发送时编码，改为orjson直接从body bytes解码、prompt token数组只编码一次复用到两份后端载荷，完成编码后以data=bytes启动P/D发送。保留应用连接池、不同P/D协议参数、共享X-Request-Id和SSE/失败取消行为，不加入人为凑批屏障。异步读body原本就非全局串行；优化对象是单事件循环上的同步解析和重复编码，未实现进程隔离或多核预处理。

54项CPU检查通过。一次30条真实HTTP、真实FastAPI/CompletionRequest、生产router/gate、10ms窗的CPU桩验证，25/50ms各三批均形成5+5+5；解码wall平均0.086ms，构造两份载荷平均0.055ms。这没有运行GPU推理，不是优化前后配对性能实验，不能直接推断真实GPU负载下同窗率或TTFT；P入口仍可能受前窗API与系统调度影响。新增读取/解码/构造编码分段时间戳用于之后定位。代码与原始CPU记录见[router优化报告](../results/router-preprocess-review-20261117-081626/REPORT.md)。


## 29. 最新router、10ms窗口GPU4/5四方案重测（2026-11-17，已完成）

在第28节router优化后版本上，使用物理GPU4(P)/5(D)、NVLink、10ms固定入口窗、C16、单槽、0.1ms poll，完成baseline、只排序、只抢占、排序＋抢占四方案×25/50ms×3轮。方案每轮全新启动并轮换顺序，第二轮先50ms后25ms，每个正式场景前预热相同负载。15条请求为8 short/3 medium/4 long，每条64输出token；24次正式360条与24次预热360条均通过逐token/SSE/DONE/usage校验。

三轮TTFT均值±回放均值样本标准差，单位ms；全部按8/3/4条请求加权：

|间隔|方案|Short|Medium|Long|全部|
|---|---|---|---|---|---|
|25ms|baseline|2603.51±17.18|3055.35±7.63|2875.30±14.81|2766.35±9.37|
|25ms|只窗口排序|2103.70±27.62|2143.73±27.57|3164.09±26.01|2394.48±25.81|
|25ms|只抢占|2750.12±24.20|3216.10±23.02|2999.62±26.14|2909.85±24.46|
|25ms|排序＋抢占|2055.37±7.34|2180.76±27.11|3144.97±12.31|2371.01±10.82|
|50ms|baseline|2569.49±31.48|3010.33±24.77|2837.99±17.85|2729.26±26.39|
|50ms|只窗口排序|2047.19±12.44|2087.04±11.13|3138.48±7.88|2346.17±10.29|
|50ms|只抢占|2713.63±11.08|3168.87±12.59|2963.14±9.56|2871.22±9.83|
|50ms|排序＋抢占|2011.97±6.88|2110.99±48.82|3101.04±10.86|2322.20±10.86|

**本轮所有排序回放均形成5+5+5。** 只排序与组合各6次正式，共12次、36个入口窗口，36/36个客户端批次完整同窗，没有拆窗或跨批合窗。10ms固定边界、截止后才预处理、稳定排序、整窗注册先于调度均通过；只排序与组合的6组同轮窗口成员及顺序完全一致。实际P同批跨度最大9.482ms，仍接近10ms边界；这是本轮观察结果，不是任意负载下五条必同窗的保证，也不是router优化前后同条件配对消融。

**组合的Short和全部TTFT最低，只排序的Medium最低，baseline的Long最低。** 组合Short相对baseline在25/50ms分别降低21.05%/21.70%，全部降低14.29%/14.91%；只排序Short降低19.20%/20.33%，全部降低13.44%/14.04%。排序使Short发起→首次P调度从baseline的1905.24/1860.11ms降到1222.70/1181.15ms，约679–683ms的计算排队收益是主要来源。组合Long仍比baseline慢9.38%/9.27%，体现短请求提前后的等待取舍。

**抢占的局部效果明确，不能视为免费增加。** 组合相对只排序，Short完整发送条件就绪→WRITE由58.75/57.19ms降至5.28/4.83ms，Short最终TTFT分别再降低48.33/35.21ms；6组同轮对照组合Short均更低。只抢占Short仍比baseline慢146.61/144.14ms（5.63%/5.61%）：没有窗内计算重排，且基于NIXL的流式发布、CUDA event观察与writer路径不同于baseline原生整请求。首次P调度→P观察传输完成，只抢占为626.43/624.06ms，baseline为514.07/510.61ms；该混合阶段差约112–113ms，不能全算成DMA或单次轮询。仲裁发生在chunk边界，不中断正在执行的DMA；保留长请求保护，不能要求每次选择short。

**50ms较低不由拆窗解释。** Short的50−25ms变化分别为baseline −34.01ms、只排序 −56.51ms、只抢占 −36.48ms、组合 −43.39ms。两间隔的排序成员/顺序相同；Short平均发起偏移增加约25ms，以首条发起共同原点计算，首文本只提前8.65/31.19/11.51/18.64ms。首次P调度前等待分别减少45.13/41.54/39.91/38.36ms，包含更晚发起时已经执行的计算；其余调度后阶段净变化+11.12/−14.97/+3.42/−5.04ms。不支持将差异唯一归结为25ms拥堵、第一次冷启动或NVLink提速。

平均E2E在25/50ms：baseline 5415.65/5385.17ms，只排序4991.72/4942.75ms，只抢占5577.81/5538.93ms，组合4975.30/4925.17ms。组合makespan为6972.25/6957.35ms，baseline为6925.59/6899.13ms：降低平均TTFT并不代表缩短全部请求完成时间。

540条流式含预热请求、6732对WRITE/DONE、单槽及最终通知通过；48次负载NVLink P Tx/D Rx均匹配每次5,817,499,648bytes KV payload（0.1%容差）。运行前54项CPU检查通过；运行源码开始/结束哈希一致，实验上下文仅4/5，两卡已释放。三轮仅作描述统计，trace与API loop观测在本轮各方案一致，未进行无观测配对或更细的开销消融。历史数据不混入本轮均值。

完整逐轮表、窗口成员、E2E/makespan、时序与仲裁明细及源码快照见[最新router四方案完整报告](../results/nvlink-router-window10-fourway-gpu4-5-20261117-082408/REPORT.md)。
