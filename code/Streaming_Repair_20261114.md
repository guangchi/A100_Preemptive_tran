# 2026-11-14 流式传输修复与验收

修复目标是使实验中的 C8、提前注册、源 KV 可见性和单 WRITE 并发生效，并通过逐 token 输出校验。历史结果 `results/nvlink-local-sort-v2-25_50ms-20261114-041044` 保留，不能用修复后的实现为历史数据补充因果解释。

## 已定位的原因与修复

| 原因 | 修复 |
|---|---|
| Connector scheduler 及父类均没有 `self.requests`，活动 prefill 被跳过 | 在首次分配保存 Request；从每步 SchedulerOutput 完整表/增量表持续更新 block table |
| 只保存首次分配的表，后续 blocks 丢失；游标会超过实际已发送数据 | 每步累加新块，使用调度输出中的 computed token 快照；按真实块长度推进游标并检查切片长度 |
| Router 等 P HTTP 响应后才提交 D，无法在 prefill 期间注册目标内存 | P/D 固定 engine ID，共享 X-Request-Id，并发提交请求；所有对照方案使用相同路由 |
| 每个 partial WRITE 拿完整 D block 列表，缺少目标 offset | metadata 带 request_id、offset、block IDs、总块数、final；目标表按 offset 切片 |
| registration 先到时被消费并丢失，随后 chunk 无法匹配 | 按去除随机后缀的 base request ID 持久保留 registration，到终态才清理 |
| forward 前提交 WRITE，源 KV 可能尚未写完 | post-forward `wait_for_save` 在当前 CUDA stream 记录 event，writer 只选择 event 已完成的块 |
| 每块通知都被当作请求完成，或 final 因异步握手越过 partial | 非 final 用独立 transfer ID 和带标记通知；D 忽略 partial；同请求按已完成 offset 串行提交，final 是唯一请求完成通知 |
| 只读了 max-in-flight 参数；deferred handshake 路径会绕过队列 | 握手完成前 chunk 留在优先队列；所有实际 WRITE 经同一 handle 容量检查，当前 TP=PP=1 一块对应一个 WRITE |
| 每步只发一块 C8，尾块膨胀成数百 blocks | 每个就绪 frontier 和终态尾部都完整拆分；long 每块最多 8 blocks |
| 保护进度按提交计数，等待保护只在恢复时更新 | 按实际 WRITE DONE 累加完成块；候选选择时刷新等待时间；每条 long 每次 bypass 只计一次 |

`start_load_kv` 只接收 metadata；`wait_for_save` 将 event 绑定到 chunk 并送入 writer inbox。队列、目标注册、下一 offset、长请求保护由 writer 管理；主线程仅按父类锁轮询 transfer handle，并经 inbox 回报完成。

## 已完成的修复验收

CPU 回归测试：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=code \
/data01/home/zjj/miniconda3/envs/cc/bin/python -m unittest discover -s code -p 'test_pd_streaming*.py'
```

10/10 通过，覆盖 offset 切片边界、partial 通知过滤、C8 尾块、单槽容量，以及实际 scheduler/worker 的增量 block table、registration 先到和 final 越序回归。

GPU 并发 smoke：`results/smoke-protocol-v2-20261114/`。仅使用物理 GPU6/7、按 UUID 绑定，Llama-3.1-8B-Instruct BF16，TP=PP=1，16-token block，计算预算2048，prefix cache 关闭。每场景15请求（8×128、3×1107、4×10000），每条输出64 tokens，temperature=0，seed1500–1514，批起点分别为[0,25,50]ms和[0,50,100]ms。

- 25ms：15/15 请求通过，makespan 7960.161ms。
- 50ms：15/15 请求通过，makespan 8814.996ms。
- 30 条输出均与原有参考逐 token ID 完全一致，包含 DONE、usage、输入/输出数量和 finish_reason 检查。
- 686 个实际 WRITE/DONE 轨迹审计通过：每请求 offset 连续且无重复、唯一 final、long 每块≤8、最大活动 WRITE=1。

以上是正确性 smoke，不能替代预热后多轮性能对比。早期失败记录 `results/nvlink-streamfix-c8-20261114/` 和 `results/smoke-registration-order-20261114.csv` 保留用于故障追溯，不并入性能均值。

## 适用范围

当前实现显式限制为本实验的 TP=PP=1、单 device KV cache group。prefix cache 关闭。源请求发生 GPU 内存抢占并重新分配且此前已发 partial 时会报错，需要重启对应远端请求；本负载没有该场景。尚未验收取消、传输失败后的远端重试，以及多 TP/PP、host buffer、混合 KV group。逐 WRITE 活跃数与块轨迹可以证明软件协议生效，不能单独证明 UCX 每次选择了 NVLink，或证明理论链路带宽。1ms writer 轮询及每块由主线程完成回报存在 CPU 调度开销，需由正式实验实测。

## 正式实验最终验收

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


## 后续完成推进优化

本文件的v3性能表为历史修复验收，后续已解决主要逐chunk推进延迟。v4完成18/18正式运行、270/270正式及270/270预热逐token一致、8239 WRITE/DONE审计通过。new相对本轮baseline的makespan增加缩小到0.64%/0.70%，short/medium TTFT改善约35%–48%，long TTFT增加约9%。设计及完整结果见 [Writer_Progress_Optimization_20261114.md](Writer_Progress_Optimization_20261114.md)。
