# 2026-11-14 writer 完成推进优化

上一轮已修复协议并通过逐token验收，但C8单槽实现的makespan比baseline慢约29%–32%。本轮保持模型、负载、并发Router、C8、单槽及保护参数不变，优化完成推进路径。

## 定位与修改

- 原partial WRITE仅在模型主线程的get_transfer_results中检查完成；writer的下一块依赖主线程回报。块数增加时，模型步/主线程调度延迟进入每块的关键路径。
- writer现在在父类sending锁下独立轮询所有WRITE handles。它调用父类释放和失败处理逻辑，再把完成送入原有inbox；主线程不再轮询WRITE handle，只消费writer发布的结果，防止重复释放及提交日志前被回收的竞争。
- final handle完成后writer释放其传输槽位，把final ID保存在完成集合；主线程只消费一次该集合，沿父类流程报告请求完成并释放KV lease。不会从writer直接释放请求KV。
- 活跃等待改为响应wake事件，回退轮询默认0.1ms；此前等待的是stop事件，wake无法中断1ms等待。
- 新增submit_ms、observed_ms、queue_ms日志。observed_ms包含Python设置、传输、轮询及完成回报延迟，不等同纯DMA时间；queue_ms也包含源KV就绪、注册及前序chunk等待。

CPU回归13/13通过，覆盖无模型步推进、final恰好一次交接、失败/进行中handle保留、partial不释放请求、无完成时不得自唤醒空转。探测结果目录：`results/nvlink-writer-progress-probe-20261114/`。单轮探测用于定位；正式性能结论见下文。

## 探测结果

仅partial独立推进版本的25/50ms单轮各15请求全部逐token一致，makespan为6962.291/7007.320ms；含预热60请求、1372 WRITE/DONE审计通过。partial observed_ms均值3.431ms，final均值77.251ms、p95 186.515ms，因此进一步将final handle完成推进移入writer，保留主线程的请求完成交接。该探测不是三轮正式性能结论。

最终版本正式归档：`results/nvlink-writer-progress-v4-20261114/`；已保持原参数完成三方案×两场景×三轮及每场景预热。

正式前两次服务启动期间主动停止，未发正式请求；记录保留在 `nvlink-writer-progress-v4-preflight-20261114` 和 `nvlink-writer-progress-v4-preflight2-20261114`，不纳入结果。实验启动器已补充中断时停止launcher子进程的清理。最终正式期间运行时代码冻结并核对哈希。

## 正式优化实验完成

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
