#!/usr/bin/env python3
"""Publish the completed local experiment into the project documentation."""
import argparse,json,re,datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--result-dir',required=True);a=p.parse_args()
root=Path(a.result_dir).resolve();code=Path(__file__).resolve().parent
result=json.loads((root/'audit.json').read_text());s=result['summary']
if not result['complete']:raise SystemExit('Refusing to mark an incomplete/invalid experiment complete')
relative='../results/'+root.name
heading='## 12. 2026-11-14 v2 正式实验执行记录'
rows=[heading,'',f'状态：**已完成**。18/18 次正式运行、270/270 条正式请求通过精确输出 token IDs 校验；18 次预热共270条请求也全部通过。结果目录：`{root}`。','',
      f'完整数据和分析：[REPORT.md]({relative}/REPORT.md)、[audit.json]({relative}/audit.json)、[逐请求数据]({relative}/requests.csv)、[逐轮数据]({relative}/runs.csv)、[逐条long]({relative}/long_requests.csv)。','',
      '**GPU 范围**：仅物理 GPU6（P）和 GPU7（D），服务分别绑定对应 UUID，其他实验子进程继承 `CUDA_VISIBLE_DEVICES=6,7`。启动前、轮次间清理和最终释放均检查两张卡；NVML PID→GPU UUID 采样未发现本实验进程访问其他卡。GPU0–5 的其他业务未被停止或修改。','',
      '**本轮参数**：Llama-3.1-8B-Instruct BF16；TP=1；context32768；gpu_memory_utilization=0.90；block16；max_num_seqs64；计算预算2048（显式传参）；prefix cache关闭；VLLM_BATCH_INVARIANT=1；UCX_TLS=cuda_copy,cuda_ipc,tcp；UCX_NET_DEVICES=lo。固定15请求：8 short×128、3 medium×1107、4 long×10000，合计44345输入tokens，每条max_tokens64、temperature0、seed1500–1514。new窗口5ms；C8、单槽和85%/3次/50ms保护仅作为已配置参数记录，不能假定全部生效（见下文）。','',
      '**设计**：三方案×两场景×三轮，三批起点分别为[0,25,50]ms、[0,50,100]ms；每批5条并发；方案/场景顺序轮换，每个正式运行前预热一次。','',
      '**采集修正**：TTFT从客户端实际提交计时，到首次解析出非空文本SSE；E2E到流完整结束；makespan从首条实际提交到最后一条结束。保存真实输出token IDs、完整文本和SSE时间戳，核验DONE、finish_reason、usage、输入/输出数量以及逐请求参考一致性。旧的文本哈希和SSE事件计数已不再当作token ID和token数量。','',
      '| 批间隔 | 方案 | makespan(s) | short TTFT(ms) | medium TTFT(ms) | long TTFT(ms) | overall E2E(ms) |','|---|---|---:|---:|---:|---:|---:|']
for scenario in ['25ms','50ms']:
 for scheme in ['baseline','old','new']:
  data=s[scheme][scenario]
  cells=[]
  for key,div in [('makespan_ms',1000),('short_ttft_ms',1),('medium_ttft_ms',1),('long_ttft_ms',1),('overall_e2e_ms',1)]:
   d=data[key];cells.append(f"{d['mean']/div:.2f}±{d['std']/div:.2f}")
  rows.append('| '+scenario+' | '+scheme+' | '+' | '.join(cells)+' |')
rows.extend(['','表中为三轮均值±样本标准差。', '', 'new 的 short/medium TTFT 重复间波动明显，特别是25ms场景；应结合原始逐轮数据和窗口日志解读三轮均值。这里没有通过同窗口FCFS消融单独隔离收集等待与长度排序的贡献。'])
for scenario in ['25ms','50ms']:
 changes=[]
 for key,label in [('short_ttft_ms','short TTFT'),('medium_ttft_ms','medium TTFT'),('long_ttft_ms','long TTFT'),('makespan_ms','makespan'),('overall_e2e_ms','overall E2E')]:
  changes.append(f"{label} {(s['new'][scenario][key]['mean']/s['baseline'][scenario][key]['mean']-1)*100:+.2f}%")
 rows.extend(['',f"{scenario}场景，new相对baseline："+'；'.join(changes)+'。'])
rows.extend(['','**解释边界及下一步**：',''])
for line in result['implementation_limits']: rows.append('- '+line)
waits=[v['wait_ms_max'] for v in result['local_sort_observations_including_warmup'].values()]
rows.extend(['',f'P局部排序日志的实际最大等待（含预热）为{max(waits):.2f}ms；日志内窗口均按输入长度非降序排列。',
             '', '**后续任务**：先修复活动请求追踪、D提前注册、按offset增量WRITE、最终完成通知/源KV可见性及lease、真实C8分块和并发限制，并补齐逐chunk执行轨迹，再重新验收完整流式算法。当前结果可用于描述当前实现以及P局部排序的表现，不能作为完整三项优化均已实现的性能结论。',
             '', '实验环境允许其他GPU业务继续运行，因此没有隔离共享CPU、内存和互联资源；三轮重复和轮换不能替代更大样本或跨负载验证。',
             '', f'复现归档见结果目录的 `environment.json`、`source-runtime/`、`source-vllm/`、`source-analysis/`、`SHA256SUMS`、`gpu_telemetry.jsonl`、`gpu_contexts.jsonl`。'])
report=code/'Thesis_Technical_Report.md';text=report.read_text()
text=text.split('\n\n'+heading)[0].split('\n'+heading)[0]
old='代码已修复；新一轮采用 5 ms 局部排序窗口、每条请求 `max_tokens=64`，并以 `[0,25,50]` ms 和 `[0,50,100]` ms 作为三批起始时间。启动前必须确认物理 GPU 6/7 可见且空闲。当前 Codex 命令沙箱的 `/proc/driver/nvidia/gpus` 能看到 8 张卡（含 GPU6/GPU7 对应 PCI 地址 `c9:00.0`/`cf:00.0`），但沙箱没有挂载 `/dev/nvidia*` 设备节点，导致 `nvidia-smi` 返回错误码 9；因此这里无法检查占用状态，也不能安全启动 GPU 实验。'
new='2026-11-14 已恢复 GPU 访问，并完成 25/50ms、64输出token、5ms窗口的三轮对比；详细数据和实现审计见第12节。仅使用物理 GPU6/7。下列 v1→v2 修复记录保留为历史，不代表完整流式/C8算法已经验收。'
text=text.replace(old,new)
text=text.replace('### 11.3 下一步：运行 25/50 ms 正式实验','### 11.3 25/50 ms 实验复现命令（已执行）')
text=text.replace('采用 max_tokens=64、25/50 ms 间隔的新正式实验尚未运行。','采用 max_tokens=64、25/50 ms 间隔的新一轮测量已完成，见第12节；完整流式算法仍需修复验收。')
text=text.replace('### 11.4 实验完成后的分析重点','### 11.4 历史分析计划（实际结论以第12节为准）')
text=text.replace('### 3.1 stream_by_batch — 按批次提前发送\n',
                  '### 3.1 stream_by_batch — 按批次提前发送\n\n> 此节描述设计目标和代码框架。本次审计发现活动请求追踪和D提前注册路径不完整，不能视为当前已生效功能，详见第12节。\n')
text=text.replace('5. **正确性通过**：270/270 请求成功，输出哈希多样，KV transfer 未造成数据损坏。',
                  '5. **历史成功计数**：270/270 HTTP请求成功；输出哈希多样不能证明与参考输出一致，因此该旧记录不构成KV正确性验收。本次v2采用精确token IDs逐条校验，见第12节。')
begin=text.index('## 10. 结论\n')
end=text.index('## 11. 当前状态与待续工作\n')
conclusion=('## 10. 结论\n\n2026-11-14已完成当前v2实现的三轮测量，正式270条和预热270条请求均精确匹配A100参考输出。新方案的P局部排序日志及延迟表现见第12节；该收益同时包含收集窗口和排序行为，尚未通过同窗口FCFS消融单独分离。\n\n实现审计发现提前发送、有效C8分块和单WRITE限制尚未完整生效，现有结果不能支持“完整三项优化已验证”或“old无收益是因为NVLink传输太快”的因果结论。后续应先修复并验收完整传输路径，再做模块消融和性能归因。\n\n---\n\n')
text=text[:begin]+conclusion+text[end:]
banner='> **最新状态（2026-11-14）**：v2 三轮测量已完成，270/270正式请求精确输出一致。第12节为本次执行记录；此前有关完整流式传输、C8和单槽已生效的表述需按第12节实现审计修正。\n\n'
text=text.replace('## 目录\n',banner+'## 目录\n',1)
report.write_text(text+'\n\n'+'\n'.join(rows)+'\n')
plan=code/'Experiment_Plan_NVLink.md';text=plan.read_text()
text=text.replace('714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14','ad9d732ba5a55de1ac9e4c682a8171df38f0513de74c1d23251c2cbc2746e94a')
notice=f'> **2026-11-14 执行更新**：已在物理GPU6/7完成25/50ms、5ms窗口、64输出token的三轮测量，270/270正式请求和270/270预热请求均精确匹配A100参考。见[本轮报告]({relative}/REPORT.md)及技术报告第12节。下文为方案设计，CUDA_IPC独立plugin设想已由第12节UCX配置取代；提前发送、C8及单槽尚未形成有效完整路径，不能把计划或配置值当成已验收实现。当前workload因输出长度改为64而更换SHA256，原Plan.txt哈希仅适用于旧文件。\n\n'
text=text.replace('修订日期：2026-11-13。',notice+'修订日期：2026-11-13。',1)
plan.write_text(text)
changes=code/'CHANGES_NVLink.md';text=changes.read_text()
notice=f'> **当前执行状态（2026-11-14）**：本文是历史变更摘要。其中250/500ms、20ms窗口及独立CUDA_IPC backend等内容已过时。当前参数为25/50ms批间隔、5ms窗口、64输出token、UCX cuda_ipc配置；三轮测量已完成。实际结果及实现限制见[本轮报告]({relative}/REPORT.md)和技术报告第12节。\n\n'
text=text.replace('**生成日期**',notice+'**生成日期**',1);changes.write_text(text)
print('Updated',report,plan,changes,sep='\n')
