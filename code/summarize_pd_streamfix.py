#!/usr/bin/env python3
"""Summarize validated, fixed-protocol experiment without changing old data."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(root):
    audit = json.loads((root/'streaming_audit.json').read_text())
    if not audit['complete_formal_experiment']:
        raise ValueError('Independent protocol/output audit must pass before publishing results')
    groups = defaultdict(list)
    checks = []
    requests = []
    for path in sorted((root/'runs').glob('*/requests.csv')):
        scheme, scenario, run = path.parent.name.split('_')
        with path.open(newline='') as file:
            rows = list(csv.DictReader(file))
        valid = len(rows) == 15 and all(r['success']=='True' and r['reference_match']=='True' for r in rows)
        if not valid:
            raise ValueError(f'Invalid run: {path}')
        stats = {'run':path.parent.name, 'scheme':scheme, 'scenario':scenario,
                 'makespan_ms':float(rows[0]['makespan_ms'])}
        for cat in ['short','medium','long','overall']:
            selected = rows if cat == 'overall' else [r for r in rows if r['category']==cat]
            for metric in ['ttft_ms','e2e_ms']:
                stats[f'{cat}_{metric}'] = statistics.mean(float(r[metric]) for r in selected)
        checks.append(stats)
        groups[(scheme,scenario)].append(stats)
        requests.extend(dict(scheme=scheme,scenario=scenario,round=run,**r) for r in rows)
    if len(checks) != 18 or len(groups) != 6 or any(len(runs) != 3 for runs in groups.values()):
        raise ValueError('Expected six scheme/scenario groups with three valid runs each')
    summary = {}
    for (scheme,scenario), runs in groups.items():
        metrics = {k:{'mean':statistics.mean(r[k] for r in runs),
                      'std':statistics.stdev(r[k] for r in runs) if len(runs)>1 else 0}
                   for k in runs[0] if k not in ['run','scheme','scenario']}
        summary[f'{scheme}_{scenario}'] = dict(rounds=len(runs), **metrics)
    result = dict(formal_runs=len(checks),formal_requests=len(requests),summary=summary)
    (root/'performance_summary.json').write_text(json.dumps(result,indent=2))
    for name,rows in [('runs.csv',checks),('requests.csv',requests)]:
        with (root/name).open('w', newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 修复后 P/D 正式实验结果','',f'正式运行：{len(checks)}/18；正式请求：{len(requests)}/270。',
           '', '所有请求均保存实际 token IDs，并与参考逐 token 匹配。每个场景三轮均值±样本标准差。',
           '', '| 批间隔 | 方案 | makespan(s) | short TTFT(ms) | medium TTFT(ms) | long TTFT(ms) | overall E2E(ms) |',
           '|---|---|---:|---:|---:|---:|---:|']
    def fmt(s,key,scale=1):
        v=s[key];return f"{v['mean']/scale:.2f}±{v['std']/scale:.2f}"
    for scenario in ['25ms','50ms']:
        for scheme in ['baseline','old','new']:
            if f'{scheme}_{scenario}' not in summary:
                continue
            s=summary[f'{scheme}_{scenario}']
            lines.append('| '+ ' | '.join([scenario,scheme,fmt(s,'makespan_ms',1000),fmt(s,'short_ttft_ms'),fmt(s,'medium_ttft_ms'),fmt(s,'long_ttft_ms'),fmt(s,'overall_e2e_ms')])+' |')
    slower = all(summary[f'{scheme}_{scenario}']['makespan_ms']['mean'] > summary[f'baseline_{scenario}']['makespan_ms']['mean']
                 for scheme in ['old', 'new'] for scenario in ['25ms', '50ms'])
    if slower:
        lines += ['', '本轮 old/new 的 makespan 均高于 baseline，当前流式实现没有取得整体完成时间收益。']
    lines += ['', '相对本轮 baseline 的 makespan 变化（正值表示变慢）：', '']
    for scenario in ['25ms', '50ms']:
        base = summary[f'baseline_{scenario}']['makespan_ms']['mean']
        changes = [f"{scheme} {(summary[f'{scheme}_{scenario}']['makespan_ms']['mean']/base-1)*100:+.2f}%" for scheme in ['old', 'new']]
        lines.append(f"- {scenario}：{'；'.join(changes)}。")
    lines += ['', '参数：Llama-3.1-8B-Instruct BF16；TP=PP=1；context32768；block16；计算预算2048；max_num_seqs64；memory0.90；prefix cache关闭；VLLM_BATCH_INVARIANT=1。固定15请求，8×128、3×1107、4×10000输入，每条64输出，temperature0、seed1500–1514。new排序窗口5ms；old/new长请求C8、单槽，长请求保护85%/3次/50ms。',
              '', 'baseline/old/new 均使用并发 P/D Router，因此与历史串行 Router 归档不能直接合并。old/new有逐 WRITE、DONE 和 D final 通知记录；验收见 streaming_audit.json。',
              '', '只绑定物理 GPU6/7 的 UUID。gpu_telemetry.jsonl 记录两卡上下文、显存与利用率；其他 GPU 业务继续运行，共享 CPU/内存/互联资源未隔离。',
              '', 'C8 单槽实现存在逐块提交与完成处理的 CPU/轮询开销；本次没有单独消融这些开销、local-sort收集等待、排序和传输仲裁，不能据结果分离各模块贡献。UCX允许cuda_ipc不等于逐 WRITE 实测NVLink带宽。']
    audit_path = root / 'streaming_audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if not audit['complete_formal_experiment']:
            raise ValueError('Formal protocol audit is incomplete')
        lines += ['', f"协议审计通过：18次正式共270条请求及18次预热共270条请求全部输出一致；old/new（含预热）{audit['trace_requests']}条流式请求、{audit['trace_chunks']}个 WRITE/DONE 无覆盖或顺序错误，long每块≤8 blocks、最大活动 WRITE={audit['max_active']}，每请求唯一final通知。CPU回归结果见 cpu_regressions.log。",
                  '', 'GPU6/7 最终释放证据见 gpu_final_state.json，监控汇总见 gpu_monitor_summary.json；运行时代码与实验快照一致性见 runtime_source_consistency.json。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('result_dir',type=Path);args=p.parse_args()
    result=summarize(args.result_dir)
    print(json.dumps(result,indent=2))
