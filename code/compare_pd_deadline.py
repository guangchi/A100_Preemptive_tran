#!/usr/bin/env python3
"""Compare deadline admission against frozen v5 with explicit timing scopes."""
import argparse
import json
from pathlib import Path
from analyze_pd_deadline import distribution


def compare(previous, current):
    for root in (previous, current):
        if not json.loads((root/'streaming_audit.json').read_text())['complete_formal_experiment']:
            raise ValueError('Complete independently audited experiments required')
    prior = json.loads((previous/'review_analysis.json').read_text())
    now = json.loads((current/'deadline_analysis.json').read_text())
    if not now['passed']:
        raise ValueError('Deadline audit must pass')
    old = json.loads((previous/'performance_summary.json').read_text())['summary']
    new = json.loads((current/'performance_summary.json').read_text())['summary']
    previous_waits = [r['wait_ms'] for window in prior['windows'] for r in window]
    previous_lates = [(r['released']-r['deadline'])*1000 for window in prior['windows'] for r in window]
    result = dict(previous=previous.name, current=current.name,
        previous_request_wait_ms=distribution(previous_waits),
        current_request_wait_ms=now['request_wait_ms'],
        previous_request_lateness_ms_rounded=distribution(previous_lates),
        current_request_lateness_ms=now['request_lateness_ms'],
        current_ready_queue_ms=now['ready_queue_ms'], performance={}, versus_current_baseline={})
    lines = ['# 5ms窗口独立到期提交优化对比', '',
        f'旧归档：`{previous.name}`；新归档：`{current.name}`。', '',
        '本轮运行改动仅为本地窗口deadline线程与线程安全FCFS队列；保持5ms窗口、跨窗FCFS、idle singleton策略、Router、模型、负载、C8/单槽与writer轮询参数。', '',
        '| 指标（含预热） | v5 | v6 |', '|---|---:|---:|',
        f"| 正常窗口数/请求数 | {prior['admission_windows']}/{prior['window_requests']} | {now['normal_windows']}/{now['normal_window_requests']} |",
        f"| 入窗至提交平均(ms) | {prior['wait_ms_mean']:.3f} | {now['request_wait_ms']['mean']:.3f} |",
        f"| 入窗至提交最大(ms) | {prior['wait_ms_max']:.3f} | {now['request_wait_ms']['max']:.3f} |",
        f"| 请求加权到期延迟平均(ms) | {result['previous_request_lateness_ms_rounded']['mean']:.3f} | {now['request_lateness_ms']['mean']:.3f} |",
        f"| 请求加权到期延迟p95(ms) | {result['previous_request_lateness_ms_rounded']['p95']:.3f} | {now['request_lateness_ms']['p95']:.3f} |",
        f"| 到期延迟最大(ms) | {result['previous_request_lateness_ms_rounded']['max']:.3f} | {now['request_lateness_ms']['max']:.3f} |",
        f"| timer正常释放窗口数 | 0 | {now['timer_windows']} |", '',
        'v5的lateness根据归档中毫秒精度的released/deadline重算；v6时间戳为微秒精度、日志耗时为0.001ms精度。等待与lateness按请求加权；p95采用nearest rank。窗口成员和数量可随实际engine入窗时间变化，不把两轮窗口当作逐窗配对样本。', '',
        f"v6全部{now['released_requests']}条new请求（包含singleton与预热）的提交后首次schedule观测等待平均{now['ready_queue_ms']['mean']:.3f}ms，p95 {now['ready_queue_ms']['p95']:.3f}ms，最大{now['ready_queue_ms']['max']:.3f}ms。该值包含已有P计算/调度预算等待，不能解释为窗口仍未释放；也不能声称GPU在5ms开始计算。", '',
        '| 批间隔 | 方案 | v5 makespan(s) | v6 makespan(s) | 变化 | v6 short TTFT(ms) | v6 long TTFT(ms) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for scenario in ['25ms','50ms']:
        for scheme in ['baseline','old','new']:
            key=f'{scheme}_{scenario}'
            metrics={m:(new[key][m]['mean']/old[key][m]['mean']-1)*100 for m in old[key] if m!='rounds'}
            result['performance'][key]=metrics
            lines.append(f"| {scenario} | {scheme} | {old[key]['makespan_ms']['mean']/1000:.3f} | {new[key]['makespan_ms']['mean']/1000:.3f} | {metrics['makespan_ms']:+.2f}% | {new[key]['short_ttft_ms']['mean']:.2f} | {new[key]['long_ttft_ms']['mean']:.2f} |")
    lines += ['', 'new相对本轮baseline：', '']
    for scenario in ['25ms','50ms']:
        deltas={m:(new[f'new_{scenario}'][m]['mean']/new[f'baseline_{scenario}'][m]['mean']-1)*100 for m in ['makespan_ms','short_ttft_ms','medium_ttft_ms','long_ttft_ms','overall_e2e_ms']}
        result['versus_current_baseline'][scenario]=deltas
        lines.append('- '+scenario+'：'+'；'.join(f'{m} {v:+.2f}%' for m,v in deltas.items())+'。')
    lines += ['', '本轮解决到期发布对下一次schedule的依赖，CPU回归验证摘除到期窗口、取消同步与线程回收。Python/OS定时器仍有调度误差，不提供严格硬实时保证。', '',
        '端到端性能只报告三轮均值与标准差（见RESULTS.md），不声称统计显著性；跨归档共享CPU/内存/互联条件未隔离，不把所有TTFT变化归因于线程机制。完整输出/WRITE/DONE审计见streaming_audit.json。']
    (current/'OPTIMIZATION_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    (current/'optimization_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('previous_dir',type=Path)
    parser.add_argument('current_dir',type=Path)
    args=parser.parse_args()
    print(json.dumps(compare(args.previous_dir,args.current_dir),indent=2))
