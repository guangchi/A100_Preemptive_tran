#!/usr/bin/env python3
"""Compare archived rounds; do not infer isolated module causality."""
import argparse
import json
from pathlib import Path


def compare(previous, current):
    for root in [previous, current]:
        audit = json.loads((root / 'streaming_audit.json').read_text())
        if not audit['complete_formal_experiment']:
            raise ValueError(f'Incomplete experiment: {root}')
    old = json.loads((previous / 'performance_summary.json').read_text())['summary']
    new = json.loads((current / 'performance_summary.json').read_text())['summary']
    environment = json.loads((current / 'environment.json').read_text())
    reviewed = bool(environment.get('review_fixes'))
    title = '边界复核修复前后对比' if reviewed else 'writer 完成推进优化对比'
    description = ('本轮修复Router SSE转发与异常清理、局部排序窗口顺序与取消、长度分类边界及partial失败标记，并强化数据独立审计；本轮未单独消融这些修改。'
                   if reviewed else '新版本把WRITE完成推进移到writer，响应wake并使用0.1ms回退轮询；本轮未单独消融这些修改。')
    lines = ['# ' + title, '',
             '旧归档：`' + previous.name + '`；新归档：`' + current.name + '`。', '',
             '两次归档均为三方案×两场景×三轮，模型、负载、并发P/D路由方式、C8、单槽和保护参数相同。' + description, '',
             '| 批间隔 | 方案 | 旧 makespan(s) | 新 makespan(s) | 变化 | 新 short TTFT(ms) | 新 long TTFT(ms) |',
             '|---|---|---:|---:|---:|---:|---:|']
    changes = {}
    for scenario in ['25ms', '50ms']:
        for scheme in ['baseline', 'old', 'new']:
            key = scheme + '_' + scenario
            prior = old[key]['makespan_ms']['mean']
            now = new[key]['makespan_ms']['mean']
            delta = (now / prior - 1) * 100
            changes[key] = dict(previous_makespan_ms=prior, current_makespan_ms=now,
                                makespan_change_percent=delta)
            lines.append(f"| {scenario} | {scheme} | {prior/1000:.3f} | {now/1000:.3f} | {delta:+.2f}% | "
                         f"{new[key]['short_ttft_ms']['mean']:.2f} | {new[key]['long_ttft_ms']['mean']:.2f} |")
    lines += ['', '变化为三轮均值之比，负值表示时间缩短。详细均值±样本标准差见 [RESULTS.md](RESULTS.md)。', '',
              '本轮new相对本轮baseline：', '']
    for scenario in ['25ms', '50ms']:
        deltas = {metric: (new[f'new_{scenario}'][metric]['mean'] / new[f'baseline_{scenario}'][metric]['mean'] - 1) * 100
                  for metric in ['makespan_ms', 'short_ttft_ms', 'medium_ttft_ms', 'long_ttft_ms', 'overall_e2e_ms']}
        changes[f'new_vs_baseline_{scenario}'] = deltas
        labels = {'makespan_ms': 'makespan', 'short_ttft_ms': 'short TTFT', 'medium_ttft_ms': 'medium TTFT',
                  'long_ttft_ms': 'long TTFT', 'overall_e2e_ms': 'overall E2E'}
        lines.append('- ' + scenario + '：' + '；'.join(f'{labels[k]} {v:+.2f}%' for k, v in deltas.items()) + '。')
    lines += ['',
              '两次实验在不同时间执行，共享CPU、内存和互联资源未隔离；不能把跨归档差异全部归因于某个模块。软件observed_ms包含设置、传输、轮询和回报延迟，不等同NVLink带宽。', '',
              '正确性与C8/单槽证据见 streaming_audit.json，阶段耗时见 writer_timings.json，GPU6/7释放状态见 gpu_final_state.json。']
    (current / 'OPTIMIZATION_COMPARISON.md').write_text('\n'.join(lines) + '\n')
    (current / 'optimization_comparison.json').write_text(json.dumps(changes, indent=2) + '\n')
    return changes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('previous_dir', type=Path)
    parser.add_argument('current_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.previous_dir, args.current_dir), indent=2))
