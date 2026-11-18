#!/usr/bin/env python3
"""Publish the NVLink ablation only after independent output/protocol audits."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import statistics

from analyze_pd_deadline import distribution

SCHEMES = ['baseline', 'sort_only', 'preempt_only', 'stream_fifo']
LABELS = {'baseline': 'baseline', 'sort_only': '只开窗口排序',
          'preempt_only': '只开抢占传输', 'stream_fifo': '流式FIFO对照'}


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path):
    return json.loads(path.read_text())


def scheme_log(root, scheme, rep):
    return root / f'logs_round{rep}_{scheme}'


def formal_trace(log_dir, scheme, start, end, replay_index):
    """Select requests by offset-zero schedule time, then all their events.

    Log timestamps have one-second precision. Only the initial SCHEDULED
    events identify the run; selecting every line by time would mix in late
    completion lines from the preceding warmup. All selected WRITE/DONE events
    are subsequently associated by unique internal request ID.
    """
    text = (log_dir / f'p_{scheme}.log').read_text(errors='replace')
    start_second = start.replace(microsecond=0)
    ordered_ids = []
    all_ordered_ids = []
    for line in text.splitlines():
        scheduled = re.search(r'PUSH_CHUNK_SCHEDULED req=(\S+) offset=0 ', line)
        timestamp = re.search(r'(11-15 \d\d:\d\d:\d\d)', line)
        if scheduled and timestamp:
            if scheduled[1] in all_ordered_ids:
                raise ValueError(f'Duplicate offset-zero schedule: {scheduled[1]}')
            all_ordered_ids.append(scheduled[1])
            when = datetime.strptime('2026-' + timestamp[1], '%Y-%m-%d %H:%M:%S')
            if start_second <= when <= end:
                if scheduled[1] not in ordered_ids:
                    ordered_ids.append(scheduled[1])
    if scheme != 'baseline' and len(ordered_ids) < 15:
        raise ValueError(f'Expected 15 formal trace IDs in {log_dir}: {len(ordered_ids)}')
    # Workload replays never overlap: the runner waits for all 15 responses
    # before the next replay. Its first few requests can start in the same
    # wall-clock second as the preceding replay ends. Keep the first 15
    # offset-zero schedules, then associate subsequent events by ID.
    ids = set(ordered_ids[:15])
    if scheme != 'baseline':
        expected_ids = all_ordered_ids[replay_index * 15:(replay_index + 1) * 15]
        if ids != set(expected_ids) or len(all_ordered_ids) != 60:
            raise ValueError(f'Formal IDs disagree with sequential 15-request replays: {log_dir}')
    selected = []
    for line in text.splitlines():
        if 'Preemption: short req ' in line:
            m = re.search(r'Preemption: short req (\S+) bypasses long req (\S+)', line)
            if m and m[1] in ids and m[2] in ids:
                selected.append(line)
        elif (m := re.search(r'\breq=(\S+)', line)) and m[1] in ids:
            selected.append(line)
    content = '\n'.join(selected)
    return dict(request_ids=sorted(ids), end_second_next_replay_ids_excluded=len(ordered_ids[15:]),
                bypass_events=content.count('Preemption: short req '),
                submit_ms=[float(v) for v in re.findall(r'submit_ms=([\d.]+)', content)],
                observed_ms=[float(v) for v in re.findall(r'observed_ms=([\d.]+)', content)],
                queue_ms=[float(v) for v in re.findall(r'(?<!ready_)queue_ms=([\d.]+)', content)])


def analyze(root):
    control = root / 'fifo_control'
    for source in (root, control):
        audit = load_json(source / 'streaming_audit.json')
        if not audit['complete_formal_experiment']:
            raise ValueError(f'Output/protocol audit did not pass: {source}')

    workload = [json.loads(v) for v in (root / 'workload.jsonl').read_text().splitlines()]
    # Llama-3.1-8B: 32 layers, K and V, 8 KV heads, 128 head dim, BF16.
    bytes_per_token = 32 * 2 * 8 * 128 * 2
    expected_bytes = sum(((len(r['prompt_token_ids']) + 15) // 16) * 16 * bytes_per_token
                         for r in workload)
    groups = defaultdict(list)
    rows_all, run_rows, nvlink_rows, trace_rows, long_rows = [], [], [], [], []
    trace_groups = defaultdict(lambda: dict(submit_ms=[], observed_ms=[], queue_ms=[], bypass_events=0))
    for scheme in SCHEMES:
        source = control if scheme == 'stream_fifo' else root
        for path in sorted((source / 'runs').glob(f'{scheme}_*/requests.csv')):
            _, scenario, rep = path.parent.name.rsplit('_', 2)
            rep = int(rep.removeprefix('run'))
            rows = list(csv.DictReader(path.open()))
            if len(rows) != 15 or not all(r['success'] == r['reference_match'] == 'True' for r in rows):
                raise ValueError(f'Invalid run: {path}')
            info = load_json(path.with_suffix('.run.json'))
            start = datetime.fromisoformat(info['started_utc'])
            end = datetime.fromisoformat(info['ended_utc'])
            log_dir = scheme_log(source, scheme, rep)
            p_durations = {}
            for line in (log_dir / 'router.log').read_text().splitlines():
                match = re.search(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) INFO pd_router: '
                                  r'req=(\S+) p=([\d.]+)ms', line)
                if match and start <= datetime.strptime(match[1], '%Y-%m-%d %H:%M:%S,%f') <= end:
                    if match[2] in p_durations:
                        raise ValueError(f'Duplicate router observation: {path}')
                    p_durations[match[2]] = float(match[3])
            if set(p_durations) != {r['request_id'] for r in rows}:
                raise ValueError(f'Router P-response request set mismatch: {path}')
            stats = dict(scheme=scheme, scenario=scenario, round=rep,
                         makespan_ms=float(rows[0]['makespan_ms']))
            for category in ['short', 'medium', 'long', 'overall']:
                selected = rows if category == 'overall' else [r for r in rows if r['category'] == category]
                for metric in ['ttft_ms', 'e2e_ms']:
                    stats[f'{category}_{metric}'] = statistics.mean(float(r[metric]) for r in selected)
                stats[f'{category}_p_response_ms'] = statistics.mean(
                    p_durations[r['request_id']] for r in selected)
            groups[(scheme, scenario)].append(stats)
            run_rows.append(stats)
            rows_all.extend(dict(scheme=scheme, scenario=scenario, round=rep, **r) for r in rows)
            long_rows.extend(dict(scheme=scheme, scenario=scenario, round=rep, **r)
                             for r in rows if r['category'] == 'long')
            measurement = load_json(path.with_suffix('.nvlink.json'))
            delta = measurement['delta']
            for key in ['gpu6_tx_gb', 'gpu7_rx_gb']:
                if abs(delta[key] * 1e9 - expected_bytes) > expected_bytes * .001:
                    raise ValueError(f'NVLink {key} does not match payload: {path}')
            nvlink_rows.append(dict(scheme=scheme, scenario=scenario, round=rep,
                                    expected_gb=expected_bytes / 1e9, **delta))
            replay_metadata = list(log_dir.glob('warmup_*.run.json'))
            replay_metadata += list((source / 'runs').glob(f'{scheme}_*_run{rep}/requests.run.json'))
            replay_starts = sorted(load_json(p)['started_utc'] for p in replay_metadata)
            if len(replay_starts) != 4:
                raise ValueError(f'Expected warmup/formal metadata for four replays: {log_dir}')
            replay_index = replay_starts.index(info['started_utc'])
            trace = formal_trace(log_dir, scheme,
                                 datetime.fromisoformat(info['started_utc']),
                                 datetime.fromisoformat(info['ended_utc']), replay_index)
            record = dict(scheme=scheme, scenario=scenario, round=rep, **trace)
            trace_rows.append(record)
            grouped = trace_groups[scheme]
            for metric in ['submit_ms', 'observed_ms', 'queue_ms']:
                grouped[metric].extend(trace[metric])
            grouped['bypass_events'] += trace['bypass_events']

    if len(run_rows) != 24 or len(groups) != 8 or any(len(g) != 3 for g in groups.values()):
        raise ValueError('Expected 18 primary and 6 FIFO-control runs, three repetitions per cell')
    summary = {}
    for (scheme, scenario), runs in groups.items():
        summary[f'{scheme}_{scenario}'] = {
            metric: dict(mean=statistics.mean(r[metric] for r in runs),
                         std=statistics.stdev(r[metric] for r in runs))
            for metric in runs[0] if metric not in ('scheme', 'scenario', 'round')}
    traces = {s: dict(bypass_events=t['bypass_events'],
                     **{metric: distribution(t[metric]) for metric in ['submit_ms', 'observed_ms', 'queue_ms']})
              for s, t in trace_groups.items()}
    deltas = {}
    for reference in ['baseline', 'stream_fifo']:
        for scheme in SCHEMES:
            if scheme == reference:
                continue
            for scenario in ['25ms', '50ms']:
                other = summary[f'{scheme}_{scenario}']
                base = summary[f'{reference}_{scenario}']
                deltas[f'{scheme}_vs_{reference}_{scenario}'] = {
                    k: (other[k]['mean'] / base[k]['mean'] - 1) * 100 for k in base}
    result = dict(formal_runs=len(run_rows), formal_requests=len(rows_all),
                  expected_payload_bytes_per_load=expected_bytes,
                  summary=summary, relative_changes_percent=deltas, formal_transfer_traces=traces)
    (root / 'performance_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (root / 'formal_trace_details.json').write_text(json.dumps(trace_rows, indent=2) + '\n')
    write_csv(root / 'runs.csv', run_rows)
    write_csv(root / 'requests.csv', rows_all)
    write_csv(root / 'long_requests.csv', long_rows)
    write_csv(root / 'nvlink_transfers.csv', nvlink_rows)

    deadline = load_json(root / 'deadline_analysis.json')
    formal_windows = []
    for trace in trace_rows:
        if trace['scheme'] != 'sort_only':
            continue
        ids = set(trace['request_ids'])
        windows = defaultdict(list)
        for record in deadline['requests']:
            if record['request_id'] in ids:
                windows[(record['log'], record['window'])].append(record['tokens'])
        formal_windows.append(dict(scenario=trace['scenario'], round=trace['round'],
                                   normal_window_members=list(windows.values())))
    (root / 'formal_windows.json').write_text(json.dumps(formal_windows, indent=2) + '\n')
    def fmt(s, key, scale=1):
        value = s[key]
        return f"{value['mean']/scale:.2f} ± {value['std']/scale:.2f}"

    lines = ['# NVLink 窗口排序与抢占传输单项实验', '',
        '日期：2026-11-15 UTC；物理 GPU6（P）/GPU7（D），按 UUID 绑定。', '',
        '**验收完成**：主实验18次正式运行270条请求；补充FIFO对照6次正式运行90条请求。'
        '相同数量的预热运行也全部通过。实际输出 token IDs、SSE 文本/时序、DONE、usage 与参考逐条核验，'
        '所有请求均正确。C8覆盖、连续offset、单槽 WRITE/DONE 和唯一final通知独立审计通过。', '',
        '| 方案 | 窗口排序 | 优先级抢占 | 传输 |', '|---|---|---|---|',
        '| baseline | 关 | 关 | 标准 NixlPushConnector，整请求传输 |',
        '| sort_only | 开，5 ms | 关 | 同一流式 connector，长请求 C8，eligible FIFO |',
        '| preempt_only | 关 | 开 | 同一流式 connector，长请求 C8，短请求插队与长请求保护 |',
        '| stream_fifo（补充对照） | 关 | 关 | 同一流式 connector，长请求 C8，eligible FIFO |', '',
        '单项表示两个功能开关只开一个；两个单项方案仍保留流式传输基础设施。'
        '因此对整请求baseline的差异同时含流式分块成本，不能全归因于某个开关。'
        '补充stream_fifo使排序/抢占能与相同传输基础设施比较；它在主实验之后顺序补测，'
        '没有与主实验交错，跨时间的共享CPU/内存条件仍是混杂因素。'
        '本轮未重跑两个开关同时开启组，尚不构成完整2×2因子实验，不能估计交互作用或分配独立贡献百分比。', '',
        '固定 Llama-3.1-8B-Instruct BF16，TP=PP=1，context32768，memory0.90，block16，'
        'max_seqs64，计算预算2048，prefix cache关闭，VLLM_BATCH_INVARIANT=1。'
        '15请求：8×128、3×1107、4×10000输入tokens，64输出，temperature0，seed1500–1514。'
        '每5条一批，批开始时刻为0/25/50 ms或0/50/100 ms；三轮轮换服务顺序，'
        '每次新启P/D，每场景一次预热、一次正式测量。FIFO补测沿用相同参数。', '',
        '流式方案限制单个活动WRITE；长请求每块≤8 blocks，短/中请求按可用批聚合；'
        'writer等待配置0.1 ms，长请求保护85%/3次/50 ms。'
        'TTFT为客户端提交至首个非空文本SSE，E2E为完整响应结束，makespan为首个实际提交到最后响应结束。'
        '表中为每次运行均值的三轮平均±样本标准差；三轮不足以证明统计显著性。', '',
        '## NVLink 验证', '',
        'UCX_TLS=cuda,tcp，UCX_NET_DEVICES=lo。主实验和FIFO对照每个正式场景均在负载前后'
        '采集GPU6 TX/GPU7 RX的12条链路计数；每边增加约5.817500 GB，'
        f'与预期填充KV负载 {expected_bytes:,} bytes（Llama配置与16-token块对齐）相符。'
        '允许的核验误差0.1%，实际差异仅计数取整量级。预热也保留独立计数文件。', '',
        '本轮请求UCX info级别，但vLLM模型worker日志没有输出UCX endpoint lane信息；'
        '不能声称在正式运行日志里直接观测了cuda_ipc lane。'
        '路径结论基于正式场景中与全部KV payload匹配的NVLink TX/RX硬件计数，'
        '以及此前同机双进程NIXL测试的CUDA IPC lane证据；不能仅凭环境变量判断。'
        '纯TCP/host-staging不可能解释这些GPU间NVLink payload计数。', '',
        '## 主要结果', '']
    for scenario in ['25ms', '50ms']:
        lines += [f'### {scenario} 批间隔', '',
            '| 方案 | makespan(s) | short TTFT(ms) | medium TTFT(ms) | long TTFT(ms) | overall E2E(ms) |',
            '|---|---:|---:|---:|---:|---:|']
        for scheme in SCHEMES:
            s = summary[f'{scheme}_{scenario}']
            lines.append('| ' + ' | '.join([LABELS[scheme], fmt(s, 'makespan_ms', 1000),
                fmt(s, 'short_ttft_ms'), fmt(s, 'medium_ttft_ms'), fmt(s, 'long_ttft_ms'),
                fmt(s, 'overall_e2e_ms')]) + ' |')
        lines += ['', '相对baseline（负值为改善）：', '',
            '| 方案 | makespan | short TTFT | medium TTFT | long TTFT | overall E2E |',
            '|---|---:|---:|---:|---:|---:|']
        for scheme in SCHEMES[1:]:
            d = deltas[f'{scheme}_vs_baseline_{scenario}']
            lines.append('| ' + LABELS[scheme] + ' | ' + ' | '.join(
                f'{d[k]:+.2f}%' for k in ['makespan_ms', 'short_ttft_ms', 'medium_ttft_ms',
                                          'long_ttft_ms', 'overall_e2e_ms']) + ' |')
        lines += ['', '同流式基础设施，相对stream_fifo：', '',
            '| 方案 | makespan | short TTFT | medium TTFT | long TTFT | overall E2E |',
            '|---|---:|---:|---:|---:|---:|']
        for scheme in ['sort_only', 'preempt_only']:
            d = deltas[f'{scheme}_vs_stream_fifo_{scenario}']
            lines.append('| ' + LABELS[scheme] + ' | ' + ' | '.join(
                f'{d[k]:+.2f}%' for k in ['makespan_ms', 'short_ttft_ms', 'medium_ttft_ms',
                                          'long_ttft_ms', 'overall_e2e_ms']) + ' |')
        lines += ['']

    lines += ['## 差距原因与证据', '',
        '**窗口排序改变的是计算入队顺序。** sort_only的日志验证窗口内按prompt长度稳定排序、'
        '跨窗口FCFS，且抢占事件为0。短/中请求提前得到P端计算资源，并更早进入D端响应，'
        '减少排在长prefill后的队列等待；长请求相应后移，因此short/medium收益伴随long首token代价。'
        '排序包含5ms收集、释放与调度机制，本次未增加同窗口FCFS，不能进一步分离收集和排序贡献。', '',
        'Router记录的P HTTP响应耗时（包含P排队、prefill和响应处理，不是纯GPU计算时间）'
        '提供了同方向证据：', '',
        '| 批间隔 | 方案 | short P响应(ms) | medium P响应(ms) | long P响应(ms) |',
        '|---|---|---:|---:|---:|']
    for scenario in ['25ms', '50ms']:
        for scheme in SCHEMES:
            s = summary[f'{scheme}_{scenario}']
            lines.append('| ' + ' | '.join([scenario, LABELS[scheme],
                fmt(s, 'short_p_response_ms'), fmt(s, 'medium_p_response_ms'),
                fmt(s, 'long_p_response_ms')]) + ' |')
    lines += ['',
        '**25ms场景的排序收益存在较大轮间波动。** short TTFT为967.04±323.34ms。'
        '引擎窗口日志显示前两轮正式请求均为1条idle singleton加剩余14条形成同一正常窗口；'
        '第三轮为1条singleton加4条、10条两个正常窗口。'
        '50ms三轮均为1条singleton加4条、10条两个窗口。'
        '窗口起点是引擎接纳时间而非HTTP提交时间，请求在前一计算步骤期间聚集后可以一起入窗，'
        '因此客户端分批间隔相同也不保证引擎窗口成员相同。成员变化改变可重排范围，'
        '与25ms场景的方差一致；详见formal_windows.json，不能宣称每次都固定获得相同比例收益。', '',
        f"窗口日志含预热共{deadline['released_requests']}条排序请求；正常窗口请求发布等待平均"
        f"{deadline['request_wait_ms']['mean']:.3f}ms，p95 {deadline['request_wait_ms']['p95']:.3f}ms，"
        f"最大{deadline['request_wait_ms']['max']:.3f}ms；发布时间不是GPU开算时刻。"
        f"发布后首次schedule完成观测等待平均{deadline['ready_queue_ms']['mean']:.1f}ms，"
        f"p95 {deadline['ready_queue_ms']['p95']:.1f}ms。模型执行/计算队列的等待仍远大于5ms窗口本身。", '',
        '**抢占确实触发，但不能跳过P端计算。** preempt_only在完整正式请求轨迹中发生'
        f"{traces['preempt_only']['bypass_events']}条short-bypasses-long日志；sort_only和FIFO均为0。"
        '这里是长chunk边界处的队列插队，不会中断已经提交的DMA，也不抢占P的GPU kernel。'
        '短请求若尚未完成prefill、KV未就绪，传输优先级不能消除之前的计算排队。'
        '插队日志计数按被绕过的long候选记录，一次选择可能对应多条，不能等同独立DMA中断次数。', '',
        '**C8单槽的软件成本可抵消传输调度收益。** 每个WRITE均走描述符准备、native提交、'
        'handle轮询与完成回报；max_active=1时下一块要等前块被观测完成。'
        '正式轨迹统计如下；submit与observed是包含软件/线程调度的host观测时间，'
        'observed从提交前开始，包含submit，二者不能相加或解释为纯DMA时间。', '',
        '| 方案 | 正式WRITE数 | submit均值/p95(ms) | observed均值/p95(ms) | queue均值(ms) | 插队日志条数 |',
        '|---|---:|---:|---:|---:|---:|']
    for scheme in SCHEMES[1:]:
        t = traces[scheme]
        sub, obs, que = t['submit_ms'], t['observed_ms'], t['queue_ms']
        lines.append(f"| {LABELS[scheme]} | {sub['count']} | {sub['mean']:.3f}/{sub['p95']:.3f} | "
                     f"{obs['mean']:.3f}/{obs['p95']:.3f} | {que['mean']:.3f} | {t['bypass_events']} |")
    lines += ['',
        '长请求一个C8块约16MiB；此前单buffer NIXL NVLink microbenchmark的16MiB约77GB/s'
        '（每次约0.22ms，含API/轮询），而这里大量分散KV描述符与线程交互下的host观测耗时为数ms。'
        '不能用约201GB/s的大buffer参考值推算本次KV端到端耗时；'
        '也不能仅由本轮日志把描述符、Python/GIL、native代码、轮询各自的成本精确拆开。'
        'FIFO对照在25/50ms下相对baseline的short TTFT分别增加9.32%/9.61%；'
        '打开抢占后，相对FIFO分别下降2.82%/2.94%，却仍比baseline增加6.24%/6.38%。'
        '这与“抢占带来小幅优先级调度收益，但不足以抵消当前流式路径成本”一致；'
        '因为FIFO补测没有交错及样本有限，不把这些差异当作严格的独立因果估计。', '',
        '**makespan与短请求TTFT是不同目标。** 对同一15请求工作量，优先服务短请求主要重排等待；'
        '最后一个长请求及64-token decode仍决定尾部完成时间。'
        '因此短请求首token大幅提前，不必带来makespan改善；当long被推迟且分块有额外开销时，'
        '整体结束时间可能增加。所有差异均据实报告，不把没有收益解释成已证明的带宽饱和或绝对无效。', '',
        '## 复现与数据范围', '',
        '在项目根目录使用cc环境Python。主实验：`python code/run_pd_local_sort_comparison.py '
        '--result-dir <新目录> --workload results/nvlink-isolated-ablation-20261115/workload.jsonl --rounds 3`。'
        '补充对照：同一程序、同一workload使用`--schemes stream_fifo --rounds 3`并指定独立新目录。'
        '启动器默认两个GPU分别绑定物理6/7的UUID，独立进程；'
        '重放固定归档workload，不修改历史结果。', '',
        '数据：`runs.csv`逐轮统计，`requests.csv`全部360条正式请求，`long_requests.csv`逐条long，'
        '`nvlink_transfers.csv`各次正式传输计数，`performance_summary.json`均值/标准差与变化百分比，'
        '`formal_trace_details.json`正式请求ID及传输观测值。'
        '原始CSV/SSE/计数/验证文件在`runs/`及`fifo_control/runs/`；'
        '两份`streaming_audit.json`核验预热和正式的输出/协议；窗口见`deadline_analysis.json`。', '',
        '最终归档核验：`nvlink_validation.json`将24次正式及24次预热的全部12链路原始计数'
        '重新求和，48次均与预期KV payload匹配；30项CPU回归通过，见`cpu_regressions.log`。'
        '最终GPU6/7均0MiB、0%利用率、无计算进程，8000/8100/8200无监听，'
        '查询时间与原始输出见`gpu_final_state.json`。', '',
        '执行源码归档在`source-runtime/`和`source-vllm/`，快照时点见`source-runtime/SNAPSHOT_NOTES.md`；'
        '最终分析源码在`source-analysis/`，文档快照在`docs/`，全量文件校验见`SHA256SUMS`。'
        '未在本轮采集连续GPU进程环境监控，GPU绑定由启动日志和启动器实现记录，'
        '负载期间由NVLink计数核验流量，最终释放状态单独查询。'
        '三轮样本有限，单机共享CPU/内存等未完全隔离；仅对本固定workload、TP=PP=1成立。'
        '旧实验没有硬件计数，不能把本轮复测反推为过去所有实验已走NVLink。', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    (root / 'RESULTS.md').write_text('# 实验结果\n\n见[完整报告](REPORT.md)及[runs.csv](runs.csv)。\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=Path)
    args = parser.parse_args()
    data = analyze(args.result_dir)
    print(json.dumps({k: data[k] for k in ['formal_runs', 'formal_requests', 'formal_transfer_traces']}, indent=2))
