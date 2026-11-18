#!/usr/bin/env python3
"""Audit and report the exact archived P/D comparison, without using GPUs."""
from __future__ import annotations
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import re
import statistics as stats
from pathlib import Path


def mean(values): return stats.mean(values) if values else None

def sd(values): return stats.stdev(values) if len(values)>1 else 0.0

def audit(root):
    reference = {r['request_id']: r for r in csv.DictReader((root/'reference_outputs_a100.csv').open())}
    workload = [json.loads(s) for s in (root/'workload.jsonl').read_text().splitlines() if s.strip()]
    expected = {r['request_id']: r for r in workload}
    expected_runs={f'{scheme}_{scenario}_run{rep}' for scheme in ['baseline','old','new']
                   for scenario in ['25ms','50ms'] for rep in [1,2,3]}
    checks, groups, combined, individual = [], collections.defaultdict(list), [], collections.defaultdict(list)
    for run in sorted((root/'runs').iterdir()):
        if not (run/'requests.csv').exists(): continue
        scheme, scenario, rep = run.name.split('_')
        rows = list(csv.DictReader((run/'requests.csv').open()))
        errors = []
        if len(rows)!=15 or collections.Counter(r['request_id'] for r in rows)!=collections.Counter(expected.keys()):
            errors.append('request set/count mismatch')
        for row in rows:
            rid = row['request_id']
            ids = json.loads(row['output_token_ids'])
            if rid not in reference or ids!=json.loads(reference[rid]['output_token_ids']): errors.append(rid+': output mismatch')
            if len(ids)!=64 or int(row['num_output_tokens'])!=64: errors.append(rid+': output count')
            if row['success']!='True' or row['reference_match']!='True': errors.append(rid+': invalid status')
            if not (0<float(row['ttft_ms'])<=float(row['e2e_ms'])): errors.append(rid+': invalid timings')
            if int(row['num_prompt_tokens'])!=len(expected[rid]['prompt_token_ids']): errors.append(rid+': input count')
            if row['token_ids_hash']!=hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]: errors.append(rid+': hash mismatch')
            combined.append(dict(scheme=scheme,scenario=scenario,round=rep.removeprefix('run'),**row))
            if row['category']=='long': individual[scheme,scenario,rid].append(row)
        events = {r['request_id']: r['events'] for r in map(json.loads,(run/'requests.sse.jsonl').read_text().splitlines())}
        for row in rows:
            nonempty = [e for e in events[row['request_id']] if any(c.get('text') or (c.get('delta') or {}).get('content') for c in e['payload'].get('choices',[]))]
            if not nonempty or abs(float(row['ttft_ms'])-nonempty[0]['elapsed_ms'])>.001: errors.append(row['request_id']+': first text timing')
        span=(max(float(r['completed_at']) for r in rows)-min(float(r['submitted_at']) for r in rows))*1000
        summary={'run':run.name,'valid':not errors,'errors':errors,'requests':len(rows),'makespan_ms':span}
        for kind in ['short','medium','long','overall']:
            cat=[r for r in rows if kind=='overall' or r['category']==kind]
            for metric in ['ttft_ms','e2e_ms']:
                summary[kind+'_'+metric]=mean([float(r[metric]) for r in cat])
        summary['batch_submissions_ms']={str(b):[float(r['submitted_at_rel_ms']) for r in rows if int(r['batch_idx'])==b] for b in range(3)}
        checks.append(summary);groups[scheme,scenario].append(summary)
    warmups=[]
    for p in sorted(root.glob('logs_round*/warmup_*.csv')):
        rows=list(csv.DictReader(p.open()))
        valid=len(rows)==15 and all(r['success']=='True' and json.loads(r['output_token_ids'])==json.loads(reference[r['request_id']]['output_token_ids']) for r in rows)
        warmups.append({'path':str(p.relative_to(root)),'valid':valid,'requests':len(rows)})
    env=json.loads((root/'environment.json').read_text())
    allowed_uuids=set()
    for line in env['gpu_snapshot_before'].splitlines()[1:]:
        parts=[v.strip() for v in line.split(',')]
        if parts and parts[0] in ('6','7'): allowed_uuids.add(parts[1])
    observed_masks=collections.Counter();gpu_errors=[];samples=0;unknown_masks=set()
    telemetry=root/'gpu_telemetry.jsonl'
    for line in telemetry.read_text().splitlines():
        entry=json.loads(line);samples+=1
        for proc in entry.get('processes',[]):
            mask=proc.get('CUDA_VISIBLE_DEVICES');observed_masks[str(mask)]+=1
            if mask is None:
                # setproctitle in an EngineCore can overwrite /proc/environ;
                # validate actual GPU contexts separately rather than infer usage.
                unknown_masks.add(proc['pid'])
            elif mask not in {'6,7',*allowed_uuids}:
                gpu_errors.append({'pid':proc['pid'],'mask':mask})
    context_counts=collections.Counter();context_violations=[];context_samples=0
    if (root/'gpu_contexts.jsonl').exists():
        for line in (root/'gpu_contexts.jsonl').read_text().splitlines():
            entry=json.loads(line);context_samples+=1
            for app in entry['experiment_contexts']:
                context_counts[app['gpu_uuid']]+=1
                if app['gpu_uuid'] not in allowed_uuids:context_violations.append(app)
    preemptions={};runtime_errors={};sort_observations={}
    for p in sorted(root.glob('logs_round*/p_*.log')):
        text=p.read_text(errors='replace')
        preemptions[str(p.relative_to(root))]=text.count('Preemption: short req')
        if 'PD local admission:' in text:
            waits=[float(s) for s in re.findall(r'PD local admission:.*?wait_ms=([\d.]+)',text)]
            windows=[];current=[]
            for rank,tokens in re.findall(r'PD local admission: window=\d+ rank=(\d+).*?tokens=(\d+)',text):
                if int(rank)==0 and current:windows.append(current);current=[]
                current.append(int(tokens))
            if current:windows.append(current)
            sort_observations[str(p.relative_to(root))]={
                'admission_records_including_warmup':len(waits),
                'wait_ms_min':min(waits),'wait_ms_mean':mean(waits),'wait_ms_max':max(waits),
                'window_sizes':[len(w) for w in windows],
                'all_logged_windows_sorted_by_prompt_length':all(w==sorted(w) for w in windows),
            }
    for p in sorted(root.glob('logs_round*/*_*.log')):
        if p.name.startswith(('p_','d_')):
            # Engine termination at cleanup may log a dead worker; preserve separately.
            lines=[s for s in p.read_text(errors='replace').splitlines() if 'preemptive-push-writer error' in s or 'registration for request' in s and 'timed out' in s]
            if lines:runtime_errors[str(p.relative_to(root))]=lines
    summary={}
    for (scheme,scenario),runs in groups.items():
        summary.setdefault(scheme,{})[scenario]={'rounds':len(runs)}
        for key in ['makespan_ms']+[k+'_'+m for k in ['short','medium','long','overall'] for m in ['ttft_ms','e2e_ms']]:
            vals=[r[key] for r in runs]
            summary[scheme][scenario][key]={'mean':mean(vals),'std':sd(vals)}
    result={'complete':{r['run'] for r in checks}==expected_runs and all(r['valid'] for r in checks) and len(warmups)==18 and all(r['valid'] for r in warmups) and context_samples>0 and not gpu_errors and not context_violations and not runtime_errors,
            'formal_runs':len(checks),'formal_requests':len(combined),'warmup_runs':len(warmups),
            'warmup_requests':sum(r['requests'] for r in warmups),'runs':checks,'warmups':warmups,
            'summary':summary,'gpu':{'allowed_physical_indices':[6,7],'allowed_uuids':sorted(allowed_uuids),'telemetry_samples':samples,'observed_process_masks':dict(observed_masks),'mask_violations':gpu_errors,'unreadable_environment_pids':sorted(unknown_masks),'actual_context_samples':context_samples,'actual_context_uuids':dict(context_counts),'context_violations':context_violations},
            'preemption_events_including_warmup':preemptions,'transfer_runtime_errors':runtime_errors,
            'local_sort_observations_including_warmup':sort_observations,
            'implementation_limits':[
                '提前发送未形成有效路径：归档版 PreemptivePushScheduler 通过 self.requests 查找活动请求，但本类初始化及父类均未创建该属性；因此活动 prefill 请求被跳过。',
                'Router 在收到 P 完成响应后才向 D 提交请求，P prefill 期间不存在 D 提前注册接收地址的路径。',
                'CHUNK_SIZE_BLOCKS=8 和 PD_STREAM_MAX_IN_FLIGHT=1 是配置值，尚不能证明 C8 提前发送和单 WRITE 并发限制生效；归档 Worker 的 max-in-flight 仅被读取及打印。',
                'UCX 配置允许 cuda_ipc，但未逐 WRITE 观测实际 transport、NVLink 流量或纯传输带宽，不能把理论约4.5ms当作实测数据。',
                '插队日志表示软件队列 bypass 决策，汇总次数包含预热，不能据此证明在活动 C8 DMA 块之间抢占。',
                '5ms 是收集窗口配置，释放由调度循环检查触发，不能保证实际仅等待5ms；具体等待分布见 local_sort_observations_including_warmup。',
            ]}
    (root/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    for name,rows in [('requests.csv',combined),('runs.csv',checks)]:
        with (root/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    long_rows=[]
    for (scheme,scenario,rid),rows in sorted(individual.items()):
        long_rows.append(dict(scheme=scheme,scenario=scenario,request_id=rid,ttft_ms=mean([float(r['ttft_ms']) for r in rows]),e2e_ms=mean([float(r['e2e_ms']) for r in rows])))
    with (root/'long_requests.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(long_rows[0]));writer.writeheader();writer.writerows(long_rows)
    lines=['# v2 P/D 正式实验记录','',f'结果目录：`{root}`。','',
           f"验收：{len(checks)}/18 正式运行，{len(combined)}/270 正式请求；{len(warmups)}/18 预热运行。全部检查通过：**{result['complete']}**。",'',
           '仅使用物理 GPU6（P）和 GPU7（D），按 UUID 分别绑定；完整进程可见设备采样见 `gpu_telemetry.jsonl`。',
           '', '本轮测量当前归档代码。实现审计发现提前发送、C8 分块、单槽并发尚未形成完整有效路径，因此以下结果只能说明当前实现及 P 排序的表现，不能当作完整流式传输算法的验证。','',
           '固定参数：Llama-3.1-8B-Instruct BF16，TP=1，context=32768，memory=0.90，block=16，max_seqs=64，计算预算=2048，prefix cache 关闭；VLLM_BATCH_INVARIANT=1。15 请求（8×128、3×1107、4×10000 输入 tokens），每条64输出，temperature=0，seed1500–1514。3轮轮换，每方案每场景一次预热和一次正式运行。',
           '', 'TTFT 从客户端提交到首个非空文本 SSE；E2E 到流完整结束；makespan 从首条实际提交到最后一条流结束。保存真实输出 token IDs、完整文本与 SSE 事件，并逐条与参考 token IDs 比较。标准差为三轮均值的样本标准差。','']
    for scenario in ['25ms','50ms']:
        lines.extend([f'## {scenario} 批间隔','', '| 方案 | makespan (s) | short TTFT (ms) | medium TTFT (ms) | long TTFT (ms) | overall E2E (ms) |','|---|---:|---:|---:|---:|---:|'])
        for scheme in ['baseline','old','new']:
            if scheme not in summary or scenario not in summary[scheme]: continue
            s=summary[scheme][scenario]
            def fmt(key,div=1):return f"{s[key]['mean']/div:.2f} ± {s[key]['std']/div:.2f}"
            lines.append('| '+scheme+' | '+ ' | '.join([fmt('makespan_ms',1000),fmt('short_ttft_ms'),fmt('medium_ttft_ms'),fmt('long_ttft_ms'),fmt('overall_e2e_ms')])+' |')
        if 'new' in summary and scenario in summary['new']:
            for base in ['baseline','old']:
                if base not in summary or scenario not in summary[base]:continue
                diffs=[]
                for key,label in [('makespan_ms','makespan'),('short_ttft_ms','short TTFT'),('medium_ttft_ms','medium TTFT'),('long_ttft_ms','long TTFT'),('overall_e2e_ms','overall E2E')]:
                    a=summary['new'][scenario][key]['mean'];b=summary[base][scenario][key]['mean']
                    diffs.append(f'{label} {(a/b-1)*100:+.2f}%')
                lines.extend(['',f'new 相对 {base}：'+ '；'.join(diffs)+'。'])
        lines.extend(['','| 方案 | short E2E (ms) | medium E2E (ms) | long E2E (ms) |','|---|---:|---:|---:|'])
        for scheme in ['baseline','old','new']:
            if scheme not in summary or scenario not in summary[scheme]:continue
            s=summary[scheme][scenario]
            cells=[f"{s[k+'_e2e_ms']['mean']:.2f} ± {s[k+'_e2e_ms']['std']:.2f}" for k in ['short','medium','long']]
            lines.append('| '+scheme+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines.extend(['## 逐条 long（三轮均值，ms）','','| 场景 | 请求 | 方案 | TTFT | E2E |','|---|---|---|---:|---:|'])
    for r in sorted(long_rows,key=lambda r:(r['scenario'],r['request_id'],r['scheme'])):
        lines.append(f"| {r['scenario']} | {r['request_id']} | {r['scheme']} | {r['ttft_ms']:.2f} | {r['e2e_ms']:.2f} |")
    lines.extend(['','## 已确认的实现限制',''])
    lines.extend('- '+s for s in result['implementation_limits'])
    lines.extend(['','插队日志次数（含预热，按每次服务启动汇总）：`'+json.dumps(preemptions)+'`。',
                  '', '归档：`environment.json` 参数及拓扑；`source-before/` 修改前文件；`source-runtime/` 实验源码；`source-vllm/` connector 运行时源码；`SHA256SUMS`；`runs/*/requests.csv` 原始数据及对应 SSE、负载日志与时间边界；`audit.json` 全量验收。',
                  '', '此轮在同机 GPU0–5 存在其他业务的条件下进行；本实验未在这些卡上建立计算上下文，但未隔离共享 CPU、内存及互联资源的影响。'])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ['complete','formal_runs','formal_requests','warmup_runs','warmup_requests','gpu']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result-dir',required=True);a=p.parse_args();audit(Path(a.result_dir))
