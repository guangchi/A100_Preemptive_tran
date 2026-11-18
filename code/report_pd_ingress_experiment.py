"""Summarize fixed-ingress ablations and paired 25/50ms timing evidence."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics as st

SCHEMES = ['baseline', 'sort_only', 'preempt_only', 'new']
LABELS = dict(baseline='baseline', sort_only='只窗口排序', preempt_only='只抢占', new='排序+抢占')
CATEGORIES = ['short', 'medium', 'long', 'all']


def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|'] +
                     ['|'+'|'.join(map(str, row))+'|' for row in rows])


def report(root):
    def load(name):
        with (root/name).open() as file:
            return list(csv.DictReader(file))
    requests, runs, windows = [load(n) for n in ['pipeline_requests.csv','pipeline_runs.csv','pipeline_windows.csv']]
    validation = {name:json.loads((root/(name+'.json')).read_text()) for name in
                  ['pipeline_validation','streaming_audit','nvlink_validation']}
    assert validation['pipeline_validation']['checks_passed']
    assert validation['streaming_audit']['complete_formal_experiment']
    assert validation['nvlink_validation']['all_passed']
    assert len(runs) == 24 and len(requests) == 360
    e2e_rows, run_metrics = [], []
    for run in runs:
        path = root/'runs'/f"{run['scheme']}_{run['scenario']}_run{run['round']}"/'requests.csv'
        with path.open() as file:
            raw = list(csv.DictReader(file))
        metrics = {k:run[k] for k in ('scheme','scenario','round')}
        metrics['makespan_ms'] = float(raw[0]['makespan_ms'])
        for category in CATEGORIES:
            selected = [r for r in raw if category == 'all' or r['category'] == category]
            metrics[category+'_e2e_ms'] = st.mean(float(r['e2e_ms']) for r in selected)
        run_metrics.append(metrics)
    with (root/'run_metrics.csv').open('w', newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(run_metrics[0]));writer.writeheader();writer.writerows(run_metrics)
    for gap in ['25ms','50ms']:
        for scheme in SCHEMES:
            selected = [r for r in run_metrics if r['scheme']==scheme and r['scenario']==gap]
            e2e_rows.append([gap,LABELS[scheme],*[
                f"{st.mean(r[c+'_e2e_ms'] for r in selected):.1f} ± {st.stdev(r[c+'_e2e_ms'] for r in selected):.1f}"
                for c in CATEGORIES],
                f"{st.mean(r['makespan_ms'] for r in selected)/1000:.3f} ± {st.stdev(r['makespan_ms'] for r in selected)/1000:.3f}"])
    def subset(scheme, gap, category='short', rep=None):
        return [r for r in requests if r['scheme'] == scheme and r['scenario'] == gap
                and (category == 'all' or r['category'] == category)
                and (rep is None or r['round'] == str(rep))]
    def mean(scheme,gap,metric,category='short',rep=None):
        values = [float(r[metric]) for r in subset(scheme,gap,category,rep) if r.get(metric)]
        return st.mean(values) if values else None
    def fmt(value): return f'{value:.2f}' if value is not None else '—'
    def window_rows(scheme,gap,rep):
        return sorted([w for w in windows if w['scheme']==scheme and w['scenario']==gap and w['round']==str(rep)],
                      key=lambda w:int(w['window']))
    def window_members(scheme,gap,rep): return [w['members'] for w in window_rows(scheme,gap,rep)]
    def pattern(scheme,gap,rep): return '+'.join(w['count'] for w in window_rows(scheme,gap,rep))
    summary_rows, delta_rows, paired_rows, baseline_rows = [],[],[],[]
    summary = {}
    for gap in ['25ms','50ms']:
        for scheme in SCHEMES:
            selected=[r for r in runs if r['scheme']==scheme and r['scenario']==gap]
            cells=[]
            for cat in CATEGORIES:
                values=[float(r[cat+'_ttft_ms']) for r in selected]
                cells.append(f'{st.mean(values):.1f} ± {st.stdev(values):.1f}')
            summary_rows.append([gap,LABELS[scheme],*cells])
            summary[scheme+'_'+gap]={cat:dict(mean=mean(scheme,gap,'ttft_ms',cat),
                std=st.stdev(float(r[cat+'_ttft_ms']) for r in selected)) for cat in CATEGORIES}
            baseline_rows.append([gap,LABELS[scheme],*[f'{(1-mean(scheme,gap,"ttft_ms",c)/mean("baseline",gap,"ttft_ms",c))*100:+.1f}%' for c in CATEGORIES]])
    for scheme in SCHEMES:
        delta_rows.append([LABELS[scheme],*[f'{mean(scheme,"50ms","ttft_ms",c)-mean(scheme,"25ms","ttft_ms",c):+.1f}' for c in CATEGORIES]])
        for rep in [1,2,3]:
            paired_rows.append([rep,LABELS[scheme],*[fmt(mean(scheme,g,'ttft_ms',rep=rep)) for g in ['25ms','50ms']],
                f'{mean(scheme,"50ms","ttft_ms",rep=rep)-mean(scheme,"25ms","ttft_ms",rep=rep):+.2f}'])
    membership_rows=[]
    for rep in [1,2,3]:
        for scheme in ['sort_only','new']:
            membership_rows.append([rep,LABELS[scheme],pattern(scheme,'25ms',rep),pattern(scheme,'50ms',rep),
                                   '相同' if window_members(scheme,'25ms',rep)==window_members(scheme,'50ms',rep) else '不同'])
    for rep in [1,2,3]:
        for gap in ['25ms','50ms']:
            membership_rows.append([rep,gap+'排序 vs 组合',pattern('sort_only',gap,rep),pattern('new',gap,rep),
                                   '相同' if window_members('sort_only',gap,rep)==window_members('new',gap,rep) else '不同'])
    stages=['client_to_schedule_ms','schedule_to_frontier_ms','frontier_to_ready_observed_ms',
            'ready_to_eligible_observed_ms','eligible_to_submit_ms','final_submit_to_done_ms','done_to_first_ms']
    stage_rows=[]
    for scheme in ['sort_only','preempt_only','new']:
        for gap in ['25ms','50ms']:
            stage_rows.append([gap,LABELS[scheme],*[fmt(mean(scheme,gap,m)) for m in stages],fmt(mean(scheme,gap,'ttft_ms'))])
        stage_rows.append(['50−25',LABELS[scheme],*[f'{mean(scheme,"50ms",m)-mean(scheme,"25ms",m):+.2f}' for m in stages],
                           f'{mean(scheme,"50ms","ttft_ms")-mean(scheme,"25ms","ttft_ms"):+.2f}'])
    baseline_stages=['client_to_schedule_ms','schedule_to_native_publish_ms','native_publish_to_write_ms','native_write_to_done_ms','native_done_to_first_ms']
    native_rows=[[gap,*[fmt(mean('baseline',gap,m)) for m in baseline_stages],fmt(mean('baseline',gap,'ttft_ms'))] for gap in ['25ms','50ms']]
    window_stage_rows=[]
    for scheme in ['sort_only','new']:
        for gap in ['25ms','50ms']:
            window_stage_rows.append([gap,LABELS[scheme],*[fmt(mean(scheme,gap,m,'all')) for m in
               ['ingress_to_prepared_ms','prepared_to_release_ms','deadline_to_release_ms','ready_queue_ms']]])
    batch_rows=[]
    for scheme in SCHEMES:
        for gap in ['25ms','50ms']:
            for batch in [0,1,2]:
                selected=[r for r in subset(scheme,gap) if r['batch']==str(batch)]
                batch_rows.append([gap,LABELS[scheme],batch,*[fmt(st.mean(float(r[m]) for r in selected)) for m in
                    ['client_to_schedule_ms','ttft_ms']]])
    arrival_rows=[]
    for scheme in SCHEMES:
        for gap in ['25ms','50ms']:
            spans=defaultdict(list)
            for rep in [1,2,3]:
                group=subset(scheme,gap,'all',rep)
                for batch in [0,1,2]:
                    rr=[r for r in group if r['batch']==str(batch)]
                    for stage in ['client','router','p_headers','api']:
                        stamps=[int(r[stage+'_ns']) for r in rr]
                        spans[stage].append((max(stamps)-min(stamps))/1e6)
            arrival_rows.append([gap,LABELS[scheme],*[f'{st.mean(spans[s]):.2f}/{max(spans[s]):.2f}'
                         for s in ['client','router','p_headers','api']]])
    # Actual client offsets relative to first request of each replay; this
    # avoids interpreting target send times as measured API/window arrivals.
    offsets=defaultdict(list)
    for run in runs:
        group=subset(run['scheme'],run['scenario'],'all',run['round'])
        first=min(int(r['client_ns']) for r in group)
        for r in group:
            offsets[(run['scheme'],run['scenario'],r['category'])].append((int(r['client_ns'])-first)/1e6)
            offsets[(run['scheme'],run['scenario'],'all')].append((int(r['client_ns'])-first)/1e6)
    origin_rows=[]
    for scheme in SCHEMES:
        for cat in CATEGORIES:
            shift=st.mean(offsets[(scheme,'50ms',cat)])-st.mean(offsets[(scheme,'25ms',cat)])
            ttft=mean(scheme,'50ms','ttft_ms',cat)-mean(scheme,'25ms','ttft_ms',cat)
            origin_rows.append([LABELS[scheme],cat,f'{shift:+.2f}',f'{ttft:+.2f}',f'{ttft+shift:+.2f}'])
    choices=load('writer_choice_summary.csv') if (root/'writer_choice_summary.csv').exists() else []
    choice_rows=[]
    for scheme in SCHEMES[1:]:
        for gap in ['25ms','50ms']:
            selected=[r for r in choices if r['scheme']==scheme and r['scenario']==gap]
            choice_rows.append([gap,LABELS[scheme],*[sum(int(r.get(m) or 0) for r in selected) for m in
                 ['short_over_long','protected_long_over_short','decisions_with_short_eligible']]])
    audit=validation['streaming_audit']
    nvlink=validation['nvlink_validation']
    manifest=json.loads((root/'experiment_manifest.json').read_text())
    findings = (root/'CAUSE_ANALYSIS.md').read_text() if (root/'CAUSE_ANALYSIS.md').exists() else ''
    text=f'''# GPU4/5：P HTTP 固定窗口四组消融

本轮使用已修改的 **P HTTP 入口固定 5ms 窗口**，物理 GPU4(P)/GPU5(D)，四方案 × 两种间隔 × 三轮，24 次正式回放、360 条正式请求；另 24 次预热、360 条请求。全部逐 token/SSE/DONE/usage 校验通过。三轮轮换方案顺序、第二轮反转场景顺序，每个服务重新启动。

## 固定配置与计时

Llama-3.1-8B-Instruct BF16，TP=1，max model len32768，block16，max sequences64，计算预算2048tokens，关闭 prefix cache，batch invariant开启。15 请求为 8×128、3×1107、4×10000 输入 tokens，每条输出64tokens；三次突发各5条，目标时刻为0/25/50ms或0/50/100ms。C16、单槽WRITE、0.1ms poll；长请求保护85%进度/3次绕过/50ms。四组都配置UCX cuda,tcp；baseline为原生整请求传输，其余为同一流式传输代码，分别开关窗口排序与抢占。GPU4/5为A100-SXM4-80GB、NV12，同属NUMA1。

TTFT为客户端开始请求至首个非空文本SSE。下表单位ms，为三轮回放均值±样本标准差；每格45条正式请求，类别样本为24/9/12条；全部均值按8:3:4加权。单机上其他GPU有外部任务，采样用于确认实验进程只使用4/5，未改动外部任务。trace在四组中均开启，其观察开销计入数据。

## TTFT与baseline对照

{table(['间隔','方案','Short','Medium','Long','全部'],summary_rows)}

完整响应结束时间 E2E（ms）与整体回放完成时间 makespan（s），三轮均值±样本标准差：

{table(['间隔','方案','Short E2E','Medium E2E','Long E2E','全部 E2E','Makespan (s)'],e2e_rows)}

相对同间隔baseline的TTFT降低比例，正数表示更快：

{table(['间隔','方案','Short','Medium','Long','全部'],baseline_rows)}

## 25ms与50ms的变化

50ms−25ms的TTFT，负数表示50ms更低：

{table(['方案','Short','Medium','Long','全部'],delta_rows)}

{findings}

逐轮Short对照，避免均值掩盖不一致方向：

{table(['轮次','方案','25ms','50ms','50−25'],paired_rows)}

## 入口窗口成员证据

以下窗口数目按入口顺序排列，成员对照包含每窗请求及稳定排序后的顺序。

{table(['轮次','对照','左：25ms或只排序','右：50ms或组合','成员顺序'],membership_rows)}

正式排序请求窗口{len(windows)}个，其中跨客户端批次{sum('/' in w['batches'] for w in windows)}个。全部整窗注册在该窗首次调度之前完成；全部窗口无提前发布。5ms定义为收集区间，不保证API预处理、引擎忙碌时第5ms就执行。请求体或tokenization迟到保留其原窗，不与下一窗合并。客户端每批5条也不强制每个入口窗口5条：若实际HTTP到达跨度超过5ms就应拆窗。

入口→准备就绪、准备→EngineCore注册、deadline→注册、注册→首次调度，下表为全部请求均值：

{table(['间隔','方案','入口→准备','准备→注册','deadline→注册','注册→调度'],window_stage_rows)}

同一客户端批次的5条请求在各阶段的实际时间跨度，单元格为9个批次（三轮×三批）的均值/最大值，单位ms：

{table(['间隔','方案','客户端发送跨度','Router完整body跨度','P发送headers跨度','P入口跨度'],arrival_rows)}

客户端同时创建5个任务并不等于P入口5ms内收到5条。Router逐请求解析和转发JSON、连接与API处理都可能拉开实际到达时刻；上表分阶段跨度用于定位时间扩展发生位置，不把全部扩展武断归给某一个函数。排窗以P入口为准，不能按客户端batch标签强制合并。

## Short连续阶段拆分

R为常规扫描首次观察到CUDA event就绪，E为完整发送条件就绪；R含前序chunk、注册与轮询条件，不能当作CUDA实际完成时刻。各段首尾连续，总和约等于TTFT，单位ms。

{table(['间隔','方案','发送→首次调度','调度→final事件记录','记录→R','R→E','E→WRITE','WRITE→完成','完成→首文本','TTFT'],stage_rows)}

baseline为整请求路径，使用对应的连续拆分：

{table(['间隔','发送→调度','调度→发布','发布→WRITE','WRITE→完成','完成→首文本','TTFT'],native_rows)}

Short按客户端原始批次拆分，可定位后两批是否主要等待前面的长请求；调度列是首次P调度，并非GPU真实开始执行：

{table(['间隔','方案','客户端批次','发送→首次调度','TTFT'],batch_rows)}

将客户端计时起点后移与真正完成时刻移动区分：下表“发送偏移变化”是相对首条发送的类别平均偏移；“首文本时刻变化”=TTFT变化+发送偏移变化。若绝对处理不变，本负载50ms通常仅因计时起点后移就可降低平均TTFT约25ms，但真实偏移以采集值为准。

{table(['方案','类别','发送偏移变化','TTFT变化','首文本时刻变化'],origin_rows)}

## 传输仲裁证据

仅统计正式回放，次数为chunk边界选择或候选扫描，不能当作中断DMA的次数。保护长请求时允许其先于short。

{table(['间隔','方案','short越过long','受保护long先于short','存在short候选的决策'],choice_rows)}

## 校验与文件

流式协议覆盖{audit['trace_requests']}条含预热的请求、{audit['trace_chunks']}对WRITE/DONE，单槽最大active={audit['max_active']}，连续覆盖和final通知一致。48次正式/预热回放均通过NVLink计数校验，每次填充KV负载{nvlink['expected_payload_bytes_per_load']:,}bytes，P Tx/D Rx与预期在0.1%内。

原始请求和SSE在`runs/`，服务及预热日志在`logs_round*/`，逐PID trace在`trace/`，源码版本和SHA256在`source/`和`source_sha256.json`。`pipeline_requests.csv`为逐请求阶段，`pipeline_windows.csv`为真实窗口，`pipeline_runs.csv`为逐轮TTFT，`writer_choices.csv`记录仲裁候选。配置见`experiment_manifest.json`，workload SHA256 `{manifest['workload_sha256']}`。

本轮属于修复后的入口窗口实现，与此前调度器入窗的C8/C16/C32结果不能混为同一实现的重复实验。三轮结果为描述性证据，阶段差异帮助定位此次观测，不单靠相关性断言普遍因果。
'''
    (root/'REPORT.md').write_text(text)
    (root/'ttft_summary.json').write_text(json.dumps(summary,indent=2))
    print(table(['间隔','方案','Short','Medium','Long','全部'],summary_rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    report(p.parse_args().root)
