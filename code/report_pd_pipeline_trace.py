#!/usr/bin/env python3
"""Render the completed admission diagnostic from validated per-request data."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics as st

SCHEMES = ['baseline', 'sort_only', 'preempt_only', 'new']
LABELS = dict(baseline='baseline', sort_only='只排序', preempt_only='只抢占', new='排序+抢占')


def report(root):
    load = lambda name: list(csv.DictReader((root/name).open()))
    requests, runs = load('pipeline_requests.csv'), load('pipeline_runs.csv')
    audit = json.loads((root/'streaming_audit.json').read_text())
    pipeline = json.loads((root/'pipeline_validation.json').read_text())
    nvlink = json.loads((root/'nvlink_validation.json').read_text())
    assert audit['complete_formal_experiment'] and pipeline['checks_passed'] and nvlink['all_passed']
    assert len(runs) == 24 and len(requests) == 360 and len(nvlink['loads']) == 48

    def subset(scheme, scenario, rep=None, category='short'):
        return [r for r in requests if r['scheme'] == scheme and r['scenario'] == scenario
                and (rep is None or r['round'] == str(rep))
                and (category == 'all' or r['category'] == category)]

    def mean(scheme, scenario, metric, rep=3):
        return st.mean(float(r[metric]) for r in subset(scheme, scenario, rep))

    def windows(scheme, scenario, rep):
        grouped = defaultdict(list)
        for row in subset(scheme, scenario, rep, 'all'):
            grouped[row['release_ns']].append(row)
        return [tuple(r['rid'] for r in sorted(grouped[key], key=lambda r: int(r['rank'])))
                for key in sorted(grouped, key=int)]

    def pattern(scheme, scenario, rep):
        return '+'.join(str(len(w)) for w in windows(scheme, scenario, rep))

    def table(headers, rows):
        return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|'] +
                         ['|'+'|'.join(map(str,row))+'|' for row in rows])

    ttft_rows = []
    for scenario in ['25ms', '50ms']:
        for scheme in SCHEMES:
            reps = [r for r in runs if r['scheme'] == scheme and r['scenario'] == scenario]
            cells = []
            for category in ['short', 'medium', 'long', 'all']:
                values = [float(r[category+'_ttft_ms']) for r in reps]
                cells.append(f'{st.mean(values):.1f}±{st.stdev(values):.1f}')
            ttft_rows.append([scenario, LABELS[scheme], *cells])
    third_rows = [[scenario, LABELS[scheme], *[
        f"{st.mean(float(r['ttft_ms']) for r in subset(scheme, scenario, 3, category)):.1f}"
        for category in ['short','medium','long','all']]]
        for scenario in ['25ms','50ms'] for scheme in SCHEMES]
    window_rows = []
    for rep in [1,2,3]:
        for scenario in ['25ms','50ms']:
            sort, both = windows('sort_only',scenario,rep), windows('new',scenario,rep)
            sm = mean('sort_only',scenario,'ttft_ms',rep)
            nm = mean('new',scenario,'ttft_ms',rep)
            window_rows.append([rep,scenario,pattern('sort_only',scenario,rep),pattern('new',scenario,rep),
                                '是' if sort == both else '否',f'{sm:.1f}',f'{nm:.1f}',f'{nm-sm:+.1f}'])
    stage_rows = []
    for scenario in ['25ms','50ms']:
        for scheme in ['sort_only','new','preempt_only']:
            stage_rows.append([scenario,LABELS[scheme],*[f'{mean(scheme,scenario,k):.2f}' for k in
                ['client_to_schedule_ms','schedule_to_frontier_ms','frontier_to_ready_observed_ms',
                 'ready_to_eligible_observed_ms','eligible_to_submit_ms','final_submit_to_done_ms','done_to_first_ms','ttft_ms']]])
    common_rows = []
    for scenario in ['25ms','50ms']:
        for scheme in ['baseline','preempt_only']:
            if scheme == 'baseline':
                middle = sum(mean(scheme,scenario,k) for k in ['schedule_to_native_publish_ms','native_publish_to_write_ms','native_write_to_done_ms'])
                tail = mean(scheme,scenario,'native_done_to_first_ms')
            else:
                middle = mean(scheme,scenario,'schedule_to_frontier_ms') + mean(scheme,scenario,'frontier_to_done_ms')
                tail = mean(scheme,scenario,'done_to_first_ms')
            common_rows.append([scenario,LABELS[scheme],f'{mean(scheme,scenario,"client_to_schedule_ms"):.2f}',
                                f'{middle:.2f}',f'{tail:.2f}',f'{mean(scheme,scenario,"ttft_ms"):.2f}'])
    choice_rows = [[r['round'],r['scenario'],LABELS[r['scheme']],r['short_over_long'],r['protected_long_over_short']]
                   for r in load('writer_choice_summary.csv')]
    lateness = [(int(r['release_ns'])-int(r['deadline_ns']))/1e6 for r in requests if r['deadline_ns']]
    win_stats = load('pipeline_windows.csv')
    new50, sort50 = mean('new','50ms','ttft_ms'), mean('sort_only','50ms','ttft_ms')
    eligible_delta = mean('new','50ms','eligible_to_submit_ms')-mean('sort_only','50ms','eligible_to_submit_ms')
    frontier_delta = mean('new','50ms','frontier_to_ready_observed_ms')-mean('sort_only','50ms','frontier_to_ready_observed_ms')
    same50 = windows('new','50ms',3) == windows('sort_only','50ms',3)
    text = f'''# C16 入窗时序诊断：GPU5/6 起跑、GPU2/3 补齐

原计划 **24/24 次正式运行全部完成**，正式360请求、预热360请求均通过逐token输出、SSE/DONE、usage和时间校验。新增的后12次全部使用物理GPU2(P)/GPU3(D)。本报告替代此前12/24次的阶段报告；原报告、数据和校验值保留在 `checkpoints/before_gpu2-3_20261116-160435/`。

实验证实的主要原因是：**5ms窗口在调度器接纳请求后才开始，EngineCore执行模型步期间积累的输入请求会集中出队。因此客户端相隔25/50ms的批次可能合入同一窗口，也可能切成不同窗口。** 窗口成员改变后，后续短请求能否排到长请求前面发生变化，引起远大于5ms的TTFT差异。传输侧另有就绪/发布等待和长请求保护，开启抢占并不保证所有短请求只快不慢。

## 配置与卡组

Llama-3.1-8B-Instruct BF16，TP=PP=1；block16，最大计算预算2048tokens，最大并发64，prefix caching关闭，batch invariant开启。固定负载15请求：8×128tokens、3×1107tokens、4×10000tokens；每条生成64tokens；分三批、每批5条。窗口5ms，C16长chunk（约32MiB）、单槽WRITE、0.1ms poll；保护阈值85%/3次绕过/50ms。workload SHA256为 `ad9d732ba5a55de1ac9e4c682a8171df38f0513de74c1d23251c2cbc2746e94a`。

|阶段|物理GPU|正式/预热负载数|所含服务组|
|---|---|---|---|
|中断前|5(P)/6(D)|12/12|第一轮四组；第二轮sort_only、preempt_only|
|本次补跑|2(P)/3(D)|12/12|第二轮new、baseline；第三轮全部四组|

两对卡均为A100-SXM4-80GB，卡间均为NV12；5/6属NUMA1，2/3属NUMA0。实际使用UUID绑定，逐回放卡号由NVLink快照和run元数据核对。GPU6外部训练与GPU7原有服务均未操作。因为更换卡组和主机时段，三轮均值是描述性统计；应同时查看同在2/3上完成的第三轮，不把跨卡组的小差距当作稳定净收益。

## TTFT结果

单位ms；三轮均值±样本标准差，每方案每间隔45条请求，short/medium/long分别24/9/12条。“全部”按8:3:4请求数加权，不是三种类别平均值的简单平均。

{table(['间隔','方案','Short','Medium','Long','全部'],ttft_rows)}

第三轮全部在GPU2/3，以下为每次回放内的均值（单轮，没有跨轮显著性含义）：

{table(['间隔','方案','Short','Medium','Long','全部'],third_rows)}

## 5ms为何还能合入相隔25/50ms的批次

所有采集使用同机 `monotonic_ns`，通过replay UUID→客户端request ID→Router shared ID→引擎内部ID关联。P API入口、完整body、EngineCore输入线程接收/入队、主循环出队/add、入窗/发布/首次调度均有独立时间戳；不以秒级日志筛选正式请求。

第一轮sort_only/25ms的14条请求，客户端发送跨度51.366ms、P API入口跨度64.141ms、引擎入队跨度55.262ms，但主循环出队跨度仅1.798ms，入窗跨度仅1.756ms。

|请求|客户端发送|P API入口|引擎入队|引擎出队|admit|
|---|---:|---:|---:|---:|---:|
|01-long-1501，批0|0.353|23.796|45.461|109.857|109.965|
|06-short-1506，批1|25.915|61.938|81.063|110.600|110.666|
|10-short-1510，批2|50.349|79.480|91.764|111.173|111.234|
|14-short-1514，批2|51.719|87.936|100.723|111.655|111.720|

上表单位ms，相对该回放首条客户端发送。此前一个EngineCore执行步占据主循环约37.05–109.70ms；首批长请求45.461ms已经入队，但等待64.396ms才出队。其中前一执行步剩余64.244ms，仅差0.152ms。模型执行步是主机侧执行/等待总区间，不能等同纯GPU kernel时间。

该窗口真正开始于109.965ms，deadline114.965ms，115.236ms发布，迟到仅0.271ms。客户端发出三批时，这个窗口尚未开始；并非把已打开的5ms窗口延长到50ms。全24次正式回放共{len(win_stats)}个窗口（含singleton），其中{sum('/' in w['batches'] for w in win_stats)}个跨客户端批次；有deadline的发布迟到按请求加权均值{st.mean(lateness):.3f}ms、最大{max(lateness):.3f}ms。

代码依据：vLLM `process_input_sockets` 在输入线程预处理后入队；主循环 `_process_input_queue` 集中出队并调用 `add_request`；`_process_engine_step` 运行 `step_fn`，执行路径可等待 `future.result()`。本项目 `LocalWindowBuffer.admit` 才设置 `now+5ms` deadline。独立deadline线程解决的是已入窗请求的到期发布，无法绕过EngineCore的接纳队列。

## 固定顺序为什么没有固定的排序结果

实际窗口成员数及Short TTFT如下。相同成员指窗口数量、窗口内请求及最终排序顺序均相同；“组合−排序”为负表示组合更快。

{table(['轮次','间隔','只排序窗口','组合窗口','成员顺序相同','排序Short','组合Short','组合−排序'],window_rows)}

六组“轮次×间隔”对照中，五组窗口成员顺序相同，这五组组合Short TTFT均更低（35.0–62.0ms）；唯一不同的是第一轮25ms，只排序把14条跨批次请求放在一起排序，而组合是1+4+10，造成只排序领先509.9ms。它足以拉低三轮25ms排序均值。尤其是同在GPU2/3上的第三轮，两场景都为1+4+10且成员相同，组合在25/50ms均更快。因此“25ms应当只排序最好、50ms才适合组合”不是固定规律；目前观察到的均值反转主要由入窗分组差异及其计算排队位置解释。第二轮两组来自不同卡组，不能把其35ms差值单独解释为仲裁净收益。

最直接的重复证据是sort_only/25ms：第一轮1+14条，Short TTFT789.8ms；第二轮1+9+5条，1909.4ms。末批3条short从发送到首次P调度均值由73.2ms变成2737.9ms，TTFT由857.5ms变成3816.5ms。它们进入后一窗口，无法越过前一窗口的长请求。这是计算排队位置的变化，不是多等了5ms。

客户端顺序/目标间隔相同，仍会受到连接建立、API处理、输入线程、模型执行步及主机线程唤醒时机影响。25/50ms改变到达时刻相对于模型执行步的相位，进而改变集中出队的分组。此前概括为“25ms拥堵”不准确；本实验不能推出50ms普遍更优。仅把发送间隔由25增至50ms、假设处理绝对时刻不变，本负载各类/全部平均TTFT就会因计时起点后移减少25ms；观察到的数百至上千ms差异远超过这一算术效应。

## 50ms组合更快：具体发生在哪一段

第三轮GPU2/3的50ms场景，排序组窗口为{pattern('sort_only','50ms',3)}，组合为{pattern('new','50ms',3)}，成员顺序{'相同' if same50 else '不同'}。Short TTFT分别{sort50:.2f}/{new50:.2f}ms，组合相对排序变化{new50-sort50:+.2f}ms。

下面为同卡第三轮的连续阶段均值。`R` 是既有仲裁扫描首次观察到CUDA event完成（且前序chunk/注册条件允许查询）；`E` 是首次观察到完整发送条件满足。**R不是CUDA真实完成时刻**，采集不增加CUDA查询。`完成`为P端观察到final WRITE完成；其后的D处理/输出/网络仍计入TTFT。

{table(['间隔','方案','发送→调度','调度→final事件记录','记录→R','R→E','E→提交','提交→完成','完成→首文本','TTFT'],stage_rows)}

50ms组合−排序的E→提交变化为{eligible_delta:+.2f}ms，记录→R变化为{frontier_delta:+.2f}ms。可发送之后的排队是可以直接观察的仲裁影响区间；记录→R仍混合CUDA未完成、前序chunk、槽位占用和扫描时机等条件，不能全部归给抢占。第一轮5/6上的相同窗口对照中，组合快62.04ms，主要落在final事件记录到完成观察的64.05ms缩短；本次细分避免将那64ms直接称为纯抢占收益。

组合的作用有前提：排序先让short更早计算，产生可传的KV；仲裁才有机会让这些chunk越过long。若窗口分组不同，计算排队变化通常比仲裁变化更大，不能把总TTFT差值都解释为两个开关的协同收益。

## 为什么只抢占仍可能慢于baseline

第三轮同卡比较，25/50ms中只抢占比baseline的Short TTFT分别高{mean('preempt_only','25ms','ttft_ms')-mean('baseline','25ms','ttft_ms'):.2f}/{mean('preempt_only','50ms','ttft_ms')-mean('baseline','50ms','ttft_ms'):.2f}ms。分解共同可比较的阶段：

{table(['间隔','方案','发送→首次P调度','调度→P观察传输完成','观察完成→首文本','TTFT'],common_rows)}

只抢占路径没有P窗口排序，short仍可能在long计算后面；仲裁无法传输尚未就绪的KV。此外baseline是原生整请求NIXL，其他三组使用流式分块、保留最后block、post-forward CUDA event和单槽writer，**不是在完全相同执行路径上免费增加一个插队开关**。

代码显示，流式final块由`push_finished_blocks`转成chunk，再由`wait_for_save`后的`mark_stream_chunks_ready`记录当前流上的CUDA event；writer必须满足前序offset、注册、event、握手及槽位条件。原生baseline直接把整请求finished blocks交给writer。因而需要检查的是完整计算/发布/就绪路径，而非假设抢占收益一定大于所有额外等待。只抢占第三轮25/50ms的final E→提交均值仅{mean('preempt_only','25ms','eligible_to_submit_ms'):.2f}/{mean('preempt_only','50ms','eligible_to_submit_ms'):.2f}ms，而记录→R为{mean('preempt_only','25ms','frontier_to_ready_observed_ms'):.2f}/{mean('preempt_only','50ms','frontier_to_ready_observed_ms'):.2f}ms，说明单看插队阶段不足以解释端到端差距。

这些是主机观察区间；原生baseline的WRITE入口→完成观察还含其完成轮询时机，流式路径则主动推进handle完成。不能把这些区间当作纯DMA耗时。本轮没有额外做stream_fifo消融，也没有捕获CUDA真实完成时刻，因此不把baseline差值全判给“抢占算法”或“CUDA门控”中的单一因素。

## 保护规则使short不保证永远优先

实际仲裁先选受保护的long；只有没有受保护long、且首个候选是long时，才优先选short。long达到85%进度、累计3次绕过或50ms等待后会受保护；medium也不是short优先规则的被绕过对象。

新采集记录了完整候选与最终选择。下表统计每次决策，并非中断DMA次数；“保护long越过short”指当时首个可发送候选为short，但选择了后面的受保护long。

{table(['轮次','间隔','方案','short越过long决策','保护long越过short决策'],choice_rows)}

这直接反驳“开启后short只会更快”的前提：现有防饥饿策略确实可能让已可发送short等待long。仲裁选择会改变后续队列，不能把某次候选集上的FIFO选择当成完整系统的反事实回放。

## 验证与后续设计依据

- 24正式+24预热，共720条请求输出通过；360条正式请求关键时间线齐全有序。
- {audit['trace_requests']}条流式请求、{audit['trace_chunks']}次WRITE/DONE，offset覆盖、唯一final、P/D通知及最大active=1通过独立审计。
- 48/48次NVLink计数匹配每负载5,817,499,648bytes填充KV，容差0.1%；前24次GPU5TX/6RX，后24次GPU2TX/3RX。
- 采集可关闭，无共享安装vLLM改动，无算法/参数修复混入本实验；记录本身仍有观察开销，未做严格的有/无采集配对。
- 如果目标是让5ms代表真正P入口的到达窗口，需要在输入阶段记录单调到达时间/窗口边界，并在主循环保留该分组；只加独立发布线程不能固定入窗分组。需要同时保持主循环对Request/KV状态的所有权，不能直接让输入线程并发操作调度器。
- 若目标是隔离仲裁收益，应在同一流式FIFO路径、相同窗口成员下切换抢占，再结合event-ready和保护状态；本轮按用户要求补齐原有四组，没有擅自改动排序/保护策略。

历史WRITE分母更正仍有效：C8主实验8231次覆盖360条流式请求，C16/C32总量6739/3950覆盖540条并额外包含new。共同sort_only/preempt_only、同为360条时应比较8231/4492/2630次。

## 原始数据

[逐回放结果](pipeline_runs.csv)、[逐请求时间线](pipeline_requests.csv)、[窗口成员](pipeline_windows.csv)、[按卡组汇总](pipeline_by_gpu_pair.csv)、[完整分组均值](pipeline_aggregate.csv)、[仲裁选择](writer_choices.csv)、[仲裁选择汇总](writer_choice_summary.csv)、[正式writer日志统计](writer_formal.csv)。

[完整运行配置](experiment_summary.json)、[输出/协议审计](streaming_audit.json)、[时间线审计](pipeline_validation.json)、[NVLink审计](nvlink_validation.json)、[原阶段资源中断](resource_interruption.json)、[GPU2/3恢复配置](gpu2-3_resume_manifest.json)。原始JSONL在`trace/`，实际分阶段源码分别保存在`source/`、`source_resume_gpu2-3/`、`source_native_trace_gpu2-3/`；最终分析工具在`source_analysis_final/`。所有正式请求用唯一ID关联，预热不混入正式统计。
'''
    (root/'REPORT.md').write_text(text)
    (root/'RESULTS.md').write_text('# C16 入窗时序诊断完成\n\n24/24正式运行，360正式+360预热请求通过校验。前12次使用GPU5/6，后12次在GPU2/3补齐，第三轮四组均在GPU2/3。数据、原因与测量边界见[完整报告](REPORT.md)。\n')
    print(root/'REPORT.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    report(parser.parse_args().root)
