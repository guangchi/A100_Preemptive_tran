#!/usr/bin/env python3
"""Analyze the reversed interval experiment without treating ASGI as NIC arrival."""
import csv
import json
from pathlib import Path
import statistics
import sys

from analyze_pd_pipeline_trace import analyze, write_csv


def main(root):
    analyze(root)
    # Even-numbered log directories alias the preceding service, rather than
    # representing another warmup. Count each physical counter capture once.
    nvpath = root / 'nvlink_validation.json'
    nv = json.loads(nvpath.read_text())
    seen, unique = set(), []
    for capture in nv['loads']:
        path = (root / capture['path']).resolve()
        if path not in seen:
            seen.add(path)
            unique.append(capture)
    nv['loads'] = unique
    nvpath.write_text(json.dumps(nv, indent=2))
    events = [json.loads(line) for p in (root / 'trace').glob('*.jsonl')
              for line in p.read_text().splitlines()]
    shared = {}
    for e in events:
        if e.get('shared') and e['role'] == 'P':
            shared.setdefault(e['shared'], []).append(e)
    requests = list(csv.DictReader((root / 'pipeline_requests.csv').open()))
    detail = []
    for r in requests:
        es = shared[r['shared']]
        row = {k: r[k] for k in ('round','scenario','replay','rid','category','ttft_ms','window','rank')}
        row.update(startup=(int(r['round'])+1)//2, pair=1 if int(r['round'])%2 else 2)
        for k in ('client','p_headers','api','scheduled','first_token'):
            row[k+'_ns'] = int(r[k+'_ns'])
        for event, label in [('http_protocol_headers','protocol'),('http_asgi_task_start','asgi_task'),
                             ('api_ingress_forward','forward'),('api_enter','inner_api')]:
            found = [e for e in es if e['event'] == event]
            if len(found) != 1:
                raise ValueError((row, event, len(found)))
            row[label+'_ns'] = found[0]['t_ns']
        for e in es:
            if e['event'] in ('api_sync_span','api_read_body_span'):
                label = e.get('phase', 'read_body')
                row[label+'_ms'] = (e['end_ns']-e['start_ns'])/1e6
                if 'cpu_ns' in e:
                    row[label+'_cpu_ms'] = e['cpu_ns']/1e6
        for a,b in [('p_headers','protocol'),('protocol','asgi_task'),('asgi_task','api'),
                    ('api','forward'),('forward','scheduled'),('client','scheduled')]:
            row[a+'_to_'+b+'_ms'] = (row[b+'_ns']-row[a+'_ns'])/1e6
        row['pid'] = es[0]['pid']
        detail.append(row)
    write_csv(root/'api_request_timing.csv', detail)
    pairs, stats = [], []
    progress = json.loads((root/'progress.json').read_text())
    for run in progress:
        subset = [r for r in detail if int(r['round']) == run['round'] and r['scenario'] == run['scenario']]
        a = next(r for r in subset if r['rid'].startswith('05-'))
        b = next(r for r in subset if r['rid'].startswith('06-'))
        row = {k: a[k] for k in ('round','scenario','startup','pair','replay')}
        row.update(same_window=a['window']==b['window'], long_window=a['window'], short_window=b['window'],
                   short06_ttft_ms=float(b['ttft_ms']), long05_ttft_ms=float(a['ttft_ms']))
        for name in ('client','p_headers','protocol','asgi_task','api','forward','scheduled'):
            row[name+'_gap_ms'] = (b[name+'_ns']-a[name+'_ns'])/1e6
        for k,v in a.items():
            if k.endswith('_ms') and not k.startswith('ttft'):
                row['long05_'+k] = v
        for k in ('p_headers_to_protocol_ms','protocol_to_asgi_task_ms','api_to_forward_ms','client_to_scheduled_ms'):
            row['short06_'+k] = b[k]
        gaps = [e for e in events if e['event']=='api_loop_gap' and e['pid']==a['pid']
                and e['start_ns'] < b['api_ns'] and e['end_ns'] > a['api_ns']]
        row['overlapping_loop_gaps'] = len(gaps)
        row['max_loop_gap_ms'] = max((e['elapsed_ns']/1e6 for e in gaps), default=0)
        row['max_loop_gap_cpu_ms'] = max((e['cpu_ns']/1e6 for e in gaps), default=0)
        pairs.append(row)
        stats_row = {k: row[k] for k in ('round','scenario','startup','pair','replay')}
        for cat in ('short','medium','long','all'):
            sample = [float(r['ttft_ms']) for r in subset if cat=='all' or r['category']==cat]
            stats_row[cat+'_ttft_ms'] = statistics.mean(sample)
        stats.append(stats_row)
    write_csv(root/'order_ttft.csv',stats)
    write_csv(root/'boundary_05_06.csv',pairs)
    aggregate = []
    for gap in ('50ms','25ms'):
        group = [r for r in stats if r['scenario']==gap]
        aggregate.append(dict(scenario=gap, runs=len(group), **{
            cat:dict(mean=statistics.mean(r[cat+'_ttft_ms'] for r in group),
                     stdev=statistics.stdev(r[cat+'_ttft_ms'] for r in group))
            for cat in ('short','medium','long','all')}))
    (root/'order_summary.json').write_text(json.dumps(aggregate,indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
