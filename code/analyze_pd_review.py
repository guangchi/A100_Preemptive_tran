#!/usr/bin/env python3
"""Audit real admission windows and report each long request's measured cost."""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ADMISSION = re.compile(r'PD local admission: window=(\d+) rank=(\d+) req=(\S+) tokens=(\d+) '
                       r'admitted=([\d.]+) deadline=([\d.]+) released=([\d.]+) wait_ms=([\d.]+)')


def analyze(root):
    audit = json.loads((root/'streaming_audit.json').read_text())
    if not audit['complete_formal_experiment']:
        raise ValueError('Full independent audit required')
    errors, windows, waits = [], [], []
    singleton_count = 0
    for path in sorted(root.glob('logs_round*/p_new.log')):
        current = []
        previous_deadline = -1
        for line in path.read_text(errors='replace').splitlines():
            if match := ADMISSION.search(line):
                window, rank, rid, tokens, admitted, deadline, released, wait = match.groups()
                if int(rank) == 0:
                    if current:
                        windows.append(current)
                    current = []
                    if float(deadline) < previous_deadline - .001:
                        errors.append(f'{path.name}: window deadline went backwards')
                    previous_deadline = float(deadline)
                current.append(dict(log=str(path.relative_to(root)), rank=int(rank), request_id=rid,
                    tokens=int(tokens), admitted=float(admitted), deadline=float(deadline),
                    released=float(released), wait_ms=float(wait)))
                waits.append(float(wait))
            elif 'PD local admission: singleton released early' in line:
                singleton_count += 1
                if current:
                    windows.append(current)
                    current = []
        if current:
            windows.append(current)
    for window in windows:
        if [r['rank'] for r in window] != list(range(len(window))):
            errors.append('window ranks are not contiguous')
        if [r['tokens'] for r in window] != sorted(r['tokens'] for r in window):
            errors.append('window is not sorted by prompt tokens')
        if len({r['deadline'] for r in window}) != 1:
            errors.append('window members do not share deadline')
        if any(r['released'] < r['deadline'] - .0011 for r in window):
            errors.append('multi-request window released before deadline')
    groups = defaultdict(list)
    for path in sorted(root.glob('runs/*/requests.csv')):
        scheme, scenario, rep = path.parent.name.split('_')
        with path.open(newline='') as file:
            for row in csv.DictReader(file):
                if row['category'] == 'long':
                    groups[(scheme, scenario, row['request_id'])].append(row)
    long_rows = []
    for (scheme, scenario, rid), rows in sorted(groups.items()):
        data = dict(scheme=scheme, scenario=scenario, request_id=rid,
                    rounds=len(rows), ttft_ms=statistics.mean(float(r['ttft_ms']) for r in rows),
                    e2e_ms=statistics.mean(float(r['e2e_ms']) for r in rows))
        baseline = groups[('baseline', scenario, rid)]
        for metric in ['ttft_ms', 'e2e_ms']:
            base = statistics.mean(float(r[metric]) for r in baseline)
            data[metric+'_vs_baseline_percent'] = (data[metric]/base - 1)*100
        long_rows.append(data)
    with (root/'long_requests.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(long_rows[0])); writer.writeheader(); writer.writerows(long_rows)
    result = dict(admission_windows=len(windows), window_requests=len(waits), singleton_early_releases=singleton_count,
        wait_ms_mean=statistics.mean(waits) if waits else None, wait_ms_max=max(waits) if waits else None,
        errors=errors, passed=bool(windows) and not errors, windows=windows,
        scope='Admission logs include warmup. Rounded timestamps have 1ms resolution. Long rows aggregate three runs per request ID; no SLA or significance claim.')
    (root/'review_analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    if not result['passed']:
        raise ValueError(result['errors'])
    return {k: v for k, v in result.items() if k != 'windows'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=Path)
    print(json.dumps(analyze(parser.parse_args().result_dir), indent=2))
