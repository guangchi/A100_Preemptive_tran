#!/usr/bin/env python3
"""Measure actual atomic window publication separately from scheduler readiness."""
import argparse
import json
import math
import re
import statistics
from pathlib import Path

WINDOW = re.compile(
    r'PD local admission: window=(\d+) rank=(\d+) req=(\S+) tokens=(\d+) '
    r'admitted=([\d.]+) deadline=([\d.]+) released=([\d.]+) wait_ms=([\d.]+) '
    r'late_ms=([\d.]+) release_source=(timer|engine)')
EARLY = re.compile(r'PD local admission: singleton released early req=(\S+)')
SCHEDULED = re.compile(r'PD admission scheduled: req=(\S+) released=([\d.]+) '
                      r'scheduled=([\d.]+) ready_queue_ms=([\d.]+)')


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return dict(count=len(values), mean=statistics.mean(values),
                p50=values[math.ceil(len(values)*.5)-1],
                p95=values[math.ceil(len(values)*.95)-1], max=values[-1])


def analyze(root, expected_requests=None, scheme='new'):
    windows, rows, ready, errors = [], [], [], []
    early_count = released_count = 0
    for path in sorted(root.glob(f'logs_round*/p_{scheme}.log')):
        batches, releases, scheduled = {}, set(), set()
        for line in path.read_text(errors='replace').splitlines():
            if m := WINDOW.search(line):
                win, rank, rid, tokens, admitted, deadline, released, wait, late, source = m.groups()
                if rid in releases:
                    errors.append(f'duplicate release {rid}')
                releases.add(rid)
                row = dict(log=str(path.relative_to(root)), window=int(win), rank=int(rank),
                           request_id=rid, tokens=int(tokens), admitted=float(admitted),
                           deadline=float(deadline), released=float(released),
                           wait_ms=float(wait), late_ms=float(late), source=source)
                rows.append(row)
                batches.setdefault(int(win), []).append(row)
                if abs((row['released']-row['deadline'])*1000-row['late_ms']) > .003:
                    errors.append(f'inconsistent lateness {rid}')
                if row['released'] < row['deadline']:
                    errors.append(f'premature normal release {rid}')
                if row['admitted'] >= row['deadline']:
                    errors.append(f'member admitted after deadline {rid}')
            elif m := EARLY.search(line):
                rid = m.group(1)
                if rid in releases:
                    errors.append(f'duplicate early release {rid}')
                releases.add(rid)
                early_count += 1
            elif m := SCHEDULED.search(line):
                rid, released, observed, wait = m.groups()
                if rid in scheduled or rid not in releases:
                    errors.append(f'duplicate/unreleased scheduled request {rid}')
                scheduled.add(rid)
                ready.append(float(wait))
                if float(observed) < float(released):
                    errors.append(f'negative ready queue wait {rid}')
        if releases != scheduled:
            errors.append(f'{path}: release/schedule sets differ')
        released_count += len(releases)
        last_release = last_deadline = -1
        for win, members in batches.items():
            if [r['rank'] for r in members] != list(range(len(members))):
                errors.append(f'{path}:{win} invalid ranks')
            if [(r['tokens'], r['admitted']) for r in members] != sorted((r['tokens'], r['admitted']) for r in members):
                errors.append(f'{path}:{win} unstable prompt ordering')
            if len({r['deadline'] for r in members}) != 1 or len({r['released'] for r in members}) != 1:
                errors.append(f'{path}:{win} non-atomic window metadata')
            head = members[0]
            if head['deadline'] < last_deadline or head['released'] < last_release:
                errors.append(f'{path}:{win} inter-window ordering')
            last_deadline, last_release = head['deadline'], head['released']
            windows.append(dict(log=str(path.relative_to(root)), window=win,
                                requests=len(members), late_ms=head['late_ms'], source=head['source']))
    if expected_requests is not None and released_count != expected_requests:
        errors.append(f'expected {expected_requests} new requests, found {released_count}')
    result = dict(passed=bool(rows) and not errors, errors=errors,
                  released_requests=released_count, singleton_early_releases=early_count,
                  normal_windows=len(windows), normal_window_requests=len(rows),
                  timer_windows=sum(w['source']=='timer' for w in windows),
                  engine_windows=sum(w['source']=='engine' for w in windows),
                  request_wait_ms=distribution([r['wait_ms'] for r in rows]),
                  request_lateness_ms=distribution([r['late_ms'] for r in rows]),
                  window_lateness_ms=distribution([w['late_ms'] for w in windows]),
                  ready_queue_ms=distribution(ready), windows=windows, requests=rows,
                  scope='Includes warmup. Deadlines start at engine admission, not HTTP arrival. '
                        'Publication means atomic append into actual scheduler waiting queue. '
                        'ready_queue_ms ends at first schedule completion observation; it is not GPU kernel start. '
                        'p95 uses nearest rank; Python/OS timers provide no hard realtime guarantee.')
    (root/'deadline_analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    if not result['passed']:
        raise ValueError(errors or 'No deadline trace')
    return {k:v for k,v in result.items() if k not in ('windows', 'requests')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--expected-requests', type=int)
    parser.add_argument('--scheme', default='new')
    args = parser.parse_args()
    print(json.dumps(analyze(args.result_dir, args.expected_requests, args.scheme), indent=2))
