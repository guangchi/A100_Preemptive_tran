#!/usr/bin/env python3
"""Audit actual WRITE/completion traces, independently of config values."""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from validate_pd_run import validate_run

WRITE = re.compile(r'PUSH_CHUNK_WRITE req=(\S+) offset=(\d+) blocks=(\d+) total=(\d+) final=(\d+) active=(\d+) event_ready=1')
DONE = re.compile(r'PUSH_CHUNK_DONE req=(\S+) offset=(\d+) blocks=(\d+) final=(\d+)')


def audit(root):
    requests = []
    errors = []
    summary_path = root / 'experiment_summary.json'
    if not summary_path.exists():
        summary_path = root / 'experiment_partial_summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    config = summary.get('config', {})
    max_chunk_blocks = int(config.get('chunk_size_blocks', 8))
    for path in sorted(root.glob('logs_round*/p_*.log')):
        text = path.read_text(errors='replace')
        written, completed = defaultdict(list), defaultdict(list)
        active_chunks = {}
        for line in text.splitlines():
            if match := WRITE.search(line):
                rid, offset, blocks, total, final, active = match.groups()
                if active_chunks or int(active) != 1:
                    errors.append(f'{path.name}: WRITE before preceding DONE / invalid active count')
                active_chunks[(rid, int(offset))] = (int(blocks), int(final))
            elif match := DONE.search(line):
                rid, offset, blocks, final = match.groups()
                if active_chunks.pop((rid, int(offset)), None) != (int(blocks), int(final)):
                    errors.append(f'{path.name}: orphan/duplicate/out-of-order DONE')
        if active_chunks:
            errors.append(f'{path.name}: unreaped WRITE at end of trace')
        for match in WRITE.finditer(text):
            rid, offset, blocks, total, final, active = match.groups()
            written[rid].append(tuple(map(int, (offset, blocks, total, final, active))))
        for match in DONE.finditer(text):
            rid, offset, blocks, final = match.groups()
            completed[rid].append(tuple(map(int, (offset, blocks, final))))
        for rid, chunks in written.items():
            cursor = 0
            for offset, blocks, total, final, active in chunks:
                if (offset != cursor or blocks <= 0 or active != 1
                        or total != chunks[0][2]
                        or (total > 512 and blocks > max_chunk_blocks)):
                    errors.append(f'{path.name}:{rid}: invalid chunk {offset, blocks, total, final, active}; expected offset {cursor}')
                cursor = offset + blocks
            if cursor != chunks[-1][2] or sum(x[3] for x in chunks) != 1 or not chunks[-1][3]:
                errors.append(f'{path.name}:{rid}: incomplete/final coverage')
            expected = [(o, b, f) for o, b, t, f, a in chunks]
            if completed[rid] != expected:
                errors.append(f'{path.name}:{rid}: WRITE/DONE mismatch')
            requests.append(dict(log=str(path.relative_to(root)), request_id=rid,
                chunks=len(chunks), blocks=cursor, max_active=max(x[4] for x in chunks),
                final_count=sum(x[3] for x in chunks)))
        for line in text.splitlines():
            if any(x in line for x in ('preemptive-push-writer error', 'PUSH_CHUNK_SUBMISSION_FAILED',
                                      'Releasing expired KV', 'block slice mismatch')):
                errors.append(f'{path.name}:{line}')
    notifications = []
    for path in sorted(root.glob('logs_round*/d_*.log')):
        matches = re.findall(r"PUSH_FINAL_NOTIFICATION b'([^']+):1'", path.read_text(errors='replace'))
        counts = Counter(re.sub(r'-[0-9a-f]{8}$', '', rid) for rid in matches)
        if matches:
            notifications.append(dict(log=str(path.relative_to(root)), count=len(matches)))
            if any(n != 1 for n in counts.values()):
                errors.append(f'{path.name}: duplicate final notification')
    for path in sorted(root.glob('logs_round*/*_*.log')):
        if path.name.startswith(('p_', 'd_')):
            for line in path.read_text(errors='replace').splitlines():
                if any(marker in line for marker in ('NIXL transfer failure:', 'PUSH handshake failed',
                        'Unrecognized request ', 'preemptive-push-writer error')):
                    errors.append(f'{path.relative_to(root)}: {line}')
    # P and D randomize their internal suffix independently; compare base IDs.
    write_bases = Counter(re.sub(r'-[0-9a-f]{8}$', '', r['request_id']) for r in requests)
    recv_bases = Counter()
    for path in sorted(root.glob('logs_round*/d_*.log')):
        for rid in re.findall(r"PUSH_FINAL_NOTIFICATION b'([^']+):1'", path.read_text(errors='replace')):
            recv_bases[re.sub(r'-[0-9a-f]{8}$', '', rid)] += 1
    if write_bases != recv_bases:
        errors.append('producer request / consumer final-notification mismatch')
    csv_checks = []
    workload = [json.loads(line) for line in (root/'workload.jsonl').read_text().splitlines() if line.strip()]
    references = {r['request_id']: r for r in csv.DictReader((root/'reference_outputs_a100.csv').open())}
    # Only per-run inputs; root requests.csv is a generated aggregate.
    csv_paths = list((root / 'runs').glob('*/requests.csv'))
    csv_paths += list(root.glob('logs_round*/warmup_*.csv'))
    for path in sorted(csv_paths):
        check = validate_run(path, workload, references)
        csv_checks.append(dict(path=str(path.relative_to(root)), **check))
        errors.extend(f'{path.relative_to(root)}: {error}' for error in check['errors'])
    formal = [c for c in csv_checks if c['path'].startswith('runs/')]
    schemes = list(config.get('schemes', ['baseline', 'old', 'new']))
    gaps = list(config.get('batch_start_times_ms', {'25ms': [], '50ms': []}))
    rounds = range(1, summary.get('rounds', 3) + 1)
    expected_formal = {f'runs/{scheme}_{gap}_run{rep}/requests.csv'
        for scheme in schemes for gap in gaps for rep in rounds}
    expected_warmups = {f'logs_round{rep}_{scheme}/warmup_{gap}.csv'
        for scheme in schemes for gap in gaps for rep in rounds} if config.get('warmup', True) else set()
    complete = {c['path'] for c in csv_checks} == expected_formal | expected_warmups
    streaming_schemes = [s for s in schemes if s != 'baseline']
    expected_streaming = (len(streaming_schemes) * len(gaps) * len(rounds)
                          * len(workload) * (2 if config.get('warmup', True) else 1))
    if complete and (len(requests) != expected_streaming or sum(n['count'] for n in notifications) != expected_streaming):
        errors.append(f'expected {expected_streaming} streaming requests and final notifications including warmups')
    return dict(trace_requests=len(requests), trace_chunks=sum(r['chunks'] for r in requests),
                max_active=max((r['max_active'] for r in requests), default=0),
                formal_runs=len(formal), formal_requests=sum(c['requests'] for c in formal),
                csv_checks=csv_checks, trace_checks=requests, final_notifications=notifications, errors=errors,
                checks_passed=bool(requests) and not errors,
                complete_formal_experiment=complete and bool(requests) and not errors)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('result_dir', type=Path)
    args = p.parse_args()
    result = audit(args.result_dir)
    (args.result_dir / 'streaming_audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('csv_checks', 'trace_checks')}, indent=2))
    raise SystemExit(0 if result['checks_passed'] else 1)
