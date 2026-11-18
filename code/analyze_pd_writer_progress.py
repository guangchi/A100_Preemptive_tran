#!/usr/bin/env python3
"""Describe software-observed WRITE timing, including warmups."""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def describe(values):
    values = sorted(values)
    return dict(count=len(values), mean=statistics.mean(values),
                median=statistics.median(values), p95=values[int((len(values)-1)*.95)],
                maximum=max(values))


def analyze(root):
    groups = defaultdict(lambda: defaultdict(list))
    for path in root.glob('logs_round*/p_*.log'):
        scheme = path.stem.removeprefix('p_')
        for line in path.read_text(errors='replace').splitlines():
            if 'PUSH_CHUNK_WRITE ' in line:
                fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
                if 'submit_ms' in fields:
                    group = groups[scheme + ('_long' if int(fields['total']) > 512 else '_short_medium')]
                    group['submit_ms'].append(float(fields['submit_ms']))
                    group['queue_ms'].append(float(fields['queue_ms']))
            if 'PUSH_CHUNK_DONE ' in line:
                fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
                if 'observed_ms' in fields:
                    group = groups[scheme + ('_final' if fields['final'] == '1' else '_partial')]
                    group['observed_ms'].append(float(fields['observed_ms']))
    result = {key: {metric: describe(values) for metric, values in metrics.items()}
              for key, metrics in groups.items()}
    result['interpretation'] = ('Includes warmup. submit_ms is Python WRITE setup/submission wall time. '
        'observed_ms includes setup, transport, polling and completion-report delay; it is not pure DMA time. '
        'queue_ms includes source readiness, registration and preceding chunks, not only arbitration.')
    (root / 'writer_timings.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=Path)
    print(json.dumps(analyze(parser.parse_args().result_dir), indent=2))
