#!/usr/bin/env python3
"""Sample the selected GPU pair and contexts belonging to the runner's tree."""
import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

import psutil


def monitor(root):
    devices = f"{os.environ.get('PD_P_GPU', '6')},{os.environ.get('PD_D_GPU', '7')}"
    runner = psutil.Process(int((root / 'runner_pid.txt').read_text()))
    known = set()
    with (root / 'gpu_telemetry.jsonl').open('w') as output:
        while runner.is_running() and runner.status() != psutil.STATUS_ZOMBIE:
            row = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            try:
                known.update(p.pid for p in runner.children(recursive=True))
                row['devices'] = subprocess.check_output([
                    'nvidia-smi', '-i', devices, '--query-gpu=index,uuid,memory.used,utilization.gpu',
                    '--format=csv,noheader'], text=True, timeout=10).splitlines()
                contexts = subprocess.check_output([
                    'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                    '--format=csv,noheader'], text=True, timeout=10).splitlines()
                row['experiment_contexts'] = [line for line in contexts
                    if line.split(',')[0].strip().isdigit()
                    and int(line.split(',')[0].strip()) in known]
            except psutil.NoSuchProcess:
                break
            except Exception as error:
                row['error'] = str(error)
            output.write(json.dumps(row) + '\n')
            output.flush()
            time.sleep(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=Path)
    monitor(parser.parse_args().result_dir)
