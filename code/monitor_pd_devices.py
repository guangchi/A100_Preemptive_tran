#!/usr/bin/env python3
"""Read-only NVML PID/GPU audit for one experiment process tree."""
import argparse,datetime,json,subprocess,time
from pathlib import Path
import psutil
p=argparse.ArgumentParser();p.add_argument('--result-dir',required=True);a=p.parse_args();root=Path(a.result_dir)
runner=int((root/'runner_pid.txt').read_text());known=set()
with (root/'gpu_contexts.jsonl').open('a') as f:
 while psutil.pid_exists(runner):
  try:
   known.update(c.pid for c in psutil.Process(runner).children(recursive=True))
   # Include already sampled workers, including a worker briefly reparented on exit.
   for line in (root/'gpu_telemetry.jsonl').read_text().splitlines():
    try:known.update(c['pid'] for c in json.loads(line).get('processes',[]))
    except json.JSONDecodeError:pass
   q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
   apps=[]
   for line in q.stdout.splitlines():
    parts=[s.strip() for s in line.split(',')]
    if len(parts)>=3 and parts[0].isdigit() and int(parts[0]) in known:
     apps.append(dict(pid=int(parts[0]),gpu_uuid=parts[1],memory_mib=parts[2]))
   f.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),experiment_contexts=apps))+'\n');f.flush()
  except psutil.NoSuchProcess:break
  time.sleep(2)
print('GPU context audit monitor finished')
