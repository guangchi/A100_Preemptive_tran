#!/usr/bin/env python3
"""Replay the fixed workload; measure first nonempty text SSE and exact IDs."""
from __future__ import annotations
import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import time
from pathlib import Path
import aiohttp
from pd_experiment_paths import DEFAULT_WORKLOAD, RESULTS_DIR, reference_path as default_reference_path
from pd_pipeline_trace import emit, aiohttp_trace

logger = logging.getLogger('pd_load')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def load_workload(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def classify(n):
    return 'short' if n < 256 else ('medium' if n <= 8192 else 'long')


async def send_request(session, router_url, req, batch_idx, reference=None):
    ids = req.get('prompt_token_ids') or req.get('token_ids') or []
    started = time.perf_counter()
    replay = os.environ.get("PD_TRACE_REPLAY", "")
    context = dict(replay=replay, rid=req["request_id"], batch=batch_idx, stage="client")
    emit("client_send", **context)
    result = dict(request_id=req['request_id'], category=classify(len(ids)),
                  num_prompt_tokens=len(ids), num_output_tokens=0, batch_idx=batch_idx,
                  replay_id=replay, submitted_mono_ns=time.monotonic_ns(),
                  submitted_at=started, completed_at=0.0, ttft_ms=-1.0, e2e_ms=-1.0,
                  token_ids_hash='', output_token_ids='[]', output_text='',
                  reference_match=False, finish_reason='', sse_done_received=False, success=False, error='')
    events, output_ids, texts = [], [], []
    done = False
    usage = {}
    prompt_count = None
    body = dict(model='llama3-p', prompt_token_ids=ids,
                max_tokens=req.get('max_tokens', 64), temperature=0.0,
                seed=req.get('seed', 0), stream=True, return_token_ids=True,
                request_id=req['request_id'])
    try:
        async with session.post(router_url+'/v1/chat/completions', json=body,
                                headers={"X-PD-Replay": replay, "X-PD-Batch": str(batch_idx)},
                                trace_request_ctx=context,
                                timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status != 200:
                raise RuntimeError(f'HTTP {resp.status}: {(await resp.text())[:500]}')
            # StreamReader iteration yields complete lines even if TCP splits an SSE.
            async for raw_line in resp.content:
                elapsed = (time.perf_counter()-started)*1000
                line = raw_line.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    events.append(dict(elapsed_ms=elapsed, payload='[DONE]'))
                    done = True
                    result['sse_done_received'] = True
                    continue
                obj = json.loads(data)
                events.append(dict(elapsed_ms=elapsed, payload=obj))
                if obj.get('error'):
                    raise RuntimeError(str(obj['error']))
                if obj.get('usage'):
                    usage = obj['usage']
                for ch in obj.get('choices', []):
                    text = ch.get('text') or (ch.get('delta') or {}).get('content') or ''
                    if text and result['ttft_ms'] < 0:
                        result['ttft_ms'] = round(elapsed, 3)
                        emit('client_first_token', **context)
                    texts.append(text)
                    output_ids.extend(ch.get('token_ids') or [])
                    if ch.get('prompt_token_ids') is not None:
                        prompt_count = len(ch['prompt_token_ids'])
                    if ch.get('finish_reason'):
                        result['finish_reason'] = ch['finish_reason']
        expected = json.loads(reference['output_token_ids']) if reference else None
        result['reference_match'] = output_ids == expected if expected is not None else False
        result['num_output_tokens'] = len(output_ids)
        result['output_token_ids'] = json.dumps(output_ids)
        result['token_ids_hash'] = hashlib.sha256(json.dumps(output_ids).encode()).hexdigest()[:16]
        result['output_text'] = ''.join(texts)
        errors = []
        if not done: errors.append('missing DONE')
        if result['ttft_ms'] < 0: errors.append('no nonempty text SSE')
        if len(output_ids) != body['max_tokens']: errors.append(f'output count {len(output_ids)}')
        if result['finish_reason'] != 'length': errors.append(f"finish={result['finish_reason']}")
        if prompt_count != len(ids): errors.append(f'prompt count {prompt_count}')
        if usage.get('completion_tokens') != len(output_ids): errors.append(f'usage={usage}')
        if not result['reference_match']: errors.append('reference token IDs mismatch')
        result['error'] = '; '.join(errors)
        result['success'] = not errors
    except Exception as exc:
        result['error'] = str(exc)
    result['completed_at'] = time.perf_counter()
    result['e2e_ms'] = round((result['completed_at']-started)*1000, 3)
    logger.info('%s %s TTFT=%.1f E2E=%.1f tokens=%d match=%s error=%s',
                result['request_id'], result['category'], result['ttft_ms'],
                result['e2e_ms'], result['num_output_tokens'],
                result['reference_match'], result['error'])
    return result, events


async def run_experiment(workload, router_url, arrival_gaps_ms, output_path,
                         group_size=5, reference_path=None):
    with Path(reference_path or default_reference_path()).open(newline='') as file:
        references = {r['request_id']: r for r in csv.DictReader(file)}
    for req in workload:
        if req['request_id'] not in references:
            raise ValueError(f"Missing reference: {req['request_id']}")
    start = time.perf_counter()
    tasks = []
    async with aiohttp.ClientSession(trace_configs=aiohttp_trace()) as session:
        for i in range(0, len(workload), group_size):
            batch = i//group_size
            offset = arrival_gaps_ms[min(batch, len(arrival_gaps_ms)-1)]
            await asyncio.sleep(max(0, start+offset/1000-time.perf_counter()))
            emit("client_batch", replay=os.environ.get("PD_TRACE_REPLAY", ""),
                 batch=batch, target_ns=int((start+offset/1000)*1e9), offset_ms=offset)
            tasks.extend(asyncio.create_task(send_request(
                session, router_url, req, batch, references[req['request_id']]))
                for req in workload[i:i+group_size])
        outputs = await asyncio.gather(*tasks)
    results = [row for row, events in outputs]
    makespan = (max(r['completed_at'] for r in results)-min(r['submitted_at'] for r in results))*1000
    for row in results:
        row['submitted_at_rel_ms'] = round((row['submitted_at']-start)*1000, 3)
        row['submitted_at'] = round(row['submitted_at']-start, 6)
        row['completed_at'] = round(row['completed_at']-start, 6)
        row['makespan_ms'] = round(makespan, 3)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with path.with_suffix('.sse.jsonl').open('w') as f:
        for row, events in outputs:
            f.write(json.dumps(dict(request_id=row['request_id'], events=events))+'\n')
    logger.info('Saved %s; %d/%d valid, makespan=%.3fms', path,
                sum(r['success'] for r in results), len(results), makespan)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workload', default=DEFAULT_WORKLOAD)
    p.add_argument('--router-url', default='http://127.0.0.1:8000')
    p.add_argument('--arrival-gaps-ms', nargs='+', type=int, default=[0,25,50])
    p.add_argument('--group-size', type=int, default=5)
    p.add_argument('--reference', type=Path, default=default_reference_path())
    p.add_argument('--output', default=RESULTS_DIR / 'run/requests.csv')
    a = p.parse_args()
    results = asyncio.run(run_experiment(load_workload(a.workload), a.router_url,
                         a.arrival_gaps_ms, a.output, a.group_size, a.reference))
    raise SystemExit(0 if all(r['success'] for r in results) else 1)

if __name__ == '__main__':
    main()
