"""Independently validate saved tokens, SSE payloads and client timings."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def validate_run(path, workload, references):
    path = Path(path)
    with path.open(newline='') as file:
        rows = list(csv.DictReader(file))
    expected = {r['request_id']: r for r in workload}
    errors = []
    if Counter(r['request_id'] for r in rows) != Counter(expected.keys()):
        errors.append('request set/count mismatch')
    sse = path.with_suffix('.sse.jsonl')
    records = [json.loads(line) for line in sse.read_text().splitlines()]
    if Counter(r['request_id'] for r in records) != Counter(expected.keys()):
        errors.append('SSE request set/count mismatch')
    events = {r['request_id']: r['events'] for r in records}
    for row in rows:
        rid = row['request_id']
        if rid not in expected or rid not in references or rid not in events:
            errors.append(rid + ': missing workload/reference/SSE')
            continue
        req = expected[rid]
        prompt = req.get('prompt_token_ids') or req.get('token_ids') or []
        ids = json.loads(row['output_token_ids'])
        if ids != json.loads(references[rid]['output_token_ids']):
            errors.append(rid + ': actual token IDs differ from reference')
        if len(ids) != req['max_tokens'] or int(row['num_output_tokens']) != len(ids):
            errors.append(rid + ': output count')
        if int(row['num_prompt_tokens']) != len(prompt):
            errors.append(rid + ': prompt count')
        category = 'short' if len(prompt) < 256 else 'medium' if len(prompt) <= 8192 else 'long'
        if row['category'] != category:
            errors.append(rid + ': category')
        if row['token_ids_hash'] != hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]:
            errors.append(rid + ': token hash')
        if row['success'] != 'True' or row['reference_match'] != 'True' or row['error']:
            errors.append(rid + ': status')
        times = [float(row[k]) for k in ['ttft_ms', 'e2e_ms', 'submitted_at', 'completed_at', 'makespan_ms']]
        if not all(math.isfinite(x) for x in times) or not 0 < times[0] <= times[1]:
            errors.append(rid + ': invalid timings')
        if abs((times[3] - times[2])*1000 - times[1]) > .003:
            errors.append(rid + ': E2E timestamps')
        emitted, texts, first, usage, finish, prompt_count, done = [], [], None, {}, None, None, 0
        last_time = -1
        for event in events[rid]:
            elapsed = event['elapsed_ms']
            if not math.isfinite(elapsed) or elapsed < last_time or elapsed > times[1] + .002:
                errors.append(rid + ': SSE event time/order')
            last_time = elapsed
            payload = event['payload']
            if payload == '[DONE]':
                done += 1
                continue
            if payload.get('error'):
                errors.append(rid + ': SSE error')
            usage = payload.get('usage') or usage
            for choice in payload.get('choices', []):
                text = choice.get('text') or (choice.get('delta') or {}).get('content') or ''
                if text and first is None:
                    first = elapsed
                texts.append(text)
                emitted.extend(choice.get('token_ids') or [])
                if choice.get('prompt_token_ids') is not None:
                    prompt_count = len(choice['prompt_token_ids'])
                finish = choice.get('finish_reason') or finish
        if emitted != ids or ''.join(texts) != row['output_text']:
            errors.append(rid + ': SSE tokens/text disagree with CSV')
        if first is None or abs(first - times[0]) > .001:
            errors.append(rid + ': TTFT differs from first text SSE')
        if (finish != 'length' or row['finish_reason'] != finish or prompt_count != len(prompt)
                or usage.get('completion_tokens') != len(ids) or usage.get('prompt_tokens') != len(prompt)):
            errors.append(rid + ': SSE finish/usage/prompt counts')
        # Older archives did not persist DONE; new collectors do.
        if 'sse_done_received' in row and (done != 1 or row['sse_done_received'] != 'True'):
            errors.append(rid + ': missing/duplicate saved DONE')
    if rows:
        span = (max(float(r['completed_at']) for r in rows) - min(float(r['submitted_at']) for r in rows))*1000
        if any(abs(float(r['makespan_ms']) - span) > .003 for r in rows):
            errors.append('makespan differs from timestamps')
    return dict(requests=len(rows), valid=not errors, errors=errors)
