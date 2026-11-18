"""CPU regressions for review findings, including real local HTTP streams."""
import asyncio
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import aiohttp
from aiohttp import web

import run_pd_router as router
from pd_preemptive_worker import TransferItem
from validate_pd_run import validate_run


class WindowReviewTests(unittest.TestCase):
    def test_transfer_classification_uses_tokens_at_256_boundary(self):
        for tokens, expected in [(241, 'short'), (255, 'short'), (256, 'medium'), (8192, 'medium'), (8193, 'long')]:
            item = TransferItem('r', ([1],), (tokens+15)//16, False, prompt_tokens=tokens)
            self.assertEqual(item.category(), expected)


class RouterReviewTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runners = []
        self.old_urls = router.P_URL, router.D_URL

    async def asyncTearDown(self):
        router.P_URL, router.D_URL = self.old_urls
        for runner in reversed(self.runners):
            await runner.cleanup()

    async def serve(self, app):
        runner = web.AppRunner(app, handler_cancellation=True)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        self.runners.append(runner)
        return f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'

    async def test_first_sse_is_forwarded_before_prefill_http_finishes(self):
        release_p = asyncio.Event()
        async def prefill(request):
            await release_p.wait()
            return web.json_response({'kv_transfer_params': {'remote_request_id': 'r'}})
        async def decode(request):
            resp = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await resp.prepare(request)
            await resp.write(b'data: {"choices":[{"text":"hello","token_ids":[1]}]}\n\n')
            await release_p.wait()
            await resp.write(b'data: [DONE]\n\n')
            return resp
        p, d = web.Application(), web.Application()
        p.router.add_post('/v1/completions', prefill)
        d.router.add_post('/v1/completions', decode)
        router.P_URL, router.D_URL = await self.serve(p), await self.serve(d)
        url = await self.serve(router.build_app())
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url+'/v1/completions', json={'stream': True, 'prompt_token_ids': [1]}) as response:
                    line = await asyncio.wait_for(response.content.readline(), 2)
                    self.assertIn(b'hello', line)
                    self.assertFalse(release_p.is_set())
                    release_p.set()
                    self.assertIn(b'[DONE]', await response.read())
        finally:
            release_p.set()

    async def test_preencoded_payloads_preserve_backend_contract_and_request_ids(self):
        received = {}
        async def backend(request):
            role = request.app['role']
            self.assertEqual(request.content_type, 'application/json')
            received[role] = (await request.json(), request.headers['X-Request-Id'])
            return web.json_response({'kv_transfer_params':{'remote_request_id':received[role][1]}})
        apps = [web.Application(),web.Application()]
        for role, app in zip(('P','D'),apps):
            app['role'] = role
            app.router.add_post('/v1/completions',backend)
        router.P_URL, router.D_URL = await self.serve(apps[0]),await self.serve(apps[1])
        url = await self.serve(router.build_app())
        ids = list(range(10000))
        body = {'prompt_token_ids':ids,'seed':7,'max_tokens':64,'temperature':.25,
                'return_token_ids':True,'request_id':'test-雪"\\'}
        async with aiohttp.ClientSession() as session:
            async with session.post(url+'/v1/completions',json=body) as response:
                self.assertEqual(response.status,200)
                await response.read()
        p, p_id = received['P']; d, d_id = received['D']
        self.assertEqual(p_id,d_id)
        self.assertEqual(p['prompt'],ids); self.assertEqual(d['prompt'],ids)
        self.assertEqual(p['model'],'llama3-p'); self.assertEqual(d['model'],'llama3-d')
        self.assertEqual(p['max_tokens'],1); self.assertEqual(d['max_tokens'],64)
        for key in ('seed','temperature'):
            self.assertEqual(p[key],body[key]); self.assertEqual(d[key],body[key])
        self.assertTrue(p['kv_transfer_params']['do_remote_decode'])
        self.assertEqual(p['kv_transfer_params']['remote_port'],5601)
        self.assertEqual(d['kv_transfer_params']['remote_num_tokens'],10000)
        self.assertEqual(d['kv_transfer_params']['remote_request_id'],p_id)
        self.assertEqual(d['kv_transfer_params']['transfer_mode'],'push')
        self.assertFalse(d['stream']); self.assertIsNone(d['stream_options'])
        self.assertTrue(d['return_token_ids'])

    async def test_prompt_array_is_encoded_once_for_both_payloads(self):
        ids = list(range(10000))
        with patch.object(router.orjson,'dumps',wraps=router.orjson.dumps) as encode:
            p,d = router.prepare_backend_payloads({'prompt_token_ids':ids,'stream':True},'shared')
            self.assertEqual(sum(call.args[0] is ids for call in encode.call_args_list),1)
        self.assertEqual(json.loads(p)['prompt'],ids)
        decoded=json.loads(d)
        self.assertEqual(decoded['prompt'],ids)
        self.assertEqual(decoded['stream_options'],{'include_usage':True})

    async def test_invalid_input_rejected_without_starting_backend_tasks(self):
        url = await self.serve(router.build_app())
        async with aiohttp.ClientSession() as session:
            for body in (b'{',b'[]',b'{"prompt_token_ids":12}',
                         b'{"prompt_token_ids":[1],"temperature":NaN}'):
                async with session.post(url+'/v1/completions',data=body) as response:
                    self.assertEqual(response.status,400,await response.text())
            async with session.post(url+'/v1/completions',data=b' '*1_048_577) as response:
                self.assertEqual(response.status,413)

    async def test_backend_pool_reuses_connections_and_closes_on_cleanup(self):
        connections = {'P': [], 'D': []}
        ids = {'P': [], 'D': []}
        async def backend(request):
            role = request.app['role']
            connections[role].append(request.transport)
            ids[role].append(request.headers['X-Request-Id'])
            await request.json()
            return web.json_response({'kv_transfer_params': {'remote_request_id': ids[role][-1]}})
        apps = [web.Application(), web.Application()]
        for role, app in zip(('P','D'), apps):
            app['role'] = role
            app.router.add_post('/v1/completions', backend)
        router.P_URL, router.D_URL = await self.serve(apps[0]), await self.serve(apps[1])
        app = router.build_app()
        url = await self.serve(app)
        pool = app[router.BACKEND_SESSION]
        async with aiohttp.ClientSession() as session:
            for _ in range(2):
                async with session.post(url+'/v1/completions', json={'prompt_token_ids':[1]}) as response:
                    self.assertEqual(response.status, 200)
                    await response.read()
        for role in ('P','D'):
            self.assertIs(connections[role][0], connections[role][1])
        self.assertEqual(ids['P'], ids['D'])
        self.assertNotEqual(ids['P'][0], ids['P'][1])
        await self.runners[-1].cleanup()
        self.runners.pop()
        self.assertTrue(pool.closed)

    async def test_failed_request_does_not_close_shared_backend_session(self):
        attempts = []
        async def prefill(request):
            attempts.append(request.headers['X-Request-Id'])
            await request.json()
            if len(attempts) == 1:
                return web.Response(status=500, text='test failure')
            return web.json_response({'kv_transfer_params': {'remote_request_id': attempts[-1]}})
        async def decode(request):
            await request.json()
            return web.json_response({'choices': []})
        p, d = web.Application(), web.Application()
        p.router.add_post('/v1/completions', prefill)
        d.router.add_post('/v1/completions', decode)
        router.P_URL, router.D_URL = await self.serve(p), await self.serve(d)
        app = router.build_app()
        url = await self.serve(app)
        async with aiohttp.ClientSession() as session:
            for expected in (502, 200):
                async with session.post(url+'/v1/completions', json={'prompt_token_ids':[1]}) as response:
                    self.assertEqual(response.status, expected)
                    await response.read()
                self.assertFalse(app[router.BACKEND_SESSION].closed)

    async def test_prefill_failure_cancels_pending_decode_and_reaps_tasks(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()
        async def decode():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        async def prefill():
            await entered.wait()
            raise RuntimeError('prefill failed')
        with self.assertRaisesRegex(RuntimeError, 'prefill failed'):
            await asyncio.wait_for(router._run_pair(asyncio.create_task(prefill()), decode()), 2)
        self.assertTrue(cancelled.is_set())

    async def test_decode_failure_cancels_pending_prefill(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()
        async def prefill():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        async def decode():
            await entered.wait()
            raise RuntimeError('decode failed')
        with self.assertRaisesRegex(RuntimeError, 'decode failed'):
            await asyncio.wait_for(router._run_pair(asyncio.create_task(prefill()), decode()), 2)
        self.assertTrue(cancelled.is_set())

    async def test_outer_cancellation_reaps_both_backend_tasks(self):
        entered = [asyncio.Event(), asyncio.Event()]
        cancelled = [asyncio.Event(), asyncio.Event()]
        async def backend(i):
            entered[i].set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled[i].set()
        pair = asyncio.create_task(router._run_pair(asyncio.create_task(backend(0)), backend(1)))
        await asyncio.gather(*(e.wait() for e in entered))
        pair.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pair
        self.assertTrue(all(e.is_set() for e in cancelled))


class DataReviewTests(unittest.TestCase):
    def test_independent_validator_rejects_actual_ids_despite_success_flags(self):
        workload = [dict(request_id='r', max_tokens=1, prompt_token_ids=[10])]
        reference = {'r': {'output_token_ids': '[20]'}}
        row = dict(request_id='r', category='short', num_prompt_tokens=1, num_output_tokens=1,
                   output_token_ids='[20]', output_text='text', finish_reason='length',
                   token_ids_hash=hashlib.sha256(b'[20]').hexdigest()[:16],
                   success=True, reference_match=True, error='', sse_done_received=True,
                   ttft_ms=10, e2e_ms=20, submitted_at=0, completed_at=.02, makespan_ms=20)
        events = [
            dict(elapsed_ms=10, payload=dict(choices=[dict(text='text', token_ids=[20],
                prompt_token_ids=[10], finish_reason='length')])),
            dict(elapsed_ms=15, payload=dict(usage=dict(completion_tokens=1, prompt_tokens=1))),
            dict(elapsed_ms=20, payload='[DONE]'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)/'requests.csv'
            dest.with_suffix('.sse.jsonl').write_text(json.dumps(dict(request_id='r', events=events)) + '\n')

            def save():
                with dest.open('w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)

            save()
            self.assertTrue(validate_run(dest, workload, reference)['valid'])
            row['output_token_ids'] = '[21]'
            save()
            checked = validate_run(dest, workload, reference)
            self.assertFalse(checked['valid'])
            self.assertTrue(any('actual token IDs' in e for e in checked['errors']))


if __name__ == '__main__':
    unittest.main()
