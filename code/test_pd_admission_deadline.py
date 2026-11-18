"""CPU regressions: ingress deadlines, backlog, cancellations and real vLLM IPC."""
import asyncio
import json
from queue import Queue
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import pd_ingress_window as ingress

async def turn():
    for _ in range(8):
        await asyncio.sleep(0)

def request(rid, tokens=128):
    return NS(request_id=rid, prompt_token_ids=[1] * tokens)

class GateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.gates, self.tasks, self.calls = [], [], []
        self.now = [0]
    async def asyncTearDown(self):
        for gate in self.gates:
            await gate.close()
        for task in self.tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
    def gate(self, dispatch=None, real_time=False):
        async def record(frontend, window, tickets):
            self.calls.append([t.request.request_id for t in tickets])
        gate = ingress.IngressGate(dispatch=dispatch or record,
                       clock=time.monotonic_ns if real_time else lambda: self.now[0])
        self.gates.append(gate)
        return gate
    def submit(self, gate, ticket, rid, tokens=128, client=None):
        task = asyncio.create_task(gate.submit(ticket, client, request(rid, tokens)))
        self.tasks.append(task)
        return task
    def advance(self, gate, ns):
        self.now[0] = ns
        gate.event.set()

    async def test_singleton_waits_full_five_ms_when_idle(self):
        stamps = []
        async def dispatch(frontend, window, tickets):
            stamps.append(time.monotonic_ns())
        gate = self.gate(dispatch, real_time=True)
        ticket = gate.enter()
        task = self.submit(gate, ticket, 'alone')
        await turn()
        self.assertFalse(stamps)
        await asyncio.wait_for(task, 1)
        self.assertGreaterEqual(stamps[0], ticket.window.deadline_ns)
        self.assertEqual(ticket.window.deadline_ns-ticket.arrival_ns, 5_000_000)

    async def test_five_plus_five_plus_five_preserved_under_engine_backlog(self):
        for gap_ms in (25, 50):
            calls, blocked = [], asyncio.Event()
            async def dispatch(frontend, window, tickets):
                calls.append([t.request.request_id for t in tickets])
                if len(calls) == 1: await blocked.wait()
            self.now[0] = 0
            gate = self.gate(dispatch)
            tasks = []
            for batch in range(3):
                for i, tokens in enumerate([10000, 128, 1107, 128, 500]):
                    self.now[0] = batch*gap_ms*1_000_000 + i*100_000
                    tasks.append(self.submit(gate, gate.enter(), f'{batch}-{i}', tokens))
                self.advance(gate, batch*gap_ms*1_000_000 + 5_000_000)
                await turn()
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(gate.windows), 2)
            blocked.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 1)
            self.assertEqual(calls, [[f'{b}-{i}' for i in [1,3,4,2,0]] for b in range(3)])

    async def test_exact_deadline_starts_new_window_despite_delayed_timer(self):
        gate = self.gate()
        first = gate.enter()
        self.now[0] = 4_999_999
        same = gate.enter()
        self.now[0] = 5_000_000
        later = gate.enter()
        self.assertIs(first.window, same.window)
        self.assertIsNot(first.window, later.window)
        self.assertEqual(later.window.deadline_ns, 10_000_000)

    async def test_late_body_keeps_original_window_and_blocks_overtaking(self):
        gate = self.gate()
        slow = gate.enter()
        self.now[0] = 1_000_000
        fast = self.submit(gate, gate.enter(), 'fast', 1000)
        self.now[0] = 25_000_000
        later = self.submit(gate, gate.enter(), 'later', 10)
        self.advance(gate, 30_000_000)
        await turn()
        self.assertFalse(self.calls)
        slow_task = self.submit(gate, slow, 'slow', 128)
        await asyncio.wait_for(asyncio.gather(fast, later, slow_task), 1)
        self.assertEqual(self.calls, [['slow','fast'], ['later']])

    async def test_invalid_http_does_not_block_neighbours(self):
        gate = self.gate()
        invalid = gate.enter()
        valid = self.submit(gate, gate.enter(), 'valid')
        gate.skip(invalid)
        self.advance(gate, 5_000_000)
        await asyncio.wait_for(valid, 1)
        self.assertEqual(self.calls, [['valid']])

    async def test_cancel_before_dispatch_never_resurrects_request(self):
        gate = self.gate()
        cancelled = self.submit(gate, gate.enter(), 'cancelled')
        valid = self.submit(gate, gate.enter(), 'valid')
        await turn()
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError): await cancelled
        self.advance(gate, 5_000_000)
        await asyncio.wait_for(valid, 1)
        self.assertEqual(self.calls, [['valid']])

    async def test_cancel_during_ipc_aborts_after_ack(self):
        sent, ack = asyncio.Event(), asyncio.Event()
        events = []
        async def dispatch(frontend, window, tickets):
            events.append('send'); sent.set()
            await ack.wait()
            events.append('ack')
        class Client:
            async def abort_requests_async(self, ids): events.append(('abort', ids))
        gate = self.gate(dispatch, real_time=True)
        task = self.submit(gate, gate.enter(), 'cancelled', client=Client())
        await asyncio.wait_for(sent.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(events, ['send'])
        ack.set()
        await turn()
        self.assertEqual(events, ['send','ack',('abort',['cancelled'])])

    async def test_failed_window_unblocks_waiters_and_next_window(self):
        async def dispatch(frontend, window, tickets):
            if window.sequence == 1: raise RuntimeError('broken IPC')
        gate = self.gate(dispatch)
        tasks = [self.submit(gate, gate.enter(), rid) for rid in ['a','b']]
        self.now[0] = 25_000_000
        tasks.append(self.submit(gate, gate.enter(), 'c'))
        self.advance(gate, 30_000_000)
        with self.assertLogs('vllm.pd_ingress_window', level='ERROR'):
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1)
        self.assertTrue(all(isinstance(r, RuntimeError) for r in results[:2]))
        self.assertIsNone(results[2])

    async def test_shutdown_releases_ready_and_unready_windows(self):
        gate = self.gate()
        gate.enter()
        task = self.submit(gate, gate.enter(), 'ready')
        await turn()
        await gate.close()
        with self.assertRaisesRegex(RuntimeError, 'gate stopped'): await task
        self.assertTrue(all(w.timer.cancelled() for w in gate.windows))

    async def test_invalid_window_length_rejected(self):
        for value in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): ingress.IngressGate(value)

class ASGITests(unittest.IsolatedAsyncioTestCase):
    async def test_window_starts_before_body_and_context_reaches_child_task(self):
        calls, responses = [], []
        async def dispatch(frontend, window, tickets): calls.append(window)
        async def app(scope, receive, send):
            body = await receive()
            self.assertEqual(json.loads(body['body']), {'prompt':[1,2]})
            ticket = ingress._ticket.get()
            await asyncio.create_task(ticket.gate.submit(ticket, None, request('r', 2)))
            await send({'type':'http.response.body', 'body':b'ok'})
        wrapper = ingress.IngressASGI(app)
        wrapper.gate.dispatch = dispatch
        async def receive():
            self.assertIsNotNone(ingress._ticket.get())
            await asyncio.sleep(.008)
            return {'type':'http.request', 'body':b'{"prompt":[1,2]}'}
        async def send(msg): responses.append(msg)
        try:
            await wrapper({'type':'http','method':'POST','path':'/v1/completions'}, receive, send)
            self.assertEqual(responses[-1]['body'], b'ok')
            self.assertEqual(len(calls), 1)
            self.assertIsNone(ingress._ticket.get())
        finally: await wrapper.gate.close()

    async def test_collects_five_before_any_body_or_preprocessing(self):
        now, calls, prepared = [0], [], []
        async def dispatch(frontend, window, tickets):
            calls.append([t.request.request_id for t in tickets])
        gate = ingress.IngressGate(clock=lambda: now[0], dispatch=dispatch)
        tokens = [10000,128,1107,128,500]
        async def app(scope, receive, send):
            # Even preprocessing which exceeds 5ms cannot split members that
            # were collected before it began.
            prepared.append(scope['index'])
            now[0] += 10_000_000
            ticket = ingress._ticket.get()
            await ticket.gate.submit(ticket, None,
                                     request(str(scope['index']), tokens[scope['index']]))
        wrapper, tasks = ingress.IngressASGI(app), []
        wrapper.gate = gate
        async def receive(): self.fail('wrapper consumed body')
        async def send(message): pass
        try:
            for i in range(5):
                now[0] = i*100_000
                tasks.append(asyncio.create_task(wrapper(
                    {'type':'http','method':'POST','path':'/v1/completions','index':i}, receive, send)))
                await turn()
            self.assertFalse(prepared)
            self.assertEqual(len(gate.windows[0].tickets), 5)
            now[0] = 5_000_000
            gate._deadline(gate.windows[0])
            await asyncio.wait_for(asyncio.gather(*tasks), 1)
            self.assertEqual(calls, [['1','3','4','2','0']])
        finally:
            await gate.close()
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_shutdown_unblocks_collection_and_early_cancelled_dispatcher(self):
        wrapper = ingress.IngressASGI(None)
        ticket = wrapper.gate.enter()
        # close() can run before the gate's dispatcher gets its first turn.
        await wrapper.gate.close()
        with self.assertRaisesRegex(RuntimeError, 'gate stopped'):
            await asyncio.wait_for(wrapper.gate.wait_collection(ticket), 1)

    async def test_cancel_during_collection_does_not_hold_valid_member(self):
        now, calls = [0], []
        async def dispatch(frontend, window, tickets):
            calls.append([t.request.request_id for t in tickets])
        async def app(scope, receive, send):
            ticket = ingress._ticket.get()
            await ticket.gate.submit(ticket, None, request(scope['rid']))
        wrapper = ingress.IngressASGI(app)
        wrapper.gate = ingress.IngressGate(clock=lambda: now[0], dispatch=dispatch)
        async def receive(): self.fail('unexpected body read')
        async def send(message): pass
        tasks = [asyncio.create_task(wrapper(
            {'type':'http','method':'POST','path':'/v1/completions','rid':rid}, receive, send))
            for rid in ['cancelled','valid']]
        try:
            await turn()
            window = wrapper.gate.windows[0]
            tasks[0].cancel()
            with self.assertRaises(asyncio.CancelledError): await tasks[0]
            self.assertTrue(window.tickets[0].skipped)
            now[0] = 5_000_000
            wrapper.gate._deadline(window)
            await asyncio.wait_for(tasks[1], 1)
            self.assertEqual(calls, [['valid']])
        finally:
            await wrapper.gate.close()
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class ValidatedAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from vllm.entrypoints.openai.completion.protocol import CompletionRequest
        self.rendered, self.calls = [], []
        rendered, calls = self.rendered, self.calls
        class Serving:
            async def render_completion_request(self, req):
                rendered.append(req)
                ticket = ingress._ticket.get()
                length = len(req.prompt)
                await ticket.gate.submit(ticket, None, request(str(len(rendered)), length))
                return {'ok': True}
            def create_error_response(self, message):
                return JSONResponse({'error': message}, status_code=400)
        ingress.patch_serving(NS(OpenAIServingCompletion=Serving))
        self.serving = Serving()
        app = FastAPI()
        @app.post('/v1/completions')
        async def complete(req: CompletionRequest):
            return await self.serving.render_completion_request(req)
        async def dispatch(frontend, window, tickets):
            calls.append([t.request.request_id for t in tickets])
        self.wrapper = ingress.IngressASGI(app)
        self.wrapper.gate.dispatch = dispatch

    async def asyncTearDown(self):
        await self.wrapper.gate.close()

    async def post(self, **kwargs):
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.wrapper),
                                     base_url='http://cpu-test') as client:
            return await client.post('/v1/completions', **kwargs)

    async def test_fanout_rejected_after_typed_validation_before_rendering(self):
        for body in [{'prompt':['a','b']}, {'prompt':[[1],[2]]}, {'prompt':[]},
                     {'prompt':[1],'n':2}, {'prompt':'a','best_of':2},
                     {'prompt':[1],'use_beam_search':True}]:
            response = await self.post(json=body)
            self.assertEqual(response.status_code, 400, (body, response.text))
        self.assertFalse(self.rendered)
        self.assertFalse(self.calls)
        self.assertTrue(all(t.skipped for w in self.wrapper.gate.windows for t in w.tickets))

    async def test_long_token_body_decoded_once_by_fastapi(self):
        body = json.dumps({'prompt':[1]*10000})
        with (patch('starlette.requests.json.loads', wraps=json.loads) as decode,
              patch.object(ingress, '_detail_timing', True),
              patch.object(ingress, 'emit') as trace):
            response = await self.post(content=body, headers={'content-type':'application/json'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(decode.call_count, 1)
            read_spans = [c.kwargs for c in trace.call_args_list if c.args[0]=='api_read_body_span']
            self.assertEqual(len(read_spans), 1)
            self.assertEqual(read_spans[0]['bytes'], len(body.encode()))
        self.assertEqual(len(self.rendered[0].prompt), 10000)
        self.assertEqual(len(self.calls), 1)

    async def test_original_validation_errors_do_not_hold_next_valid_request(self):
        for body in ['{', '{"prompt":[-1]}', '{"prompt":[{}]}']:
            response = await self.post(content=body, headers={'content-type':'application/json'})
            self.assertEqual(response.status_code, 422)
        response = await self.post(json={'prompt':[1,2]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.calls), 1)

    async def test_no_extra_token_traversal_after_model_validation(self):
        class ValidatedTokens(list):
            def __iter__(self): raise AssertionError('extra token scan')
        ticket = self.wrapper.gate.enter()
        token = ingress._ticket.set(ticket)
        try:
            req = NS(prompt=ValidatedTokens([1]*10000), n=1, use_beam_search=False)
            result = await self.serving.render_completion_request(req)
            self.assertTrue(result['ok'])
        finally:
            ingress._ticket.reset(token)

class AtomicEngineTests(unittest.IsolatedAsyncioTestCase):
    def real_request(self, rid, tokens):
        from vllm import SamplingParams
        from vllm.v1.engine import EngineCoreRequest
        return EngineCoreRequest(request_id=rid, prompt_token_ids=[1]*tokens, mm_features=None,
            sampling_params=SamplingParams(max_tokens=1, extra_args={'kv_transfer_params':{'do_remote_decode':True}}),
            pooling_params=None, arrival_time=time.time(), lora_request=None,
            cache_salt=None, data_parallel_rank=None, client_index=0)
    def core(self):
        from vllm.v1.engine.core import EngineCore, EngineCoreProc
        from vllm.v1.core.sched.request_queue import FCFSRequestQueue
        class Core:
            preprocess_add_request = EngineCore.preprocess_add_request
            _convert_msgspec_args = staticmethod(EngineCoreProc._convert_msgspec_args)
            _invoke_utility_method = staticmethod(EngineCoreProc._invoke_utility_method)
            _handle_client_request = EngineCoreProc._handle_client_request
            pd_add_ingress_window = ingress.add_ingress_window
            def __init__(self):
                self.mm_receiver_cache = self.request_block_hasher = self.fail_at = None
                self.waiting, self.output_queue = FCFSRequestQueue(), Queue()
            def _reject_utility_in_shutdown(self, *args): return False
            def _reject_add_in_shutdown(self, req): return False
            def _reject_add_while_paused(self, req): return False
            def add_request(self, req, wave):
                self.waiting.add_request(req)
                if req.request_id == self.fail_at: raise RuntimeError('registration failed')
            def abort_requests(self, ids):
                self.waiting.remove_requests([r for r in self.waiting if r.request_id in ids])
        return Core()

    async def test_real_vllm_serde_and_main_loop_utility_admit_complete_window(self):
        from vllm.v1.engine import EngineCoreRequestType
        from vllm.v1.serial_utils import MsgpackEncoder, MsgpackDecoder
        core = self.core()
        class Client:
            async def call_utility_async(self, method, *args):
                wire = MsgpackEncoder().encode((0, 17, method, args))
                core._handle_client_request(EngineCoreRequestType.UTILITY, MsgpackDecoder().decode(wire))
                out = core.output_queue.get_nowait()[1].utility_output
                if out.failure_message: raise RuntimeError(out.failure_message)
                self.registered_before_schedule = [r.request_id for r in core.waiting]
        client, gate = Client(), ingress.IngressGate()
        try:
            tickets = [gate.enter() for _ in range(5)]
            await asyncio.wait_for(asyncio.gather(*[
                gate.submit(t, client, self.real_request(str(i), n))
                for i,(t,n) in enumerate(zip(tickets,[10000,128,1107,128,500]))]), 2)
            self.assertEqual(client.registered_before_schedule, ['1','3','4','2','0'])
            self.assertEqual(len(core.waiting), 5)
            self.assertEqual(core.waiting.pop_request().kv_transfer_params, {'do_remote_decode':True})
            self.assertEqual([r.num_prompt_tokens for r in core.waiting], [128,500,1107,10000])
        finally: await gate.close()

    async def test_registration_failure_rolls_back_whole_window(self):
        core = self.core()
        core.fail_at = 'b'
        reqs = [self.real_request('a', 128), self.real_request('b', 256)]
        metadata = dict(window=1, opened_ns=0, deadline_ns=5_000_000,
            members=[dict(rid=r.request_id, arrival_ns=i) for i,r in enumerate(reqs)])
        with self.assertRaisesRegex(RuntimeError, 'registration failed'):
            ingress.add_ingress_window(core, metadata, ingress.encode_requests(reqs))
        self.assertFalse(core.waiting)

    async def test_unsorted_window_is_rejected_before_registering_any_member(self):
        core = self.core()
        reqs = [self.real_request('long', 10000), self.real_request('short', 128)]
        metadata = dict(window=1, opened_ns=0, deadline_ns=5_000_000,
            members=[dict(rid=r.request_id, arrival_ns=i) for i,r in enumerate(reqs)])
        with self.assertRaisesRegex(ValueError, 'not sorted'):
            ingress.add_ingress_window(core, metadata, ingress.encode_requests(reqs))
        self.assertFalse(core.waiting)

    async def test_real_client_hook_and_utility_ack_with_http_context(self):
        import vllm.v1.engine.core_client as module
        from vllm.v1.engine import EngineCoreRequestType
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
        core = self.core()
        client = object.__new__(module.AsyncMPClient)
        client.core_engines, client.core_engine, client.client_index = [b'0'], b'0', 0
        client.utility_results = {}
        client.encoder = MsgpackEncoder()
        client._ensure_output_queue_task = lambda: None
        wire_types = []
        async def send_input(message, engine):
            kind = EngineCoreRequestType(message[0])
            wire_types.append(kind)
            core._handle_client_request(kind, MsgpackDecoder().decode(message[1:]))
            output = core.output_queue.get_nowait()[1].utility_output
            module._process_utility_output(output, client.utility_results)
        client._send_input_message = send_input
        req = self.real_request('http', 128)
        async def app(scope, receive, send):
            await receive()
            # OpenAI's merge_async_iterators also creates child tasks.
            await asyncio.create_task(client.add_request_async(req))
            await send({'type':'http.response.body', 'body':b'ok'})
        wrapper = ingress.IngressASGI(app)
        async def receive(): return {'type':'http.request', 'body':b'{"prompt":[1]}'}
        async def send(message): pass
        try:
            with patch.object(module.AsyncMPClient, 'add_request_async', module.AsyncMPClient.add_request_async):
                ingress.patch_client(module)
                await asyncio.wait_for(wrapper(
                    {'type':'http','method':'POST','path':'/v1/completions'}, receive, send), 2)
            self.assertEqual(wire_types, [EngineCoreRequestType.UTILITY])
            self.assertEqual([r.request_id for r in core.waiting], ['http'])
            self.assertFalse(client.utility_results)
        finally:
            await wrapper.gate.close()

    async def test_async_llm_failed_admission_cleans_output_registration(self):
        import vllm.v1.engine.async_llm as module
        llm = object.__new__(module.AsyncLLM)
        llm.shutdown = lambda: None  # No resources were initialized in this CPU fixture.
        registered, aborted = [], []
        llm.check_admission = lambda **kwargs: None
        llm.output_processor = NS(has_request=lambda rid: False,
                                 add_request=lambda *args: registered.append(args[0].request_id))
        async def failed(request): raise RuntimeError('ingress failed')
        llm.engine_core = NS(add_request_async=failed)
        async def abort(rid, internal=False): aborted.append((rid, internal))
        llm.abort = abort
        with patch.object(module.AsyncLLM, '_add_request', module.AsyncLLM._add_request):
            ingress.patch_async_llm(module)
            with self.assertRaisesRegex(RuntimeError, 'ingress failed'):
                await llm._add_request(self.real_request('cleanup', 128), None, None, 0, None)
        self.assertEqual(registered, ['cleanup'])
        self.assertEqual(aborted, [('cleanup', True)])

if __name__ == '__main__': unittest.main()
