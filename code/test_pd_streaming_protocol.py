"""CPU regressions for actual scheduler/worker lifecycle (no GPU context)."""
import logging
import queue
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from unittest.mock import Mock

from pd_preemptive_scheduler import PreemptivePushScheduler
from pd_preemptive_worker import PreemptivePushWorker, TransferItem, ReqProtectionState
from pd_streaming_metadata import PreemptiveNixlMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import NixlPushConnectorScheduler
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import NixlPushConnectorWorker


class StreamingProtocolTests(unittest.TestCase):
    def scheduler(self, prompt):
        obj = PreemptivePushScheduler.__new__(PreemptivePushScheduler)
        obj.block_size = 16
        obj._active_requests = {'r': NS(num_prompt_tokens=prompt)}
        obj._active_block_ids = {'r': ([],)}
        obj._partial_cursor = {}
        return obj

    def step(self, obj, computed, count, blocks, first=False, finished=False):
        meta = PreemptiveNixlMetadata()
        if finished:
            meta.push_finished_blocks = {'r': (list(range((obj._active_requests['r'].num_prompt_tokens+15)//16)),)}
        output = NS(
            scheduled_new_reqs=[NS(req_id='r', num_computed_tokens=computed, block_ids=(blocks,))] if first else [],
            scheduled_cached_reqs=NS(req_ids=[] if first or finished else ['r'],
                num_computed_tokens=[] if first or finished else [computed], resumed_req_ids=set(),
                new_block_ids=[] if first or finished else [(blocks,)]),
            num_scheduled_tokens={'r': count})
        with patch.object(NixlPushConnectorScheduler, 'build_connector_meta', return_value=meta):
            return obj.build_connector_meta(output).push_chunks

    def test_medium_growing_table_has_no_unsent_gap(self):
        obj = self.scheduler(1107)
        chunks = self.step(obj, 0, 32, [0, 1], first=True)
        chunks += self.step(obj, 32, 1075, list(range(2, 70)))
        chunks += self.step(obj, 1107, 0, [], finished=True)
        self.assertEqual([c.block_offset for c in chunks], [0, 2, 69])
        self.assertEqual([b for c in chunks for b in c.block_ids[0]], list(range(70)))
        self.assertEqual(sum(c.is_final for c in chunks), 1)
        self.assertFalse(obj._active_requests)

    def test_long_growing_table_keeps_every_write_at_most_c8(self):
        obj = self.scheduler(10000)
        chunks = self.step(obj, 0, 2048, list(range(128)), first=True)
        chunks += self.step(obj, 2048, 2048, list(range(128, 256)))
        chunks += self.step(obj, 4096, 5904, list(range(256, 625)))
        chunks += self.step(obj, 10000, 0, [], finished=True)
        self.assertTrue(all(0 < len(c.block_ids[0]) <= 8 for c in chunks))
        self.assertEqual([b for c in chunks for b in c.block_ids[0]], list(range(625)))
        self.assertEqual(sum(c.is_final for c in chunks), 1)

    def worker(self):
        obj = PreemptivePushWorker.__new__(PreemptivePushWorker)
        obj._failed_streams = set()
        obj._busy_requests = set()
        obj._next_offset = {}
        obj._prot_state = {}
        obj._stream_done_transfers = set()
        obj._stream_failed_transfers = set()
        obj._stream_registrations = {'cmpl-shared-0': dict(decode_engine_id='d', decode_host='localhost',
            decode_port=5601, decode_tp_size=1)}
        obj._ensure_handshake = lambda *args: None
        return obj

    def test_early_registration_survives_random_request_suffix(self):
        obj = self.worker()
        item = TransferItem('cmpl-shared-0-1234abcd', ([1],), 2, False)
        self.assertTrue(obj._eligible(item))
        obj._busy_requests.add(item.req_id)
        self.assertFalse(obj._eligible(item))

    def test_ready_trace_observes_normal_queries_once_without_changing_eligibility(self):
        obj = self.worker()
        event = Mock()
        event.query.side_effect = [False, True, True]
        item = TransferItem('cmpl-shared-0-1234abcd', ([1],), 1, True, ready_event=event)
        with patch('pd_preemptive_worker._TRACE_PIPELINE', True), \
                patch('pd_preemptive_worker.emit') as emit:
            self.assertFalse(obj._eligible(item))
            self.assertTrue(obj._eligible(item))
            self.assertTrue(obj._eligible(item))
        self.assertEqual(event.query.call_count, 3)
        self.assertEqual([call.args[0] for call in emit.call_args_list],
                         ['chunk_ready_observed', 'chunk_eligible_observed'])

    def test_final_cannot_overtake_partial_during_handshake(self):
        obj = self.worker()
        partial = TransferItem('cmpl-shared-0-1234abcd', ([1],), 2, False)
        final = TransferItem(partial.req_id, ([2],), 2, True, block_offset=1)
        obj._transfer_queue = [final, partial]
        self.assertIs(obj._select_next_item(), partial)
        self.assertIsNone(obj._select_next_item())
        obj._next_offset[partial.req_id] = 1
        self.assertIs(obj._select_next_item(), final)

    def test_partial_completion_does_not_release_request(self):
        obj = self.worker()
        obj._sending_transfers = {'synthetic#chunk-0': [1]}
        item = TransferItem('r', ([1, 2],), 3, False)
        obj._chunk_transfers = {'synthetic#chunk-0': item}
        obj._chunk_complete_inbox = queue.Queue()
        obj._send_failures = set()
        obj._push_writer_wake = threading.Event()
        def complete(transfers):
            transfers.clear()
            return {'synthetic#chunk-0'}, set()
        obj._sending_transfers_lock = threading.Lock()
        with patch.object(NixlPushConnectorWorker, '_pop_done_transfers', side_effect=complete):
            obj._poll_stream_transfers()
        with patch.object(NixlPushConnectorWorker, '_pop_done_transfers', side_effect=AssertionError('engine must not poll WRITE handles')):
            done, failures = obj._pop_done_transfers(obj._sending_transfers)
        self.assertEqual(done, set())
        self.assertEqual(obj._chunk_complete_inbox.get_nowait(), (item, False, True))

    def test_writer_reaps_handles_without_model_step_and_reports_final_once(self):
        obj = self.worker()
        item = TransferItem('r', ([1, 2],), 3, False)
        obj._sending_transfers = {'r#chunk-0': [11], 'final': [22]}
        obj._sending_transfers_lock = threading.Lock()
        final = TransferItem('final', ([3],), 1, True)
        obj._chunk_transfers = {'r#chunk-0': item, 'final': final}
        obj._chunk_complete_inbox = queue.Queue()
        obj._send_failures = set()
        obj._push_writer_wake = threading.Event()
        def complete(transfers):
            self.assertEqual(transfers, {'r#chunk-0': [11], 'final': [22]})
            transfers.clear()
            return {'r#chunk-0', 'final'}, set()
        with patch.object(NixlPushConnectorWorker, '_pop_done_transfers', side_effect=complete):
            obj._poll_stream_transfers()
        self.assertEqual(obj._sending_transfers, {})
        self.assertEqual(obj._stream_done_transfers, {'final'})
        completed = [obj._chunk_complete_inbox.get_nowait() for _ in range(2)]
        self.assertCountEqual([x[0].req_id for x in completed], ['r', 'final'])
        with patch.object(NixlPushConnectorWorker, '_pop_done_transfers', side_effect=lambda _: (set(), set())):
            self.assertEqual(obj._pop_done_transfers(obj._sending_transfers), ({'final'}, set()))
            self.assertEqual(obj._pop_done_transfers(obj._sending_transfers), (set(), set()))
        self.assertTrue(obj._chunk_complete_inbox.empty())
        self.assertTrue(obj._push_writer_wake.is_set())

    def test_writer_retains_in_progress_and_failed_dma_handles(self):
        obj = self.worker()
        item = TransferItem('r', ([1],), 2, False)
        obj._sending_transfers = {'r#chunk-0': [11]}
        obj._sending_transfers_lock = threading.Lock()
        obj._chunk_transfers = {'r#chunk-0': item}
        obj._chunk_complete_inbox = queue.Queue()
        obj._send_failures = set()
        obj._push_writer_wake = threading.Event()
        with patch.object(NixlPushConnectorWorker, '_pop_done_transfers', return_value=(set(), {'r#chunk-0'})):
            obj._poll_stream_transfers()
        self.assertEqual(obj._sending_transfers, {'r#chunk-0': [11]})
        self.assertEqual(obj._chunk_complete_inbox.get_nowait(), (item, True, False))
        self.assertIn('r#chunk-0', obj._send_failures)
        self.assertEqual(obj._stream_failed_transfers, set())
        def finish_failed(transfers):
            transfers.clear()
            return {'r#chunk-0'}, set()
        with patch.object(NixlPushConnectorWorker, '_pop_done_transfers', side_effect=finish_failed):
            obj._poll_stream_transfers()
        self.assertEqual(obj._chunk_complete_inbox.get_nowait(), (item, True, True))
        self.assertNotIn('r#chunk-0', obj._send_failures)
        self.assertEqual(obj._stream_done_transfers, set())

    def test_pending_transfer_poll_does_not_wake_itself_into_busy_spin(self):
        obj = self.worker()
        obj._sending_transfers = {'r#chunk-0': [11]}
        obj._sending_transfers_lock = threading.Lock()
        obj._chunk_transfers = {}
        obj._chunk_complete_inbox = queue.Queue()
        obj._send_failures = set()
        obj._push_writer_wake = threading.Event()
        with patch.object(NixlPushConnectorWorker, '_pop_done_transfers', return_value=(set(), set())):
            obj._poll_stream_transfers()
        self.assertFalse(obj._push_writer_wake.is_set())
        self.assertEqual(obj._sending_transfers, {'r#chunk-0': [11]})


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    unittest.main()
