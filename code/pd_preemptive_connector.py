"""Custom NixlPushConnector using preemptive scheduler and worker.

Usage in kv-transfer-config JSON:
  P side (kv_producer):
    {
      "kv_connector": "PreemptiveNixlPushConnector",
      "kv_connector_module_path": "pd_preemptive_connector",
      "kv_role": "kv_producer"
    }
  D side (kv_consumer):
    {
      "kv_connector": "PreemptiveNixlPushConnector",
      "kv_connector_module_path": "pd_preemptive_connector",
      "kv_role": "kv_consumer"
    }

For the baseline (整请求传输, no streaming/preemption), set:
  PD_STREAM_BY_BATCH=0  PD_PRIORITY_PREEMPTION=0

and use the standard NixlPushConnector instead of this class.
"""
from __future__ import annotations

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPushConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from pd_preemptive_scheduler import PreemptivePushScheduler
from pd_preemptive_worker import PreemptivePushWorker


class PreemptiveNixlPushConnector(NixlPushConnector):
    """NixlPushConnector with stream_by_batch + preemptive transfer scheduling.

    Drop-in replacement for NixlPushConnector. Activated via
    kv_connector_module_path pointing to this module.
    """

    def __init__(self, vllm_config, role, kv_cache_config):
        # Call grandparent (NixlBaseConnector) to avoid NixlPushConnector's
        # hardcoded scheduler/worker instantiation.
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
            NixlBaseConnector,
        )
        NixlBaseConnector.__init__(self, vllm_config, role, kv_cache_config)

        self.connector_scheduler = None
        self.connector_worker = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = PreemptivePushScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = PreemptivePushWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")

    def wait_for_save(self):
        super().wait_for_save()
        if self.connector_worker is not None:
            self.connector_worker.mark_stream_chunks_ready(self._connector_metadata)
