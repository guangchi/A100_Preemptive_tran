"""FCFS queue with atomic window publication for the deadline feeder.

Only the engine pops/prepends/removes requests. The feeder only appends
complete windows, so peek/pop head identity remains stable across appends.
"""
from collections import deque
from threading import RLock

from vllm.v1.core.sched.request_queue import RequestQueue


class AdmissionRequestQueue(RequestQueue):
    def __init__(self, requests=()):
        self._requests = deque(requests)
        self._lock = RLock()

    def add_requests(self, requests):
        with self._lock:
            self._requests.extend(requests)

    def add_request(self, request):
        self.add_requests((request,))

    def pop_request(self):
        with self._lock:
            return self._requests.popleft()

    def peek_request(self):
        with self._lock:
            return self._requests[0]

    def prepend_request(self, request):
        with self._lock:
            self._requests.appendleft(request)

    def prepend_requests(self, requests):
        with self._lock:
            self._requests.extendleft(requests)

    def remove_request(self, request):
        with self._lock:
            self._requests.remove(request)

    def remove_requests(self, requests):
        removed = set(requests)
        with self._lock:
            self._requests = deque(r for r in self._requests if r not in removed)

    def __bool__(self):
        return len(self) > 0

    def __len__(self):
        with self._lock:
            return len(self._requests)

    def __iter__(self):
        # The scheduler may iterate while the feeder publishes a new window.
        with self._lock:
            return iter(tuple(self._requests))
