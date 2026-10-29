"""Volatile local queue: references only, not customer data or predictions."""

from collections import deque
from collections.abc import Mapping
from threading import Lock

from churn_retention.modules.scoring.batch_ports import message_job_id


class InProcessJobQueue:
    def __init__(self) -> None:
        self._messages: deque[str] = deque()
        self._pending: set[str] = set()
        self._lock = Lock()

    def enqueue(self, message: Mapping[str, object]) -> None:
        job_id = message_job_id(message)
        with self._lock:
            if job_id not in self._pending:
                self._messages.append(job_id)
                self._pending.add(job_id)

    def dequeue(self) -> dict[str, str] | None:
        with self._lock:
            if not self._messages:
                return None
            job_id = self._messages.popleft()
            self._pending.remove(job_id)
            return {"job_id": job_id}
