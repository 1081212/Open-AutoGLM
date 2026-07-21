"""Cooperative cancellation shared by the worker parent and task executor."""

from __future__ import annotations

import threading

from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode


class CancellationToken:
    """Thread-safe cancellation token with a stable reason."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = "Cancellation requested"

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def cancel(self, reason: str = "Cancellation requested") -> None:
        with self._lock:
            if not self._event.is_set():
                self._reason = reason
                self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise ExecutionError(
                code=ExecutionErrorCode.CANCELLED,
                message=self.reason,
                retryable=False,
            )
