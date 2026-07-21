"""Lifecycle event sink used by both local reporting and the worker protocol."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, TextIO
from uuid import UUID, uuid4


class LifecycleSink(Protocol):
    def emit(self, event_type: str, data: dict[str, Any]) -> None: ...


class NullLifecycleSink:
    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        del event_type, data


@dataclass
class JsonLinesLifecycleSink:
    """Write protocol-only JSON lines to a dedicated stream."""

    stream: TextIO
    task_run_id: UUID
    producer_id: UUID = field(default_factory=uuid4)
    _sequence: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            envelope = {
                "task_run_id": str(self.task_run_id),
                "producer_id": str(self.producer_id),
                "producer_seq": self._sequence,
                "type": event_type,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }
            self.stream.write(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.stream.flush()
