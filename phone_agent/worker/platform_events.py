"""Persist structured lifecycle events before any platform send attempt."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from phone_agent.worker.identity import uuid7
from phone_agent.worker.outbox import DurableOutbox

logger = logging.getLogger(__name__)

# This allowlist mirrors platform TaskRunEventService.EVENT_TYPES. Internal
# lifecycle telemetry must be filtered before allocating producer_seq;
# filtering during send would create permanent sequence gaps.
PLATFORM_EVENT_TYPES = frozenset(
    {
        "RUN_STARTED",
        "CASE_STARTED",
        "CASE_ATTEMPT_STARTED",
        "STEP_STARTED",
        "SCREEN_CAPTURED",
        "ACTION_RECORDED",
        "STEP_FINISHED",
        "CASE_ATTEMPT_FINISHED",
        "CASE_FINISHED",
        "ARTIFACT_STAGED",
        "ARTIFACT_UPLOADED",
        "CANCEL_ACKNOWLEDGED",
        "RUN_FINISHED",
        "RUN_ERROR",
    }
)

_SENSITIVE_KEY = re.compile(
    r"(?i)(thinking|password|passwd|cookie|authorization|api[_-]?key|"
    r"(?:^|[_-])token(?:$|[_-])|secret|credential)"
)
_LABELLED_SECRET = re.compile(
    r"(?i)(password|passwd|cookie|authorization|api[_ -]?key|"
    r"gitlab[_ -]?token|lease[_ -]?token|worker[_ -]?credential|secret)"
    r"\s*[:=]\s*[^\s,;]+"
)
_INPUT_ACTIONS = frozenset({"type", "type_name", "typeintoelement", "input"})


class OutboxLifecycleSink:
    def __init__(
        self,
        *,
        outbox: DurableOutbox,
        task_run_id: UUID,
        producer_id: UUID,
        lease_credential_ref: str,
        fencing_token: int,
        adhoc_execution_item_id: UUID | None = None,
    ) -> None:
        self.outbox = outbox
        self.task_run_id = task_run_id
        self.producer_id = producer_id
        self.lease_credential_ref = lease_credential_ref
        self.fencing_token = fencing_token
        self.adhoc_execution_item_id = (
            str(adhoc_execution_item_id) if adhoc_execution_item_id else None
        )

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type not in PLATFORM_EVENT_TYPES:
            logger.debug(
                "Lifecycle event retained as local telemetry only "
                "task_run_id=%s event_type=%s",
                self.task_run_id,
                event_type,
            )
            return
        event_data = _sanitize_event_data(data)
        if self.adhoc_execution_item_id:
            forbidden = (
                "execution_case_id",
                "case_attempt_id",
                "case_attempt_no",
                "step_id",
            )
            if any(event_data.get(name) is not None for name in forbidden):
                raise ValueError(
                    "ADHOC events must not reference Case, Attempt, or step_id"
                )
            reported_item_id = event_data.get("execution_item_id")
            if reported_item_id not in (None, self.adhoc_execution_item_id):
                raise ValueError(
                    "ADHOC event execution_item_id does not match the Plan"
                )
            event_data["execution_item_id"] = self.adhoc_execution_item_id
        sequence = self.outbox.next_producer_sequence(str(self.producer_id))
        idempotency_key = f"{self.task_run_id}:{self.producer_id}:{sequence}"
        payload = {
            "schema_version": "autoglm.event.v1",
            "event_id": str(uuid7()),
            "idempotency_key": idempotency_key,
            "task_run_id": str(self.task_run_id),
            "producer_id": str(self.producer_id),
            "producer_seq": sequence,
            "fencing_token": self.fencing_token,
            "type": event_type,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "execution_case_id": event_data.pop("execution_case_id", None),
            "case_attempt_id": event_data.pop("case_attempt_id", None),
            "case_attempt_no": event_data.pop("case_attempt_no", None),
            "step_id": event_data.pop("step_id", None),
            "data": event_data,
        }
        self.outbox.enqueue(
            idempotency_key=idempotency_key,
            kind="EVENT",
            payload=payload,
            task_run_id=str(self.task_run_id),
            lease_credential_ref=self.lease_credential_ref,
            fencing_token=self.fencing_token,
            producer_id=str(self.producer_id),
            producer_seq=sequence,
        )


def _sanitize_event_data(data: dict[str, Any]) -> dict[str, Any]:
    """Remove model reasoning and bearer/user secrets before durable storage."""
    known_secrets = tuple(
        value
        for key, value in os.environ.items()
        if value and _SENSITIVE_KEY.search(key) and len(value) >= 4
    )

    def clean(value: Any, key: str | None = None) -> Any:
        if key and _SENSITIVE_KEY.search(key):
            return "[REDACTED]"
        if isinstance(value, dict):
            result = {str(k): clean(v, str(k)) for k, v in value.items()}
            action_name = str(
                value.get("action") or value.get("_metadata") or ""
            ).lower()
            if action_name in _INPUT_ACTIONS:
                for input_key in ("text", "input_text"):
                    if input_key in result:
                        result[input_key] = "[REDACTED]"
            return result
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, str):
            rendered = value
            for secret in known_secrets:
                rendered = rendered.replace(secret, "[REDACTED]")
            return _LABELLED_SECRET.sub(
                lambda match: f"{match.group(1)}=[REDACTED]", rendered
            )
        return value

    return clean(dict(data))


class OutboxPump:
    """Best-effort sender; durable state remains authoritative locally until ACK."""

    def __init__(self, outbox: DurableOutbox, api, sealer) -> None:
        self.outbox = outbox
        self.api = api
        self.sealer = sealer

    def flush_once(self, limit: int = 100) -> int:
        sent = 0
        for item in self.outbox.due(limit):
            try:
                if item.kind == "EVENT":
                    payload = dict(item.payload)
                    if not item.lease_credential_ref:
                        raise ValueError("EVENT is missing lease credential reference")
                    payload["lease_token"] = self.outbox.load_credential(
                        item.lease_credential_ref, self.sealer
                    )
                    self.api.events_batch(
                        item.task_run_id,
                        {"events": [payload]},
                    )
                elif item.kind == "RUN_COMPLETE":
                    payload = dict(item.payload)
                    if not item.lease_credential_ref:
                        raise ValueError(
                            "RUN_COMPLETE is missing lease credential reference"
                        )
                    payload["lease_token"] = self.outbox.load_credential(
                        item.lease_credential_ref, self.sealer
                    )
                    self.api.complete_run(item.task_run_id, payload)
                else:
                    continue
                self.outbox.acknowledge(item.id)
                sent += 1
            except Exception as error:
                code = getattr(error, "code", None)
                if getattr(code, "value", code) == "LEASE_LOST":
                    self.outbox.mark_orphaned_for_run(item.task_run_id, str(error))
                elif getattr(error, "retryable", False):
                    self.outbox.retry(item.id, str(error))
                else:
                    self.outbox.mark_failed(item.id, str(error))
        return sent
