"""Single-message Redis Stream consumer; BUSY workers never prefetch."""

from __future__ import annotations

import json
from uuid import UUID

from pydantic import ValidationError
from redis import Redis
from redis.exceptions import ResponseError

from phone_agent.worker.models import DispatchNotification, RedisMessage
from phone_agent.worker.config import RuntimeEnvironment, parse_runtime_environment


class RedisDispatchNotifier:
    def __init__(
        self,
        redis_url: str,
        worker_id: UUID,
        instance_id: UUID,
        runtime_environment: RuntimeEnvironment,
        *,
        client: Redis | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.runtime_environment = parse_runtime_environment(runtime_environment)
        self.stream = f"autoglm:{self.runtime_environment}:v1:dispatch:{worker_id}"
        self.group = f"worker:{self.runtime_environment}:{worker_id}"
        self.consumer = str(instance_id)
        self.client = client or Redis.from_url(redis_url, decode_responses=True)
        try:
            self.client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def read_one(self, *, worker_busy: bool, block_ms: int = 1000) -> RedisMessage | None:
        if worker_busy:
            return None
        rows = self.client.xreadgroup(
            self.group,
            self.consumer,
            {self.stream: ">"},
            count=1,
            block=block_ms,
        )
        if not rows:
            return None
        _, messages = rows[0]
        redis_id, fields = messages[0]
        payload = _decode_payload(fields)
        try:
            notification = DispatchNotification.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"invalid dispatch notification {redis_id}: {exc}") from exc
        if notification.worker_id != self.worker_id:
            raise ValueError("dispatch notification belongs to another worker")
        return RedisMessage(redis_id=redis_id, notification=notification, raw_fields=fields)

    def acknowledge(self, message: RedisMessage) -> None:
        self.client.xack(self.stream, self.group, message.redis_id)

    def close(self) -> None:
        self.client.close()


def _decode_payload(fields: dict[str, str]) -> dict[str, object]:
    if "payload" in fields:
        value = json.loads(fields["payload"])
        if not isinstance(value, dict):
            raise ValueError("Redis payload must be a JSON object")
        return value
    return dict(fields)
