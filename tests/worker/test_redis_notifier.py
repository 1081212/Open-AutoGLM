from __future__ import annotations

import json
from uuid import uuid4

import pytest

from phone_agent.worker.redis_notifier import RedisDispatchNotifier


class FakeRedis:
    def __init__(self):
        self.created = None
        self.read_request = None
        self.rows = []

    def xgroup_create(self, stream, group, id, mkstream):
        self.created = (stream, group, id, mkstream)

    def xreadgroup(self, group, consumer, streams, count, block):
        self.read_request = (group, consumer, streams, count, block)
        return self.rows


@pytest.mark.parametrize("environment", ["dev", "prod"])
def test_dispatch_stream_contains_runtime_environment(environment):
    worker_id = uuid4()
    client = FakeRedis()

    notifier = RedisDispatchNotifier(
        "redis://unused/0",
        worker_id,
        uuid4(),
        environment,
        client=client,
    )

    assert notifier.stream == f"autoglm:{environment}:v1:dispatch:{worker_id}"
    assert client.created[0] == notifier.stream
    assert environment in notifier.group


def test_dispatch_stream_rejects_invalid_environment():
    with pytest.raises(ValueError, match="lowercase dev or prod"):
        RedisDispatchNotifier(
            "redis://unused/0",
            uuid4(),
            uuid4(),
            "staging",
            client=FakeRedis(),
        )


def test_dev_notifier_reads_only_dev_stream():
    worker_id = uuid4()
    device_uid = uuid4()
    client = FakeRedis()
    notifier = RedisDispatchNotifier(
        "redis://unused/0",
        worker_id,
        uuid4(),
        "dev",
        client=client,
    )
    payload = {
        "schema_version": "autoglm.dispatch.v1",
        "dispatch_id": str(uuid4()),
        "task_id": str(uuid4()),
        "task_type": "TEST_RUN",
        "plan_id": str(uuid4()),
        "plan_canonical_sha256": "sha256:test",
        "worker_id": str(worker_id),
        "device_uid": str(device_uid),
    }
    client.rows = [(notifier.stream, [("1-0", {"payload": json.dumps(payload)})])]

    message = notifier.read_one(worker_busy=False, block_ms=1)

    assert message.notification.device_uid == device_uid
    assert client.read_request[2] == {f"autoglm:dev:v1:dispatch:{worker_id}": ">"}
    assert "prod" not in next(iter(client.read_request[2]))
