from __future__ import annotations

import pytest

from phone_agent.worker.api_client import WorkerApiClient


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"registered": True}


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.posts = []

    def post(self, url, *, json, timeout, headers):
        self.posts.append((url, json, timeout, headers))
        return FakeResponse()


def test_register_requires_and_sends_runtime_environment():
    session = FakeSession()
    client = WorkerApiClient(
        "http://platform.internal",
        "test-credential",
        "prod",
        session=session,
    )

    response = client.register({"worker_id": "worker"})

    assert response == {"registered": True}
    assert session.posts[0][0].endswith("/worker/v1/register")
    assert session.posts[0][1]["runtime_environment"] == "prod"


def test_register_rejects_cross_environment_override_before_http():
    session = FakeSession()
    client = WorkerApiClient(
        "http://platform.internal",
        "test-credential",
        "dev",
        session=session,
    )

    with pytest.raises(ValueError, match="does not match"):
        client.register({"worker_id": "worker", "runtime_environment": "prod"})

    assert session.posts == []


def test_control_heartbeat_is_environment_bound_by_client():
    session = FakeSession()
    client = WorkerApiClient(
        "http://platform.internal",
        "test-credential",
        "dev",
        session=session,
    )

    client.heartbeat({"worker_id": "worker"})

    assert session.posts[0][0].endswith("/worker/v1/heartbeat")
    assert session.posts[0][1]["runtime_environment"] == "dev"
