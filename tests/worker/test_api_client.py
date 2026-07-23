from __future__ import annotations

import pytest

from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
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


class JsonResponse(FakeResponse):
    def __init__(self, status_code, payload):
        import json

        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class OneResponseSession(FakeSession):
    def __init__(self, response):
        super().__init__()
        self.response = response

    def post(self, url, *, json, timeout, headers):
        self.posts.append((url, json, timeout, headers))
        return self.response


def _successful_claim_payload():
    return {
        "claimed": True,
        "task_id": "376c0f29-d699-48c6-8322-2f108277b68c",
        "worker_id": "fd6b33cd-5896-489f-92bb-bcf0cba7ca75",
        "instance_id": "c3ac8227-c464-4be2-a45b-508328681db2",
        "device_uid": "3195eefd-16a6-4518-a4dc-e4d01b862921",
        "task_run_id": "5c890bcc-550e-4b6e-b310-77905eb00028",
        "lease_token": "lease-secret",
        "fencing_token": 1,
        "run_started_at": "2026-07-22T09:59:57.123456+08:00",
        "lease_expires_at": "2026-07-22T10:00:45+08:00",
        "renew_after_seconds": 10,
        "plan": {
            "schema_version": "autoglm.execution.v1",
            "plan_id": "b19356d3-4cf1-40fa-8138-39e20c931d31",
            "download_url": "/worker/v1/task-runs/run/plan",
            "wire_media_type": "application/vnd.autoglm.execution-plan+gzip",
            "inner_media_type": "application/vnd.autoglm.execution-plan+json",
            "wire_format": "gzip",
            "compressed_sha256": "sha256:" + "1" * 64,
            "compressed_size": 100,
            "canonical_sha256": "sha256:" + "2" * 64,
            "canonical_size": 200,
            "item_count": 1,
            "case_count": 1,
        },
    }


def test_claim_parses_timezone_aware_run_started_at():
    response = JsonResponse(200, _successful_claim_payload())
    client = WorkerApiClient(
        "http://platform.internal",
        "credential",
        "dev",
        session=OneResponseSession(response),
    )

    assert (
        client.claim("dispatch", {}).run_started_at
        == "2026-07-22T09:59:57.123456+08:00"
    )


def test_claim_accepts_platform_five_digit_fraction_on_python_310():
    payload = _successful_claim_payload()
    payload["run_started_at"] = "2026-07-23T06:22:32.19851Z"
    client = WorkerApiClient(
        "http://platform.internal",
        "credential",
        "dev",
        session=OneResponseSession(JsonResponse(200, payload)),
    )

    assert client.claim("dispatch", {}).run_started_at == payload["run_started_at"]


def test_successful_claim_missing_run_started_at_is_plan_invalid():
    payload = _successful_claim_payload()
    payload.pop("run_started_at")
    response = JsonResponse(200, payload)
    client = WorkerApiClient(
        "http://platform.internal",
        "credential",
        "dev",
        session=OneResponseSession(response),
    )

    with pytest.raises(ExecutionError) as caught:
        client.claim("dispatch", {})

    assert caught.value.code is ExecutionErrorCode.PLAN_INVALID
    assert "lease-secret" not in str(caught.value)


def test_platform_json_error_is_diagnostic_and_redacts_secrets():
    response = JsonResponse(
        400,
        {
            "code": "INVALID_RESULT_TIME",
            "message": (
                "Result time invalid; lease_token=lease-secret; "
                "credential=worker-secret; GitLab Token=gitlab-secret"
            ),
            "request_id": "request-123",
        },
    )
    client = WorkerApiClient(
        "http://platform.internal",
        "worker-secret",
        "dev",
        session=OneResponseSession(response),
    )

    with pytest.raises(ExecutionError) as caught:
        client.complete_run("run", {"lease_token": "lease-secret"})

    error = caught.value
    assert error.code is ExecutionErrorCode.EXECUTION_ERROR
    assert error.retryable is False
    assert "HTTP 400 INVALID_RESULT_TIME" in error.message
    assert "Result time invalid" in error.message
    assert "request_id=request-123" in error.message
    assert "lease-secret" not in error.message
    assert "worker-secret" not in error.message
    assert "gitlab-secret" not in error.message


def test_409_protocol_error_is_not_misclassified_as_lease_lost():
    response = JsonResponse(
        409,
        {
            "code": "INVALID_EVENT",
            "message": "事件信封不符合 autoglm.event.v1",
            "request_id": "request-409",
        },
    )
    client = WorkerApiClient(
        "http://platform.internal",
        "worker-secret",
        "dev",
        session=OneResponseSession(response),
    )

    with pytest.raises(ExecutionError) as caught:
        client.events_batch("run", {"events": []})

    assert caught.value.code is ExecutionErrorCode.EXECUTION_ERROR
    assert caught.value.retryable is False
    assert "INVALID_EVENT" in caught.value.message


def test_409_lease_lost_keeps_lease_classification():
    response = JsonResponse(
        409,
        {
            "code": "LEASE_LOST",
            "message": "Run lease 已失效",
            "request_id": "request-lease",
        },
    )
    client = WorkerApiClient(
        "http://platform.internal",
        "worker-secret",
        "dev",
        session=OneResponseSession(response),
    )

    with pytest.raises(ExecutionError) as caught:
        client.events_batch("run", {"events": []})

    assert caught.value.code is ExecutionErrorCode.LEASE_LOST
