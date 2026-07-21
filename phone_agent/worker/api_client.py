"""Worker API client with fixed v1 paths and bounded timeouts."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import urljoin

import requests

from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.worker.models import ClaimResponse, RunHeartbeatResponse
from phone_agent.worker.config import RuntimeEnvironment, parse_runtime_environment


class WorkerApiClient:
    def __init__(
        self,
        base_url: str,
        worker_credential: str,
        runtime_environment: RuntimeEnvironment,
        *,
        timeout: tuple[float, float] = (5, 30),
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.runtime_environment = parse_runtime_environment(runtime_environment)
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {worker_credential}",
                "Accept": "application/json",
                "User-Agent": "open-autoglm-worker/0.3",
            }
        )

    def heartbeat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(
            "worker/v1/heartbeat",
            self._environment_bound_payload(payload, "heartbeat"),
        )

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        """First-install bootstrap; payload must include runtime_environment."""
        return self._post(
            "worker/v1/register",
            self._environment_bound_payload(payload, "register"),
        )

    def _environment_bound_payload(
        self, payload: dict[str, Any], operation: str
    ) -> dict[str, Any]:
        declared = payload.get("runtime_environment")
        if declared is not None and declared != self.runtime_environment:
            raise ValueError(
                f"{operation} runtime_environment does not match WorkerApiClient environment"
            )
        return {**payload, "runtime_environment": self.runtime_environment}

    def claim(self, dispatch_id: str, payload: dict[str, Any]) -> ClaimResponse:
        response = self._post(f"worker/v1/dispatches/{dispatch_id}:claim", payload)
        return ClaimResponse.model_validate(response)

    @contextmanager
    def plan_chunks(self, download_url: str, *, chunk_size: int = 1024 * 1024) -> Iterator[Iterator[bytes]]:
        url = self._url(download_url)
        try:
            response = self.session.get(
                url,
                stream=True,
                timeout=self.timeout,
                headers={"Accept-Encoding": "identity"},
            )
            response.raise_for_status()
            response.raw.decode_content = False
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
            if content_type != "application/vnd.autoglm.execution-plan+gzip":
                raise ExecutionError(
                    ExecutionErrorCode.PLAN_INVALID,
                    f"unexpected plan Content-Type: {content_type or '<missing>'}",
                )
            yield iter(lambda: response.raw.read(chunk_size), b"")
        except requests.RequestException as exc:
            raise ExecutionError(
                ExecutionErrorCode.RETRYABLE_ERROR,
                f"plan download failed: {type(exc).__name__}",
                retryable=True,
            ) from exc
        finally:
            if "response" in locals():
                response.close()

    def plan_accepted(self, task_run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"worker/v1/task-runs/{task_run_id}:plan-accepted", payload)

    def run_heartbeat(self, task_run_id: str, payload: dict[str, Any]) -> RunHeartbeatResponse:
        data = self._post(f"worker/v1/task-runs/{task_run_id}/heartbeat", payload)
        return RunHeartbeatResponse.model_validate(data)

    def events_batch(self, task_run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"worker/v1/task-runs/{task_run_id}/events:batch", payload)

    def begin_attempt(self, task_run_id: str, execution_case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(
            f"worker/v1/task-runs/{task_run_id}/cases/{execution_case_id}/attempts:begin",
            payload,
        )

    def checkpoint(self, task_run_id: str, execution_case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(
            f"worker/v1/task-runs/{task_run_id}/cases/{execution_case_id}:checkpoint",
            payload,
        )

    def complete_attempt(self, attempt_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"worker/v1/case-attempts/{attempt_id}:complete", payload)

    def initiate_artifact(self, task_run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"worker/v1/task-runs/{task_run_id}/artifacts:initiate", payload)

    def refresh_artifact_upload(self, artifact_id: str, token: str) -> dict[str, Any]:
        return self._post(
            f"worker/v1/artifacts/{artifact_id}:refresh-upload",
            {},
            extra_headers={"X-Artifact-Upload-Token": token},
        )

    def complete_artifact(self, artifact_id: str, payload: dict[str, Any], token: str) -> dict[str, Any]:
        return self._post(
            f"worker/v1/artifacts/{artifact_id}:complete",
            payload,
            extra_headers={"X-Artifact-Upload-Token": token},
        )

    def complete_run(self, task_run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"worker/v1/task-runs/{task_run_id}:complete", payload)

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self.session.post(
                self._url(path),
                json=payload,
                timeout=self.timeout,
                headers=extra_headers,
            )
        except requests.RequestException as exc:
            raise ExecutionError(
                ExecutionErrorCode.RETRYABLE_ERROR,
                f"Worker API request failed: {type(exc).__name__}",
                retryable=True,
            ) from exc
        if response.status_code in {401, 403}:
            raise ExecutionError(ExecutionErrorCode.CLAIM_REJECTED, "Worker API authorization rejected")
        if response.status_code == 409:
            raise ExecutionError(ExecutionErrorCode.LEASE_LOST, "Worker API rejected lease or fence")
        if response.status_code == 429 or response.status_code >= 500:
            raise ExecutionError(
                ExecutionErrorCode.RETRYABLE_ERROR,
                f"Worker API temporary status {response.status_code}",
                retryable=True,
            )
        try:
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ExecutionError(
                ExecutionErrorCode.EXECUTION_ERROR,
                f"Worker API invalid response ({response.status_code})",
            ) from exc
        if not isinstance(data, dict):
            raise ExecutionError(ExecutionErrorCode.EXECUTION_ERROR, "Worker API response must be an object")
        return data

    def _url(self, path_or_url: str) -> str:
        return urljoin(self.base_url, path_or_url.lstrip("/"))
