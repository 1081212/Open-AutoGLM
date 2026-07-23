"""Worker API client with fixed v1 paths and bounded timeouts."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import json
import re
from typing import Any, Iterator
from urllib.parse import urljoin

import requests
from pydantic import ValidationError

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
        try:
            return ClaimResponse.model_validate(response)
        except ValidationError as exc:
            raise ExecutionError(
                ExecutionErrorCode.PLAN_INVALID,
                "Worker API claim response violates the Claim contract",
            ) from exc

    @contextmanager
    def plan_chunks(
        self, download_url: str, *, chunk_size: int = 1024 * 1024
    ) -> Iterator[Iterator[bytes]]:
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

    def plan_accepted(
        self, task_run_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._post(f"worker/v1/task-runs/{task_run_id}:plan-accepted", payload)

    def run_heartbeat(
        self, task_run_id: str, payload: dict[str, Any]
    ) -> RunHeartbeatResponse:
        data = self._post(f"worker/v1/task-runs/{task_run_id}/heartbeat", payload)
        return RunHeartbeatResponse.model_validate(data)

    def events_batch(self, task_run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"worker/v1/task-runs/{task_run_id}/events:batch", payload)

    def begin_attempt(
        self, task_run_id: str, execution_case_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._post(
            f"worker/v1/task-runs/{task_run_id}/cases/{execution_case_id}/attempts:begin",
            payload,
        )

    def checkpoint(
        self, task_run_id: str, execution_case_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._post(
            f"worker/v1/task-runs/{task_run_id}/cases/{execution_case_id}:checkpoint",
            payload,
        )

    def complete_attempt(
        self, attempt_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._post(f"worker/v1/case-attempts/{attempt_id}:complete", payload)

    def initiate_artifact(
        self, task_run_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._post(
            f"worker/v1/task-runs/{task_run_id}/artifacts:initiate", payload
        )

    def refresh_artifact_upload(self, artifact_id: str, token: str) -> dict[str, Any]:
        return self._post(
            f"worker/v1/artifacts/{artifact_id}:refresh-upload",
            {},
            extra_headers={"X-Artifact-Upload-Token": token},
        )

    def complete_artifact(
        self, artifact_id: str, payload: dict[str, Any], token: str
    ) -> dict[str, Any]:
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
        if not 200 <= response.status_code < 300:
            message = _safe_error_message(
                response,
                _request_secret_values(payload, self.session.headers, extra_headers),
            )
            if response.status_code in {401, 403}:
                raise ExecutionError(ExecutionErrorCode.CLAIM_REJECTED, message)
            if response.status_code == 409:
                if _platform_error_code(response) == "LEASE_LOST":
                    raise ExecutionError(ExecutionErrorCode.LEASE_LOST, message)
                raise ExecutionError(ExecutionErrorCode.EXECUTION_ERROR, message)
            if response.status_code == 429 or response.status_code >= 500:
                raise ExecutionError(
                    ExecutionErrorCode.RETRYABLE_ERROR,
                    message,
                    retryable=True,
                )
            raise ExecutionError(
                ExecutionErrorCode.EXECUTION_ERROR,
                message,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ExecutionError(
                ExecutionErrorCode.EXECUTION_ERROR,
                f"Worker API invalid response ({response.status_code})",
            ) from exc
        if not isinstance(data, dict):
            raise ExecutionError(
                ExecutionErrorCode.EXECUTION_ERROR,
                "Worker API response must be an object",
            )
        return data

    def _url(self, path_or_url: str) -> str:
        return urljoin(self.base_url, path_or_url.lstrip("/"))


_MAX_ERROR_RESPONSE_CHARS = 4096
_MAX_ERROR_FIELD_CHARS = 512


def _platform_error_code(response: requests.Response) -> str | None:
    """Read only the bounded platform error code for status classification."""
    try:
        text = response.text
    except (AttributeError, UnicodeError):
        return None
    if len(text) > _MAX_ERROR_RESPONSE_CHARS:
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    code = payload.get("code")
    if not isinstance(code, str) or len(code) > _MAX_ERROR_FIELD_CHARS:
        return None
    return code


def _safe_error_message(
    response: requests.Response, secrets: set[str] | None = None
) -> str:
    """Expose only bounded fields from the platform's standard ErrorResponse."""
    fallback = f"Worker API rejected request: HTTP {response.status_code}"
    try:
        text = response.text
    except (AttributeError, UnicodeError):
        return fallback
    if len(text) > _MAX_ERROR_RESPONSE_CHARS:
        return fallback
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    code = _bounded_error_field(payload.get("code"), secrets)
    message = _bounded_error_field(payload.get("message"), secrets)
    request_id = _bounded_error_field(payload.get("request_id"), secrets)
    if not code or not message:
        return fallback
    rendered = f"{fallback} {code}: {message}"
    if request_id:
        rendered += f" (request_id={request_id})"
    return rendered


def _bounded_error_field(value: object, secrets: set[str] | None = None) -> str | None:
    if not isinstance(value, str):
        return None
    # Prevent forged multi-line log entries while retaining the useful diagnosis.
    bounded = " ".join(value.split())[:_MAX_ERROR_FIELD_CHARS]
    for secret in secrets or ():
        if secret:
            bounded = bounded.replace(secret, "[REDACTED]")
    # Also redact labelled credentials if a server accidentally echoes one that
    # was not part of this particular request.
    bounded = re.sub(
        r"(?i)(lease[_ -]?token|credential|gitlab[_ -]?token|authorization)"
        r"\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        bounded,
    )
    return bounded or None


def _request_secret_values(*sources: object) -> set[str]:
    secrets: set[str] = set()

    def collect(value: object, sensitive: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                collect(
                    child,
                    sensitive
                    or any(
                        marker in lowered
                        for marker in (
                            "token",
                            "credential",
                            "secret",
                            "authorization",
                            "plaintext_dek",
                        )
                    ),
                )
        elif sensitive and isinstance(value, str):
            secrets.add(value)
            if value.lower().startswith("bearer "):
                secrets.add(value[7:])

    for source in sources:
        collect(source)
    return secrets
