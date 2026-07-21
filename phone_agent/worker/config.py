"""Validated worker configuration loaded from environment and CLI overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlparse
from uuid import UUID


RuntimeEnvironment = Literal["dev", "prod"]


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    worker_id: UUID
    runtime_environment: RuntimeEnvironment
    platform_base_url: str
    worker_credential: str
    redis_url: str
    spool_root: Path
    worker_id_path: Path
    sealing_key_path: Path
    model_profiles_path: Path
    heartbeat_seconds: int = 10
    discovery_seconds: int = 10
    lease_safety_seconds: int = 5
    device_allowlist: frozenset[str] | None = None
    claim_enabled: bool = True
    platform_reporter_enabled: bool = True
    encrypted_cos_enabled: bool = True
    legacy_local_report_enabled: bool = True

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        runtime_environment = parse_runtime_environment(
            os.getenv("AUTOGLM_RUNTIME_ENVIRONMENT", "")
        )
        spool_base = Path(
            os.getenv("AUTOGLM_WORKER_SPOOL_ROOT", "worker_spool")
        ).expanduser()
        _reject_unscoped_legacy_spool(spool_base)
        spool_root = spool_base / runtime_environment
        if spool_root.is_symlink():
            raise ValueError(
                "environment-specific Worker spool directory must not be a symlink"
            )
        credential = os.getenv("AUTOGLM_WORKER_CREDENTIAL", "")
        if not credential:
            raise ValueError("AUTOGLM_WORKER_CREDENTIAL is required")
        allowlist_raw = os.getenv("AUTOGLM_DEVICE_ALLOWLIST", "").strip()
        allowlist = (
            frozenset(item.strip() for item in allowlist_raw.split(",") if item.strip())
            if allowlist_raw
            else None
        )
        return cls(
            worker_id=_required_uuid("AUTOGLM_WORKER_ID"),
            runtime_environment=runtime_environment,
            platform_base_url=_required_url("AUTOGLM_PLATFORM_BASE_URL"),
            worker_credential=credential,
            redis_url=_redis_url("AUTOGLM_REDIS_URL"),
            spool_root=spool_root,
            worker_id_path=spool_root / "worker-id",
            sealing_key_path=Path(
                os.getenv("AUTOGLM_WORKER_SEALING_KEY", str(spool_root / "sealing-key"))
            ).expanduser(),
            model_profiles_path=Path(
                os.getenv("AUTOGLM_MODEL_PROFILES_FILE", "worker-model-profiles.yaml")
            ).expanduser(),
            heartbeat_seconds=_positive_int("AUTOGLM_HEARTBEAT_SECONDS", 10),
            discovery_seconds=_positive_int("AUTOGLM_DISCOVERY_SECONDS", 10),
            lease_safety_seconds=_positive_int("AUTOGLM_LEASE_SAFETY_SECONDS", 5),
            device_allowlist=allowlist,
            claim_enabled=_flag("AUTOGLM_WORKER_CLAIM_ENABLED", True),
            platform_reporter_enabled=_flag("AUTOGLM_PLATFORM_REPORTER_ENABLED", True),
            encrypted_cos_enabled=_flag("AUTOGLM_ENCRYPTED_COS_ENABLED", True),
            legacy_local_report_enabled=_flag(
                "AUTOGLM_LEGACY_LOCAL_REPORT_ENABLED", True
            ),
        )


def _required_url(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value.rstrip("/")


def _required_uuid(name: str) -> UUID:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    try:
        return UUID(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a UUID") from exc


def _redis_url(name: str) -> str:
    value = _required_url(name)
    parsed = urlparse(value)
    if parsed.scheme not in {"redis", "rediss"}:
        raise ValueError(f"{name} must use redis:// or rediss://")
    database = parsed.path.strip("/")
    if database not in {"", "0"}:
        raise ValueError(
            f"{name} must use Redis database 0; environment isolation uses key prefixes"
        )
    return value


def parse_runtime_environment(value: str) -> RuntimeEnvironment:
    if value not in {"dev", "prod"}:
        raise ValueError(
            "AUTOGLM_RUNTIME_ENVIRONMENT is required and must be lowercase dev or prod"
        )
    return cast(RuntimeEnvironment, value)


def _reject_unscoped_legacy_spool(spool_base: Path) -> None:
    legacy_entries = (
        "worker.db",
        "worker-id",
        "sealing-key",
        "task-runs",
        "device-locks",
    )
    found = [name for name in legacy_entries if (spool_base / name).exists()]
    if found:
        raise ValueError(
            "unscoped legacy Worker spool detected; verify whether it belongs to dev or prod "
            "and migrate it into that environment subdirectory before startup: "
            + ", ".join(found)
        )


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")
