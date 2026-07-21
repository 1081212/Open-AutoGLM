from __future__ import annotations

import pytest
from uuid import uuid4

from phone_agent.worker.config import WorkerConfig


def base_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOGLM_PLATFORM_BASE_URL", "http://platform.internal")
    monkeypatch.setenv("AUTOGLM_REDIS_URL", "redis://redis.internal:6379/0")
    monkeypatch.setenv("AUTOGLM_WORKER_CREDENTIAL", "test-worker-credential")
    monkeypatch.setenv("AUTOGLM_WORKER_ID", str(uuid4()))
    monkeypatch.setenv("AUTOGLM_WORKER_SPOOL_ROOT", str(tmp_path / "spool"))


def test_runtime_environment_is_required_and_lowercase(monkeypatch, tmp_path):
    base_environment(monkeypatch, tmp_path)
    monkeypatch.delenv("AUTOGLM_RUNTIME_ENVIRONMENT", raising=False)
    with pytest.raises(ValueError, match="required"):
        WorkerConfig.from_env()

    monkeypatch.setenv("AUTOGLM_RUNTIME_ENVIRONMENT", "DEV")
    with pytest.raises(ValueError, match="lowercase"):
        WorkerConfig.from_env()


@pytest.mark.parametrize("environment", ["dev", "prod"])
def test_environment_scopes_local_spool(monkeypatch, tmp_path, environment):
    base_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AUTOGLM_RUNTIME_ENVIRONMENT", environment)

    config = WorkerConfig.from_env()

    assert config.runtime_environment == environment
    assert config.spool_root == tmp_path / "spool" / environment
    assert config.worker_id_path.parent == config.spool_root


def test_redis_database_other_than_zero_is_rejected(monkeypatch, tmp_path):
    base_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AUTOGLM_RUNTIME_ENVIRONMENT", "dev")
    monkeypatch.setenv("AUTOGLM_REDIS_URL", "redis://redis.internal:6379/1")

    with pytest.raises(ValueError, match="database 0"):
        WorkerConfig.from_env()


def test_unscoped_legacy_spool_blocks_environment_upgrade(monkeypatch, tmp_path):
    base_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AUTOGLM_RUNTIME_ENVIRONMENT", "dev")
    legacy_root = tmp_path / "spool"
    legacy_root.mkdir()
    (legacy_root / "worker.db").write_bytes(b"legacy")

    with pytest.raises(ValueError, match="unscoped legacy Worker spool"):
        WorkerConfig.from_env()


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_claim_enabled_boolean(monkeypatch, tmp_path, value, expected):
    base_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("AUTOGLM_RUNTIME_ENVIRONMENT", "dev")
    monkeypatch.setenv("AUTOGLM_WORKER_CLAIM_ENABLED", value)

    assert WorkerConfig.from_env().claim_enabled is expected
