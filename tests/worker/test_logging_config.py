from __future__ import annotations

import logging

from phone_agent.worker.logging_config import configure_worker_logging


def test_worker_log_writes_to_spool_and_redacts_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOGLM_WORKER_CREDENTIAL", "worker-secret-value")
    monkeypatch.setenv("AUTOGLM_GITLAB_TOKEN", "gitlab-secret-value")
    logger = configure_worker_logging(spool_root=tmp_path)
    child_logger = logging.getLogger("phone_agent.worker.test")

    child_logger.info("Worker ready credential=%s", "worker-secret-value")
    try:
        raise RuntimeError("request failed with gitlab-secret-value")
    except RuntimeError:
        child_logger.exception("GitLab operation failed")
    for handler in logger.handlers:
        handler.flush()

    content = (tmp_path / "worker.log").read_text(encoding="utf-8")
    assert "Worker ready" in content
    assert "GitLab operation failed" in content
    assert "worker-secret-value" not in content
    assert "gitlab-secret-value" not in content
    assert "[REDACTED]" in content
