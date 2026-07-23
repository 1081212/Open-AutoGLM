"""Safe Worker logging to the console and environment-scoped spool."""

from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "phone_agent.worker"
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = (
    "%(asctime)s %(levelname)s pid=%(process)d thread=%(threadName)s "
    "%(name)s - %(message)s"
)
_LABELLED_SECRET = re.compile(
    r"(?i)(token|credential|api[_ -]?key|authorization|secret)" r"\s*[:=]\s*[^\s,;]+"
)


class _SecretRedactionFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.secrets = tuple(
            value
            for name, value in os.environ.items()
            if len(value) >= 8
            and any(
                marker in name.upper()
                for marker in (
                    "TOKEN",
                    "CREDENTIAL",
                    "API_KEY",
                    "SECRET",
                    "AUTHORIZATION",
                    "PLAINTEXT_DEK",
                )
            )
        )

    def filter(self, record: logging.LogRecord) -> bool:
        rendered = record.getMessage()
        for secret in self.secrets:
            rendered = rendered.replace(secret, "[REDACTED]")
        record.msg = _LABELLED_SECRET.sub(r"\1=[REDACTED]", rendered)
        record.args = ()
        return True


class _RedactingFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(LOG_FORMAT)
        self.secrets = _SecretRedactionFilter().secrets

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        for secret in self.secrets:
            rendered = rendered.replace(secret, "[REDACTED]")
        return _LABELLED_SECRET.sub(r"\1=[REDACTED]", rendered)


def configure_worker_logging(
    *,
    spool_root: Path | None = None,
    child_process: bool = False,
) -> logging.Logger:
    """Configure only Worker loggers, leaving library/application logging alone."""
    level_name = os.getenv("AUTOGLM_WORKER_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = _RedactingFormatter()
    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.setFormatter(formatter)
    stream.addFilter(_SecretRedactionFilter())
    logger.addHandler(stream)

    if spool_root is not None and not child_process:
        spool_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        log_path = spool_root / "worker.log"
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=20 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_SecretRedactionFilter())
        logger.addHandler(file_handler)
        try:
            os.chmod(log_path, 0o600)
        except OSError:
            pass
    return logger
