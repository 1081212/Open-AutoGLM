"""SQLite WAL durable outbox and sealed local credential references."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass(frozen=True, slots=True)
class OutboxItem:
    id: int
    idempotency_key: str
    task_run_id: str | None
    lease_credential_ref: str | None
    artifact_upload_credential_ref: str | None
    fencing_token: int | None
    producer_id: str | None
    producer_seq: int | None
    kind: str
    payload: dict[str, Any]
    local_path: str | None
    retry_count: int


class DurableOutbox:
    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._migrate()
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass

    def _migrate(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    task_run_id TEXT,
                    lease_credential_ref TEXT,
                    artifact_upload_credential_ref TEXT,
                    fencing_token INTEGER,
                    producer_id TEXT,
                    producer_seq INTEGER,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    local_path TEXT,
                    state TEXT NOT NULL DEFAULT 'PENDING',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    next_retry_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_error TEXT,
                    acknowledged_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_outbox_due
                    ON outbox(state, next_retry_at, id);
                CREATE TABLE IF NOT EXISTS credentials (
                    credential_ref TEXT PRIMARY KEY,
                    ciphertext BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_run_local_state (
                    task_run_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    claim_json TEXT NOT NULL,
                    plan_ready INTEGER NOT NULL DEFAULT 0,
                    plan_accepted INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS producer_sequences (
                    producer_id TEXT PRIMARY KEY,
                    last_sequence INTEGER NOT NULL
                );
                """
            )

    def enqueue(
        self,
        *,
        idempotency_key: str,
        kind: str,
        payload: dict[str, Any],
        task_run_id: str | None = None,
        lease_credential_ref: str | None = None,
        artifact_upload_credential_ref: str | None = None,
        fencing_token: int | None = None,
        producer_id: str | None = None,
        producer_seq: int | None = None,
        local_path: str | None = None,
    ) -> int:
        now = _now()
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO outbox(
                    idempotency_key, task_run_id, lease_credential_ref,
                    artifact_upload_credential_ref, fencing_token, producer_id,
                    producer_seq, kind, payload_json, local_path,
                    state, next_retry_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING
                """,
                (
                    idempotency_key,
                    task_run_id,
                    lease_credential_ref,
                    artifact_upload_credential_ref,
                    fencing_token,
                    producer_id,
                    producer_seq,
                    kind,
                    serialized,
                    local_path,
                    now,
                    now,
                ),
            )
            row = self._connection.execute(
                "SELECT id FROM outbox WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            assert row is not None
            return int(row["id"])

    def due(self, limit: int = 100) -> tuple[OutboxItem, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM outbox
                WHERE state = 'PENDING' AND next_retry_at <= ?
                ORDER BY id LIMIT ?
                """,
                (_now(), limit),
            ).fetchall()
        return tuple(_to_item(row) for row in rows)

    def acknowledge(self, item_id: int) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE outbox SET state='ACKED', acknowledged_at=? WHERE id=? AND state='PENDING'",
                (_now(), item_id),
            )

    def retry(self, item_id: int, error: str, *, base_seconds: int = 2) -> None:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT retry_count FROM outbox WHERE id=? AND state='PENDING'",
                (item_id,),
            ).fetchone()
            if row is None:
                return
            retry_count = int(row["retry_count"]) + 1
            delay = min(300, base_seconds * (2 ** min(retry_count - 1, 8)))
            jitter = secrets.randbelow(max(1, delay * 250)) / 1000
            next_retry = datetime.now(timezone.utc) + timedelta(seconds=delay + jitter)
            self._connection.execute(
                """
                UPDATE outbox
                SET retry_count=?, next_retry_at=?, last_error=?
                WHERE id=? AND state='PENDING'
                """,
                (retry_count, next_retry.isoformat(), error[:2000], item_id),
            )

    def mark_failed(self, item_id: int, error: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE outbox SET state='FAILED', last_error=?
                WHERE id=? AND state='PENDING'
                """,
                (error[:2000], item_id),
            )

    def next_producer_sequence(self, producer_id: str) -> int:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO producer_sequences(producer_id, last_sequence)
                VALUES (?, 0) ON CONFLICT(producer_id) DO NOTHING
                """,
                (producer_id,),
            )
            self._connection.execute(
                "UPDATE producer_sequences SET last_sequence=last_sequence+1 WHERE producer_id=?",
                (producer_id,),
            )
            row = self._connection.execute(
                "SELECT last_sequence FROM producer_sequences WHERE producer_id=?",
                (producer_id,),
            ).fetchone()
            assert row is not None
            return int(row["last_sequence"])

    def producer_positions(self) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT producer_id, last_sequence FROM producer_sequences"
            ).fetchall()
        return {str(row["producer_id"]): int(row["last_sequence"]) for row in rows}

    def producer_positions_for_run(self, task_run_id: str) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT producer_id, MAX(producer_seq) AS last_sequence
                FROM outbox
                WHERE task_run_id=? AND kind='EVENT' AND producer_id IS NOT NULL
                GROUP BY producer_id
                """,
                (task_run_id,),
            ).fetchall()
        return {str(row["producer_id"]): int(row["last_sequence"]) for row in rows}

    def mark_orphaned_for_run(self, task_run_id: str, reason: str) -> int:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE outbox SET state='ORPHANED', last_error=?
                WHERE task_run_id=? AND state='PENDING'
                  AND kind NOT IN ('ARTIFACT_UPLOAD', 'ARTIFACT_COMPLETE')
                """,
                (reason[:2000], task_run_id),
            )
            return cursor.rowcount

    def pending_count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM outbox WHERE state='PENDING'"
            ).fetchone()
            return int(row["count"])

    def pending_count_for_run(
        self, task_run_id: str, *, kind: str | None = None
    ) -> int:
        query = "SELECT COUNT(*) AS count FROM outbox WHERE state='PENDING' AND task_run_id=?"
        parameters: tuple[object, ...] = (task_run_id,)
        if kind is not None:
            query += " AND kind=?"
            parameters = (task_run_id, kind)
        with self._lock:
            row = self._connection.execute(query, parameters).fetchone()
            return int(row["count"])

    def events_for_run(self, task_run_id: str) -> tuple[dict[str, Any], ...]:
        """Return event payloads in their original durable append order."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload_json FROM outbox
                WHERE task_run_id=? AND kind='EVENT'
                ORDER BY id
                """,
                (task_run_id,),
            ).fetchall()
        return tuple(json.loads(row["payload_json"]) for row in rows)

    def unacknowledged_event_count_for_run(self, task_run_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count FROM outbox
                WHERE task_run_id=? AND kind='EVENT' AND state!='ACKED'
                """,
                (task_run_id,),
            ).fetchone()
        return int(row["count"])

    def unacknowledged_count_for_run(self, task_run_id: str, *, kind: str) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count FROM outbox
                WHERE task_run_id=? AND kind=? AND state!='ACKED'
                """,
                (task_run_id, kind),
            ).fetchone()
        return int(row["count"])

    def save_task_run_state(
        self,
        task_run_id: str,
        *,
        state: str,
        claim: dict[str, Any],
        plan_ready: bool = False,
        plan_accepted: bool = False,
    ) -> None:
        serialized = json.dumps(claim, ensure_ascii=False, separators=(",", ":"))
        lowered = serialized.lower()
        if any(
            name in lowered
            for name in ("lease_token", "artifact_upload_token", "plaintext_dek")
        ):
            raise ValueError(
                "task run state must contain credential references, not bearer secrets"
            )
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO task_run_local_state(
                    task_run_id, state, claim_json, plan_ready, plan_accepted, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_run_id) DO UPDATE SET
                    state=excluded.state,
                    claim_json=excluded.claim_json,
                    plan_ready=excluded.plan_ready,
                    plan_accepted=excluded.plan_accepted,
                    updated_at=excluded.updated_at
                """,
                (
                    task_run_id,
                    state,
                    serialized,
                    int(plan_ready),
                    int(plan_accepted),
                    _now(),
                ),
            )

    def active_task_runs(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM task_run_local_state
                WHERE state IN ('ACTIVE', 'FINALIZING')
                ORDER BY updated_at
                """
            ).fetchall()
        return tuple(
            {
                "task_run_id": row["task_run_id"],
                "state": row["state"],
                "claim": json.loads(row["claim_json"]),
                "plan_ready": bool(row["plan_ready"]),
                "plan_accepted": bool(row["plan_accepted"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        )

    def save_credential(
        self, credential_ref: str, secret: str, sealer: "LocalSealer"
    ) -> None:
        encrypted = sealer.seal(credential_ref, secret.encode())
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO credentials(credential_ref, ciphertext, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(credential_ref) DO UPDATE SET ciphertext=excluded.ciphertext
                """,
                (credential_ref, encrypted, _now()),
            )

    def load_credential(self, credential_ref: str, sealer: "LocalSealer") -> str:
        with self._lock:
            row = self._connection.execute(
                "SELECT ciphertext FROM credentials WHERE credential_ref=?",
                (credential_ref,),
            ).fetchone()
        if row is None:
            raise KeyError(credential_ref)
        return sealer.open(credential_ref, bytes(row["ciphertext"])).decode()

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class LocalSealer:
    def __init__(self, key_path: str | os.PathLike[str]) -> None:
        self.key_path = Path(key_path)
        self.key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._key = self._load_or_create_key()

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            key = self.key_path.read_bytes()
            if len(key) != 32:
                raise ValueError("local sealing key must be exactly 32 bytes")
            return key
        key = AESGCM.generate_key(bit_length=256)
        fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(key)
            handle.flush()
            os.fsync(handle.fileno())
        return key

    def seal(self, credential_ref: str, plaintext: bytes) -> bytes:
        nonce = os.urandom(12)
        return nonce + AESGCM(self._key).encrypt(
            nonce, plaintext, credential_ref.encode()
        )

    def open(self, credential_ref: str, sealed: bytes) -> bytes:
        if len(sealed) < 28:
            raise ValueError("sealed credential is truncated")
        return AESGCM(self._key).decrypt(
            sealed[:12], sealed[12:], credential_ref.encode()
        )


def _to_item(row: sqlite3.Row) -> OutboxItem:
    return OutboxItem(
        id=int(row["id"]),
        idempotency_key=str(row["idempotency_key"]),
        task_run_id=row["task_run_id"],
        lease_credential_ref=row["lease_credential_ref"],
        artifact_upload_credential_ref=row["artifact_upload_credential_ref"],
        fencing_token=row["fencing_token"],
        producer_id=row["producer_id"],
        producer_seq=row["producer_seq"],
        kind=str(row["kind"]),
        payload=json.loads(row["payload_json"]),
        local_path=row["local_path"],
        retry_count=int(row["retry_count"]),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
