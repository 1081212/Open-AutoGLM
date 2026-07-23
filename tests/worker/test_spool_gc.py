from __future__ import annotations

import sqlite3
from collections import namedtuple
from datetime import datetime, timedelta, timezone

from phone_agent.worker.outbox import DurableOutbox, LocalSealer
from phone_agent.worker.spool_gc import SpoolGarbageCollector


def save_run(
    root,
    outbox,
    run_id,
    state,
    *,
    updated_at,
    with_artifact=False,
):
    lease_ref = f"lease:{run_id}"
    sealer = LocalSealer(root / "sealing-key")
    outbox.save_credential(lease_ref, f"secret-{run_id}", sealer)
    outbox.save_task_run_state(
        run_id,
        state=state,
        claim={"lease_credential_ref": lease_ref},
    )
    with sqlite3.connect(outbox.db_path) as connection:
        connection.execute(
            "UPDATE task_run_local_state SET updated_at=? WHERE task_run_id=?",
            (updated_at.isoformat(), run_id),
        )
    for parent in ("task-runs", "local-reports"):
        path = root / parent / run_id
        path.mkdir(parents=True)
        (path / "data.bin").write_bytes(b"x" * 64)
    if with_artifact:
        outbox.enqueue(
            idempotency_key=f"artifact:{run_id}",
            kind="ARTIFACT_UPLOAD",
            payload={"url_credential_ref": f"artifact-url:{run_id}"},
            task_run_id=run_id,
            lease_credential_ref=lease_ref,
        )


def collector(root, outbox, *, retention_days=7, max_bytes=10**9, min_free=1):
    return SpoolGarbageCollector(
        spool_root=root,
        outbox=outbox,
        retention_days=retention_days,
        max_bytes=max_bytes,
        min_free_bytes=min_free,
    )


def test_gc_removes_expired_terminal_run_and_durable_rows(tmp_path):
    root = tmp_path / "spool"
    root.mkdir()
    outbox = DurableOutbox(root / "worker.db")
    now = datetime.now(timezone.utc)
    save_run(
        root,
        outbox,
        "old-completed",
        "COMPLETED",
        updated_at=now - timedelta(days=8),
        with_artifact=True,
    )

    result = collector(root, outbox).collect(now=now)

    assert result.deleted_runs == ("old-completed",)
    assert not (root / "task-runs" / "old-completed").exists()
    assert not (root / "local-reports" / "old-completed").exists()
    assert outbox.terminal_task_runs() == ()
    assert outbox.find_by_idempotency_key("artifact:old-completed") is None


def test_gc_never_removes_active_or_finalizing_runs(tmp_path):
    root = tmp_path / "spool"
    root.mkdir()
    outbox = DurableOutbox(root / "worker.db")
    now = datetime.now(timezone.utc)
    for run_id, state in (("active", "ACTIVE"), ("finalizing", "FINALIZING")):
        save_run(
            root,
            outbox,
            run_id,
            state,
            updated_at=now - timedelta(days=30),
        )

    result = collector(root, outbox, max_bytes=1).collect(now=now)

    assert result.deleted_runs == ()
    assert not result.can_claim
    assert (root / "task-runs" / "active").exists()
    assert (root / "task-runs" / "finalizing").exists()
    assert {run["task_run_id"] for run in outbox.active_task_runs()} == {
        "active",
        "finalizing",
    }


def test_disk_pressure_removes_recent_terminal_run_oldest_first(tmp_path, monkeypatch):
    root = tmp_path / "spool"
    root.mkdir()
    outbox = DurableOutbox(root / "worker.db")
    now = datetime.now(timezone.utc)
    save_run(
        root,
        outbox,
        "older",
        "COMPLETED",
        updated_at=now - timedelta(hours=2),
    )
    save_run(
        root,
        outbox,
        "newer",
        "COMPLETED",
        updated_at=now - timedelta(hours=1),
    )
    usage = namedtuple("usage", "total used free")
    calls = iter((50, 50, 200, 200, 200))
    monkeypatch.setattr(
        "phone_agent.worker.spool_gc.shutil.disk_usage",
        lambda _path: usage(1000, 0, next(calls, 200)),
    )

    result = collector(
        root,
        outbox,
        max_bytes=10**9,
        min_free=100,
    ).collect(now=now)

    assert result.deleted_runs == ("older", "newer")
    assert result.can_claim


def test_gc_refuses_symlinked_run_directory(tmp_path):
    root = tmp_path / "spool"
    root.mkdir()
    outbox = DurableOutbox(root / "worker.db")
    now = datetime.now(timezone.utc)
    save_run(
        root,
        outbox,
        "unsafe",
        "COMPLETED",
        updated_at=now - timedelta(days=8),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    run_path = root / "task-runs" / "unsafe"
    for child in run_path.iterdir():
        child.unlink()
    run_path.rmdir()
    run_path.symlink_to(outside, target_is_directory=True)

    result = collector(root, outbox).collect(now=now)

    assert result.deleted_runs == ()
    assert (outside / "keep.txt").read_text() == "keep"
    assert [run.task_run_id for run in outbox.terminal_task_runs()] == ["unsafe"]
