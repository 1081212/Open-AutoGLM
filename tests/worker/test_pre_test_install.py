from __future__ import annotations

import io
import json
import shutil
import zipfile
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest
import requests

from phone_agent.execution.cancellation import CancellationToken
from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.execution.models import ExecutionPlan
from phone_agent.gitlab_install import (
    FrozenArtifactError,
    download_frozen_job_artifact,
    extract_single_apk_secure,
)
from phone_agent.worker.outbox import DurableOutbox
from phone_agent.worker.pre_test_install import (
    FrozenGitLabApkInstaller,
    WorkerGitLabConfig,
)
from phone_agent.worker.spool import SpoolPlan


class FakeResponse:
    def __init__(self, status: int, body: bytes = b"", headers=None):
        self.status_code = status
        self.body = body
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset : offset + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeAdb:
    def __init__(
        self,
        order=None,
        fail=False,
        *,
        install_output="Success",
        package_present=True,
        install_existing_succeeds=False,
    ):
        self.calls = []
        self.order = order
        self.fail = fail
        self.install_output = install_output
        self.package_present = package_present
        self.install_existing_succeeds = install_existing_succeeds

    def run(self, serial, args, *, timeout, check=True):
        del check
        self.calls.append((serial, list(args), timeout))
        if args[0] == "install" and self.order is not None:
            self.order.append("install")
        if args[0] == "install" and self.fail:
            raise ExecutionError(
                ExecutionErrorCode.DEVICE_COMMAND_TIMEOUT, "sanitized adb failure"
            )
        if args == ["shell", "am", "get-current-user"]:
            stdout = "0\n"
        elif args[:3] == ["shell", "pm", "path"]:
            package = args[-1]
            stdout = (
                f"package:/data/app/{package}/base.apk\n"
                if self.package_present
                else ""
            )
        elif args[:4] == ["shell", "cmd", "package", "install-existing"]:
            if not self.install_existing_succeeds:
                raise ExecutionError(ExecutionErrorCode.ACTION_ERROR, "Unknown package")
            self.package_present = True
            stdout = "Package installed for user: 0\n"
        else:
            stdout = self.install_output
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="", duration_ms=12)

    @property
    def install_calls(self):
        return [call for call in self.calls if call[1][0] == "install"]


class ForbiddenMetadataReader:
    def read(self, _path):
        raise AssertionError("APK metadata inspection must remain disabled")


def zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return buffer.getvalue()


def v2_plan(test_run_plan, job_ids=(10,)) -> ExecutionPlan:
    data = deepcopy(test_run_plan.model_dump(mode="json"))
    data["schema_version"] = "autoglm.execution.v2"
    data["pre_test_install"] = {
        "type": "GITLAB_CI_ANDROID_APK",
        "ci_build_id": str(uuid4()),
        "repository_url": "https://gitlab.example.com/android/app.git",
        "gitlab_project_path": "android/app",
        "ref": "feature/install",
        "expected_commit_sha": "0123456789abcdef0123456789abcdef01234567",
        "build_variant": "demoDebug",
        "pipeline_id": 123,
        "pipeline_sha": "0123456789abcdef0123456789abcdef01234567",
        "pipeline_web_url": None,
        "artifact_candidates": [
            {
                "job_id": job_id,
                "job_name": f"job-{job_id}",
                "artifact_filename": "artifacts.zip",
                "artifact_size": 0,
            }
            for job_id in job_ids
        ],
        "download_strategy": "FIRST_SINGLE_APK",
        "install_strategy": "ADB_REPLACE_DOWNGRADE",
    }
    return ExecutionPlan.model_validate(data)


def build_installer(tmp_path, responses, *, adb=None, token="read-only-secret"):
    outbox = DurableOutbox(tmp_path / "worker.db")
    session = FakeSession(responses)
    adb = adb or FakeAdb()
    installer = FrozenGitLabApkInstaller(
        config=WorkerGitLabConfig(
            base_url="https://gitlab.example.com",
            token=token,
            max_artifact_bytes=10 * 1024 * 1024,
            max_apk_bytes=5 * 1024 * 1024,
            min_free_bytes=1,
        ),
        outbox=outbox,
        adb=adb,
        metadata_reader=ForbiddenMetadataReader(),
        session=session,
    )
    return installer, outbox, session, adb


def run_install(tmp_path, plan, installer):
    run_root = tmp_path / "task-runs" / str(uuid4())
    run_root.mkdir(parents=True)
    canonical = run_root / "plan.json"
    canonical.write_text(json.dumps(plan.model_dump(mode="json")))
    stored = SpoolPlan(run_root, run_root / "plan.json.gz", canonical, plan)
    claimed = SimpleNamespace(
        task_run_id=uuid4(),
        device_uid=uuid4(),
        adb_serial="LOCKED-SERIAL",
        fencing_token=9,
    )
    attempt = installer.install(
        stored,
        claimed,
        CancellationToken(),
        lease_credential_ref="lease-ref",
        guard=lambda: None,
    )
    return attempt, stored, claimed


def test_candidates_are_tried_in_order_and_actual_job_is_recorded(
    tmp_path, test_run_plan
):
    plan = v2_plan(test_run_plan, (10, 20))
    installer, outbox, session, adb = build_installer(
        tmp_path,
        [
            FakeResponse(200, zip_bytes({"readme.txt": b"none"})),
            FakeResponse(200, zip_bytes({"outputs/app.apk": b"apk-content"})),
        ],
    )

    attempt, _stored, _claimed = run_install(tmp_path, plan, installer)

    assert attempt is not None and attempt.error is None
    assert attempt.fact.job_id == 20
    assert [call[0].rsplit("/", 2)[-2] for call in session.calls] == ["10", "20"]
    assert "%2F" in session.calls[0][0]
    assert adb.calls[0][0] == "LOCKED-SERIAL"
    install_args = adb.install_calls[0][1]
    assert install_args[:3] == ["install", "-r", "-d"]
    assert install_args[3].endswith("job-20.apk")
    assert attempt.fact.package_name == plan.target_requirements.app_package
    assert attempt.fact.version_name is None
    assert attempt.fact.version_code is None
    fact_item = outbox.find_by_idempotency_key(attempt.fact.idempotency_key)
    assert fact_item is not None and fact_item.kind == "INSTALLATION_FACT"
    events = outbox.events_for_run(attempt.fact.task_run_id)
    assert events[0]["type"] == "RUN_STARTED"
    assert events[0]["data"]["pre_test_install"]["job_id"] == 20
    assert list(_stored.root.glob("pre-test-install/*")) == []


def test_worker_path_never_triggers_or_queries_pipeline(
    tmp_path, test_run_plan, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Worker must consume only platform-frozen Job IDs")

    monkeypatch.setattr("phone_agent.gitlab_install.trigger_pipeline", forbidden)
    monkeypatch.setattr("phone_agent.gitlab_install.wait_for_pipeline", forbidden)
    monkeypatch.setattr("phone_agent.gitlab_install.get_pipeline_jobs", forbidden)
    plan = v2_plan(test_run_plan)
    installer, _outbox, _session, adb = build_installer(
        tmp_path, [FakeResponse(200, zip_bytes({"app.apk": b"apk"}))]
    )

    attempt, _stored, _claimed = run_install(tmp_path, plan, installer)

    assert attempt is not None and attempt.error is None
    assert len(adb.install_calls) == 1


@pytest.mark.parametrize(
    "first",
    [
        FakeResponse(404),
        FakeResponse(200, zip_bytes({"readme.txt": b"none"})),
        FakeResponse(200, zip_bytes({"a.apk": b"a", "b.apk": b"b"})),
    ],
)
def test_only_candidate_local_failures_continue(tmp_path, test_run_plan, first):
    plan = v2_plan(test_run_plan, (10, 20))
    installer, _outbox, session, _adb = build_installer(
        tmp_path,
        [first, FakeResponse(200, zip_bytes({"app.apk": b"good"}))],
    )

    attempt, _stored, _claimed = run_install(tmp_path, plan, installer)

    assert attempt is not None and attempt.error is None
    assert attempt.fact.job_id == 20
    assert len(session.calls) == 2


@pytest.mark.parametrize(
    "failure",
    [
        FakeResponse(403, b"response body must not escape"),
        requests.exceptions.SSLError("tls details"),
    ],
)
def test_authentication_and_tls_failures_do_not_try_later_candidates(
    tmp_path, test_run_plan, failure
):
    plan = v2_plan(test_run_plan, (10, 20))
    installer, outbox, session, adb = build_installer(
        tmp_path,
        [failure, FakeResponse(200, zip_bytes({"app.apk": b"good"}))],
    )

    attempt, _stored, _claimed = run_install(tmp_path, plan, installer)

    assert attempt is not None and attempt.error is not None
    assert attempt.fact.install_result == "FAILED"
    assert len(session.calls) == 1
    assert adb.calls == []
    assert "response body" not in str(attempt.error)
    assert "tls details" not in str(attempt.error)
    assert [
        event["type"] for event in outbox.events_for_run(attempt.fact.task_run_id)
    ] == ["RUN_STARTED", "RUN_ERROR"]


def test_success_fact_prevents_duplicate_adb_install(tmp_path, test_run_plan):
    plan = v2_plan(test_run_plan)
    apk = zip_bytes({"app.apk": b"same-apk"})
    installer, _outbox, _session, adb = build_installer(
        tmp_path, [FakeResponse(200, apk), FakeResponse(200, apk)]
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    canonical = run_root / "plan.json"
    canonical.write_text("{}")
    stored = SpoolPlan(run_root, run_root / "plan.json.gz", canonical, plan)
    claimed = SimpleNamespace(
        task_run_id=uuid4(),
        device_uid=uuid4(),
        adb_serial="SERIAL",
        fencing_token=3,
    )

    for _ in range(2):
        result = installer.install(
            stored,
            claimed,
            CancellationToken(),
            lease_credential_ref="lease",
            guard=lambda: None,
        )
        assert result is not None and result.error is None

    assert len(adb.install_calls) == 1


def test_adb_failure_is_structured_and_does_not_report_success(tmp_path, test_run_plan):
    plan = v2_plan(test_run_plan)
    installer, _outbox, _session, adb = build_installer(
        tmp_path,
        [FakeResponse(200, zip_bytes({"app.apk": b"apk"}))],
        adb=FakeAdb(fail=True),
    )

    attempt, _stored, _claimed = run_install(tmp_path, plan, installer)

    assert attempt is not None and attempt.error is not None
    assert attempt.error.code is ExecutionErrorCode.PRE_TEST_INSTALL_FAILED
    assert attempt.fact.install_result == "FAILED"
    assert len(adb.install_calls) == 1


def test_adb_zero_exit_without_success_is_install_failure(tmp_path, test_run_plan):
    plan = v2_plan(test_run_plan)
    installer, _outbox, _session, adb = build_installer(
        tmp_path,
        [FakeResponse(200, zip_bytes({"app.apk": b"apk"}))],
        adb=FakeAdb(install_output="Performing Streamed Install"),
    )

    attempt, _stored, _claimed = run_install(tmp_path, plan, installer)

    assert attempt is not None and attempt.error is not None
    assert attempt.fact.install_result == "FAILED"
    assert "explicit Success" in attempt.error.message
    assert len(adb.install_calls) == 1


def test_target_package_absent_after_adb_success_is_install_failure(
    tmp_path, test_run_plan
):
    plan = v2_plan(test_run_plan)
    installer, _outbox, _session, adb = build_installer(
        tmp_path,
        [FakeResponse(200, zip_bytes({"app.apk": b"apk"}))],
        adb=FakeAdb(package_present=False),
    )

    attempt, _stored, _claimed = run_install(tmp_path, plan, installer)

    assert attempt is not None and attempt.error is not None
    assert attempt.fact.install_result == "FAILED"
    assert plan.target_requirements.app_package in attempt.error.message
    assert "could not be enabled for Android user 0" in attempt.error.message
    assert len(adb.install_calls) == 1


def test_package_enabled_only_for_other_user_is_enabled_for_current_user(
    tmp_path, test_run_plan
):
    plan = v2_plan(test_run_plan)
    installer, _outbox, _session, adb = build_installer(
        tmp_path,
        [FakeResponse(200, zip_bytes({"app.apk": b"apk"}))],
        adb=FakeAdb(
            package_present=False,
            install_existing_succeeds=True,
        ),
    )

    attempt, _stored, _claimed = run_install(tmp_path, plan, installer)

    assert attempt is not None and attempt.error is None
    assert attempt.fact.install_result == "SUCCESS"
    assert any(
        call[1][:4] == ["shell", "cmd", "package", "install-existing"]
        and call[1][-3:] == ["0", "--wait", plan.target_requirements.app_package]
        for call in adb.calls
    )


def test_manifest_metadata_is_skipped_and_adb_install_still_runs(
    tmp_path, test_run_plan
):
    plan = v2_plan(test_run_plan)

    class ExplodingMetadataReader:
        def read(self, _path):
            raise AssertionError("metadata reader must not be called")

    installer, _outbox, _session, adb = build_installer(
        tmp_path, [FakeResponse(200, zip_bytes({"app.apk": b"apk"}))]
    )
    installer.metadata_reader = ExplodingMetadataReader()

    attempt, _stored, _claimed = run_install(tmp_path, plan, installer)

    assert attempt is not None and attempt.error is None
    assert attempt.fact.package_name == plan.target_requirements.app_package
    assert len(adb.install_calls) == 1


def test_pre_cancelled_install_does_not_download_or_touch_adb(tmp_path, test_run_plan):
    plan = v2_plan(test_run_plan)
    installer, _outbox, session, adb = build_installer(
        tmp_path, [FakeResponse(200, zip_bytes({"app.apk": b"apk"}))]
    )
    run_root = tmp_path / "cancelled-run"
    run_root.mkdir()
    canonical = run_root / "plan.json"
    canonical.write_text("{}")
    stored = SpoolPlan(run_root, run_root / "plan.json.gz", canonical, plan)
    claimed = SimpleNamespace(
        task_run_id=uuid4(),
        device_uid=uuid4(),
        adb_serial="SERIAL",
        fencing_token=3,
    )
    cancellation = CancellationToken()
    cancellation.cancel("platform requested cancellation")

    with pytest.raises(ExecutionError) as raised:
        installer.install(
            stored,
            claimed,
            cancellation,
            lease_credential_ref="lease",
            guard=lambda: None,
        )

    assert raised.value.code is ExecutionErrorCode.CANCELLED
    assert session.calls == []
    assert adb.calls == []


def test_token_never_enters_spool_outbox_event_or_error(tmp_path, test_run_plan):
    token = "highly-sensitive-gitlab-token"
    plan = v2_plan(test_run_plan)
    installer, outbox, _session, _adb = build_installer(
        tmp_path,
        [FakeResponse(403, b"server-secret-body")],
        token=token,
    )

    attempt, _stored, _claimed = run_install(tmp_path, plan, installer)

    assert attempt is not None and attempt.error is not None
    assert token not in str(attempt.error)
    assert token not in json.dumps(attempt.fact.as_payload())
    assert token not in json.dumps(outbox.events_for_run(attempt.fact.task_run_id))
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert token.encode() not in path.read_bytes()


def test_cancellation_and_post_install_guard_stop_before_cases(tmp_path, test_run_plan):
    plan = v2_plan(test_run_plan)
    installer, outbox, _session, adb = build_installer(
        tmp_path, [FakeResponse(200, zip_bytes({"app.apk": b"apk"}))]
    )
    run_root = tmp_path / "guard-run"
    run_root.mkdir()
    canonical = run_root / "plan.json"
    canonical.write_text("{}")
    stored = SpoolPlan(run_root, run_root / "plan.json.gz", canonical, plan)
    claimed = SimpleNamespace(
        task_run_id=uuid4(),
        device_uid=uuid4(),
        adb_serial="SERIAL",
        fencing_token=3,
    )

    def guard():
        if adb.install_calls:
            raise ExecutionError(ExecutionErrorCode.LEASE_LOST, "fence changed")

    with pytest.raises(ExecutionError) as raised:
        installer.install(
            stored,
            claimed,
            CancellationToken(),
            lease_credential_ref="lease",
            guard=guard,
        )

    assert raised.value.code is ExecutionErrorCode.LEASE_LOST
    assert len(adb.install_calls) == 1
    facts = [item for item in outbox.due(100) if item.kind == "INSTALLATION_FACT"]
    assert facts == []  # retained ACKED, never queued as an API operation
    assert any(
        item.kind == "INSTALLATION_FACT"
        for item in [
            outbox.find_by_idempotency_key(
                event["data"]["pre_test_install"]["idempotency_key"]
            )
            for event in outbox.events_for_run(str(claimed.task_run_id))
        ]
        if item is not None
    )


def test_zip_slip_and_symlink_are_rejected(tmp_path):
    traversal = tmp_path / "traversal.zip"
    traversal.write_bytes(zip_bytes({"../escape.apk": b"apk"}))
    with pytest.raises(FrozenArtifactError, match="unsafe entry"):
        extract_single_apk_secure(
            traversal,
            tmp_path / "out-a",
            output_filename="app.apk",
            max_apk_bytes=1024,
            min_free_bytes=1,
        )

    symlink = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink, "w") as archive:
        info = zipfile.ZipInfo("linked.apk")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        archive.writestr(info, "target")
    with pytest.raises(FrozenArtifactError, match="unsafe entry"):
        extract_single_apk_secure(
            symlink,
            tmp_path / "out-b",
            output_filename="app.apk",
            max_apk_bytes=1024,
            min_free_bytes=1,
        )


def test_zip_bomb_apk_limit_and_disk_limit_are_rejected(tmp_path, monkeypatch):
    bomb = tmp_path / "bomb.zip"
    bomb.write_bytes(zip_bytes({"app.apk": b"0" * 100_000}))
    with pytest.raises(FrozenArtifactError, match="compression ratio"):
        extract_single_apk_secure(
            bomb,
            tmp_path / "bomb-out",
            output_filename="app.apk",
            max_apk_bytes=200_000,
            min_free_bytes=1,
            max_compression_ratio=2,
        )
    with pytest.raises(FrozenArtifactError, match="APK exceeds"):
        extract_single_apk_secure(
            bomb,
            tmp_path / "apk-out",
            output_filename="app.apk",
            max_apk_bytes=10,
            min_free_bytes=1,
            max_compression_ratio=10_000,
        )

    session = FakeSession([FakeResponse(200, b"zip")])
    monkeypatch.setattr(
        "phone_agent.gitlab_install.shutil.disk_usage",
        lambda _path: shutil._ntuple_diskusage(100, 99, 1),
    )
    with pytest.raises(FrozenArtifactError, match="insufficient free space"):
        download_frozen_job_artifact(
            session=session,
            gitlab_base_url="https://gitlab.example.com",
            project_path="android/app",
            job_id=1,
            token="secret",
            output_dir=tmp_path / "disk",
            declared_size=10,
            max_artifact_bytes=100,
            min_free_bytes=1,
            timeout_seconds=10,
            verify_ssl=True,
        )
    assert session.calls == []


def test_download_uses_atomic_part_file_and_frozen_size(tmp_path):
    body = zip_bytes({"app.apk": b"apk"})
    session = FakeSession([FakeResponse(200, body)])

    downloaded = download_frozen_job_artifact(
        session=session,
        gitlab_base_url="https://gitlab.example.com",
        project_path="android/app",
        job_id=9,
        token="secret",
        output_dir=tmp_path,
        declared_size=len(body),
        max_artifact_bytes=1024 * 1024,
        min_free_bytes=1,
        timeout_seconds=10,
        verify_ssl=True,
    )

    assert downloaded.read_bytes() == body
    assert not downloaded.with_suffix(downloaded.suffix + ".part").exists()

    mismatched = FakeSession([FakeResponse(200, body)])
    with pytest.raises(FrozenArtifactError, match="frozen Plan"):
        download_frozen_job_artifact(
            session=mismatched,
            gitlab_base_url="https://gitlab.example.com",
            project_path="android/app",
            job_id=10,
            token="secret",
            output_dir=tmp_path,
            declared_size=len(body) + 1,
            max_artifact_bytes=1024 * 1024,
            min_free_bytes=1,
            timeout_seconds=10,
            verify_ssl=True,
        )
    assert not (tmp_path / "job-10-artifacts.zip.part").exists()
