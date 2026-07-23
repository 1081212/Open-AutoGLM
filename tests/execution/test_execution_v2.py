from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import pytest
from pydantic import ValidationError

from phone_agent.execution.models import CaseOutcome, ExecutionPlan
from phone_agent.execution.result import AttemptResult
from phone_agent.execution.task_executor import TaskExecutor


def v2_data(test_run_plan) -> dict:
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
        "pipeline_web_url": "https://gitlab.example.com/pipelines/123",
        "artifact_candidates": [
            {
                "job_id": 10,
                "job_name": "assemble",
                "artifact_filename": "artifacts.zip",
                "artifact_size": 100,
            }
        ],
        "download_strategy": "FIRST_SINGLE_APK",
        "install_strategy": "ADB_REPLACE_DOWNGRADE",
    }
    return data


def test_v1_serialization_and_validation_remain_strict(test_run_plan):
    assert "pre_test_install" not in test_run_plan.model_dump(mode="json")
    invalid = test_run_plan.model_dump(mode="json")
    invalid["pre_test_install"] = None
    with pytest.raises(ValidationError, match="forbids pre_test_install"):
        ExecutionPlan.model_validate(invalid)


def test_v2_accepts_platform_frozen_install(test_run_plan):
    plan = ExecutionPlan.model_validate(v2_data(test_run_plan))
    assert plan.schema_version == "autoglm.execution.v2"
    assert plan.pre_test_install is not None
    assert plan.pre_test_install.artifact_candidates[0].job_id == 10


@pytest.mark.parametrize("mutation", ["sha", "empty", "duplicate"])
def test_v2_preflight_rejects_invalid_install_before_runtime(test_run_plan, mutation):
    data = v2_data(test_run_plan)
    install = data["pre_test_install"]
    if mutation == "sha":
        install["pipeline_sha"] = "f" * 40
    elif mutation == "empty":
        install["artifact_candidates"] = []
    else:
        install["artifact_candidates"].append(
            {**install["artifact_candidates"][0], "job_name": "duplicate"}
        )

    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate(data)


def test_v2_requires_install_and_lowercase_frozen_sha(test_run_plan):
    missing = v2_data(test_run_plan)
    del missing["pre_test_install"]
    with pytest.raises(ValidationError, match="requires pre_test_install"):
        ExecutionPlan.model_validate(missing)

    uppercase = v2_data(test_run_plan)
    uppercase["pre_test_install"]["pipeline_sha"] = "A" * 40
    uppercase["pre_test_install"]["expected_commit_sha"] = "A" * 40
    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate(uppercase)


def test_v2_test_run_boundary_events_use_a_real_execution_case_binding(
    test_run_plan,
):
    plan = ExecutionPlan.model_validate(v2_data(test_run_plan))

    class Sink:
        def __init__(self):
            self.events = []

        def emit(self, event_type, data):
            self.events.append((event_type, data))

    sink = Sink()
    result = TaskExecutor(
        attempt_runner=lambda _case, _attempt: AttemptResult(
            CaseOutcome.PASS, "passed"
        ),
        lifecycle_sink=sink,
        bind_test_run_boundaries=True,
    ).execute(plan)

    assert result.outcome.value == "PASS"
    first_case_id = str(plan.test_run.cases[0].execution_case_id)
    assert sink.events[0] == (
        "RUN_STARTED",
        {
            "task_id": str(plan.task_id),
            "plan_id": str(plan.plan_id),
            "task_type": "TEST_RUN",
            "execution_case_id": first_case_id,
        },
    )
    assert sink.events[-1][0] == "RUN_FINISHED"
    assert sink.events[-1][1]["execution_case_id"] == first_case_id
