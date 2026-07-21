from __future__ import annotations

from uuid import uuid4

import pytest

from phone_agent.execution.models import ExecutionPlan


@pytest.fixture
def test_run_plan() -> ExecutionPlan:
    source_suite_id = uuid4()
    source_suite_version_id = uuid4()
    cases = []
    for ordinal in (1, 2):
        cases.append(
            {
                "execution_case_id": str(uuid4()),
                "case_id": str(uuid4()),
                "case_revision_id": str(uuid4()),
                "case_schema_version": "autoglm.case.v1",
                "parser_version": "test/1.0",
                "semantic_hash": f"sha256:case-{ordinal}",
                "canonical_code": f"TC-WEARFIT-DEMO-{ordinal:03d}-normal",
                "source_case_id": f"TC-WEARFIT-DEMO-{ordinal:03d}-normal",
                "display_id": f"TC-WEARFIT-DEMO-{ordinal:03d}-normal",
                "repeat_index": 1,
                "provenance": [
                    {
                        "source_suite_id": str(source_suite_id),
                        "source_suite_version_id": str(source_suite_version_id),
                        "source_suite_version_no": 1,
                        "source_case_id": f"SOURCE-{ordinal}",
                        "relation": "PRIMARY",
                    }
                ],
                "ordinal": ordinal,
                "title": f"Case {ordinal}",
                "rule_type": "normal",
                "priority": "P0",
                "module": "demo",
                "goal": "verify demo",
                "preconditions": [],
                "steps": [
                    {
                        "step_id": str(uuid4()),
                        "index": 1,
                        "instruction": "open page",
                        "target_state": "page visible",
                        "expected_activity": None,
                        "conditional": False,
                    }
                ],
                "expected_result": "page visible",
                "failure_conditions": [],
                "source_excerpt": None,
                "extensions": {},
            }
        )

    return ExecutionPlan.model_validate(
        {
            "schema_version": "autoglm.execution.v1",
            "plan_id": str(uuid4()),
            "task_id": str(uuid4()),
            "project_id": str(uuid4()),
            "task_type": "TEST_RUN",
            "revision": 1,
            "normalizer": {
                "name": "test",
                "version": "1.0",
                "case_schema_version": "autoglm.case.v1",
                "source_manifest_sha256": "sha256:test",
            },
            "target_requirements": {
                "device_type": "adb",
                "app_package": "com.example",
                "reset_policy": {"type": "NONE"},
            },
            "execution_options": {
                "max_steps_per_agent_call": 20,
                "status_judge_enabled": True,
                "language": "cn",
                "task_timeout_seconds": 3600,
                "model_call_timeout_seconds": 180,
                "cancel_grace_seconds": 30,
                "local_report_compatibility": True,
            },
            "model_profiles": {
                "vision_profile": "test-profile",
                "judge_profile": "judge-profile",
            },
            "test_run": {
                "case_order": "SEQUENTIAL",
                "case_retry": {
                    "max_retries": 1,
                    "eligible_outcomes": ["FAIL", "BLOCKED", "RETRYABLE_ERROR"],
                },
                "cases": cases,
            },
            "adhoc": None,
        }
    )
