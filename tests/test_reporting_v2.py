from __future__ import annotations

import base64
import json
from pathlib import Path
from uuid import uuid4

from phone_agent.reporting import TestRunReporter as Reporter

TASK = """## TC-WEARFIT-DEMO-001-normal 重复编号测试

- 优先级：P0
- 关联模块：demo
"""


def test_duplicate_display_id_uses_distinct_execution_directories(tmp_path):
    reporter = Reporter(base_dir=tmp_path, artifact_name="run", device_type="test")
    execution_ids = [str(uuid4()), str(uuid4())]

    for index, execution_case_id in enumerate(execution_ids, start=1):
        reporter.start_case(
            TASK,
            index,
            execution_case_id=execution_case_id,
            ordinal=index,
        )
        reporter.finish_case("STATUS: PASS")

    assert len(reporter.cases) == 2
    assert {case.execution_case_id for case in reporter.cases} == set(execution_ids)
    assert len({case.artifacts_dir for case in reporter.cases}) == 2
    assert all(
        Path(case.artifacts_dir, "report.json").exists() for case in reporter.cases
    )


def test_multiple_actions_are_appended_to_same_step(tmp_path):
    reporter = Reporter(base_dir=tmp_path, artifact_name="run", device_type="test")
    reporter.start_case(TASK, 1, execution_case_id=str(uuid4()), ordinal=1)
    reporter.save_step(
        step=1,
        screenshot_base64=base64.b64encode(b"image").decode(),
        width=1,
        height=1,
        current_app="demo",
    )
    reporter.record_action(1, {"action": "Tap"}, True, "first", vision_total_tokens=3)
    reporter.record_action(1, {"action": "Wait"}, True, "second", vision_total_tokens=5)
    reporter.finish_case("STATUS: PASS")

    case = reporter.cases[0]
    assert [item.action_sequence for item in case.steps[0].actions] == [1, 2]
    assert case.steps[0].vision_total_tokens == 8
    report = json.loads(
        Path(case.artifacts_dir, "report.json").read_text(encoding="utf-8")
    )
    assert len(report["steps"][0]["actions"]) == 2


def test_model_error_is_never_classified_as_pass(tmp_path):
    reporter = Reporter(base_dir=tmp_path, artifact_name="run", device_type="test")
    reporter.start_case(TASK, 1)
    status = reporter.finish_case("Model error: timeout")
    assert status == "REVIEW"


def test_issue_summary_accepts_step_without_action(tmp_path):
    reporter = Reporter(base_dir=tmp_path, artifact_name="run", device_type="test")
    reporter.start_case(TASK, 1, execution_case_id=str(uuid4()), ordinal=1)
    reporter.save_step(
        step=1,
        screenshot_base64=base64.b64encode(b"image").decode(),
        width=1,
        height=1,
        current_app="demo",
    )

    assert reporter.finish_case("STATUS: FAIL\nREASON: target not reached") == "FAIL"


def test_legacy_html_failure_does_not_abort_structured_case(tmp_path, monkeypatch):
    reporter = Reporter(base_dir=tmp_path, artifact_name="run", device_type="test")
    reporter.start_case(TASK, 1, execution_case_id=str(uuid4()), ordinal=1)
    monkeypatch.setattr(
        reporter,
        "_write_summary_html",
        lambda: (_ for _ in ()).throw(RuntimeError("template failed")),
    )

    assert reporter.finish_case("STATUS: PASS") == "PASS"
    assert reporter.current_case is None
    reporter.finish_run()
