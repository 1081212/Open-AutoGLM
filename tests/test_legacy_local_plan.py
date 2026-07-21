from __future__ import annotations

from types import SimpleNamespace

from main import ParsedTestCase, build_legacy_local_plan
from phone_agent.model import ModelConfig


def test_plain_legacy_prompt_becomes_one_local_step_without_affecting_worker_parser():
    args = SimpleNamespace(
        device_type="ios",
        max_steps=20,
        lang="cn",
        case_retries=0,
    )
    plan, lookup = build_legacy_local_plan(
        [
            ParsedTestCase(
                case_id=None,
                title=None,
                rule_type=None,
                priority=None,
                module=None,
                task="打开设置并查看关于本机",
            )
        ],
        args=args,
        model_config=ModelConfig(model_name="local-model"),
        status_judge=None,
    )

    assert plan.target_requirements.device_type == "ios"
    assert len(plan.test_run.cases[0].steps) == 1
    assert plan.test_run.cases[0].steps[0].instruction == "打开设置并查看关于本机"
    assert len(lookup) == 1
