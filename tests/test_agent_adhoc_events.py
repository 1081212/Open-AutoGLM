from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

from phone_agent.actions.handler import ActionResult
from phone_agent.agent import AgentConfig, PhoneAgent
from phone_agent.model.client import ModelResponse


class RecordingSink:
    def __init__(self):
        self.events = []

    def emit(self, event_type, data):
        self.events.append((event_type, data))


class FakeDevice:
    def get_screenshot(self, _device_id):
        return SimpleNamespace(
            base64_data="image",
            width=1080,
            height=1920,
            is_sensitive=False,
        )

    def get_current_app(self, _device_id):
        return "Settings"


class FakeModelClient:
    def __init__(self):
        self.responses = [
            ModelResponse("", 'do(action="Home")', "", total_tokens=11),
            ModelResponse("", 'do(action="Back")', "", total_tokens=17),
        ]

    def request(self, _messages, cancellation_check=None):
        if cancellation_check:
            cancellation_check()
        return self.responses.pop(0)

    def close_active_stream(self):
        pass


class FakeActionHandler:
    def __init__(self):
        self.calls = 0

    def execute(self, _action, _width, _height):
        self.calls += 1
        return ActionResult(True, self.calls == 2)


def test_adhoc_agent_emits_stable_action_ids_and_contiguous_model_steps(monkeypatch):
    sink = RecordingSink()
    item_id = str(uuid4())
    monkeypatch.setattr("phone_agent.agent.get_device_factory", lambda: FakeDevice())
    agent = PhoneAgent.__new__(PhoneAgent)
    agent.agent_config = AgentConfig(
        max_steps=2,
        device_id="SERIAL",
        verbose=False,
        reporter=None,
        auto_manage_report_case=False,
        require_structured_finish_status=False,
        lifecycle_sink=sink,
        run_context={"execution_item_id": item_id},
    )
    agent.model_client = FakeModelClient()
    agent.action_handler = FakeActionHandler()
    agent._context = []
    agent._step_count = 0
    agent._model_step_sequence = 0
    agent._current_task = ""
    agent.custom_rules = []
    agent._rule_fire_counts = {}
    agent._remind_task = False
    agent._finish_status_repair_count = 0

    agent.run("open settings")

    actions = [
        data for event_type, data in sink.events if event_type == "ACTION_RECORDED"
    ]
    steps = [data for event_type, data in sink.events if event_type == "STEP_FINISHED"]
    assert len(actions) == 2
    assert len({UUID(action["action_id"]) for action in actions}) == 2
    assert all(action["execution_item_id"] == item_id for action in actions)
    required_action_fields = {
        "action_id",
        "agent_step",
        "action",
        "success",
        "message",
        "vision_prompt_tokens",
        "vision_completion_tokens",
        "vision_cached_tokens",
        "vision_total_tokens",
        "model_ttft_ms",
        "model_total_ms",
    }
    assert all(required_action_fields <= action.keys() for action in actions)
    assert all("thinking" not in action for action in actions)
    assert [step["step_sequence"] for step in steps] == [1, 2]
    assert [step["vision_tokens"] for step in steps] == [11, 17]
    assert all(step["judge_tokens"] == 0 for step in steps)
    assert all(step["duration_ms"] >= 0 for step in steps)
