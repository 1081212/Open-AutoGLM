from __future__ import annotations

import pytest

from phone_agent.worker.status_judge import _parse_json_object


def test_judge_json_parser_accepts_plain_and_fenced_objects():
    assert _parse_json_object('{"status":"PASS","message":"ok"}')["status"] == "PASS"
    assert _parse_json_object('```json\n{"status":"REVIEW","message":"x"}\n```')["status"] == (
        "REVIEW"
    )


def test_judge_json_parser_rejects_non_object():
    with pytest.raises(ValueError, match="JSON object"):
        _parse_json_object("[]")
