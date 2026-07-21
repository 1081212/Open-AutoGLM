from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_plan_requires_unique_ascending_ordinals(test_run_plan):
    payload = test_run_plan.model_dump(mode="json")
    payload["test_run"]["cases"][1]["ordinal"] = 1

    with pytest.raises(ValidationError, match="ordinals"):
        type(test_run_plan).model_validate(payload)


def test_plan_rejects_retry_count_above_one(test_run_plan):
    payload = test_run_plan.model_dump(mode="json")
    payload["test_run"]["case_retry"]["max_retries"] = 2

    with pytest.raises(ValidationError):
        type(test_run_plan).model_validate(payload)


def test_plan_rejects_credentials_in_model_profile(test_run_plan):
    payload = test_run_plan.model_dump(mode="json")
    payload["model_profiles"]["vision_profile"] = "token=secret"

    with pytest.raises(ValidationError, match="profile names only"):
        type(test_run_plan).model_validate(payload)

    payload["model_profiles"]["vision_profile"] = "sk-" + "x" * 32
    with pytest.raises(ValidationError, match="profile names only"):
        type(test_run_plan).model_validate(payload)
