from __future__ import annotations

import pytest

from phone_agent.worker.model_profiles import ModelProfileStore


def test_profile_name_resolves_secret_from_environment(tmp_path, monkeypatch):
    path = tmp_path / "profiles.yaml"
    path.write_text(
        """
profiles:
  autoglm-default:
    base_url: http://model.internal/v1
    model_name: phone-model
    api_key_env: TEST_MODEL_KEY
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_MODEL_KEY", "local-only-secret")

    config = ModelProfileStore.load(path).resolve(
        "autoglm-default", lang="cn", timeout_seconds=42
    )

    assert config.api_key == "local-only-secret"
    assert config.model_name == "phone-model"
    assert config.timeout_seconds == 42


def test_profile_missing_secret_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "profiles.yaml"
    path.write_text(
        "profiles:\n  default:\n    base_url: http://model/v1\n    model_name: m\n    api_key_env: ABSENT_KEY\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ABSENT_KEY", raising=False)

    with pytest.raises(ValueError, match="ABSENT_KEY"):
        ModelProfileStore.load(path).resolve("default", lang="cn", timeout_seconds=1)
