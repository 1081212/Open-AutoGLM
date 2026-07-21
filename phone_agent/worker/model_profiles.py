"""Resolve Plan model profile names from a Worker-local YAML file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from phone_agent.model import ModelConfig


class LocalModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str
    model_name: str
    api_key_env: str = "AUTOGLM_MODEL_API_KEY"
    max_tokens: int = Field(default=3000, ge=1)
    temperature: float = 0.0
    top_p: float = 0.85
    frequency_penalty: float = 0.2
    extra_body: dict[str, Any] = Field(default_factory=dict)


class ModelProfileStore:
    def __init__(self, profiles: dict[str, LocalModelProfile]) -> None:
        self._profiles = dict(profiles)

    @classmethod
    def load(cls, path: str | Path) -> "ModelProfileStore":
        source = Path(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("profiles"), dict):
            raise ValueError("model profiles file must contain a profiles mapping")
        return cls(
            {
                str(name): LocalModelProfile.model_validate(value)
                for name, value in raw["profiles"].items()
            }
        )

    def resolve(self, name: str, *, lang: str, timeout_seconds: float) -> ModelConfig:
        try:
            profile = self._profiles[name]
        except KeyError as exc:
            raise ValueError(f"unknown local model profile: {name}") from exc
        api_key = os.getenv(profile.api_key_env, "")
        if not api_key:
            raise ValueError(f"model profile {name!r} requires environment variable {profile.api_key_env}")
        return ModelConfig(
            base_url=profile.base_url,
            model_name=profile.model_name,
            api_key=api_key,
            max_tokens=profile.max_tokens,
            temperature=profile.temperature,
            top_p=profile.top_p,
            frequency_penalty=profile.frequency_penalty,
            extra_body=profile.extra_body,
            lang=lang,
            timeout_seconds=timeout_seconds,
        )
