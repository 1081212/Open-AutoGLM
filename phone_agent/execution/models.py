"""Pydantic models for immutable platform execution plans."""

from __future__ import annotations

from enum import Enum
import re
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

SUPPORTED_EXECUTION_VERSIONS = (
    "autoglm.execution.v1",
    "autoglm.execution.v2",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskType(str, Enum):
    TEST_RUN = "TEST_RUN"
    ADHOC = "ADHOC"


class CaseOutcome(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    REVIEW = "REVIEW"
    RETRYABLE_ERROR = "RETRYABLE_ERROR"
    CASE_ERROR = "CASE_ERROR"
    SKIPPED = "SKIPPED"
    INFRA_ERROR = "INFRA_ERROR"
    CANCELLED = "CANCELLED"
    WORKER_LOST = "WORKER_LOST"
    DEVICE_LOST = "DEVICE_LOST"
    TIMED_OUT = "TIMED_OUT"


class ResetPolicy(StrictModel):
    type: Literal["NONE", "ANDROID_ACTIVITY"]
    component: str | None = None
    wait_seconds: int = Field(default=0, ge=0, le=300)

    @model_validator(mode="after")
    def validate_component(self) -> "ResetPolicy":
        if self.type == "ANDROID_ACTIVITY" and not self.component:
            raise ValueError("ANDROID_ACTIVITY reset requires component")
        if self.type == "NONE" and self.component is not None:
            raise ValueError("NONE reset must not include component")
        return self


class TargetRequirements(StrictModel):
    # hdc/ios remain accepted only for the legacy local CLI compatibility path.
    # The platform Worker rejects anything except adb before device access.
    device_type: Literal["adb", "hdc", "ios"]
    app_package: str | None = None
    reset_policy: ResetPolicy


class ExecutionOptions(StrictModel):
    max_steps_per_agent_call: int = Field(default=20, ge=1, le=1000)
    status_judge_enabled: bool = True
    language: str = "cn"
    task_timeout_seconds: int = Field(default=28800, ge=1)
    model_call_timeout_seconds: int = Field(default=180, ge=1)
    cancel_grace_seconds: int = Field(default=30, ge=1, le=300)
    local_report_compatibility: bool = True


class ModelProfiles(StrictModel):
    vision_profile: str
    judge_profile: str | None = None

    @model_validator(mode="after")
    def reject_credentials(self) -> "ModelProfiles":
        for value in (self.vision_profile, self.judge_profile):
            if value:
                lowered = value.lower()
                looks_secret = (
                    any(token in lowered for token in ("api_key", "token=", "bearer "))
                    or lowered.startswith("sk-")
                    or bool(re.fullmatch(r"[0-9a-f]{24,}\.[A-Za-z0-9_-]{12,}", value))
                    or len(value) > 128
                )
                if looks_secret:
                    raise ValueError("model_profiles may contain profile names only")
        return self


class Normalizer(StrictModel):
    name: str
    version: str
    case_schema_version: Literal["autoglm.case.v1"]
    source_manifest_sha256: str


class Provenance(StrictModel):
    source_suite_id: UUID
    source_suite_version_id: UUID
    source_suite_version_no: int = Field(ge=1)
    source_case_id: str
    relation: str


class ExecutionStep(StrictModel):
    step_id: UUID
    index: int = Field(ge=1)
    instruction: str
    target_state: str | None = None
    expected_activity: str | None = None
    conditional: bool = False


class ExecutionCase(StrictModel):
    execution_case_id: UUID
    case_id: UUID
    case_revision_id: UUID
    case_schema_version: Literal["autoglm.case.v1"]
    parser_version: str
    semantic_hash: str
    canonical_code: str
    source_case_id: str
    display_id: str
    repeat_index: int = Field(default=1, ge=1)
    provenance: tuple[Provenance, ...]
    ordinal: int = Field(ge=1)
    title: str
    rule_type: str
    priority: str | None = None
    module: str | None = None
    goal: str
    preconditions: tuple[str, ...] = ()
    steps: tuple[ExecutionStep, ...] = Field(min_length=1)
    expected_result: str
    failure_conditions: tuple[str, ...] = ()
    source_excerpt: str | None = None
    extensions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_steps(self) -> "ExecutionCase":
        indexes = [step.index for step in self.steps]
        if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
            raise ValueError("case step indexes must be unique and ascending")
        return self


class CaseRetry(StrictModel):
    max_retries: int = Field(default=1, ge=0, le=1)
    eligible_outcomes: tuple[CaseOutcome, ...] = (
        CaseOutcome.FAIL,
        CaseOutcome.BLOCKED,
        CaseOutcome.RETRYABLE_ERROR,
    )

    @model_validator(mode="after")
    def validate_allowlist(self) -> "CaseRetry":
        allowed = {CaseOutcome.FAIL, CaseOutcome.BLOCKED, CaseOutcome.RETRYABLE_ERROR}
        if not set(self.eligible_outcomes).issubset(allowed):
            raise ValueError("retry outcomes exceed the v1 allowlist")
        return self


class TestRun(StrictModel):
    case_order: Literal["SEQUENTIAL"]
    case_retry: CaseRetry
    cases: tuple[ExecutionCase, ...] = Field(min_length=1, max_length=1000)


class AdhocItem(StrictModel):
    execution_item_id: UUID
    prompt: str = Field(min_length=1)


class ArtifactCandidate(StrictModel):
    job_id: int = Field(ge=1)
    job_name: str = Field(min_length=1, max_length=255)
    artifact_filename: str = Field(min_length=1, max_length=1024)
    artifact_size: int = Field(ge=0)


class PreTestInstall(StrictModel):
    type: Literal["GITLAB_CI_ANDROID_APK"]
    ci_build_id: UUID
    repository_url: str = Field(min_length=1)
    gitlab_project_path: str = Field(min_length=1, max_length=512)
    ref: str = Field(min_length=1, max_length=255)
    expected_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    build_variant: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    pipeline_id: int = Field(ge=1)
    pipeline_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    pipeline_web_url: str | None = None
    artifact_candidates: tuple[ArtifactCandidate, ...] = Field(
        min_length=1, max_length=100
    )
    download_strategy: Literal["FIRST_SINGLE_APK"]
    install_strategy: Literal["ADB_REPLACE_DOWNGRADE"]

    @model_validator(mode="after")
    def validate_frozen_build(self) -> "PreTestInstall":
        if self.pipeline_sha != self.expected_commit_sha:
            raise ValueError("pipeline_sha must equal expected_commit_sha")
        job_ids = [candidate.job_id for candidate in self.artifact_candidates]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("artifact candidate job_id values must be unique")
        repository = urlparse(self.repository_url)
        if repository.scheme != "https" or not repository.netloc:
            raise ValueError("repository_url must be an absolute HTTPS URL")
        if self.pipeline_web_url is not None:
            pipeline_url = urlparse(self.pipeline_web_url)
            if pipeline_url.scheme not in {"http", "https"} or not pipeline_url.netloc:
                raise ValueError("pipeline_web_url must be an absolute HTTP(S) URL")
        return self


class ExecutionPlan(StrictModel):
    schema_version: Literal["autoglm.execution.v1", "autoglm.execution.v2"]
    plan_id: UUID
    task_id: UUID
    project_id: UUID
    task_type: TaskType
    revision: int = Field(ge=1)
    normalizer: Normalizer | None
    target_requirements: TargetRequirements
    execution_options: ExecutionOptions
    model_profiles: ModelProfiles
    pre_test_install: PreTestInstall | None = None
    test_run: TestRun | None
    adhoc: AdhocItem | None

    @model_validator(mode="before")
    @classmethod
    def validate_version_shape(cls, value: Any) -> Any:
        if isinstance(value, dict):
            version = value.get("schema_version")
            if version == "autoglm.execution.v1" and "pre_test_install" in value:
                raise ValueError("autoglm.execution.v1 forbids pre_test_install")
            if version == "autoglm.execution.v2" and "pre_test_install" not in value:
                raise ValueError("autoglm.execution.v2 requires pre_test_install")
        return value

    @model_validator(mode="after")
    def validate_union_and_ids(self) -> "ExecutionPlan":
        if self.schema_version == "autoglm.execution.v2":
            if self.pre_test_install is None:
                raise ValueError("autoglm.execution.v2 requires pre_test_install")
        elif self.pre_test_install is not None:
            raise ValueError("autoglm.execution.v1 forbids pre_test_install")

        if self.task_type is TaskType.TEST_RUN:
            if (
                self.test_run is None
                or self.adhoc is not None
                or self.normalizer is None
            ):
                raise ValueError(
                    "TEST_RUN requires test_run/normalizer and forbids adhoc"
                )
        elif self.adhoc is None or self.test_run is not None:
            raise ValueError("ADHOC requires adhoc and forbids test_run")

        if (
            self.execution_options.status_judge_enabled
            and not self.model_profiles.judge_profile
        ):
            raise ValueError("status_judge_enabled requires judge_profile")

        if self.test_run:
            ordinals = [case.ordinal for case in self.test_run.cases]
            if ordinals != sorted(ordinals) or len(ordinals) != len(set(ordinals)):
                raise ValueError("case ordinals must be unique and ascending")
            execution_ids = [case.execution_case_id for case in self.test_run.cases]
            if len(execution_ids) != len(set(execution_ids)):
                raise ValueError("execution_case_id must be unique")
            step_ids = [
                step.step_id for case in self.test_run.cases for step in case.steps
            ]
            if len(step_ids) != len(set(step_ids)):
                raise ValueError("step_id must be globally unique within a plan")
        return self

    @model_serializer(mode="wrap")
    def serialize_plan(self, handler):
        data = handler(self)
        if self.schema_version == "autoglm.execution.v1":
            data.pop("pre_test_install", None)
        return data
