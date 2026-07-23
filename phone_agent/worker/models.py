"""Strict control-plane DTOs for Worker API and Redis notifications."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from phone_agent.worker.time_utils import parse_aware_iso8601


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkerActivity(str, Enum):
    STARTING = "STARTING"
    IDLE = "IDLE"
    CLAIMING = "CLAIMING"
    DOWNLOADING_PLAN = "DOWNLOADING_PLAN"
    BUSY = "BUSY"
    FINALIZING = "FINALIZING"
    DEGRADED = "DEGRADED"


class DispatchNotification(StrictModel):
    schema_version: Literal["autoglm.dispatch.v1"]
    dispatch_id: UUID
    task_id: UUID
    task_type: Literal["TEST_RUN", "ADHOC"]
    plan_id: UUID
    plan_canonical_sha256: str
    worker_id: UUID
    device_uid: UUID


class ClaimPlanDescriptor(StrictModel):
    schema_version: Literal["autoglm.execution.v1", "autoglm.execution.v2"]
    plan_id: UUID
    download_url: str
    wire_media_type: Literal["application/vnd.autoglm.execution-plan+gzip"]
    inner_media_type: Literal["application/vnd.autoglm.execution-plan+json"]
    wire_format: Literal["gzip"]
    compressed_sha256: str
    compressed_size: int = Field(ge=0)
    canonical_sha256: str
    canonical_size: int = Field(ge=0)
    item_count: int = Field(ge=1)
    case_count: int = Field(ge=0)


class ClaimResponse(StrictModel):
    claimed: bool
    reason: str | None = None
    ack_disposition: Literal["ACK", "RETRY", "HOLD"] | None = None
    task_id: UUID | None = None
    worker_id: UUID | None = None
    instance_id: UUID | None = None
    device_uid: UUID | None = None
    task_run_id: UUID | None = None
    lease_token: str | None = None
    fencing_token: int | None = None
    run_started_at: str | None = None
    lease_expires_at: str | None = None
    renew_after_seconds: int | None = None
    plan: ClaimPlanDescriptor | None = None

    @field_validator("run_started_at")
    @classmethod
    def validate_run_started_at(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parse_aware_iso8601(value, "run_started_at")
        return value

    @model_validator(mode="after")
    def validate_claim(self) -> "ClaimResponse":
        required = (
            self.task_id,
            self.worker_id,
            self.instance_id,
            self.device_uid,
            self.task_run_id,
            self.lease_token,
            self.fencing_token,
            self.run_started_at,
            self.lease_expires_at,
            self.renew_after_seconds,
            self.plan,
        )
        if self.claimed and any(value is None for value in required):
            raise ValueError("successful claim is missing run binding fields")
        if not self.claimed and not self.reason:
            raise ValueError("rejected claim requires a stable reason")
        return self


class RunHeartbeatResponse(StrictModel):
    lease_expires_at: str
    cancel_requested: bool = False
    fence_owned: bool = True
    accepted_producer_seq: dict[str, int] = Field(default_factory=dict)


class RedisMessage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    redis_id: str
    notification: DispatchNotification
    raw_fields: dict[str, Any]
