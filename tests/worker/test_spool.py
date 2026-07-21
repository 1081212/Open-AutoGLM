from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode
from phone_agent.worker.spool import PlanDescriptor, PlanSpool


def _payload(plan):
    canonical = json.dumps(
        plan.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    compressed = gzip.compress(canonical, mtime=0)
    descriptor = PlanDescriptor(
        plan_id=str(plan.plan_id),
        compressed_sha256="sha256:" + hashlib.sha256(compressed).hexdigest(),
        compressed_size=len(compressed),
        canonical_sha256="sha256:" + hashlib.sha256(canonical).hexdigest(),
        canonical_size=len(canonical),
        item_count=len(plan.test_run.cases),
        case_count=len(plan.test_run.cases),
    )
    return canonical, compressed, descriptor


def test_spool_verifies_and_atomically_stores_plan(tmp_path, test_run_plan):
    canonical, compressed, descriptor = _payload(test_run_plan)
    stored = PlanSpool(tmp_path).store("run-1", [compressed[:7], compressed[7:]], descriptor)
    assert stored.canonical_path.read_bytes() == canonical
    assert stored.compressed_path.read_bytes() == compressed
    assert stored.plan.plan_id == test_run_plan.plan_id
    assert not (stored.root / "plan.json.gz.part").exists()


def test_spool_rejects_tampered_compressed_plan(tmp_path, test_run_plan):
    _, compressed, descriptor = _payload(test_run_plan)
    tampered = compressed[:-1] + bytes([compressed[-1] ^ 1])
    with pytest.raises(ExecutionError) as caught:
        PlanSpool(tmp_path).store("run-1", [tampered], descriptor)
    assert caught.value.code is ExecutionErrorCode.PLAN_HASH_MISMATCH


def test_claim_json_rejects_bearer_tokens(tmp_path):
    with pytest.raises(ValueError, match="bearer"):
        PlanSpool(tmp_path).write_claim_metadata("run-1", {"lease_token": "secret"})
