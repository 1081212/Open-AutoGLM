from __future__ import annotations

import json
import zipfile

import pytest

from phone_agent.execution.errors import ExecutionError
from phone_agent.worker.report_bundle import create_local_report_bundle


def test_report_bundle_has_manifest_and_relative_entries(tmp_path):
    report = tmp_path / "report"
    (report / "case").mkdir(parents=True)
    (report / "index.html").write_text("<html/>", encoding="utf-8")
    (report / "case" / "report.json").write_text("{}", encoding="utf-8")

    bundle = create_local_report_bundle(report, tmp_path / "bundle.zip")

    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "index.html",
            "case/report.json",
        }
        manifest = json.loads(archive.read("manifest.json"))
        assert [item["path"] for item in manifest["files"]] == [
            "case/report.json",
            "index.html",
        ]


def test_report_bundle_rejects_symlink(tmp_path):
    report = tmp_path / "report"
    report.mkdir()
    outside = tmp_path / "secret"
    outside.write_text("secret", encoding="utf-8")
    (report / "leak").symlink_to(outside)

    with pytest.raises(ExecutionError, match="symlink"):
        create_local_report_bundle(report, tmp_path / "bundle.zip")
