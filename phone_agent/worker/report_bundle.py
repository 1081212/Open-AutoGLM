"""Create a safe, self-contained legacy local report bundle."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

from phone_agent.execution.errors import ExecutionError, ExecutionErrorCode


MAX_REPORT_BUNDLE_SOURCE_SIZE = 512 * 1024 * 1024


def create_local_report_bundle(report_dir: Path, target: Path) -> Path:
    root = report_dir.resolve()
    if not root.is_dir():
        raise ExecutionError(
            ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
            f"Local report directory is missing: {report_dir}",
        )
    files: list[tuple[Path, str, int, str]] = []
    total_size = 0
    for path in sorted(report_dir.rglob("*")):
        if path.is_symlink():
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                f"Local report contains a symlink: {path.name}",
            )
        if not path.is_file():
            continue
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                "Local report path escapes its root",
            )
        relative = path.relative_to(report_dir).as_posix()
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                f"Unsafe report bundle entry: {relative}",
            )
        size = path.stat().st_size
        total_size += size
        if total_size > MAX_REPORT_BUNDLE_SOURCE_SIZE:
            raise ExecutionError(
                ExecutionErrorCode.ARTIFACT_ENCRYPT_FAILED,
                "Local report bundle source exceeds 512 MiB",
            )
        files.append((path, relative, size, _sha256(path)))

    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    part = target.with_suffix(target.suffix + ".part")
    manifest = {
        "schema_version": "autoglm.local-report-bundle.v1",
        "files": [
            {"path": relative, "size": size, "sha256": digest}
            for _path, relative, size, digest in files
        ],
    }
    try:
        with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            )
            for path, relative, _size, _digest in files:
                archive.write(path, relative)
        with part.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
