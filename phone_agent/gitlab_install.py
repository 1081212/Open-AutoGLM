"""GitLab build artifact download and Android APK installation helpers."""

from __future__ import annotations

import os
import re
import shutil
import ssl
import stat
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import quote, unquote, urlencode

import requests
from requests.adapters import HTTPAdapter

DEFAULT_GITLAB = "https://gitlab-office.4pd.io"
DEFAULT_PROJECT = "android%2FWearfitPro"
TERMINAL_SUCCESS = "success"
TERMINAL_FAILURES = {"failed", "canceled", "skipped", "manual"}
MAX_ZIP_ENTRIES = 10_000
MAX_ZIP_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
MAX_ZIP_ENTRY_BYTES = 1024 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 200


class GitLabInstallError(RuntimeError):
    """Raised when build, artifact download, or APK installation fails."""


class FrozenArtifactError(GitLabInstallError):
    """Sanitized failure while consuming one platform-frozen artifact job."""

    def __init__(
        self,
        message: str,
        *,
        candidate_recoverable: bool = False,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.candidate_recoverable = candidate_recoverable
        self.retryable = retryable


class TLSHttpAdapter(HTTPAdapter):
    """Requests adapter that can force a specific TLS version."""

    def __init__(self, tls_version: str, verify_ssl: bool, *args, **kwargs):
        self.tls_version = tls_version
        self.verify_ssl = verify_ssl
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        context = (
            ssl.create_default_context()
            if self.verify_ssl
            else ssl._create_unverified_context()
        )
        if self.tls_version == "1.2":
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.maximum_version = ssl.TLSVersion.TLSv1_2
        elif self.tls_version == "1.3":
            context.minimum_version = ssl.TLSVersion.TLSv1_3
            context.maximum_version = ssl.TLSVersion.TLSv1_3
        pool_kwargs["ssl_context"] = context
        return super().init_poolmanager(
            connections, maxsize, block=block, **pool_kwargs
        )


@dataclass
class GitLabInstallConfig:
    """Configuration for the pre-test GitLab build/install flow."""

    branch: str
    variant: str
    token: str
    gitlab: str = DEFAULT_GITLAB
    project: str = DEFAULT_PROJECT
    output_dir: str = "gitlab_artifacts"
    filename: str | None = None
    job_id: int | None = None
    keep_zip: bool = False
    poll_interval: int = 20
    timeout: int = 0
    verify_ssl: bool = True
    tls_version: str = "1.2"
    use_env_proxy: bool = False
    device_id: str | None = None


def create_session(
    verify_ssl: bool = True,
    tls_version: str = "1.2",
    use_env_proxy: bool = False,
) -> requests.Session:
    session = requests.Session()
    session.trust_env = use_env_proxy
    if tls_version != "auto":
        adapter = TLSHttpAdapter(tls_version, verify_ssl=verify_ssl)
        session.mount("https://", adapter)
    return session


def request_json(
    method: str,
    url: str,
    token: str,
    data: list[tuple[str, str]] | None = None,
    verify_ssl: bool = True,
    tls_version: str = "1.2",
    use_env_proxy: bool = False,
) -> Any:
    session = create_session(
        verify_ssl=verify_ssl,
        tls_version=tls_version,
        use_env_proxy=use_env_proxy,
    )
    headers = {"PRIVATE-TOKEN": token}
    try:
        response = session.request(
            method,
            url,
            headers=headers,
            data=data,
            timeout=60,
            verify=verify_ssl,
        )
    except requests.exceptions.SSLError as exc:
        raise GitLabInstallError(f"GitLab API SSL 请求失败：{exc}") from exc
    except requests.exceptions.Timeout as exc:
        raise GitLabInstallError("GitLab API 请求超时。") from exc
    except requests.exceptions.RequestException as exc:
        raise GitLabInstallError(f"GitLab API 请求失败：{exc}") from exc

    if response.status_code >= 400:
        detail = response.text.strip()
        raise GitLabInstallError(
            f"GitLab API 返回 HTTP {response.status_code}：{detail}"
        )

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise GitLabInstallError(f"GitLab API 返回内容不是合法 JSON：{exc}") from exc


def trigger_pipeline(config: GitLabInstallConfig) -> int:
    url = f"{config.gitlab.rstrip('/')}/api/v4/projects/{config.project}/pipeline"
    payload = [
        ("ref", config.branch),
        ("variables[][key]", "BUILD_VARIANT"),
        ("variables[][value]", config.variant),
    ]
    data = request_json(
        "POST",
        url,
        config.token,
        payload,
        verify_ssl=config.verify_ssl,
        tls_version=config.tls_version,
        use_env_proxy=config.use_env_proxy,
    )
    if not isinstance(data, dict) or not isinstance(data.get("id"), int):
        raise GitLabInstallError(
            f"Pipeline response did not contain numeric id: {data}"
        )
    return data["id"]


def get_pipeline_status(config: GitLabInstallConfig, pipeline_id: int) -> str:
    url = f"{config.gitlab.rstrip('/')}/api/v4/projects/{config.project}/pipelines/{pipeline_id}"
    data = request_json(
        "GET",
        url,
        config.token,
        verify_ssl=config.verify_ssl,
        tls_version=config.tls_version,
        use_env_proxy=config.use_env_proxy,
    )
    if not isinstance(data, dict) or not isinstance(data.get("status"), str):
        raise GitLabInstallError(
            f"Pipeline status response did not contain status: {data}"
        )
    return data["status"]


def get_pipeline_jobs(
    config: GitLabInstallConfig, pipeline_id: int
) -> list[dict[str, Any]]:
    query = urlencode({"per_page": "100"})
    url = (
        f"{config.gitlab.rstrip('/')}/api/v4/projects/{config.project}/"
        f"pipelines/{pipeline_id}/jobs?{query}"
    )
    data = request_json(
        "GET",
        url,
        config.token,
        verify_ssl=config.verify_ssl,
        tls_version=config.tls_version,
        use_env_proxy=config.use_env_proxy,
    )
    if not isinstance(data, list):
        raise GitLabInstallError(f"Pipeline jobs response is not a list: {data}")
    return [job for job in data if isinstance(job, dict)]


def print_pipeline_jobs(jobs: list[dict[str, Any]], printed_ids: set[int]) -> None:
    new_jobs = []
    for job in sorted(jobs, key=lambda item: int(item.get("id") or 0)):
        job_id = job.get("id")
        if not isinstance(job_id, int) or job_id in printed_ids:
            continue
        printed_ids.add(job_id)
        new_jobs.append(job)

    if not new_jobs:
        return

    print("CI Jobs:", flush=True)
    for job in new_jobs:
        job_id = job.get("id")
        name = job.get("name") or "unknown"
        status = job.get("status") or "unknown"
        web_url = job.get("web_url") or ""
        url_text = f" {web_url}" if web_url else ""
        print(f"  - job #{job_id} {name} [{status}]{url_text}", flush=True)


def wait_for_pipeline(config: GitLabInstallConfig, pipeline_id: int) -> None:
    started_at = time.monotonic()
    printed_job_ids: set[int] = set()
    while True:
        jobs = get_pipeline_jobs(config, pipeline_id)
        print_pipeline_jobs(jobs, printed_job_ids)

        status = get_pipeline_status(config, pipeline_id)
        print(f"  {datetime.now().strftime('%H:%M:%S')} {status}", flush=True)

        if status == TERMINAL_SUCCESS:
            print("✅ 成功")
            return
        if status in TERMINAL_FAILURES:
            raise GitLabInstallError(f"GitLab pipeline 结束状态异常：{status}")
        if config.timeout > 0 and time.monotonic() - started_at >= config.timeout:
            raise GitLabInstallError(f"GitLab pipeline timeout after {config.timeout}s")

        time.sleep(config.poll_interval)


def filename_from_headers(headers: dict[str, Any], fallback: str) -> str:
    disposition = str(headers.get("Content-Disposition") or "")
    match = re.search(r"filename\*=UTF-8''([^;]+)", disposition, re.IGNORECASE)
    if match:
        return sanitize_filename(unquote(match.group(1).strip().strip('"')))
    match = re.search(r'filename="?([^";]+)"?', disposition, re.IGNORECASE)
    if match:
        return sanitize_filename(unquote(match.group(1).strip()))
    return fallback


def sanitize_filename(filename: str) -> str:
    filename = filename.replace("/", "_").replace("\\", "_").strip()
    return filename or "artifacts.zip"


def find_apk_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.apk") if path.is_file())


def download_frozen_job_artifact(
    *,
    session: requests.Session,
    gitlab_base_url: str,
    project_path: str,
    job_id: int,
    token: str,
    output_dir: Path,
    declared_size: int,
    max_artifact_bytes: int,
    min_free_bytes: int,
    timeout_seconds: int,
    verify_ssl: bool,
    cancellation_check: Callable[[], None] | None = None,
) -> Path:
    """Download exactly one frozen GitLab job artifact using an atomic rename."""

    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    expected_size = declared_size or min(max_artifact_bytes, 256 * 1024 * 1024)
    if declared_size > max_artifact_bytes:
        raise FrozenArtifactError("declared GitLab artifact exceeds the Worker limit")
    _require_disk_space(output_dir, expected_size, min_free_bytes)
    encoded_project = quote(project_path, safe="")
    url = (
        f"{gitlab_base_url.rstrip('/')}/api/v4/projects/"
        f"{encoded_project}/jobs/{job_id}/artifacts"
    )
    final_path = output_dir / f"job-{job_id}-artifacts.zip"
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    part_path.unlink(missing_ok=True)
    response = None
    try:
        response = session.get(
            url,
            headers={"PRIVATE-TOKEN": token},
            stream=True,
            timeout=timeout_seconds,
            verify=verify_ssl,
            allow_redirects=False,
        )
        if response.status_code == 404:
            raise FrozenArtifactError(
                f"GitLab job {job_id} has no downloadable artifact",
                candidate_recoverable=True,
            )
        if response.status_code in {401, 403}:
            raise FrozenArtifactError("GitLab artifact authentication was rejected")
        if response.status_code >= 400:
            raise FrozenArtifactError(
                f"GitLab artifact request failed with HTTP {response.status_code}",
                retryable=response.status_code in {408, 429}
                or response.status_code >= 500,
            )
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > max_artifact_bytes:
                    raise FrozenArtifactError(
                        "GitLab artifact response exceeds the Worker limit"
                    )
            except ValueError as exc:
                raise FrozenArtifactError(
                    "GitLab artifact Content-Length is invalid"
                ) from exc
        written = 0
        with part_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if cancellation_check:
                    cancellation_check()
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_artifact_bytes:
                    raise FrozenArtifactError(
                        "GitLab artifact response exceeds the Worker limit"
                    )
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if declared_size and written != declared_size:
            raise FrozenArtifactError(
                "GitLab artifact size does not match the frozen Plan"
            )
        os.replace(part_path, final_path)
        return final_path
    except requests.exceptions.SSLError as exc:
        raise FrozenArtifactError("GitLab artifact TLS verification failed") from exc
    except requests.exceptions.Timeout as exc:
        raise FrozenArtifactError(
            "GitLab artifact download timed out", retryable=True
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise FrozenArtifactError(
            "GitLab artifact transport failed", retryable=True
        ) from exc
    except OSError as exc:
        raise FrozenArtifactError(
            "GitLab artifact could not be stored locally"
        ) from exc
    finally:
        part_path.unlink(missing_ok=True)
        if response is not None:
            response.close()


def extract_single_apk_secure(
    zip_path: Path,
    output_dir: Path,
    *,
    output_filename: str | None,
    max_apk_bytes: int,
    min_free_bytes: int,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_uncompressed_bytes: int = MAX_ZIP_UNCOMPRESSED_BYTES,
    max_entry_bytes: int = MAX_ZIP_ENTRY_BYTES,
    max_compression_ratio: int = MAX_ZIP_COMPRESSION_RATIO,
    cancellation_check: Callable[[], None] | None = None,
) -> Path:
    """Validate an entire ZIP and extract only its unique APK atomically."""

    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            entries = archive.infolist()
            if len(entries) > max_entries:
                raise FrozenArtifactError(
                    "GitLab artifact ZIP contains too many entries"
                )
            total_size = 0
            apk_entries: list[zipfile.ZipInfo] = []
            for entry in entries:
                if cancellation_check:
                    cancellation_check()
                _validate_zip_entry(entry)
                if entry.is_dir():
                    continue
                if entry.file_size > max_entry_bytes:
                    raise FrozenArtifactError("GitLab artifact ZIP entry is too large")
                total_size += entry.file_size
                if total_size > max_uncompressed_bytes:
                    raise FrozenArtifactError(
                        "GitLab artifact ZIP expands beyond the Worker limit"
                    )
                ratio = entry.file_size / max(1, entry.compress_size)
                if ratio > max_compression_ratio:
                    raise FrozenArtifactError(
                        "GitLab artifact ZIP compression ratio is unsafe"
                    )
                if (
                    PurePosixPath(entry.filename.replace("\\", "/")).suffix.lower()
                    == ".apk"
                ):
                    apk_entries.append(entry)
            if not apk_entries:
                raise FrozenArtifactError(
                    "GitLab artifact ZIP contains no APK",
                    candidate_recoverable=True,
                )
            if len(apk_entries) != 1:
                raise FrozenArtifactError(
                    "GitLab artifact ZIP contains multiple APK files",
                    candidate_recoverable=True,
                )
            apk_entry = apk_entries[0]
            if apk_entry.file_size > max_apk_bytes:
                raise FrozenArtifactError("APK exceeds the Worker size limit")
            _require_disk_space(output_dir, apk_entry.file_size, min_free_bytes)
            output_name = sanitize_filename(
                output_filename
                or PurePosixPath(apk_entry.filename.replace("\\", "/")).name
            )
            if not output_name.lower().endswith(".apk"):
                output_name += ".apk"
            output_path = output_dir / output_name
            part_path = output_path.with_suffix(output_path.suffix + ".part")
            part_path.unlink(missing_ok=True)
            written = 0
            try:
                with archive.open(apk_entry, "r") as source, part_path.open(
                    "wb"
                ) as target:
                    while True:
                        if cancellation_check:
                            cancellation_check()
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > max_apk_bytes or written > apk_entry.file_size:
                            raise FrozenArtifactError(
                                "APK expands beyond its declared size"
                            )
                        target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                if written != apk_entry.file_size:
                    raise FrozenArtifactError("APK size does not match ZIP metadata")
                os.replace(part_path, output_path)
                return output_path
            finally:
                part_path.unlink(missing_ok=True)
    except FrozenArtifactError:
        raise
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise FrozenArtifactError("GitLab artifact is not a safe readable ZIP") from exc


def _validate_zip_entry(entry: zipfile.ZipInfo) -> None:
    normalized = entry.filename.replace("\\", "/")
    path = PurePosixPath(normalized)
    mode = entry.external_attr >> 16
    if (
        not normalized
        or "\x00" in normalized
        or path.is_absolute()
        or ".." in path.parts
        or stat.S_ISLNK(mode)
        or bool(entry.flag_bits & 0x1)
    ):
        raise FrozenArtifactError("GitLab artifact ZIP contains an unsafe entry")


def _require_disk_space(path: Path, incoming_bytes: int, min_free_bytes: int) -> None:
    try:
        free = shutil.disk_usage(path).free
    except OSError as exc:
        raise FrozenArtifactError(
            "Worker spool free space cannot be determined"
        ) from exc
    if free - incoming_bytes < min_free_bytes:
        raise FrozenArtifactError("Worker spool has insufficient free space")


def extract_single_apk(
    zip_path: Path, output_dir: Path, filename: str | None = None
) -> Path:
    try:
        return extract_single_apk_secure(
            zip_path,
            output_dir,
            output_filename=filename,
            max_apk_bytes=MAX_ZIP_ENTRY_BYTES,
            min_free_bytes=1,
        )
    except FrozenArtifactError as exc:
        raise GitLabInstallError(str(exc)) from exc


def download_job_artifacts(config: GitLabInstallConfig, job_id: int) -> Path:
    session = create_session(
        verify_ssl=config.verify_ssl,
        tls_version=config.tls_version,
        use_env_proxy=config.use_env_proxy,
    )
    url = f"{config.gitlab.rstrip('/')}/api/v4/projects/{config.project}/jobs/{job_id}/artifacts"
    headers = {"PRIVATE-TOKEN": config.token}

    try:
        response = session.get(
            url,
            headers=headers,
            stream=True,
            timeout=60,
            verify=config.verify_ssl,
        )
    except requests.exceptions.SSLError as exc:
        raise GitLabInstallError(f"GitLab artifacts SSL 请求失败：{exc}") from exc
    except requests.exceptions.Timeout as exc:
        raise GitLabInstallError("GitLab artifacts 下载请求超时。") from exc
    except requests.exceptions.RequestException as exc:
        raise GitLabInstallError(f"GitLab artifacts 下载请求失败：{exc}") from exc

    if response.status_code == 404:
        raise GitLabInstallError(
            f"job #{job_id} 没有可下载 artifacts，或当前账号无权限访问。"
        )
    if response.status_code >= 400:
        detail = response.text.strip()
        raise GitLabInstallError(
            f"GitLab artifacts 返回 HTTP {response.status_code}：{detail}"
        )

    output_root = Path(config.output_dir)
    fallback = f"job_{job_id}_artifacts.zip"
    artifact_name = filename_from_headers(response.headers, fallback)
    with tempfile.TemporaryDirectory(prefix="gitlab_artifacts_download_") as temp_dir:
        zip_path = Path(temp_dir) / artifact_name
        with zip_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

        apk_path = extract_single_apk(zip_path, output_root, filename=config.filename)
        if config.keep_zip:
            kept_zip_path = output_root / sanitize_filename(artifact_name)
            shutil.copy2(zip_path, kept_zip_path)
            print(f"保留 artifacts zip：{kept_zip_path}")
        return apk_path


def find_artifact_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for job in jobs:
        if job.get("status") != "success":
            continue
        artifacts_file = job.get("artifacts_file")
        artifacts = job.get("artifacts")
        has_file = isinstance(artifacts_file, dict) and bool(
            artifacts_file.get("filename") or artifacts_file.get("size")
        )
        has_artifacts = isinstance(artifacts, list) and bool(artifacts)
        if has_file or has_artifacts:
            candidates.append(job)
    return sorted(candidates, key=lambda item: int(item.get("id") or 0), reverse=True)


def install_apk_with_adb(apk_path: Path, device_id: str | None = None) -> None:
    cmd = ["adb"]
    if device_id:
        cmd.extend(["-s", device_id])
    cmd.extend(["install", "-r", "-d", str(apk_path)])
    print("Installing APK:")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    output = "\n".join(
        part.strip() for part in [result.stdout, result.stderr] if part.strip()
    )
    if output:
        print(output)
    if result.returncode != 0:
        raise GitLabInstallError(f"APK 安装失败，exit={result.returncode}")


def build_download_install(config: GitLabInstallConfig) -> Path:
    if config.poll_interval <= 0:
        raise GitLabInstallError("--install-poll-interval 必须是正整数。")
    if config.timeout < 0:
        raise GitLabInstallError("--install-timeout 不能为负数。")

    print("GitLab build config:")
    print(f"  gitlab: {config.gitlab}")
    print(f"  project: {config.project}")
    print(f"  branch: {config.branch}")
    print(f"  variant: {config.variant}")

    pipeline_id = trigger_pipeline(config)
    pipeline_url = (
        f"{config.gitlab.rstrip('/')}/android/WearfitPro/-/pipelines/{pipeline_id}"
    )
    print(f"▶ #{pipeline_id} 运行中: {pipeline_url}", flush=True)
    wait_for_pipeline(config, pipeline_id)

    jobs = get_pipeline_jobs(config, pipeline_id)
    if config.job_id:
        artifact_jobs = [{"id": config.job_id, "name": f"job #{config.job_id}"}]
    else:
        artifact_jobs = find_artifact_jobs(jobs)
    if not artifact_jobs:
        raise GitLabInstallError("pipeline 成功，但没有找到带 artifacts 的成功 job。")

    last_error: Exception | None = None
    apk_path: Path | None = None
    for job in artifact_jobs:
        job_id = int(job.get("id") or 0)
        job_name = job.get("name") or "unknown"
        if job_id <= 0:
            continue
        print(f"Trying artifacts from job #{job_id} {job_name}...")
        try:
            apk_path = download_job_artifacts(config, job_id)
            break
        except GitLabInstallError as exc:
            last_error = exc
            print(f"  跳过 job #{job_id}: {exc}")
    if apk_path is None:
        raise GitLabInstallError(f"没有从 artifacts 中解出 APK。最后错误：{last_error}")

    print(f"✅ APK 已下载：{apk_path}")
    install_apk_with_adb(apk_path, device_id=config.device_id)
    print("✅ APK 已安装到手机。")
    return apk_path
