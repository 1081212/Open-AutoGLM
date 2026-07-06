"""GitLab build artifact download and Android APK installation helpers."""

from __future__ import annotations

import os
import re
import shutil
import ssl
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlencode

import requests
from requests.adapters import HTTPAdapter


DEFAULT_GITLAB = "https://gitlab-office.4pd.io"
DEFAULT_PROJECT = "android%2FWearfitPro"
TERMINAL_SUCCESS = "success"
TERMINAL_FAILURES = {"failed", "canceled", "skipped", "manual"}


class GitLabInstallError(RuntimeError):
    """Raised when build, artifact download, or APK installation fails."""


class TLSHttpAdapter(HTTPAdapter):
    """Requests adapter that can force a specific TLS version."""

    def __init__(self, tls_version: str, verify_ssl: bool, *args, **kwargs):
        self.tls_version = tls_version
        self.verify_ssl = verify_ssl
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        context = ssl.create_default_context() if self.verify_ssl else ssl._create_unverified_context()
        if self.tls_version == "1.2":
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.maximum_version = ssl.TLSVersion.TLSv1_2
        elif self.tls_version == "1.3":
            context.minimum_version = ssl.TLSVersion.TLSv1_3
            context.maximum_version = ssl.TLSVersion.TLSv1_3
        pool_kwargs["ssl_context"] = context
        return super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)


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
        raise GitLabInstallError(f"GitLab API 返回 HTTP {response.status_code}：{detail}")

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
        raise GitLabInstallError(f"Pipeline response did not contain numeric id: {data}")
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
        raise GitLabInstallError(f"Pipeline status response did not contain status: {data}")
    return data["status"]


def get_pipeline_jobs(config: GitLabInstallConfig, pipeline_id: int) -> list[dict[str, Any]]:
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


def extract_single_apk(zip_path: Path, output_dir: Path, filename: str | None = None) -> Path:
    with tempfile.TemporaryDirectory(prefix="gitlab_artifacts_extract_") as temp_dir:
        extract_root = Path(temp_dir)
        try:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_root)
        except zipfile.BadZipFile as exc:
            raise GitLabInstallError(f"artifacts 不是合法 zip 文件：{zip_path}") from exc

        apk_files = find_apk_files(extract_root)
        if not apk_files:
            raise GitLabInstallError(f"artifacts 中没有找到 APK：{zip_path}")
        if len(apk_files) > 1:
            apk_list = "\n".join(f"  - {apk.relative_to(extract_root)}" for apk in apk_files)
            raise GitLabInstallError(
                "artifacts 中找到多个 APK，当前无法自动选择。请调整 CI 产物或传 --install-job-id：\n"
                f"{apk_list}"
            )

        apk_name = sanitize_filename(filename) if filename else sanitize_filename(apk_files[0].name)
        if not apk_name.lower().endswith(".apk"):
            apk_name += ".apk"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / apk_name
        shutil.move(str(apk_files[0]), output_path)
        return output_path


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
        raise GitLabInstallError(f"job #{job_id} 没有可下载 artifacts，或当前账号无权限访问。")
    if response.status_code >= 400:
        detail = response.text.strip()
        raise GitLabInstallError(f"GitLab artifacts 返回 HTTP {response.status_code}：{detail}")

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
    output = "\n".join(part.strip() for part in [result.stdout, result.stderr] if part.strip())
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
    pipeline_url = f"{config.gitlab.rstrip('/')}/android/WearfitPro/-/pipelines/{pipeline_id}"
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
