#!/usr/bin/env python3
"""Run main.py sequentially for multiple test modules.

python scripts/run_modules.py 登录 首页 我的 视频 手表 \
    --continue-on-fail \
    --limit 10 \
    --file docs/新用例.md \
    --base-url "https://open.bigmodel.cn/api/paas/v4" \
    --max-steps=8 \
    --model "autoglm-phone" \
    --apikey "6" --judge-api-key "c" --case-retries 2

少数脚本未声明的 main.py 参数仍可放在 -- 后面透传。
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "按关联模块顺序运行 main.py。示例："
            "python scripts/run_modules.py 登录 手表 -- --device-id xxx"
        )
    )
    parser.add_argument(
        "modules",
        nargs="*",
        help="要依次执行的关联模块，例如：登录 手表。也可用 --modules 登录,手表。",
    )
    parser.add_argument(
        "--modules",
        dest="modules_csv",
        help="逗号分隔的模块列表，例如：登录,手表,我的。",
    )
    parser.add_argument(
        "--file",
        default="docs/新用例.md",
        help="测试用例 Markdown 文件，默认 docs/新用例.md。",
    )
    parser.add_argument(
        "--priority",
        action="append",
        help="传给 main.py 的优先级过滤，可重复，例如 --priority P0 --priority P1。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="传给 main.py 的每个模块最大用例数。",
    )
    parser.add_argument(
        "--start",
        type=int,
        help=(
            "传给 main.py 的起始位置。注意：对每个模块分别生效，"
            "例如 --start 11 表示每个模块从筛选后的第 11 条开始。"
        ),
    )
    parser.add_argument(
        "--case-retries",
        type=int,
        help="传给 main.py 的非 PASS 用例重试次数。不传则使用 main.py 默认值。",
    )
    parser.add_argument("--base-url", help="传给 main.py 的模型 API base URL。")
    parser.add_argument("--model", help="传给 main.py 的模型名称。")
    parser.add_argument("--apikey", help="传给 main.py 的模型 API key。")
    parser.add_argument("--judge-base-url", help="传给 main.py 的 judge API base URL。")
    parser.add_argument("--judge-model", help="传给 main.py 的 judge 模型名称。")
    parser.add_argument("--judge-api-key-env", help="传给 main.py 的 judge API key 环境变量名。")
    parser.add_argument("--judge-api-key", help="传给 main.py 的 judge API key。")
    parser.add_argument(
        "--disable-status-judge",
        action="store_true",
        help="传给 main.py，禁用 judge 模型状态判断。",
    )
    parser.add_argument("--max-steps", type=int, help="传给 main.py 的每步最大操作数。")
    parser.add_argument("--device-id", "-d", help="传给 main.py 的设备 ID。")
    parser.add_argument(
        "--device-type",
        choices=["adb", "hdc", "ios"],
        help="传给 main.py 的设备类型。",
    )
    parser.add_argument("--wda-url", help="传给 main.py 的 WebDriverAgent URL。")
    parser.add_argument(
        "--lang",
        choices=["cn", "en"],
        help="传给 main.py 的提示词语言。",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="传给 main.py，减少输出。",
    )
    parser.add_argument(
        "--artifact-prefix",
        default=None,
        help="本批次 artifact 前缀。默认 batch_YYYYmmdd_HHMMSS。",
    )
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="某个模块失败后继续执行后续模块。",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python 解释器路径，默认使用当前解释器。",
    )
    parser.add_argument(
        "--main",
        default="main.py",
        help="main.py 路径，默认当前项目根目录下的 main.py。",
    )
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def normalize_modules(args: argparse.Namespace) -> list[str]:
    modules: list[str] = []
    if args.modules_csv:
        modules.extend(part.strip() for part in args.modules_csv.split(","))
    modules.extend(args.modules)
    modules = [module for module in modules if module]
    if not modules:
        raise SystemExit("必须提供至少一个模块，例如：登录 手表，或 --modules 登录,手表")
    return modules


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", value.strip())
    return value.strip("._-") or "module"


def has_option(args: list[str], option: str) -> bool:
    return option in args or any(arg.startswith(option + "=") for arg in args)


def append_option(
    cmd: list[str], extra: list[str], option: str, value: str | int | None
) -> None:
    if value is not None and not has_option(extra, option):
        cmd.extend([option, str(value)])


def append_flag(cmd: list[str], extra: list[str], option: str, enabled: bool) -> None:
    if enabled and not has_option(extra, option):
        cmd.append(option)


def build_command(
    *,
    args: argparse.Namespace,
    python_bin: str,
    main_py: str,
    file_path: str,
    module: str,
    module_index: int,
    priority: list[str] | None,
    limit: int | None,
    start: int | None,
    case_retries: int | None,
    artifact_prefix: str,
    extra: list[str],
) -> list[str]:
    cmd = [
        python_bin,
        main_py,
        "--file",
        file_path,
        "--module",
        module,
    ]
    for item in priority or []:
        cmd.extend(["--priority", item])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if start is not None:
        cmd.extend(["--start", str(start)])
    if case_retries is not None and not has_option(extra, "--case-retries"):
        cmd.extend(["--case-retries", str(case_retries)])
    if not has_option(extra, "--artifact-name"):
        cmd.extend(
            [
                "--artifact-name",
                f"{artifact_prefix}_{module_index:02d}_{safe_name(module)}",
            ]
        )
    append_option(cmd, extra, "--base-url", args.base_url)
    append_option(cmd, extra, "--model", args.model)
    append_option(cmd, extra, "--apikey", args.apikey)
    append_option(cmd, extra, "--judge-base-url", args.judge_base_url)
    append_option(cmd, extra, "--judge-model", args.judge_model)
    append_option(cmd, extra, "--judge-api-key-env", args.judge_api_key_env)
    append_option(cmd, extra, "--judge-api-key", args.judge_api_key)
    append_option(cmd, extra, "--max-steps", args.max_steps)
    append_option(cmd, extra, "--device-id", args.device_id)
    append_option(cmd, extra, "--device-type", args.device_type)
    append_option(cmd, extra, "--wda-url", args.wda_url)
    append_option(cmd, extra, "--lang", args.lang)
    append_flag(cmd, extra, "--disable-status-judge", args.disable_status_judge)
    append_flag(cmd, extra, "--quiet", args.quiet)
    cmd.extend(extra)
    return cmd


def mask_command(cmd: list[str]) -> str:
    """Return a printable command with secret values masked."""
    secret_options = {"--apikey", "--judge-api-key"}
    masked: list[str] = []
    skip_next = False
    for index, part in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if part in secret_options:
            masked.extend([part, "******"])
            skip_next = index + 1 < len(cmd)
            continue
        matched = False
        for option in secret_options:
            prefix = option + "="
            if part.startswith(prefix):
                masked.append(prefix + "******")
                matched = True
                break
        if not matched:
            masked.append(part)
    return " ".join(masked)


def main() -> int:
    args, extra = parse_args()
    modules = normalize_modules(args)
    root = Path(__file__).resolve().parents[1]
    artifact_prefix = args.artifact_prefix or dt.datetime.now().strftime(
        "batch_%Y%m%d_%H%M%S"
    )

    failures: list[tuple[str, int]] = []
    for index, module in enumerate(modules, start=1):
        cmd = build_command(
            args=args,
            python_bin=args.python,
            main_py=str((root / args.main).resolve()),
            file_path=args.file,
            module=module,
            module_index=index,
            priority=args.priority,
            limit=args.limit,
            start=args.start,
            case_retries=args.case_retries,
            artifact_prefix=artifact_prefix,
            extra=extra,
        )
        print("\n" + "=" * 80, flush=True)
        print(f"Running module {index}/{len(modules)}: {module}", flush=True)
        print("Command: " + mask_command(cmd), flush=True)
        print("=" * 80 + "\n", flush=True)
        result = subprocess.run(cmd, cwd=root)
        if result.returncode != 0:
            failures.append((module, result.returncode))
            print(
                f"\nModule failed: {module} (exit={result.returncode})",
                flush=True,
            )
            if not args.continue_on_fail:
                break

    if failures:
        print("\nFailed modules:", flush=True)
        for module, code in failures:
            print(f"- {module}: exit={code}", flush=True)
        return failures[0][1]
    print("\nAll modules completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
