#!/usr/bin/env python3
"""Run main.py sequentially for multiple test modules."""

"""
python scripts/run_modules.py 登录 首页 我的 视频 手表 \
    --continue-on-fail \
    --limit 10 \
    --file docs/yl.md \
    -- \
    --base-url "https://open.bigmodel.cn/api/paas/v4" \
    --max-steps=8 \
    --model "autoglm-phone" \
    --apikey "6" --judge-api-key "c"
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
        default="docs/yl.md",
        help="测试用例 Markdown 文件，默认 docs/yl.md。",
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


def build_command(
    *,
    python_bin: str,
    main_py: str,
    file_path: str,
    module: str,
    module_index: int,
    priority: list[str] | None,
    limit: int | None,
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
    if not has_option(extra, "--artifact-name"):
        cmd.extend(
            [
                "--artifact-name",
                f"{artifact_prefix}_{module_index:02d}_{safe_name(module)}",
            ]
        )
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
            python_bin=args.python,
            main_py=str((root / args.main).resolve()),
            file_path=args.file,
            module=module,
            module_index=index,
            priority=args.priority,
            limit=args.limit,
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
