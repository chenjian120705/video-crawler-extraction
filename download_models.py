#!/usr/bin/env python3
"""从 ModelScope 下载 Gradio UI 所需的 ASR 模型及配套 VAD。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

# 嵌入式 Python（yaowang2035）不会自动把脚本目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model_paths import (
    AUX_MODEL_SPECS,
    MODELS_DIR,
    MODEL_SPECS,
    ensure_models_layout,
    format_verification_report,
    local_model_dir,
    ui_model_specs,
    verify_all_local_models,
)

# 国内镜像（与 2035 配置一致，可按需注释）
os.environ.setdefault("MODELSCOPE_ENDPOINT", "https://mirrors.aliyun.com/modelscope/")
os.environ.setdefault("MODELSCOPE_PARALLEL_DOWNLOAD", "True")

ensure_models_layout()

MODELS: list[tuple[str, str]] = [
    (label, spec["modelscope_id"]) for label, spec in ui_model_specs().items()
]

AUX_MODELS: list[tuple[str, str]] = [
    (label, spec["modelscope_id"]) for label, spec in AUX_MODEL_SPECS.items()
]


def download_one(name: str, model_id: str, revision: str) -> str:
    """Download a single model snapshot from ModelScope into ./models."""
    from modelscope.hub.snapshot_download import snapshot_download

    print(f"\n{'=' * 60}")
    print(f"正在下载: {name}")
    print(f"ModelScope ID: {model_id}")
    print(f"目标目录: {local_model_dir(model_id)}")
    print(f"{'=' * 60}")
    # 使用 MODELSCOPE_CACHE=项目根目录，权重落在 ./models/{org}/{name}
    local_dir = snapshot_download(
        model_id,
        revision=revision,
    )
    print(f"已保存到: {local_dir}")
    return local_dir


def download_all(
    models: Iterable[tuple[str, str]],
    revision: str,
    include_aux: bool,
) -> int:
    """Download all listed models; return exit code (0 success)."""
    items = list(models)
    if include_aux:
        items.extend(AUX_MODELS)

    failed: list[str] = []
    for name, model_id in items:
        try:
            download_one(name, model_id, revision)
        except Exception as exc:
            print(f"[失败] {name} ({model_id}): {exc}", file=sys.stderr)
            failed.append(name)

    print(f"\n{'=' * 60}")
    if failed:
        print("以下模型下载失败:", ", ".join(failed))
        print("请检查网络、ModelScope 登录状态或模型 ID 是否可用。")
        return 1

    print("全部模型下载完成。")
    print(f"模型根目录: {MODELS_DIR.resolve()}")
    print("\n本地完整性检查:")
    print(format_verification_report(verify_all_local_models()))
    print("可执行 start.bat 启动 Gradio 界面")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="从 ModelScope 下载 FunASR Gradio UI 模型")
    parser.add_argument(
        "--revision",
        default="master",
        help="模型版本分支，默认 master",
    )
    parser.add_argument(
        "--no-aux",
        action="store_true",
        help="不下载配套 fsmn-vad（仅当你确定不需要长音频 VAD 时）",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="仅检查本地模型完整性，不下载",
    )
    parser.add_argument(
        "--only",
        choices=[m[0] for m in MODELS],
        nargs="+",
        help="仅下载指定模型（显示名称）",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    if args.verify_only:
        report = format_verification_report(verify_all_local_models())
        print(report)
        issues = verify_all_local_models()
        return 1 if any(issues.values()) else 0

    targets = MODELS
    if args.only:
        only_set = set(args.only)
        targets = [m for m in MODELS if m[0] in only_set]
        unknown = only_set - {m[0] for m in targets}
        if unknown:
            print(f"未知模型名: {unknown}", file=sys.stderr)
            return 2

    return download_all(targets, args.revision, include_aux=not args.no_aux)


if __name__ == "__main__":
    sys.exit(main())
