"""项目内模型目录：下载与 FunASR/ModelScope 加载共用。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, TypedDict

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_UPLOADS = OUTPUT_DIR / "uploads"
OUTPUT_URL_DOWNLOADS = OUTPUT_UPLOADS / "url_downloads"
OUTPUT_CONVERTED = OUTPUT_DIR / "converted"
OUTPUT_TRANSCRIPTS = OUTPUT_DIR / "transcripts"

LoadMode = Literal["funasr_yaml", "funasr_id"]


class ModelSpec(TypedDict, total=False):
    """UI 模型与本地/加载策略配置。"""

    key: str
    modelscope_id: str
    required_files: list[str]
    optional_weight_globs: list[str]
    load_mode: LoadMode
    uses_vad: bool
    requires_glm_arch: bool
    ui_visible: bool


# 单一数据源：UI 显示名 -> 模型规格
MODEL_SPECS: dict[str, ModelSpec] = {
    "Fun-ASR-Nano": {
        "key": "fun_asr_nano",
        "modelscope_id": "FunAudioLLM/Fun-ASR-Nano-2512",
        "required_files": ["config.yaml", "model.pt", "configuration.json"],
        "load_mode": "funasr_yaml",
        "uses_vad": True,
    },
    "Qwen3-ASR-1.7B": {
        "key": "qwen3_asr",
        "modelscope_id": "Qwen/Qwen3-ASR-1.7B",
        "required_files": ["config.json", "model.safetensors.index.json"],
        "load_mode": "funasr_id",
        "uses_vad": False,
    },
    "Whisper-large-v3-turbo": {
        "key": "whisper_turbo",
        "modelscope_id": "iic/Whisper-large-v3-turbo",
        "required_files": ["config.yaml", "large-v3-turbo.pt", "configuration.json"],
        "load_mode": "funasr_yaml",
        "uses_vad": True,
        "ui_visible": False,
    },
    "GLM-ASR-Nano-1.5B": {
        "key": "glm_asr",
        "modelscope_id": "ZhipuAI/GLM-ASR-Nano-2512",
        "required_files": ["config.json", "model.safetensors"],
        "load_mode": "funasr_id",
        "uses_vad": False,
        "requires_glm_arch": True,
        "ui_visible": False,
    },
}

AUX_MODEL_SPECS: dict[str, ModelSpec] = {
    "fsmn-vad": {
        "key": "fsmn_vad",
        "modelscope_id": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "required_files": ["config.yaml", "model.pt"],
        "load_mode": "funasr_yaml",
        "uses_vad": False,
    },
}


def ensure_output_layout() -> Path:
    """创建运行时输出目录（上传、转码、转写结果）。"""
    for folder in (
        OUTPUT_DIR,
        OUTPUT_UPLOADS,
        OUTPUT_URL_DOWNLOADS,
        OUTPUT_CONVERTED,
        OUTPUT_TRANSCRIPTS,
    ):
        folder.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def ensure_models_layout() -> Path:
    """创建 models 目录，并将 ModelScope 缓存指向项目根目录。

    ModelScope 规则：实际权重位于 ``{MODELSCOPE_CACHE}/models/{组织}/{模型名}``，
    因此将 ``MODELSCOPE_CACHE`` 设为项目根目录，即可得到 ``./models/...``。
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MODELSCOPE_CACHE"] = str(PROJECT_ROOT)
    return MODELS_DIR


def ensure_project_layout() -> None:
    """初始化模型目录与运行时输出目录。"""
    ensure_models_layout()
    ensure_output_layout()


def modelscope_model_id_to_name(model_id: str) -> str:
    """ModelScope 本地目录名（点号替换为下划线）。"""
    return model_id.split("/", 1)[1].replace(".", "___")


def local_model_dir(model_id: str) -> Path:
    """返回某个 ModelScope 模型在本项目中的预期本地路径。"""
    owner, _ = model_id.split("/", 1)
    return MODELS_DIR / owner / modelscope_model_id_to_name(model_id)


def resolve_model_path(model_id: str) -> str:
    """若已下载则返回本地目录，否则返回 ModelScope 模型 ID 供在线拉取。"""
    local = local_model_dir(model_id)
    if local.is_dir():
        return str(local)
    return model_id


def is_funasr_yaml_package(model_dir: Path) -> bool:
    """是否为 FunASR 标准包（含 config.yaml，可从本地路径解析 model 类）。"""
    return (model_dir / "config.yaml").is_file()


def ui_model_specs() -> dict[str, ModelSpec]:
    """返回 Gradio 下拉框中展示的模型（ui_visible 默认为 True）。"""
    return {
        label: spec
        for label, spec in MODEL_SPECS.items()
        if spec.get("ui_visible", True)
    }


def build_automodel_kwargs(modelscope_id: str, device: str) -> dict[str, Any]:
    """构建 FunASR AutoModel 参数（本地路径 + 已注册 ID 的统一策略）。"""
    local = resolve_model_path(modelscope_id)
    local_path = Path(local)
    kwargs: dict[str, Any] = {
        "hub": "ms",
        "device": device,
        "disable_update": True,
    }
    if local_path.is_dir():
        kwargs["model_path"] = str(local_path)
        if is_funasr_yaml_package(local_path):
            kwargs["model"] = str(local_path)
        else:
            kwargs["model"] = modelscope_id
    else:
        kwargs["model"] = modelscope_id
    return kwargs


def _has_weight_files(model_dir: Path, globs: list[str]) -> bool:
    """检查目录下是否存在任意匹配的权重文件。"""
    for pattern in globs:
        if any(model_dir.glob(pattern)):
            return True
    return False


def transformers_supports_glm_asr() -> bool:
    """当前 transformers 是否包含 GLM-ASR 架构。"""
    try:
        from transformers import GlmAsrForConditionalGeneration  # noqa: F401

        return True
    except ImportError:
        return False


def verify_local_model(label: str, specs: dict[str, ModelSpec] | None = None) -> list[str]:
    """检查单个模型本地文件与运行环境，返回问题列表（空表示通过）。"""
    catalog = specs if specs is not None else MODEL_SPECS
    if label not in catalog:
        return [f"未知模型: {label}"]

    spec = catalog[label]
    model_id = spec["modelscope_id"]
    model_dir = local_model_dir(model_id)
    issues: list[str] = []

    if not model_dir.is_dir():
        issues.append(f"本地目录不存在: {model_dir}")
        return issues

    for filename in spec.get("required_files", []):
        if not (model_dir / filename).is_file():
            issues.append(f"缺少文件: {filename}")

    optional_globs = spec.get("optional_weight_globs", [])
    if optional_globs and not _has_weight_files(model_dir, optional_globs):
        issues.append(f"未找到权重文件（匹配: {', '.join(optional_globs)}）")

    if spec.get("requires_glm_arch") and not transformers_supports_glm_asr():
        issues.append(
            "GLM-ASR 需要 transformers 5.x（含 GlmAsrForConditionalGeneration）；"
            "当前为 4.57.6（Qwen3-ASR 依赖），暂无法加载 GLM-ASR"
        )

    return issues


def verify_all_local_models(*, include_hidden: bool = False) -> dict[str, list[str]]:
    """检查全部 ASR 模型与 VAD，返回 {名称: 问题列表}。"""
    results: dict[str, list[str]] = {}
    for label, spec in MODEL_SPECS.items():
        if not include_hidden and not spec.get("ui_visible", True):
            continue
        results[label] = verify_local_model(label)
    for label in AUX_MODEL_SPECS:
        results[label] = verify_local_model(label, AUX_MODEL_SPECS)
    return results


def format_verification_report(results: dict[str, list[str]]) -> str:
    """将 verify_all_local_models 结果格式化为可读文本。"""
    lines: list[str] = []
    for label, issues in results.items():
        if issues:
            lines.append(f"[未就绪] {label}")
            lines.extend(f"  - {item}" for item in issues)
        else:
            lines.append(f"[就绪] {label}")
    return "\n".join(lines)
