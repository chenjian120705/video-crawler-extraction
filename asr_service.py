"""ASR model loading and inference for the Gradio UI."""

from __future__ import annotations

import gc
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

# 嵌入式 Python（yaowang2035）不会自动把脚本目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from model_paths import (
    MODEL_SPECS,
    OUTPUT_CONVERTED,
    OUTPUT_TRANSCRIPTS,
    OUTPUT_UPLOADS,
    build_automodel_kwargs,
    ensure_project_layout,
    local_model_dir,
    resolve_model_path,
    transformers_supports_glm_asr,
    ui_model_specs,
    verify_local_model,
)

ensure_project_layout()

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg"}

MODEL_CHOICES: dict[str, dict[str, str]] = {
    label: {
        "key": spec["key"],
        "auto_model": spec["modelscope_id"],
        "modelscope_id": spec["modelscope_id"],
    }
    for label, spec in ui_model_specs().items()
}


def resolve_device(requested: str = "auto") -> str:
    """Pick inference device from user choice or hardware availability."""
    if requested and requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，已回退到 CPU")
            return "cpu"
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_dtype(device: str) -> str:
    """Pick compute dtype; bf16 only when CUDA supports it."""
    if device == "cpu":
        return "fp32"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp16"


def is_ffmpeg_available() -> bool:
    """Return True if ffmpeg is on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except FileNotFoundError:
        return False


def prepare_audio(media_path: str, work_dir: Path | None = None) -> str:
    """Convert video to 16 kHz mono WAV when needed; return path for ASR."""
    path = Path(media_path)
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {media_path}")

    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return str(path.resolve())

    if not is_ffmpeg_available():
        raise RuntimeError(
            "当前为视频文件，但未检测到 ffmpeg。请安装 ffmpeg 并加入 PATH，或先转为 wav/mp3 再上传。"
        )

    out_dir = work_dir or OUTPUT_CONVERTED
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{path.stem}_16k.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 转码失败:\n{proc.stderr[-2000:]}")
    return str(wav_path.resolve())


def extract_text(result: Any) -> str:
    """Normalize FunASR generate() output to plain text."""
    if not result:
        return ""
    item = result[0] if isinstance(result, list) else result
    if not isinstance(item, dict):
        return str(item)

    text = item.get("text", "")
    if text:
        return str(text).strip()

    sentences = item.get("sentence_info")
    if sentences:
        lines: list[str] = []
        for seg in sentences:
            seg_text = seg.get("text") or seg.get("sentence") or ""
            spk = seg.get("spk")
            if spk is not None:
                lines.append(f"[说话人{spk}] {seg_text}")
            else:
                lines.append(str(seg_text))
        return "\n".join(lines).strip()

    return str(item)


def save_transcript(text: str, source_path: str, model_label: str) -> Path:
    """将转写结果保存到 output/transcripts/。"""
    OUTPUT_TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    stem = Path(source_path).stem
    safe_model = model_label.replace("/", "-").replace(" ", "_")
    out_path = OUTPUT_TRANSCRIPTS / f"{stem}_{safe_model}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def map_language(language_ui: str) -> Optional[str]:
    """Map UI language option to model-specific language hint."""
    mapping = {
        "自动": None,
        "中文": "Chinese",
        "英文": "English",
    }
    return mapping.get(language_ui)


def map_fun_asr_language(language_ui: str) -> Optional[str]:
    """Map UI language for Fun-ASR-Nano."""
    mapping = {
        "自动": None,
        "中文": "中文",
        "英文": "英文",
    }
    return mapping.get(language_ui)


def _audio_duration_seconds(audio_path: str) -> Optional[float]:
    """Return audio duration in seconds when readable."""
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        return float(info.duration)
    except Exception:
        return None


def _format_duration(seconds: Optional[float]) -> str:
    """Format seconds for log messages."""
    if seconds is None:
        return "未知"
    if seconds < 60:
        return f"{seconds:.1f}秒"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}分{secs:.0f}秒"


def _log_inference_result(result: Any) -> None:
    """Log a concise summary after model.generate() returns."""
    text = extract_text(result)
    if isinstance(result, list):
        logger.info("语音推理结束，返回 %d 条结果", len(result))
    else:
        logger.info("语音推理结束")

    logger.info("识别文本长度: %d 字", len(text))
    if text:
        preview = text[:200].replace("\n", " ")
        if len(text) > 200:
            preview += "..."
        logger.info("识别文本预览: %s", preview)
        return

    logger.warning("未从结果中提取到文本")
    raw_preview = repr(result)
    if len(raw_preview) > 500:
        raw_preview = raw_preview[:500] + "..."
    logger.warning("原始结果: %s", raw_preview)


def _wrap_progress_callback(
    progress_callback: Callable[[int, int], None] | None,
) -> Callable[[int, int], None]:
    """Log batch progress and optionally forward to UI callback."""
    last_logged: list[int] = [-1]

    def _callback(current: int, total: int) -> None:
        if total > 0 and current != last_logged[0]:
            last_logged[0] = current
            percent = current * 100 // total
            logger.info("语音推理进度: %d/%d (%d%%)", current, total, percent)
        if progress_callback is not None:
            progress_callback(current, total)

    return _callback


def _run_with_heartbeat(
    fn: Callable[[], Any],
    label: str,
    interval_sec: float = 30.0,
) -> Any:
    """Run a blocking call and emit periodic heartbeat logs while waiting."""
    stop = threading.Event()
    started = time.monotonic()

    def _heartbeat() -> None:
        while not stop.wait(interval_sec):
            elapsed = time.monotonic() - started
            logger.info("%s 进行中，已耗时 %s…", label, _format_duration(elapsed))

    worker = threading.Thread(target=_heartbeat, daemon=True, name="asr-heartbeat")
    worker.start()
    try:
        return fn()
    finally:
        stop.set()
        worker.join(timeout=1.0)
        elapsed = time.monotonic() - started
        logger.info("%s 阶段结束，耗时 %s", label, _format_duration(elapsed))


def _apply_disable_pbar(model: Any) -> None:
    """Ensure FunASR sub-models respect disable_pbar (VAD reads model.kwargs)."""
    if hasattr(model, "kwargs") and isinstance(model.kwargs, dict):
        model.kwargs["disable_pbar"] = True
    vad_kwargs = getattr(model, "vad_kwargs", None)
    if isinstance(vad_kwargs, dict):
        vad_kwargs["disable_pbar"] = True


def _log_model_pipeline(key: str) -> None:
    """Log model-specific inference pipeline description."""
    pipelines = {
        "fun_asr_nano": "推理流程: VAD 语音分段 → Fun-ASR-Nano 逐段 LLM 识别",
        "qwen3_asr": "推理流程: Qwen3-ASR 整段识别（长音频较慢，加载/推理均可能需数分钟）",
        "whisper_turbo": "推理流程: VAD 语音分段 → Whisper 逐段 decode",
        "glm_asr": "推理流程: GLM-ASR 逐条音频 LLM generate",
    }
    logger.info(pipelines.get(key, "推理流程: 开始语音识别"))


class ModelManager:
    """Lazy-load one ASR model at a time to limit VRAM usage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self._current_key: Optional[str] = None
        self._device: str = "cpu"

    def unload(self) -> None:
        """Release the loaded model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._current_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _resolve_vad_model(self) -> str:
        """VAD 本地路径或 ModelScope 别名。"""
        vad_id = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        return resolve_model_path(vad_id) if local_model_dir(vad_id).is_dir() else "fsmn-vad"

    def _auto_model_kwargs(self, modelscope_id: str, device: str) -> dict[str, Any]:
        """构建 AutoModel 参数（见 model_paths.build_automodel_kwargs）。"""
        return build_automodel_kwargs(modelscope_id, device)

    def _preflight_model(self, model_label: str) -> None:
        """加载前检查本地模型与环境。"""
        issues = verify_local_model(model_label)
        if issues:
            detail = "\n".join(f"  · {item}" for item in issues)
            raise RuntimeError(f"{model_label} 未通过本地检查:\n{detail}")

    def _build_model(self, model_label: str, device: str) -> Any:
        """Instantiate AutoModel for the selected label."""
        from funasr import AutoModel

        self._preflight_model(model_label)
        cfg = MODEL_CHOICES[model_label]
        key = cfg["key"]
        dtype = resolve_dtype(device)
        modelscope_id = cfg["modelscope_id"]
        common = self._auto_model_kwargs(modelscope_id, device)

        if key == "glm_asr" and not transformers_supports_glm_asr():
            raise RuntimeError(
                "GLM-ASR 需要 transformers 5.x（GlmAsrForConditionalGeneration）。"
                "当前环境为 transformers 4.57.6（Qwen3-ASR 依赖），两者暂不能并存。"
                "请使用 Fun-ASR 或 Qwen3-ASR。"
            )

        if key == "fun_asr_nano":
            # 使用 FunASR 内置 FunASRNano；模型目录无 model.py，勿设 remote_code
            return AutoModel(
                vad_model=self._resolve_vad_model(),
                vad_kwargs={"max_single_segment_time": 30000},
                **common,
            )
        if key == "qwen3_asr":
            return AutoModel(
                dtype=dtype,
                **common,
            )
        if key == "whisper_turbo":
            return AutoModel(
                vad_model=self._resolve_vad_model(),
                vad_kwargs={"max_single_segment_time": 30000},
                **common,
            )
        if key == "glm_asr":
            return AutoModel(
                dtype=dtype,
                **common,
            )
        raise ValueError(f"未知模型: {model_label}")

    def ensure_model(self, model_label: str, device: str) -> Any:
        """Load model if needed; reuse when label and device unchanged."""
        with self._lock:
            dev = resolve_device(device)
            if (
                self._model is not None
                and self._current_key == model_label
                and self._device == dev
            ):
                return self._model

            self.unload()
            logger.info("正在加载模型: %s (%s)", model_label, dev)
            logger.info("模型权重加载中，大模型首次加载可能需数分钟，请耐心等待…")
            self._model = _run_with_heartbeat(
                lambda: self._build_model(model_label, dev),
                label=f"{model_label} 模型加载",
            )
            _apply_disable_pbar(self._model)
            self._current_key = model_label
            self._device = dev
            logger.info("模型加载完成: %s (%s)", model_label, dev)
            return self._model

    def transcribe(
        self,
        media_path: Optional[str],
        model_label: str,
        language_ui: str,
        device: str = "auto",
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[str, str]:
        """Run transcription; return (text, status_message)."""
        if not media_path:
            return "", "请先上传音频或视频文件。"

        try:
            audio_path = prepare_audio(media_path, OUTPUT_CONVERTED)
        except Exception as exc:
            return "", f"媒体预处理失败: {exc}"

        duration = _audio_duration_seconds(audio_path)
        logger.info(
            "媒体预处理完成: %s | 时长 %s",
            Path(audio_path).name,
            _format_duration(duration),
        )

        try:
            model = self.ensure_model(model_label, device)
        except Exception as exc:
            return "", f"模型加载失败: {exc}\n请确认已运行 download_models.py 下载对应模型。"

        cfg = MODEL_CHOICES[model_label]
        key = cfg["key"]

        out_dir = str(OUTPUT_TRANSCRIPTS)
        gen_common: dict[str, Any] = {
            "output_dir": out_dir,
            "disable_pbar": True,
        }
        gen_common["progress_callback"] = _wrap_progress_callback(progress_callback)

        logger.info(
            "开始进行语音推理 | 模型: %s | 设备: %s | 语言: %s | 音频: %s",
            model_label,
            self._device,
            language_ui,
            audio_path,
        )
        _log_model_pipeline(key)
        if MODEL_SPECS[model_label].get("uses_vad"):
            logger.info("本模型启用 VAD 长音频分段（VAD 权重应已本地就绪）")

        try:
            if key == "fun_asr_nano":
                lang = map_fun_asr_language(language_ui)
                kwargs: dict[str, Any] = {
                    "batch_size": 1,
                    "itn": True,
                    **gen_common,
                }
                if self._device == "cpu":
                    kwargs["llm_dtype"] = "fp32"
                if lang:
                    kwargs["language"] = lang
                result = _run_with_heartbeat(
                    lambda: model.generate(input=audio_path, **kwargs),
                    label="Fun-ASR-Nano 语音推理",
                )
            elif key == "qwen3_asr":
                lang = map_language(language_ui)
                kwargs = {**gen_common}
                if lang:
                    kwargs["language"] = lang
                result = _run_with_heartbeat(
                    lambda: model.generate(input=audio_path, **kwargs),
                    label="Qwen3-ASR 语音推理",
                )
            elif key == "whisper_turbo":
                lang = map_language(language_ui)
                decoding: dict[str, Any] = {
                    "task": "transcribe",
                    "language": lang,
                    "beam_size": None,
                    "fp16": self._device != "cpu" and torch.cuda.is_available(),
                    "without_timestamps": False,
                    "prompt": None,
                }
                result = _run_with_heartbeat(
                    lambda: model.generate(
                        input=audio_path,
                        DecodingOptions=decoding,
                        batch_size_s=0,
                        **gen_common,
                    ),
                    label="Whisper 语音推理",
                )
            else:
                result = _run_with_heartbeat(
                    lambda: model.generate(input=audio_path, **gen_common),
                    label=f"{model_label} 语音推理",
                )
        except Exception as exc:
            logger.exception("转写失败")
            return "", f"转写失败: {exc}"

        _log_inference_result(result)

        text = extract_text(result)
        if not text:
            return "", "转写完成，但未识别到有效文本。"

        transcript_path = save_transcript(text, media_path, model_label)
        status = (
            f"完成 | 模型: {model_label} | 设备: {self._device} | "
            f"音频: {Path(audio_path).name} | 结果: {transcript_path}"
        )
        return text, status


_manager = ModelManager()


def get_manager() -> ModelManager:
    """Return the process-wide model manager singleton."""
    return _manager
