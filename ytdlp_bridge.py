"""精简封装 yt-dlp 目录的短视频下载能力，供 FunASR Gradio 调用。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
YTDLP_ROOT = PROJECT_ROOT / "yt-dlp"

# 转写场景优先体积较小的清晰度
DEFAULT_PRESET = "720p 及以下"


def _ensure_ytdlp_path() -> None:
    """将 vendored yt-dlp 工程加入 import 路径。"""
    root = str(YTDLP_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def is_ytdlp_available() -> bool:
    """检查 yt-dlp 子项目目录是否存在。"""
    return YTDLP_ROOT.is_dir() and (YTDLP_ROOT / "app.py").is_file()


def _load_ytdlp_app() -> Any:
    """加载 yt-dlp/app.py 模块。"""
    if not is_ytdlp_available():
        raise RuntimeError(f"未找到 yt-dlp 组件目录: {YTDLP_ROOT}")
    _ensure_ytdlp_path()
    import app as ytdlp_app  # type: ignore[import-not-found]

    return ytdlp_app


def get_douyin_cookies_dir() -> Path:
    """抖音 Cookie 与浏览器 Profile 目录（yt-dlp/cookies）。"""
    return YTDLP_ROOT / "cookies"


def douyin_login_status() -> str:
    """返回抖音登录状态说明。"""
    if not is_ytdlp_available():
        return f"未找到 yt-dlp 组件目录: {YTDLP_ROOT}"
    try:
        ytdlp_app = _load_ytdlp_app()
        return ytdlp_app.DOUYIN_LOGIN.get_status_text()
    except Exception as exc:
        logger.exception("读取抖音登录状态失败")
        return f"读取登录状态失败: {exc}"


def douyin_open_login() -> str:
    """打开 Playwright 窗口供用户登录抖音。"""
    if not is_ytdlp_available():
        return f"未找到 yt-dlp 组件目录: {YTDLP_ROOT}"
    try:
        ytdlp_app = _load_ytdlp_app()
        return ytdlp_app.DOUYIN_LOGIN.open_login_browser()
    except Exception as exc:
        logger.exception("打开抖音登录页失败")
        return f"打开登录页失败: {exc}"


def douyin_save_login() -> str:
    """保存抖音 Cookie 到 yt-dlp/cookies/douyin.txt。"""
    if not is_ytdlp_available():
        return f"未找到 yt-dlp 组件目录: {YTDLP_ROOT}"
    try:
        ytdlp_app = _load_ytdlp_app()
        return ytdlp_app.DOUYIN_LOGIN.save_cookies()
    except Exception as exc:
        logger.exception("保存抖音 Cookie 失败")
        return f"保存登录失败: {exc}"


def close_douyin_login() -> None:
    """释放 Playwright 浏览器资源。"""
    if not is_ytdlp_available():
        return
    try:
        ytdlp_app = _load_ytdlp_app()
        ytdlp_app.DOUYIN_LOGIN.close()
    except Exception:
        logger.exception("关闭抖音登录浏览器失败")


def download_from_url(
    url: str,
    dest_dir: Path,
    progress: Any = None,
    preset_name: str = DEFAULT_PRESET,
    cookies_file: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """从短视频链接下载媒体文件。

    Args:
        url: 视频页面 URL（抖音 / B站 / YouTube 等）。
        dest_dir: 保存目录（通常为 output/uploads/url_downloads）。
        progress: Gradio Progress 或兼容 ``(ratio, desc=...)`` 的可调用对象。
        preset_name: yt-dlp 画质预设（与 yt-dlp/app.py 中 FORMAT_PRESETS 键一致）。
        cookies_file: 可选 Netscape cookies.txt（手动上传备用）。

    Returns:
        (状态说明, 本地文件路径)；失败时路径为 None。
    """
    url = (url or "").strip()
    if not url:
        return "请输入视频链接。", None

    if not is_ytdlp_available():
        return f"未找到 yt-dlp 组件目录: {YTDLP_ROOT}", None

    try:
        ytdlp_app = _load_ytdlp_app()
    except (ImportError, RuntimeError) as exc:
        logger.exception("加载 yt-dlp/app.py 失败")
        return f"加载 yt-dlp 模块失败: {exc}", None

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    original_download_dir = ytdlp_app.DOWNLOAD_DIR
    ytdlp_app.DOWNLOAD_DIR = dest_dir

    try:
        browser_label = ytdlp_app._default_browser_label()
        status, filepath = ytdlp_app.download_video(
            url=url,
            preset_name=preset_name,
            browser_label=browser_label,
            cookies_file=cookies_file,
            write_subs=False,
            embed_subs=False,
            progress=progress,
        )
    except Exception as exc:
        logger.exception("短视频下载异常: %s", url)
        return f"下载失败: {exc}", None
    finally:
        ytdlp_app.DOWNLOAD_DIR = original_download_dir

    if filepath and Path(filepath).is_file():
        return status, str(Path(filepath).resolve())

    return status, None
