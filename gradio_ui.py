#!/usr/bin/env python3
"""FunASR 多模型语音转写 Gradio 界面."""

from __future__ import annotations

import argparse
import html as html_module
import logging
import os
import re
import shutil
import socket
import sys
import threading
import time
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# 嵌入式 Python（yaowang2035）不会自动把脚本目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model_paths import (
    MODELS_DIR,
    OUTPUT_DIR,
    OUTPUT_UPLOADS,
    OUTPUT_URL_DOWNLOADS,
    ensure_project_layout,
)

ensure_project_layout()

import gradio as gr
from gradio.components.base import Component as GradioComponent

from asr_service import MODEL_CHOICES, get_manager
from llm_service import (
    chat_with_history_stream,
    correct_transcript_stream,
    get_active_provider_key,
    is_correction_request,
    original_transcript_from_history,
)
from ytdlp_bridge import (
    close_douyin_login,
    douyin_login_status,
    douyin_open_login,
    douyin_save_login,
    get_douyin_cookies_dir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent
FAVICON_ICO_PATH = PROJECT_ROOT / "yaowang2035" / "img" / "head.ico"
FAVICON_ICO_FALLBACK = PROJECT_ROOT / "2035" / "img" / "head.ico"
FAVICON_TAB_PNG_PATH = PROJECT_ROOT / "yaowang2035" / "img" / "favicon-tab.png"
BTN_RUN_IDLE = "开始转写"


def _resolve_favicon_ico() -> Path | None:
    """Return the first existing head.ico path."""
    for path in (FAVICON_ICO_PATH, FAVICON_ICO_FALLBACK):
        if path.is_file():
            return path.resolve()
    return None


def _prepare_favicon_for_browser() -> Path | None:
    """Build a small PNG tab icon from head.ico for reliable browser display."""
    ico = _resolve_favicon_ico()
    if ico is None:
        return None
    try:
        from PIL import Image

        png = FAVICON_TAB_PNG_PATH
        if png.is_file() and png.stat().st_mtime >= ico.stat().st_mtime:
            return png.resolve()
        png.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(ico) as img:
            img = img.convert("RGBA")
            img.thumbnail((32, 32), Image.Resampling.LANCZOS)
            img.save(png, format="PNG")
        return png.resolve()
    except Exception as exc:
        logging.warning("无法从 head.ico 生成 PNG 标签图标，将直接使用 ICO: %s", exc)
        return ico


def _favicon_head_html(favicon_file: Path) -> str:
    """Inject cache-busted favicon links so browsers pick up custom icon."""
    version = int(favicon_file.stat().st_mtime)
    mime = "image/png" if favicon_file.suffix.lower() == ".png" else "image/x-icon"
    href = f"/favicon.ico?v={version}"
    return f"""
<link rel="icon" href="{href}" type="{mime}" sizes="32x32">
<link rel="shortcut icon" href="{href}" type="{mime}">
<script>
(function () {{
  var href = "{href}";
  var type = "{mime}";
  document.querySelectorAll("link[rel*='icon']").forEach(function (el) {{
    el.type = type;
    el.href = href;
  }});
  if (!document.querySelector("link[rel*='icon']")) {{
    var link = document.createElement("link");
    link.rel = "icon";
    link.type = type;
    link.href = href;
    document.head.appendChild(link);
  }}
}})();
</script>
"""


def _apply_favicon_to_demo(demo: gr.Blocks) -> Path | None:
    """Set Blocks.favicon_path and custom head tags for the tab icon."""
    favicon_file = _prepare_favicon_for_browser()
    if favicon_file is None:
        logging.warning(
            "未找到标签页图标，请放置 head.ico 于: %s 或 %s",
            FAVICON_ICO_PATH,
            FAVICON_ICO_FALLBACK,
        )
        return None
    demo.favicon_path = str(favicon_file)
    demo.head = (demo.head or "") + _favicon_head_html(favicon_file)
    logging.info("标签页图标: %s", favicon_file)
    return favicon_file


def _btn_run_label(percent: int) -> str:
    """Format transcribe button label with progress percentage."""
    return f"{BTN_RUN_IDLE}({max(0, min(percent, 100))}%)"


_NOTICE_ERROR_MARKERS = (
    "失败",
    "错误",
    "异常",
    "请输入",
    "请在本地上传",
    "未找到",
    "下载失败",
    "转写失败",
    "调用大模型失败",
    "中止",
    "加载 yt-dlp 模块失败",
    "链接下载失败",
)
_NOTICE_SUCCESS_MARKERS = ("成功", "完成", "已卸载", "获取成功", "结果:")


def _is_error_notice(text: str) -> bool:
    """Heuristically detect whether a status message represents an error."""
    message = (text or "").strip()
    if not message:
        return False
    if any(marker in message for marker in _NOTICE_SUCCESS_MARKERS) and not any(
        marker in message for marker in ("失败", "错误", "异常")
    ):
        return False
    return any(marker in message for marker in _NOTICE_ERROR_MARKERS)


TOAST_DURATION = 5.0
TOAST_ERROR_DURATION = 8.0


def _shorten_toast_message(message: str, max_len: int = 280) -> str:
    """Trim long backend errors (e.g. yt-dlp tracebacks) for toast display."""
    raw = (message or "").strip()
    if not raw:
        return ""
    for marker in ("\nTraceback", " Traceback (most recent call last)"):
        if marker in raw:
            raw = raw.split(marker, 1)[0].strip()
    text = " ".join(part.strip() for part in raw.splitlines() if part.strip())
    for prefix in ("下载失败：", "下载失败:", "ERROR:", "转写失败:", "转写失败："):
        if prefix in text:
            segment = text.split(prefix, 1)[1].strip()
            if segment:
                text = f"{prefix.rstrip(':：')}: {segment}"
                break
    if " is not a valid URL" in text:
        return "无效链接，请输入正确的短视频 URL"
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def show_toast(
    message: str,
    *,
    kind: str = "auto",
    duration: float | None = None,
    raise_error: bool = True,
) -> None:
    """Show an auto-dismiss toast on the page."""
    if kind == "progress":
        return
    text = _shorten_toast_message(message)
    if not text:
        return
    if kind == "auto":
        kind = "error" if _is_error_notice(text) else "info"
    if duration is None:
        duration = TOAST_ERROR_DURATION if kind == "error" else TOAST_DURATION
    if kind == "error":
        if raise_error:
            raise gr.Error(text, duration=duration, print_exception=False)
        gr.Warning(f"错误：{text}", duration=duration)
        return
    if kind == "success":
        gr.Success(text, duration=duration)
    elif kind in ("info", "warning"):
        gr.Warning(text, duration=duration)
    else:
        gr.Info(text, duration=duration)


TECH_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Noto+Sans+SC:wght@400;500;600&display=swap');

:root {
    --app-header-h: 3rem;
    --app-main-pad: 0.5rem;
    --app-col-gap: 0.65rem;
}

html {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    height: 100% !important;
    overflow: hidden !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

html::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
    background: transparent !important;
}

body {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    height: 100% !important;
    overflow: hidden !important;
    position: fixed !important;
    inset: 0 !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

body::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
    background: transparent !important;
}

#root,
#root > div,
.gradio-container,
.gradio-container > .main,
.gradio-container .contain,
.gradio-container > .wrap,
.app {
    max-width: 100% !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

#root,
#root > div {
    height: 100% !important;
    max-height: 100% !important;
    overflow: hidden !important;
}

.gradio-container {
    font-family: 'Noto Sans SC', 'Microsoft YaHei', sans-serif !important;
    background:
        radial-gradient(ellipse 90% 55% at 50% -15%, rgba(0, 180, 255, 0.14), transparent),
        radial-gradient(ellipse 50% 45% at 100% 80%, rgba(124, 77, 255, 0.1), transparent),
        linear-gradient(180deg, #0a1220 0%, #050810 100%) !important;
    color: #d8e8f8 !important;
    max-width: 100% !important;
    width: 100% !important;
    height: 100vh !important;
    height: 100dvh !important;
    min-height: 0 !important;
    max-height: 100vh !important;
    max-height: 100dvh !important;
    margin: 0 !important;
    padding: 0 !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.gradio-container::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
    background: transparent !important;
}

.gradio-container > .main,
.gradio-container > .main.fillable,
.gradio-container .wrap.fillable {
    padding: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
    max-height: 100% !important;
    height: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}

.gradio-container footer,
.gradio-container .footer {
    display: none !important;
}

.fullscreen-app {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    max-height: 100% !important;
    height: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 0 !important;
    gap: 0 !important;
}

.fullscreen-app > .gap {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
    padding: 0 !important;
}

/* 顶栏所在 block：不参与伸缩、禁止出现滚动条 */
.fullscreen-app > .gap > .block:has(.app-header),
.fullscreen-app > .gap > div:first-child,
.block.app-header-block,
.app-header-block {
    flex: 0 0 auto !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
    max-height: var(--app-header-h) !important;
    overflow: hidden !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.fullscreen-app > .gap > .block:has(.app-header)::-webkit-scrollbar,
.block.app-header-block::-webkit-scrollbar,
.app-header-block::-webkit-scrollbar,
.fullscreen-app > .gap > .block:has(.app-header) *::-webkit-scrollbar,
.block.app-header-block *::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
    background: transparent !important;
}

.fullscreen-app > .gap > .block:has(.app-header) .html-container,
.fullscreen-app > .gap > .block:has(.app-header) .prose,
.block.app-header-block .html-container,
.block.app-header-block .prose,
.block.app-header-block > .wrap {
    margin: 0 !important;
    padding: 0 !important;
    max-height: var(--app-header-h) !important;
    overflow: hidden !important;
}

.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    min-height: var(--app-header-h);
    padding: 0.45rem 1.35rem;
    box-sizing: border-box;
    border-bottom: 1px solid rgba(0, 200, 255, 0.18);
    background: rgba(6, 12, 22, 0.6);
}

.hero-title {
    font-family: 'Orbitron', 'Noto Sans SC', sans-serif !important;
    font-size: 1.28rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em;
    margin: 0 !important;
    line-height: 1.25 !important;
    background: linear-gradient(92deg, #00e5ff 0%, #a78bfa 50%, #00e5ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: #6b8aa8 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em;
    margin: 0 !important;
    line-height: 1.25 !important;
}

.block.app-main-row,
.app-main-row.block {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    max-height: 100% !important;
    overflow: hidden !important;
    margin: 0 !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

.app-main-row {
    width: 100% !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
    max-height: 100% !important;
    height: 100% !important;
    padding: var(--app-main-pad) var(--app-main-pad) var(--app-main-pad) !important;
    box-sizing: border-box !important;
    gap: var(--app-col-gap) !important;
    align-items: stretch !important;
    overflow: hidden !important;
}

.app-main-row > .column {
    min-height: 0 !important;
    max-height: 100% !important;
    height: 100% !important;
    overflow: hidden !important;
}

.app-main-row > .form {
    min-height: 0 !important;
    max-height: 100% !important;
    height: 100% !important;
    overflow: hidden !important;
}

.panel-left {
    border: 1px solid rgba(0, 200, 255, 0.18) !important;
    border-radius: 12px !important;
    background: rgba(8, 16, 28, 0.82) !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 28px rgba(0, 0, 0, 0.4) !important;
    padding: 0.55rem 0.85rem !important;
    box-sizing: border-box !important;
    align-self: stretch !important;
    gap: 0.3rem !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    overflow: hidden !important;
}

.panel-left > .form,
.panel-left > .column,
.panel-left .block,
.panel-left .tabs,
.panel-left .tabitem {
    overflow: hidden !important;
    max-height: 100% !important;
}

.panel-right {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    background: #1a1f2e !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35) !important;
    padding: 0 !important;
    height: 100% !important;
    min-height: 0 !important;
    max-height: 100% !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
    position: relative !important;
    top: auto !important;
    overflow: hidden !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.panel-right::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
}

.panel-right > .form,
.panel-right > .column {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: 100% !important;
    max-height: 100% !important;
    overflow: hidden !important;
    gap: 0 !important;
    padding: 0 !important;
}

.chat-column {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
    max-height: 100% !important;
    height: 100% !important;
    overflow: hidden !important;
    gap: 0 !important;
}

.chat-column > .form {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
    max-height: 100% !important;
    height: 100% !important;
    overflow: hidden !important;
    gap: 0 !important;
}

.chat-column .chat-compose {
    flex: 0 0 auto !important;
    flex-shrink: 0 !important;
}

#transcript-chat {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    max-height: none !important;
    height: 0 !important;
    overflow: hidden !important;
    --chatbot-text-size: 0.875rem;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

#transcript-chat::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
    background: transparent !important;
}

#transcript-chat.block,
.chat-column .block:has(#transcript-chat) {
    margin-bottom: 0 !important;
    flex: 1 1 0 !important;
    min-height: 0 !important;
    height: 0 !important;
    max-height: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}

/* 避免 Chatbot 外层 block 再套一层滚动条 */
.chat-column .block:has(#transcript-chat) > .wrap,
.chat-column .block:has(#transcript-chat) > div {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    max-height: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}

#transcript-chat > .wrap,
#transcript-chat .wrap {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    max-height: 100% !important;
    height: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}

/* 仅对话消息列表区域可纵向滚动 */
#transcript-chat .bubble-wrap,
#transcript-chat .panel-wrap {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    max-height: 100% !important;
    height: 0 !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    overscroll-behavior: contain !important;
    -webkit-overflow-scrolling: touch !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

/* 右侧对话区滚动条：透明隐藏，保留鼠标滚轮/触控滑动 */
#transcript-chat .bubble-wrap::-webkit-scrollbar,
#transcript-chat .panel-wrap::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
    background: transparent !important;
}

#transcript-chat .bubble-wrap::-webkit-scrollbar-thumb,
#transcript-chat .panel-wrap::-webkit-scrollbar-thumb,
#transcript-chat .bubble-wrap::-webkit-scrollbar-track,
#transcript-chat .panel-wrap::-webkit-scrollbar-track {
    background: transparent !important;
    border: none !important;
}

#transcript-chat .bubble-wrap::-webkit-scrollbar-button,
#transcript-chat .panel-wrap::-webkit-scrollbar-button {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

/* 对话区排版：略缩小字号，收紧字距与行距 */
#transcript-chat .bubble-wrap {
    padding-top: 0.75rem !important;
}

#transcript-chat .message-wrap,
#transcript-chat .flex-wrap,
#transcript-chat .message-content,
#transcript-chat .message-markdown-disabled {
    font-size: 0.875rem !important;
    line-height: 1.72 !important;
    letter-spacing: 0.02em !important;
    word-spacing: 0.02em !important;
}

#transcript-chat .message :global(.prose),
#transcript-chat .message-wrap :global(.prose.chatbot.md),
#transcript-chat .message-wrap :global(.md),
#transcript-chat .message-wrap :global(.prose) {
    font-size: 0.875rem !important;
    line-height: 1.72 !important;
    letter-spacing: 0.02em !important;
}

#transcript-chat .message-wrap :global(.prose p),
#transcript-chat .message-wrap :global(.md p),
#transcript-chat .message-wrap :global(p) {
    font-size: inherit !important;
    line-height: inherit !important;
    letter-spacing: inherit !important;
    margin: 0 0 0.55em 0 !important;
}

#transcript-chat .message-wrap > div :global(p:not(:first-child)) {
    margin-top: 0.55em !important;
}

#transcript-chat .message-wrap :global(.prose p:last-child),
#transcript-chat .message-wrap :global(.md p:last-child) {
    margin-bottom: 0 !important;
}

#transcript-chat .bubble {
    margin: 0.65rem 1rem !important;
}

#transcript-chat .message {
    margin-top: 0.35rem !important;
}

#transcript-chat .user,
#transcript-chat .bot {
    padding: 0.5rem 0.85rem !important;
    font-size: 0.875rem !important;
    line-height: 1.72 !important;
    letter-spacing: 0.02em !important;
}

#transcript-chat .placeholder {
    font-size: 0.8125rem !important;
    line-height: 1.65 !important;
    letter-spacing: 0.03em !important;
    color: #6d8098 !important;
    opacity: 0.9 !important;
}

/* 大模型回复 Markdown 展示（prose） */
#transcript-chat .message-wrap :global(.md),
#transcript-chat .message-wrap :global(.prose) {
    color: #dce6f2 !important;
    max-width: 100% !important;
}

#transcript-chat .message-wrap :global(.md h1),
#transcript-chat .message-wrap :global(.md h2),
#transcript-chat .message-wrap :global(.md h3),
#transcript-chat .message-wrap :global(.prose h1),
#transcript-chat .message-wrap :global(.prose h2),
#transcript-chat .message-wrap :global(.prose h3) {
    color: #e8f2fa !important;
    margin: 1em 0 0.45em !important;
    font-weight: 600 !important;
}

#transcript-chat .message-wrap :global(.md strong),
#transcript-chat .message-wrap :global(.prose strong) {
    color: #f0f6fc !important;
    font-weight: 600 !important;
}

#transcript-chat .message-wrap :global(.md a),
#transcript-chat .message-wrap :global(.prose a) {
    color: #6ec8ff !important;
    text-decoration: underline !important;
}

#transcript-chat .message-wrap :global(.md code),
#transcript-chat .message-wrap :global(.prose code) {
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace !important;
    font-size: 0.84em !important;
    background: rgba(8, 14, 24, 0.65) !important;
    padding: 0.12em 0.35em !important;
    border-radius: 4px !important;
    color: #b8e8ff !important;
}

#transcript-chat .message-wrap :global(.md pre),
#transcript-chat .message-wrap :global(.prose pre) {
    background: rgba(6, 10, 18, 0.92) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    padding: 0.75rem 0.9rem !important;
    overflow-x: auto !important;
    margin: 0.5em 0 !important;
}

#transcript-chat .message-wrap :global(.md pre code),
#transcript-chat .message-wrap :global(.prose pre code) {
    background: transparent !important;
    padding: 0 !important;
    color: #d7e6f5 !important;
}

#transcript-chat .message-wrap :global(.md ul),
#transcript-chat .message-wrap :global(.md ol),
#transcript-chat .message-wrap :global(.prose ul),
#transcript-chat .message-wrap :global(.prose ol) {
    margin: 0.35em 0 0.55em !important;
    padding-left: 1.35em !important;
    list-style-type: disc !important;
}

#transcript-chat .message-wrap :global(.md ul li),
#transcript-chat .message-wrap :global(.prose ul li),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.md ul li),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.prose ul li) {
    list-style-type: disc !important;
}

#transcript-chat .message-wrap :global(.md blockquote),
#transcript-chat .message-wrap :global(.prose blockquote) {
    border-left: 3px solid rgba(110, 200, 255, 0.55) !important;
    margin: 0.5em 0 !important;
    padding: 0.2em 0 0.2em 0.85em !important;
    color: #b8c8dc !important;
}

/* 横线（---）不显示线条，仅保留段间距（勿 display:none，避免破坏块级结构） */
#transcript-chat .message-wrap :global(.md hr),
#transcript-chat .message-wrap :global(.prose hr),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.md hr),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.prose hr) {
    display: block !important;
    border: none !important;
    border-top: none !important;
    height: 0 !important;
    margin: 0.7em 0 !important;
    padding: 0 !important;
    background: transparent !important;
    opacity: 0 !important;
    visibility: hidden !important;
    overflow: hidden !important;
}

#transcript-chat .message-wrap :global(.md p + h2),
#transcript-chat .message-wrap :global(.md p + h3),
#transcript-chat .message-wrap :global(.prose p + h2),
#transcript-chat .message-wrap :global(.prose p + h3),
#transcript-chat .message-wrap :global(.md ul + h2),
#transcript-chat .message-wrap :global(.md ul + h3),
#transcript-chat .message-wrap :global(.md ol + h2),
#transcript-chat .message-wrap :global(.md ol + h3),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.md p + h3),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.prose p + h3) {
    margin-top: 1em !important;
}

#transcript-chat .message-wrap :global(.md p + hr),
#transcript-chat .message-wrap :global(.prose p + hr),
#transcript-chat .message-wrap :global(.md hr + h1),
#transcript-chat .message-wrap :global(.md hr + h2),
#transcript-chat .message-wrap :global(.md hr + h3),
#transcript-chat .message-wrap :global(.prose hr + h1),
#transcript-chat .message-wrap :global(.prose hr + h2),
#transcript-chat .message-wrap :global(.prose hr + h3),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.md p + hr),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.prose p + hr),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.md hr + h1),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.md hr + h2),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.md hr + h3),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.prose hr + h1),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.prose hr + h2),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.prose hr + h3) {
    margin-top: 0.55em !important;
    margin-bottom: 0.55em !important;
}

/* 首条 ASR 转写：纯文本换行，不当作 Markdown 标题/列表 */
#transcript-chat .bubble-wrap > .bubble.bot-row:first-of-type .message-content :global(.md),
#transcript-chat .bubble-wrap > .bubble.bot-row:first-of-type .message-content :global(.prose) {
    white-space: pre-wrap !important;
}

/* 大模型回复：正常 Markdown 块级排版（标题、列表、段落） */
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.md),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.prose) {
    white-space: normal !important;
}

#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.llm-md-output) {
    color: #dce6f2 !important;
    font-size: 0.875rem !important;
    line-height: 1.72 !important;
}

#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.llm-md-output h1),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.llm-md-output h2),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.llm-md-output h3),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.llm-md-output h4) {
    color: #e8f2fa !important;
    margin: 0.85em 0 0.45em !important;
    font-weight: 600 !important;
    line-height: 1.35 !important;
}

#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.llm-md-output p) {
    margin: 0 0 0.55em 0 !important;
}

#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.llm-md-output ul),
#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.llm-md-output ol) {
    margin: 0.35em 0 0.65em !important;
    padding-left: 1.35em !important;
}

#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.llm-md-output li) {
    margin: 0.2em 0 !important;
}

#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.llm-md-output strong) {
    color: #f0f6fc !important;
    font-weight: 600 !important;
}

#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.llm-md-output code) {
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace !important;
    font-size: 0.84em !important;
    background: rgba(8, 14, 24, 0.65) !important;
    padding: 0.12em 0.35em !important;
    border-radius: 4px !important;
    color: #b8e8ff !important;
}

#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.llm-md-output pre) {
    background: rgba(6, 10, 18, 0.92) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    padding: 0.75rem 0.9rem !important;
    overflow-x: auto !important;
    margin: 0.5em 0 !important;
}

#transcript-chat .bubble-wrap > .bubble.bot-row:not(:first-of-type) .message-content :global(.llm-md-output blockquote) {
    border-left: 3px solid rgba(110, 200, 255, 0.55) !important;
    margin: 0.5em 0 !important;
    padding: 0.2em 0 0.2em 0.85em !important;
    color: #b8c8dc !important;
}

/* 流式生成占位与末尾光标 */
#transcript-chat .bubble-wrap > .bubble.bot-row:last-of-type .message-content :global(.md em),
#transcript-chat .bubble-wrap > .bubble.bot-row:last-of-type .message-content :global(.prose em) {
    color: #9eb4cc !important;
    font-style: normal !important;
    animation: llm-think-pulse 1.4s ease-in-out infinite;
}

@keyframes llm-think-pulse {
    0%, 100% { opacity: 0.45; }
    50% { opacity: 1; }
}

.chat-compose-inner:has(textarea:disabled) {
    opacity: 0.72 !important;
}

.chat-compose-inner:has(textarea:disabled) button.chat-send-circle {
    opacity: 0.45 !important;
    pointer-events: none !important;
}

/* 大模型 HTML 回复（gr.HTML，常规对话式排版） */
#transcript-chat .message.html .llm-reply-html,
#transcript-chat .message.html :global(.llm-reply-html) {
    color: #dce6f2 !important;
    font-size: 0.875rem !important;
    line-height: 1.72 !important;
    letter-spacing: 0.02em !important;
    word-break: break-word !important;
}

#transcript-chat .message.html :global(.llm-reply-html h1),
#transcript-chat .message.html :global(.llm-reply-html h2),
#transcript-chat .message.html :global(.llm-reply-html h3),
#transcript-chat .message.html :global(.llm-reply-html h4) {
    color: #e8f2fa !important;
    font-weight: 600 !important;
    margin: 0.9em 0 0.45em !important;
    line-height: 1.35 !important;
}

#transcript-chat .message.html :global(.llm-reply-html h1:first-child),
#transcript-chat .message.html :global(.llm-reply-html h2:first-child),
#transcript-chat .message.html :global(.llm-reply-html h3:first-child) {
    margin-top: 0.15em !important;
}

#transcript-chat .message.html :global(.llm-reply-html p) {
    margin: 0 0 0.55em 0 !important;
}

#transcript-chat .message.html :global(.llm-reply-html ul),
#transcript-chat .message.html :global(.llm-reply-html ol) {
    display: block !important;
    list-style-position: outside !important;
    margin: 0.25em 0 0.65em !important;
    padding-left: 1.35em !important;
}

#transcript-chat .message.html :global(.llm-reply-html li) {
    display: list-item !important;
    margin: 0.22em 0 !important;
}

#transcript-chat .message.html :global(.llm-reply-html strong) {
    color: #f0f6fc !important;
    font-weight: 600 !important;
}

#transcript-chat .message.html :global(.llm-reply-html code) {
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace !important;
    font-size: 0.84em !important;
    background: rgba(8, 14, 24, 0.65) !important;
    padding: 0.12em 0.35em !important;
    border-radius: 4px !important;
    color: #b8e8ff !important;
}

#transcript-chat .message.html :global(.llm-reply-html pre) {
    background: rgba(6, 10, 18, 0.92) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    padding: 0.75rem 0.9rem !important;
    overflow-x: auto !important;
    margin: 0.5em 0 !important;
}

.chat-compose {
    flex: 0 0 auto !important;
    flex-shrink: 0 !important;
    padding: 0.45rem 0.85rem 0.55rem !important;
    border-top: 1px solid rgba(255, 255, 255, 0.07) !important;
    background: linear-gradient(180deg, rgba(26, 31, 46, 0.2), rgba(26, 31, 46, 0.95)) !important;
}

.chat-compose-inner {
    display: flex !important;
    align-items: center !important;
    gap: 0.45rem !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0.22rem 0.35rem 0.22rem 0.75rem !important;
    border-radius: 22px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background: rgba(12, 18, 30, 0.92) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
    min-height: 0 !important;
}

.chat-compose-inner .block,
.chat-compose-inner .form {
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    min-height: 0 !important;
}

.chat-compose-input,
.chat-compose-input .wrap,
.chat-compose-input textarea {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    outline: none !important;
    min-height: 0 !important;
}

.chat-compose-input textarea {
    font-size: 0.875rem !important;
    line-height: 1.35 !important;
    letter-spacing: 0.02em !important;
    min-height: 1.35rem !important;
    max-height: 7rem !important;
    padding: 0.1rem 0 !important;
    color: #dce6f2 !important;
    resize: none !important;
    overflow-y: auto !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.chat-compose-input textarea::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

.chat-compose-input textarea::-webkit-scrollbar-button {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

.chat-compose-input .wrap,
.chat-compose-input .input-container {
    overflow: hidden !important;
    min-height: 0 !important;
    padding: 0 !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.chat-compose-input .wrap::-webkit-scrollbar,
.chat-compose-input .input-container::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

.chat-compose-input textarea::placeholder {
    color: #6d8098 !important;
    font-size: 0.8125rem !important;
    letter-spacing: 0.02em !important;
}

.chat-compose-inner .block:has(> button.chat-send-circle) {
    flex: 0 0 auto !important;
    width: 2rem !important;
    min-width: 2rem !important;
    max-width: 2rem !important;
    margin: 0 !important;
    padding: 0 !important;
}

.chat-compose-inner .block:has(> button.chat-send-circle) .form,
.chat-compose-inner .block:has(> button.chat-send-circle) .wrap {
    width: 2rem !important;
    min-width: 2rem !important;
    max-width: 2rem !important;
    margin: 0 !important;
    padding: 0 !important;
}

button.chat-send-circle {
    --button-large-radius: 50%;
    --button-medium-radius: 50%;
    --button-small-radius: 50%;
    width: 2rem !important;
    height: 2rem !important;
    min-width: 2rem !important;
    min-height: 2rem !important;
    max-width: 2rem !important;
    max-height: 2rem !important;
    aspect-ratio: 1 / 1 !important;
    border-radius: 50% !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    background: linear-gradient(135deg, #0077ee, #00b8e6) !important;
    color: #ffffff !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    box-shadow: 0 4px 16px rgba(0, 150, 255, 0.35) !important;
}

button.chat-send-circle:hover {
    background: linear-gradient(135deg, #0088ff, #00c8f0) !important;
    color: #ffffff !important;
}

.panel-left .block,
.panel-right .block {
    margin-bottom: 0.15rem !important;
    padding-top: 0.1rem !important;
    padding-bottom: 0.1rem !important;
}

.panel-left .form,
.panel-right .form {
    gap: 0.35rem !important;
}

.panel-left .block-label,
.panel-right .block-label,
.panel-left label span,
.panel-right label span {
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

.section-label {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #7aa8cc !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0.1rem 0 0.25rem !important;
    line-height: 1.3 !important;
}

.section-label-config {
    margin-top: 0.18rem !important;
    margin-bottom: 0.125rem !important;
}

.section-label-media {
    margin: 0.1rem 0 0.125rem !important;
}

.output-area {
    flex: 1 1 auto !important;
    min-height: 14rem !important;
    overflow: visible !important;
}

.btn-row {
    gap: 0.5rem !important;
    margin-top: 0.35rem !important;
    padding-top: 0.25rem !important;
}

.primary-btn button {
    background: linear-gradient(135deg, #0077ee, #00b8e6) !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.65rem 1rem !important;
    letter-spacing: 0.06em;
    box-shadow: 0 4px 20px rgba(0, 150, 255, 0.4) !important;
}

.secondary-btn button {
    background: rgba(30, 45, 70, 0.85) !important;
    border: 1px solid rgba(0, 200, 255, 0.28) !important;
    color: #9ec5e8 !important;
    padding: 0.65rem 1rem !important;
}

.media-tabs {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.media-tabs > .tab-nav {
    margin-bottom: 0.125rem !important;
}

.media-tabs > .tab-nav button {
    font-size: 0.85rem !important;
    letter-spacing: 0.06em;
    padding: 0.35rem 0.75rem !important;
}

.media-tabs .tabitem {
    padding-top: 0.15rem !important;
    min-height: unset !important;
}

.douyin-hint {
    font-size: 0.72rem !important;
    color: #6a8aa8 !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.4 !important;
}

.douyin-hint-block.block {
    margin: 0 !important;
    padding: 0 !important;
}

.douyin-login-accordion.block {
    margin-bottom: 0.28rem !important;
    padding: 0.75rem 0.55rem 0.85rem !important;
    box-sizing: border-box !important;
}

.douyin-login-accordion > .block,
.douyin-login-accordion details[open] > .form,
.douyin-login-accordion .form {
    gap: 0.55rem !important;
    padding: 0.15rem 0.25rem 0.1rem !important;
}

.douyin-login-accordion .form > * + * {
    margin-top: 0.55rem !important;
}

.douyin-login-accordion .column,
.douyin-login-accordion .form {
    --layout-gap: 0.55rem !important;
    gap: 0.55rem !important;
    padding-bottom: 0.15rem !important;
}

.douyin-hint-block .html-container,
.douyin-hint-block .prose {
    margin: 0 !important;
    padding: 0 !important;
}

.douyin-login-accordion .block {
    margin-bottom: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

.douyin-login-accordion .label-wrap {
    margin-bottom: 0.28rem !important;
    padding-top: 0.1rem !important;
}

.douyin-login-accordion .label-wrap.open {
    margin-bottom: 0.45rem !important;
}

.douyin-status-line.block {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}

.douyin-status-line .wrap,
.douyin-status-line .input-container {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
}

.douyin-status-line textarea {
    min-height: 1.65rem !important;
    max-height: 1.65rem !important;
    height: 1.65rem !important;
    padding: 0.22rem 0.45rem !important;
    line-height: 1.35 !important;
    overflow: hidden !important;
    white-space: nowrap !important;
    text-overflow: ellipsis !important;
    border: none !important;
    box-shadow: none !important;
    background: rgba(8, 16, 28, 0.5) !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    color: #9ec5e8 !important;
}

.douyin-login-accordion .douyin-btn-row {
    margin: 0 0 0.1rem !important;
    padding: 0 !important;
    gap: 0.45rem !important;
}

.douyin-login-accordion .douyin-btn-row button {
    padding: 0.42rem 0.45rem !important;
    font-size: 0.78rem !important;
    line-height: 1.2 !important;
}

.url-link-module.column {
    --layout-gap: 0.5rem !important;
    gap: 0.5rem !important;
    margin-top: 0.25rem !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}

.url-link-module .form {
    gap: 0.5rem !important;
}

.url-link-module .url-link-input.block {
    margin-bottom: 0.5rem !important;
    padding: 0.3rem 0.45rem 0.35rem !important;
    overflow: hidden !important;
}

.url-link-input.block.padded {
    padding: 0.3rem 0.45rem 0.35rem !important;
}

.url-link-input label,
.url-link-input .input-container {
    margin-bottom: 0 !important;
}

.url-link-input label.container.show_textbox_border textarea,
.url-link-input textarea {
    min-height: 2.75rem !important;
    max-height: 2.75rem !important;
    height: 2.75rem !important;
    line-height: 1.45 !important;
    margin-bottom: 0 !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding-left: 0.1rem !important;
    padding-right: 0.1rem !important;
}

.url-link-input label.container.show_textbox_border textarea:focus {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

.url-link-module .url-fetch-row {
    margin-bottom: 0.5rem !important;
}

.url-link-module .block {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding-top: 0.15rem !important;
    padding-bottom: 0.15rem !important;
}

.url-link-module .label-wrap {
    margin-bottom: 0.3rem !important;
}

.url-fetch-row {
    margin: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
}

.url-fetch-row button {
    padding: 0.52rem 0.75rem !important;
}

.url-link-module .url-video-preview .label-wrap {
    margin-bottom: 0.3rem !important;
}

.config-radio-row {
    flex-wrap: nowrap !important;
    gap: 0.5rem !important;
}

.config-radio-row > .block,
.config-radio-row > .form {
    min-width: 0 !important;
    flex: 1 1 0 !important;
}

.inline-radio .wrap {
    flex-wrap: nowrap !important;
    gap: 0.35rem !important;
}

.inline-radio label {
    white-space: nowrap !important;
    font-size: 0.82rem !important;
    padding: 0.2rem 0.45rem !important;
}

.panel-left .accordion,
.panel-left details {
    margin-bottom: 0.2rem !important;
}

.panel-left .row,
.panel-right .row {
    gap: 0.5rem !important;
}

.url-video-preview {
    height: 180px !important;
    min-height: 180px !important;
    max-height: 180px !important;
    overflow: hidden !important;
}

.url-video-preview > .wrap,
.url-video-preview .video-container,
.url-video-preview .container,
.url-video-preview .video-wrapper {
    height: 180px !important;
    min-height: 180px !important;
    max-height: 180px !important;
}

.url-video-preview video,
.url-video-preview .video-container video {
    width: 100% !important;
    height: 180px !important;
    max-height: 180px !important;
    object-fit: contain !important;
    background: #060c18 !important;
}

/* 禁止 Gradio 外层产生整页滚动条 */
.gradio-container .tab-nav,
.gradio-container .tabs,
.gradio-container [class*="tabitem"] {
    max-height: 100% !important;
}

footer, .footer, [data-testid="footer"] { display: none !important; }

/* Gradio 外层 wrap/contain：避免整页滚动条出现在顶栏右侧 */
.gradio-container > .main > .wrap,
.gradio-container > .main > .contain,
.gradio-container .wrap.svelte-czcr5b,
.gradio-container .contain.svelte-czcr5b {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    max-height: 100% !important;
    height: 100% !important;
    overflow: hidden !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.gradio-container > .main > .wrap::-webkit-scrollbar,
.gradio-container > .main > .contain::-webkit-scrollbar,
.gradio-container .wrap::-webkit-scrollbar,
.gradio-container .contain::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
    background: transparent !important;
}

/* 全局隐藏滚动条外观（对话区 bubble-wrap 仍可用滚轮滚动） */
* {
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

*::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
    background: transparent !important;
}

*::-webkit-scrollbar-thumb,
*::-webkit-scrollbar-track,
*::-webkit-scrollbar-button {
    display: none !important;
    background: transparent !important;
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Noto Sans SC"), "Microsoft YaHei", "sans-serif"],
).set(
    body_background_fill="transparent",
    block_background_fill="rgba(10, 18, 32, 0.85)",
    block_border_color="rgba(0, 200, 255, 0.25)",
    block_label_text_color="#9ec5e8",
    input_background_fill="rgba(6, 12, 24, 0.9)",
    button_primary_background_fill="linear-gradient(135deg, #0066ff, #00b4d8)",
    button_primary_text_color="#ffffff",
)


def save_upload(uploaded: str | None) -> str | None:
    """Copy uploaded file into output/uploads/ and return stable path."""
    if not uploaded:
        return None
    src = Path(uploaded)
    if not src.is_file():
        return uploaded
    OUTPUT_UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT_UPLOADS / src.name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return str(dest)


def _resolve_media_path(
    file_obj: gr.File | str | None,
    video_url: str,
    url_cached_path: str | None,
    progress: gr.Progress,
) -> tuple[str | None, str]:
    """解析媒体来源：本地上传、已缓存的 URL 下载、或即时下载。"""
    if file_obj is not None:
        if isinstance(file_obj, str):
            path = save_upload(file_obj)
        elif hasattr(file_obj, "name"):
            path = save_upload(getattr(file_obj, "name", None))
        else:
            path = save_upload(str(file_obj))
        if path:
            return path, ""

    if url_cached_path and Path(url_cached_path).is_file():
        return str(Path(url_cached_path).resolve()), ""

    url = (video_url or "").strip()
    if url:
        from ytdlp_bridge import download_from_url

        progress(0.05, desc="正在从链接下载视频…")
        msg, path = download_from_url(url, OUTPUT_URL_DOWNLOADS, progress=progress)
        if path:
            return path, ""
        return None, msg or "链接下载失败，未获得媒体文件。"

    return None, "请在本地上传媒体文件，或输入链接后点击「获取视频」。"


def fetch_video_url(
    video_url: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str | None, str | None]:
    """从 URL 下载短视频，返回预览路径与缓存路径。"""
    from ytdlp_bridge import download_from_url

    url = (video_url or "").strip()
    if not url:
        show_toast("请输入短视频链接。", kind="info")
        return None, None

    progress(0, desc="准备下载…")
    status, path = download_from_url(url, OUTPUT_URL_DOWNLOADS, progress=progress)
    progress(1.0, desc="完成")
    if path:
        resolved = str(Path(path).resolve())
        show_toast(f"视频下载成功：{Path(path).name}", kind="success")
        return resolved, resolved
    if "douyin" in url.lower() and "登录" not in status:
        cookie_hint = get_douyin_cookies_dir() / "douyin.txt"
        status = (
            f"{status}\n\n"
            "── 抖音下载提示 ──\n"
            "请先展开上方「抖音登录」：打开登录 → 保存登录（只需一次）。\n"
            f"Cookie 保存路径：{cookie_hint}"
        )
    raise gr.Error(
        _shorten_toast_message(status),
        duration=TOAST_ERROR_DURATION,
        print_exception=False,
    )


def _load_transcript_from_status(status: str) -> str:
    """若内存结果为空，尝试从状态栏里的文件路径读取转写文本。"""
    marker = "结果:"
    if marker not in status:
        return ""
    try:
        path = Path(status.split(marker, 1)[1].strip())
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logging.warning("从文件回读转写结果失败: %s", exc)
    return ""


def _transcript_to_chat_history(text: str) -> list[dict[str, str]]:
    """将转写文本转为聊天首条 assistant 消息。"""
    content = (text or "").strip()
    if not content:
        return []
    return [{"role": "assistant", "content": content}]


def _resolve_transcript_text(transcript_text: str, status: str) -> str:
    """合并内存文本与 status 中 transcript 文件路径回读。"""
    resolved = (transcript_text or "").strip()
    if resolved:
        return resolved
    return _load_transcript_from_status(status)


def _html_component_to_markdown(raw_html: str) -> str:
    """旧版 gr.HTML 回复转为纯文本，供 Markdown 渲染。"""
    text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.I)
    text = re.sub(r"</(p|li|h[1-6]|div|tr)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_module.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _coerce_chat_content(content: Any) -> Any:
    """统一消息内容；保留 gr.HTML 组件供对话展示。"""
    if isinstance(content, GradioComponent):
        return content
    if isinstance(content, dict) and content.get("component") == "html":
        return gr.HTML(
            value=str(content.get("value", "")),
            padding=False,
            container=False,
            show_label=False,
            elem_classes=["llm-reply-html"],
        )
    return content


def _normalize_chat_messages(
    history: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """统一 Chatbot 消息为 {role, content} 格式（保留换行；内容为 Markdown 字符串）。"""
    normalized: list[dict[str, Any]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = _coerce_chat_content(item.get("content"))
        if role not in ("user", "assistant") or content is None:
            continue
        if isinstance(content, GradioComponent):
            normalized.append({"role": role, "content": content})
            continue
        text = str(content)
        if text.strip():
            normalized.append({"role": role, "content": text})
    return normalized


def _normalize_markdown_for_render(text: str) -> str:
    """Markdown 展示层规范化：仅补空格/换行以便渲染，不改写词句（类似 ChatGPT）。"""
    prepared = (text or "").replace("\r\n", "\n")
    if not prepared.strip():
        return ""

    def _fix_atx_heading_spaces(value: str) -> str:
        return re.sub(
            r"^(#{1,6})([^\s#\n].*)$",
            lambda match: f"{match.group(1)} {match.group(2).strip()}",
            value,
            flags=re.MULTILINE,
        )

    prepared = _fix_atx_heading_spaces(prepared)
    prepared = re.sub(r"-{3,}\s*(?=#{1,6})", r"\n\n---\n\n", prepared)
    for _ in range(8):
        prev = prepared
        prepared = re.sub(
            r"([^\n#\s])(\s*)(#{1,6})(?=[^\s#\n])",
            r"\1\n\n\3",
            prepared,
        )
        prepared = re.sub(
            r"(#{1,6}\s[^\n#]+?)(#{1,6})(?=[^\s#\n])",
            r"\1\n\n\2",
            prepared,
        )
        prepared = _fix_atx_heading_spaces(prepared)
        if prepared == prev:
            break
    prepared = re.sub(
        r"^(#{1,6}\s+[^\n-]+?)(-\s*)",
        r"\1\n\2",
        prepared,
        flags=re.MULTILINE,
    )
    prepared = re.sub(
        r"([。；;!?！？:：\)])(-\s*)",
        r"\1\n\2",
        prepared,
    )
    prepared = re.sub(
        r"([\u4e00-\u9fff\)）])(-\s*)",
        r"\1\n\2",
        prepared,
    )
    prepared = re.sub(r"(?m)^-(?=[\u4e00-\u9fffA-Za-z0-9])", "- ", prepared)

    lines: list[str] = []
    for line in prepared.split("\n"):
        stripped = line.strip()
        if stripped and re.match(r"^#{1,6}\s", stripped):
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(line.rstrip())
            lines.append("")
            continue
        if stripped == "---" and lines and lines[-1].strip():
            lines.append("")
        lines.append(line.rstrip())

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _prepare_llm_reply_for_display(text: str) -> str:
    """展示用大模型回复：规范化 Markdown 以便 Chatbot 渲染标题/列表/换行。"""
    return _normalize_markdown_for_render(text)


_STREAMING_PLACEHOLDER_MD = "*正在思考…*"
_STREAM_RENDER_INTERVAL_SEC = 0.08
_STREAM_MIN_CHARS_BUMP = 16
_STREAM_CURSOR = "▌"


def _close_incomplete_markdown_fences(text: str) -> str:
    """流式展示时临时闭合未写完的代码围栏，减轻半截 Markdown 抖动。"""
    if text.count("```") % 2 == 1:
        return f"{text}\n```"
    return text


def _llm_assistant_streaming_placeholder() -> str:
    """首包到达前在 Chatbot 中显示的占位 Markdown。"""
    return _STREAMING_PLACEHOLDER_MD


def _llm_assistant_chat_content_streaming(
    text: str,
    *,
    plain_text: bool = False,
    show_cursor: bool = True,
) -> str:
    """流式进行中：轻量展示，不做完整 Markdown 规范化。"""
    raw = (text or "").replace("\r\n", "\n")
    if not raw.strip():
        return _STREAMING_PLACEHOLDER_MD
    body = raw if plain_text else _close_incomplete_markdown_fences(raw)
    if show_cursor:
        return f"{body.rstrip()}\n\n{_STREAM_CURSOR}"
    return body


def _llm_assistant_chat_content_final(text: str, *, plain_text: bool = False) -> str:
    """流式结束后：校正为纯文本；对话为完整 Markdown 规范化。"""
    if plain_text:
        return (text or "").replace("\r\n", "\n").strip()
    return _prepare_llm_reply_for_display(text)


def _should_refresh_stream_display(
    partial_len: int,
    last_rendered_len: int,
    last_render_at: float,
    *,
    force: bool = False,
) -> bool:
    """节流 UI 刷新，避免每个 token 都触发 Chatbot 全量重渲染。"""
    if force:
        return True
    if partial_len <= 0:
        return False
    if last_rendered_len <= 0:
        return True
    if partial_len - last_rendered_len >= _STREAM_MIN_CHARS_BUMP:
        return True
    return (time.monotonic() - last_render_at) >= _STREAM_RENDER_INTERVAL_SEC


def _append_or_update_assistant_message(
    chat_history: list[dict[str, Any]],
    content: str,
) -> None:
    """更新末尾 assistant 消息，不存在则追加。"""
    if chat_history and chat_history[-1].get("role") == "assistant":
        chat_history[-1] = {"role": "assistant", "content": content}
    else:
        chat_history.append({"role": "assistant", "content": content})


def _chat_compose_updates(*, locked: bool) -> tuple[Any, Any]:
    """生成中锁定输入框与发送按钮，防止并发请求。"""
    interactive = not locked
    return (
        gr.update(interactive=interactive),
        gr.update(interactive=interactive),
    )


def _render_llm_reply_html(text: str) -> str:
    """将 Markdown 转为 HTML（用于测试或服务端预览）。"""
    prepared = _prepare_llm_reply_for_display(text)
    if not prepared.strip():
        return ""
    try:
        import markdown as md_lib

        body = md_lib.markdown(
            prepared,
            extensions=["extra", "nl2br", "sane_lists"],
        )
    except ImportError:
        body = f"<pre>{html_module.escape(prepared)}</pre>"
    return f'<div class="llm-reply-html">{body}</div>'


def _llm_assistant_chat_content(text: str) -> str:
    """大模型 assistant 消息：返回 Markdown 字符串，由 Chatbot 原生渲染（同 GPT 聊天）。"""
    return _llm_assistant_chat_content_final(text, plain_text=False)


def run_transcribe_with_progress(
    file_obj: gr.File | str | None,
    video_url: str,
    url_cached_path: str | None,
    model_name: str,
    language: str,
    device_choice: str,
    progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple[Any, Any]]:
    """转写过程刷新按钮百分比；仅在最后一次 yield 写入 Chatbot，避免中间帧覆盖结果。"""
    ui_state = {"pct": 0, "status": "任务已启动…"}
    result_box: dict[str, Any] = {"done": False, "text": "", "status": ""}

    logging.info(
        "开始转写 | 模型: %s | 语言: %s | 设备: %s",
        model_name,
        language,
        device_choice,
    )

    def _worker() -> None:
        try:
            ui_state["pct"] = 5
            ui_state["status"] = "准备文件…"
            progress(0.05, desc=ui_state["status"])
            logging.info("转写进度: %s", ui_state["status"])
            path, prep_error = _resolve_media_path(
                file_obj, video_url, url_cached_path, progress
            )
            if not path:
                result_box["status"] = prep_error
                logging.warning("转写中止: %s", prep_error)
                return

            ui_state["pct"] = 12
            ui_state["status"] = "加载模型…"
            progress(0.12, desc=ui_state["status"])
            logging.info("转写进度: %s", ui_state["status"])
            mgr = get_manager()

            def _on_asr_progress(current: int, total: int) -> None:
                if total > 0:
                    pct = int(15 + 80 * current / total)
                    ui_state["pct"] = pct
                    ui_state["status"] = f"转写中… {current}/{total}"
                    progress(pct / 100, desc=ui_state["status"])

            text, status = mgr.transcribe(
                path,
                model_name,
                language,
                device_choice,
                progress_callback=_on_asr_progress,
            )
            progress(1.0, desc="完成")
            ui_state["pct"] = 100
            ui_state["status"] = "完成"
            result_box["text"] = text
            result_box["status"] = status
            logging.info("转写进度: 推理完成")
        except Exception as exc:
            logging.exception("转写任务异常")
            result_box["text"] = ""
            result_box["status"] = f"转写失败: {exc}"
            ui_state["status"] = result_box["status"]
        finally:
            result_box["done"] = True

    threading.Thread(target=_worker, daemon=True, name="asr-transcribe").start()

    yield gr.skip(), gr.update(value=_btn_run_label(0), interactive=False)

    last_pct = -1
    last_status = ""
    while not result_box.get("done"):
        pct = ui_state["pct"]
        status_msg = ui_state["status"]
        if pct != last_pct or status_msg != last_status:
            last_pct = pct
            last_status = status_msg
            logging.info("转写进度: %s (%d%%)", status_msg, pct)
            yield gr.skip(), gr.update(value=_btn_run_label(pct), interactive=False)
        time.sleep(0.4)

    resolved = _resolve_transcript_text(
        str(result_box.get("text") or ""),
        str(result_box.get("status") or ""),
    )
    status = str(result_box.get("status") or "")
    if _is_error_notice(status) or (not resolved and status):
        error_message = status or "转写失败，未获得识别文本。"
        logging.warning("转写失败或未获得文本: %s", error_message)
        yield gr.skip(), gr.update(value=BTN_RUN_IDLE, interactive=True)
        raise gr.Error(
            _shorten_toast_message(error_message),
            duration=TOAST_ERROR_DURATION,
            print_exception=False,
        )

    history = _transcript_to_chat_history(resolved)
    logging.info(
        "聊天区写入: %d 字, %d 条消息",
        len(resolved),
        len(history),
    )
    show_toast(f"转写完成，共 {len(resolved)} 字。", kind="success")
    yield history, gr.update(value=BTN_RUN_IDLE, interactive=True)


def handle_chat_submit(
    user_message: str,
    history: list[dict[str, Any]] | None,
) -> Iterator[tuple[list[dict[str, Any]], str, Any, Any]]:
    """处理聊天发送：流式输出 assistant 回复（节流刷新 + 结束后再规范化 Markdown）。"""
    message = (user_message or "").strip()
    chat_history = _normalize_chat_messages(history)
    compose_unlock = _chat_compose_updates(locked=False)

    if not message:
        yield chat_history, "", *compose_unlock
        return

    chat_history.append({"role": "user", "content": message})
    compose_lock = _chat_compose_updates(locked=True)
    yield chat_history, "", *compose_lock

    provider_key = get_active_provider_key()
    toast_message: str | None = None
    toast_kind = "info"
    plain_text_reply = False

    try:
        if is_correction_request(message):
            plain_text_reply = True
            transcript = original_transcript_from_history(chat_history)
            if not transcript:
                reply = "当前没有可校正的转写文本，请先完成语音转写。"
                toast_message = reply
                toast_kind = "info"
                chat_history.append({"role": "assistant", "content": reply})
                if toast_message:
                    show_toast(toast_message, kind=toast_kind, raise_error=False)
                yield chat_history, "", *compose_unlock
                return
            stream_iter = correct_transcript_stream(
                transcript, provider_key=provider_key
            )
        else:
            stream_iter = chat_with_history_stream(
                chat_history, provider_key=provider_key
            )

        _append_or_update_assistant_message(
            chat_history, _llm_assistant_streaming_placeholder()
        )
        yield chat_history, "", *compose_lock

        partial = ""
        last_rendered_len = 0
        last_render_at = 0.0
        for chunk in stream_iter:
            partial += chunk
            if not _should_refresh_stream_display(
                len(partial),
                last_rendered_len,
                last_render_at,
            ):
                continue
            rendered = _llm_assistant_chat_content_streaming(
                partial,
                plain_text=plain_text_reply,
                show_cursor=True,
            )
            _append_or_update_assistant_message(chat_history, rendered)
            last_rendered_len = len(partial)
            last_render_at = time.monotonic()
            yield chat_history, "", *compose_lock

        if not partial.strip():
            fallback = "大模型未返回内容，请稍后重试。"
            _append_or_update_assistant_message(chat_history, fallback)
            yield chat_history, "", *compose_unlock
            return

        if _should_refresh_stream_display(
            len(partial),
            last_rendered_len,
            last_render_at,
            force=True,
        ):
            penultimate = _llm_assistant_chat_content_streaming(
                partial,
                plain_text=plain_text_reply,
                show_cursor=True,
            )
            _append_or_update_assistant_message(chat_history, penultimate)
            yield chat_history, "", *compose_lock

        final_content = _llm_assistant_chat_content_final(
            partial,
            plain_text=plain_text_reply,
        )
        _append_or_update_assistant_message(chat_history, final_content)
        yield chat_history, "", *compose_unlock

    except Exception as exc:
        logging.exception("大模型调用失败")
        reply = f"调用大模型失败: {exc}"
        toast_message = str(exc)
        toast_kind = "error"
        _append_or_update_assistant_message(chat_history, reply)
        if toast_message:
            show_toast(toast_message, kind=toast_kind, raise_error=False)
        yield chat_history, "", *compose_unlock


def unload_models() -> None:
    """Release GPU memory."""
    get_manager().unload()
    show_toast("已卸载当前模型，显存已释放。", kind="success")


def _format_douyin_status_line(message: str) -> str:
    """将抖音状态格式化为单行「登录状态：…」。"""
    one_line = " ".join(part.strip() for part in message.splitlines() if part.strip())
    return f"登录状态：{one_line}"


def douyin_status_line() -> str:
    """读取当前抖音登录状态（单行）。"""
    return _format_douyin_status_line(douyin_login_status())


def douyin_open_login_line() -> str:
    """打开抖音登录并返回单行状态说明。"""
    return _format_douyin_status_line(douyin_open_login())


def douyin_save_login_line() -> str:
    """保存抖音登录并返回单行状态说明。"""
    return _format_douyin_status_line(douyin_save_login())


def build_app() -> gr.Blocks:
    """Construct the Gradio application."""
    model_names = list(MODEL_CHOICES.keys())

    with gr.Blocks(
        title="遥望·音视频转写",
        theme=THEME,
        css=TECH_CSS,
        fill_height=True,
        fill_width=True,
        elem_classes=["fullscreen-app"],
    ) as demo:
        gr.HTML(
            """
            <div class="app-header">
                <div class="hero-title">音视频转写</div>
                <div class="hero-sub">封装 · 遥望2035</div>
            </div>
            """,
            elem_classes=["app-header-block"],
        )

        with gr.Row(elem_classes=["app-main-row"]):
            with gr.Column(scale=3, min_width=360, elem_classes=["panel-left"]):
                gr.HTML('<div class="section-label section-label-media">媒体来源</div>')
                url_media_path = gr.State(value=None)

                with gr.Tabs(elem_classes=["media-tabs"]):
                    with gr.Tab("本地上传"):
                        media = gr.File(
                            label="拖拽或点击上传音频 / 视频",
                            file_types=["audio", "video"],
                            type="filepath",
                            height=130,
                        )
                    with gr.Tab("链接获取"):
                        with gr.Accordion(
                            "抖音登录（下载抖音视频需先登录一次）",
                            open=False,
                            elem_classes=["douyin-login-accordion"],
                        ):
                            cookie_dir = get_douyin_cookies_dir()
                            douyin_status = gr.Textbox(
                                value=douyin_status_line(),
                                lines=1,
                                max_lines=1,
                                interactive=False,
                                show_label=False,
                                elem_classes=["douyin-status-line"],
                            )
                            gr.HTML(
                                '<div class="douyin-hint">'
                                "1. 点击「打开抖音登录」→ 在弹出窗口扫码/手机号登录<br>"
                                "2. 登录成功后点击「保存登录」<br>"
                                f"3. Cookie 自动保存至 <code>{cookie_dir / 'douyin.txt'}</code>"
                                "</div>",
                                elem_classes=["douyin-hint-block"],
                            )
                            with gr.Row(elem_classes=["douyin-btn-row"]):
                                douyin_open_btn = gr.Button(
                                    "1. 打开抖音登录",
                                    scale=1,
                                    elem_classes=["secondary-btn"],
                                )
                                douyin_save_btn = gr.Button(
                                    "2. 保存登录",
                                    scale=1,
                                    elem_classes=["primary-btn"],
                                )
                                douyin_check_btn = gr.Button(
                                    "检查状态",
                                    scale=1,
                                    elem_classes=["secondary-btn"],
                                )
                        with gr.Column(elem_classes=["url-link-module"]):
                            video_url = gr.Textbox(
                                label="短视频链接",
                                placeholder="https://v.douyin.com/... 或完整视频页 URL",
                                lines=2,
                                elem_classes=["url-link-input"],
                            )
                            with gr.Row(elem_classes=["url-fetch-row"]):
                                btn_fetch = gr.Button(
                                    "获取视频",
                                    scale=2,
                                    elem_classes=["secondary-btn"],
                                )
                            url_video_preview = gr.Video(
                                label="视频预览",
                                height=220,
                                interactive=False,
                                show_download_button=False,
                                elem_classes=["url-video-preview"],
                            )

                gr.HTML('<div class="section-label section-label-config">识别配置</div>')
                model_dd = gr.Dropdown(
                    label="模型",
                    choices=model_names,
                    value=model_names[0],
                )
                with gr.Row(elem_classes=["config-radio-row"]):
                    language_dd = gr.Radio(
                        label="语言",
                        choices=["自动", "中文", "英文"],
                        value="自动",
                        scale=1,
                        elem_classes=["inline-radio"],
                    )
                    device_dd = gr.Radio(
                        label="设备",
                        choices=["auto", "cuda:0", "cpu"],
                        value="auto",
                        scale=1,
                        elem_classes=["inline-radio"],
                    )

                with gr.Row(elem_classes=["btn-row"]):
                    btn_run = gr.Button(
                        "开始转写",
                        variant="primary",
                        scale=2,
                        elem_classes=["primary-btn"],
                    )
                    btn_unload = gr.Button(
                        "释放显存",
                        scale=1,
                        elem_classes=["secondary-btn"],
                    )

            with gr.Column(scale=7, elem_classes=["panel-right"]):
                with gr.Column(elem_classes=["chat-column"]):
                    chatbot = gr.Chatbot(
                        label="",
                        type="messages",
                        layout="bubble",
                        show_label=False,
                        placeholder="转写完成后，识别文本将显示在这里…",
                        height=360,
                        autoscroll=True,
                        show_copy_button=True,
                        render_markdown=True,
                        line_breaks=True,
                        allow_tags=True,
                        sanitize_html=True,
                        avatar_images=(None, None),
                        elem_id="transcript-chat",
                        elem_classes=["transcript-chat-md"],
                    )
                    with gr.Column(elem_classes=["chat-compose"]):
                        with gr.Row(elem_classes=["chat-compose-inner"]):
                            chat_input = gr.Textbox(
                                label="",
                                placeholder="询问任何问题，或发送「校正」纠正识别错误",
                                show_label=False,
                                lines=1,
                                max_lines=8,
                                scale=10,
                                container=False,
                                elem_classes=["chat-compose-input"],
                            )
                            btn_chat_send = gr.Button(
                                "↑",
                                variant="primary",
                                scale=0,
                                min_width=36,
                                elem_classes=["chat-send-circle"],
                            )

        btn_fetch.click(
            fn=fetch_video_url,
            inputs=[video_url],
            outputs=[url_video_preview, url_media_path],
        )

        douyin_open_btn.click(fn=douyin_open_login_line, outputs=douyin_status)
        douyin_save_btn.click(fn=douyin_save_login_line, outputs=douyin_status)
        douyin_check_btn.click(fn=douyin_status_line, outputs=douyin_status)

        chat_submit_inputs = [chat_input, chatbot]
        chat_submit_outputs = [chatbot, chat_input, btn_chat_send]

        btn_chat_send.click(
            fn=handle_chat_submit,
            inputs=chat_submit_inputs,
            outputs=chat_submit_outputs,
        )
        chat_input.submit(
            fn=handle_chat_submit,
            inputs=chat_submit_inputs,
            outputs=chat_submit_outputs,
        )

        btn_run.click(
            fn=run_transcribe_with_progress,
            inputs=[
                media,
                video_url,
                url_media_path,
                model_dd,
                language_dd,
                device_dd,
            ],
            outputs=[chatbot, btn_run],
            show_progress="hidden",
        )
        btn_unload.click(fn=unload_models)

        demo.queue(default_concurrency_limit=1)

    _apply_favicon_to_demo(demo)
    return demo


def is_port_free(port: int, host: str = "0.0.0.0") -> bool:
    """检测端口是否可绑定（与 Gradio launch 使用相同 host）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _is_port_bind_error(exc: BaseException) -> bool:
    """判断是否为端口占用导致的启动失败。"""
    message = str(exc).lower()
    return (
        "cannot find empty port" in message
        or "10048" in message
        or "address already in use" in message
        or "already in use" in message
    )


def find_available_port(preferred: int, max_tries: int = 50) -> int:
    """从 preferred 起向后查找可用 TCP 端口。"""
    for port in range(preferred, preferred + max_tries):
        if is_port_free(port):
            return port
    raise OSError(
        f"在端口 {preferred}–{preferred + max_tries - 1} 范围内未找到可用端口，"
        "请关闭占用进程或使用 --port 指定其他起始端口。"
    )


def launch_app_with_port_fallback(
    app: gr.Blocks,
    preferred_port: int,
    *,
    open_browser: bool,
    max_tries: int = 50,
) -> None:
    """从首选端口起依次尝试启动；成功后由 Gradio 打开浏览器到实际地址。"""
    last_error: Exception | None = None

    for port in range(preferred_port, preferred_port + max_tries):
        if not is_port_free(port):
            if port == preferred_port:
                print(f"[提示] 端口 {port} 已被占用，正在尝试其他端口…")
            continue

        browser_url = f"http://127.0.0.1:{port}/"
        if port != preferred_port:
            print(f"[提示] 已改用端口 {port}")
        print()
        print("=" * 60)
        print("  FunASR 语音转写  ·  封装  ·  遥望2035")
        print(f"  UI 地址: {browser_url}")
        print("=" * 60)
        print()

        try:
            favicon = getattr(app, "favicon_path", None)
            if not favicon:
                prepared = _prepare_favicon_for_browser()
                favicon = str(prepared) if prepared else None
            app.launch(
                server_name="0.0.0.0",
                server_port=port,
                inbrowser=open_browser,
                show_error=True,
                show_api=False,
                favicon_path=favicon,
                allowed_paths=[str(OUTPUT_DIR), str(MODELS_DIR), str(PROJECT_ROOT)],
            )
            return
        except OSError as exc:
            if not _is_port_bind_error(exc):
                raise
            last_error = exc
            logging.warning("端口 %d 启动失败，尝试下一端口: %s", port, exc)
            if getattr(app, "is_running", False):
                app.close()
            continue

    raise OSError(
        f"在端口 {preferred_port}–{preferred_port + max_tries - 1} 范围内均无法启动，"
        "请关闭占用端口的进程后重试。"
    ) from last_error


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="FunASR 语音转写 Gradio 界面")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="首选端口（默认 7880；被占用时自动尝试后续端口）",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="启动后不自动打开浏览器",
    )
    return parser.parse_args()


def main() -> None:
    """启动 Gradio；7880 被占用时自动换端口，并打开浏览器。"""
    args = parse_args()
    preferred = args.port or int(os.environ.get("GRADIO_SERVER_PORT", "7880"))

    # Gradio 5.x：theme/css 仍放在 Blocks；抑制 6.0 迁移提示
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*'theme' parameter in the Blocks constructor.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*'css' parameter in the Blocks constructor.*",
    )

    from model_paths import format_verification_report, verify_all_local_models

    for line in format_verification_report(verify_all_local_models()).splitlines():
        logging.info("模型检查 | %s", line)

    app = build_app()
    try:
        launch_app_with_port_fallback(
            app,
            preferred,
            open_browser=not args.no_browser,
        )
    finally:
        close_douyin_login()


if __name__ == "__main__":
    main()
