"""大模型对话与 ASR 转写校正（OpenAI 兼容 Chat Completions API）。"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "llm_config.json"

# Qwen3 / vLLM 思考块标签（流式输出中需过滤，否则界面空白）
_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "<" + "/" + "think" + ">"
_THINK_BLOCK_RE = re.compile(
    "<" + "think" + ">.*?</" + "think" + ">",
    re.DOTALL,
)

CORRECTION_SYSTEM_PROMPT = """你是一个专业的语音识别（ASR）文本校对助手。

你的唯一任务：修正给定转写文本中的识别错误。

严格规则（必须全部遵守）：
1. 只修正明显的 ASR 错误：错别字、同音字误识别、标点缺失或错误、断句问题、英文/数字/专有名词拼写错误。
2. 不得改变原文的语义、语气、立场和说话风格。
3. 不得添加原文中不存在的内容、句子、解释、总结或评论。
4. 不得删除原文中实际存在的有效信息；对口误赘词、明显重复可极保守地删减。
5. 若某处无法确定是否为识别错误，保持原样，不要猜测改写。
6. 保持原文段落结构；不要合并或拆分段落，除非仅为修正断句错误。
7. 直接输出校正后的完整文本；不要输出标题、前缀（如「校正后：」）、Markdown 代码块或任何额外说明。"""

CHAT_SYSTEM_PROMPT = """你是语音转写工作台的对话助手。
对话开头通常有一条由系统提供的语音转写文本（assistant 消息）。
请基于该转写内容与用户后续问题，用简洁、准确的中文回答。
若用户问题与转写内容无关，也可正常作答；不要编造转写中不存在的事实。

排版：请使用标准 Markdown 输出（界面会按 Markdown 渲染）：
- 段落之间空一行；无序列表统一用 `- `（减号+空格）作行首，每条单独一行，不要用 *、•、◦ 混用；
- 小节标题用 `### 标题`（# 后必须有空格），标题前后各空一行；
- 小节之间可空一行分隔；若使用 `---` 仅作分段（界面不会显示横线，只保留间距）；
- 需要强调时使用 **粗体**；不要输出 HTML 标签或把整段包在代码块里。"""

# 本地 vLLM / Qwen 等模型更易输出粘连文本，需额外强调标准 Markdown
LOCAL_VLLM_MARKDOWN_APPEND = """

【输出格式 - 必须严格遵守（本地 vLLM）】
回复将按 Markdown 渲染；格式不规范会导致界面不换行、列表错位。你必须：
1. 段落之间空一行；不要把多段挤在同一行。
2. 无序列表：每行以「- 」（减号+空格）开头，一条一行；禁止 *、•、◦；禁止多条列表写在同一行。
3. 有序列表：每行以「1. 」「2. 」（数字+点+空格）开头，一条一行。
4. 小节标题：必须单独一行写「### 标题」（# 与标题间有空格），例如「### 概述」，禁止只写标题文字而不加 ###。
5. 小节之间空一行；列表每条单独一行，以「- 」开头；勿写 ---### 粘连。
6. 强调用 **粗体**；禁止 HTML；禁止用 ``` 包裹整段回复。

格式示例（请模仿）：
### 概述

- 第一条说明文字
- 第二条说明文字

### 详细说明

1. 第一点内容
2. 第二点内容

（小节之间空一行即可，不必画横线。）
"""

CORRECTION_TRIGGER = "校正"
CONNECT_TIMEOUT_SEC = 20
READ_TIMEOUT_SEC = 600


class LLMConfigError(RuntimeError):
    """LLM 配置无效或缺失。"""


def load_config() -> dict[str, Any]:
    """读取 llm_config.json。"""
    if not CONFIG_PATH.is_file():
        raise LLMConfigError(f"未找到配置文件: {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise LLMConfigError("llm_config.json 根节点必须是 JSON 对象")
    return data


def save_config(config: dict[str, Any]) -> None:
    """写回 llm_config.json。"""
    with CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def provider_choices() -> list[tuple[str, str]]:
    """返回 Gradio Dropdown 用的 (显示名, provider_key) 列表。"""
    config = load_config()
    providers = config.get("providers", {})
    return [
        (str(meta.get("display_name", key)), key)
        for key, meta in providers.items()
    ]


def get_active_provider_key() -> str:
    """返回当前启用的 provider key。"""
    config = load_config()
    active = config.get("active_provider")
    providers = config.get("providers", {})
    if active not in providers:
        if providers:
            return next(iter(providers))
        raise LLMConfigError("llm_config.json 中未配置任何 provider")
    return str(active)


def set_active_provider_key(provider_key: str) -> None:
    """切换当前 provider 并持久化到配置文件。"""
    config = load_config()
    providers = config.get("providers", {})
    if provider_key not in providers:
        raise LLMConfigError(f"未知 provider: {provider_key}")
    config["active_provider"] = provider_key
    save_config(config)
    logging.info("已切换大模型 provider: %s", provider_key)


def _resolve_provider(provider_key: str | None) -> tuple[str, dict[str, Any]]:
    """解析 provider 配置，返回 (key, meta)。"""
    config = load_config()
    providers = config.get("providers", {})
    key = provider_key or config.get("active_provider")
    if key not in providers:
        raise LLMConfigError(f"未知或未配置的 provider: {key}")
    meta = providers[key]
    for field in ("model", "api_url"):
        if not meta.get(field):
            raise LLMConfigError(f"provider「{key}」缺少字段: {field}")
    return str(key), meta


def _chat_system_prompt(provider_key: str | None = None) -> str:
    """按 provider 返回系统提示；本地 vLLM 附加严格 Markdown 说明。"""
    key, meta = _resolve_provider(provider_key)
    prompt = CHAT_SYSTEM_PROMPT
    if key == "local_vllm" or meta.get("strict_markdown_output"):
        prompt = f"{prompt.rstrip()}\n{LOCAL_VLLM_MARKDOWN_APPEND}"
    return prompt


def _strip_think_tags(text: str) -> str:
    """移除完整 … 区块。"""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    return cleaned.replace(_THINK_OPEN, "").replace(_THINK_CLOSE, "").strip()


def _should_disable_thinking(provider_key: str, meta: dict[str, Any]) -> bool:
    """是否关闭模型思考输出（不展示 reasoning / think 内容）。"""
    explicit = meta.get("disable_thinking")
    if explicit is not None:
        return bool(explicit)
    if provider_key == "deepseek":
        return True
    model = str(meta.get("model", "")).lower()
    if "deepseek" in model:
        return True
    return False


def _uses_qwen_no_think_suffix(provider_key: str, meta: dict[str, Any]) -> bool:
    """Qwen / 本地 vLLM 需在 user 消息末尾追加 /no_think。"""
    if meta.get("append_no_think"):
        return True
    return provider_key == "local_vllm"


def _apply_disable_thinking_to_payload(
    provider_key: str,
    meta: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    """按 provider 向请求体注入「关闭思考」参数。"""
    if not _should_disable_thinking(provider_key, meta):
        return
    if provider_key == "deepseek" or "deepseek.com" in str(meta.get("api_url", "")):
        payload["thinking"] = {"type": "disabled"}
        return
    extra = dict(meta.get("extra_body") or {})
    chat_kwargs = dict(extra.pop("chat_template_kwargs", {}) or {})
    chat_kwargs["enable_thinking"] = False
    payload["chat_template_kwargs"] = chat_kwargs
    payload.update(extra)


def _prepare_messages_for_provider(
    messages: list[dict[str, str]],
    meta: dict[str, Any],
    *,
    provider_key: str,
) -> list[dict[str, str]]:
    """按 provider 配置调整请求消息（如 Qwen3 关闭思考）。"""
    if not _should_disable_thinking(provider_key, meta):
        return messages
    if not _uses_qwen_no_think_suffix(provider_key, meta):
        return messages
    prepared: list[dict[str, str]] = []
    for item in messages:
        msg = dict(item)
        if msg.get("role") == "user":
            content = str(msg.get("content", "")).rstrip()
            if "/no_think" not in content and _THINK_OPEN not in content:
                content = f"{content} /no_think"
            msg["content"] = content
        prepared.append(msg)
    return prepared


class _ThinkTagStreamFilter:
    """流式过滤 …，避免思考内容导致界面无可见文字。"""

    def __init__(self) -> None:
        self._in_think = False
        self._buf = ""

    def feed(self, text: str) -> str:
        """输入原始增量，返回可展示给用户的增量。"""
        if not text:
            return ""
        self._buf += text
        visible: list[str] = []
        while self._buf:
            if self._in_think:
                close_at = self._buf.find(_THINK_CLOSE)
                if close_at < 0:
                    keep = max(len(_THINK_CLOSE) - 1, 0)
                    if len(self._buf) > keep:
                        self._buf = self._buf[-keep:]
                    break
                self._buf = self._buf[close_at + len(_THINK_CLOSE) :]
                self._in_think = False
                continue
            open_at = self._buf.find(_THINK_OPEN)
            if open_at < 0:
                keep = max(len(_THINK_OPEN) - 1, 0)
                if keep and len(self._buf) > keep:
                    visible.append(self._buf[:-keep])
                    self._buf = self._buf[-keep:]
                else:
                    visible.append(self._buf)
                    self._buf = ""
                break
            if open_at > 0:
                visible.append(self._buf[:open_at])
            self._buf = self._buf[open_at + len(_THINK_OPEN) :]
            self._in_think = True
        return "".join(visible)

    def flush(self) -> str:
        """释放缓冲区尾部可见文本。"""
        if self._in_think:
            return ""
        tail = self._buf
        self._buf = ""
        return tail


def _build_request(
    meta: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    provider_key: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> tuple[str, dict[str, str], dict[str, Any], tuple[float, float]]:
    """构造 Chat Completions 请求的 URL、headers、JSON body 与超时。"""
    headers = {"Content-Type": "application/json"}
    api_key = str(meta.get("api_key") or "").strip()
    if api_key and api_key != "sk-no-key-required":
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {
        "model": meta["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": False}

    _apply_disable_thinking_to_payload(provider_key, meta, payload)

    connect_timeout = float(meta.get("connect_timeout", CONNECT_TIMEOUT_SEC))
    read_timeout = float(meta.get("read_timeout", READ_TIMEOUT_SEC))
    url = str(meta["api_url"]).rstrip("/")
    return url, headers, payload, (connect_timeout, read_timeout)


def _iter_sse_payloads(response: requests.Response) -> Iterator[str]:
    """从流式 HTTP 响应中增量解析 SSE data 字段（不依赖 iter_lines 整行缓冲）。"""
    buffer = ""
    for chunk in response.iter_content(chunk_size=2048, decode_unicode=True):
        if not chunk:
            continue
        buffer += chunk.replace("\r\n", "\n")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                yield line[5:].strip()
    tail = buffer.strip()
    if tail.startswith("data:"):
        yield tail[5:].strip()


def _extract_delta_from_chunk(
    chunk: dict[str, Any],
    *,
    include_reasoning: bool = True,
) -> str:
    """从单个 stream chunk JSON 提取可见文本增量。"""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return ""
    delta = choice0.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if content is not None and str(content):
            return str(content)
        if include_reasoning:
            reasoning = delta.get("reasoning_content")
            if reasoning is not None and str(reasoning):
                return str(reasoning)
    message = choice0.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if content is not None and str(content):
            return str(content)
    text = choice0.get("text")
    return str(text) if text is not None else ""


def _extract_delta_from_sse_payload(
    data: str,
    *,
    include_reasoning: bool = True,
) -> str:
    """从 OpenAI 兼容 SSE data 行解析文本增量。"""
    if not data or data == "[DONE]":
        return ""
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return ""
    if not isinstance(chunk, dict):
        return ""
    return _extract_delta_from_chunk(chunk, include_reasoning=include_reasoning)


def _non_stream_completion(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeouts: tuple[float, float],
) -> str:
    """非流式请求，返回完整 assistant 文本。"""
    body_payload = {**payload, "stream": False}
    body_payload.pop("stream_options", None)
    response = requests.post(
        url,
        headers=headers,
        json=body_payload,
        timeout=timeouts,
    )
    response.raise_for_status()
    body = response.json()
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"API 响应格式异常: {body}") from exc
    return _strip_think_tags(str(content))


def iter_chat_completion_chunks(
    messages: list[dict[str, str]],
    *,
    provider_key: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 8192,
) -> Iterator[str]:
    """流式调用 Chat Completions，逐块 yield 文本增量。"""
    key, meta = _resolve_provider(provider_key)
    messages = _prepare_messages_for_provider(messages, meta, provider_key=key)
    strip_thinking = _should_disable_thinking(key, meta)
    url, headers, payload, timeouts = _build_request(
        meta,
        messages,
        provider_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    logging.info(
        "流式调用大模型 provider=%s model=%s url=%s",
        key,
        meta["model"],
        url,
    )
    logging.info("正在连接大模型 API（连接超时 %ss）…", timeouts[0])

    try:
        with requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeouts,
            stream=True,
        ) as response:
            response.raise_for_status()
            logging.info(
                "大模型 API 已连接 HTTP %s，等待首 token（读取超时 %ss）…",
                response.status_code,
                timeouts[1],
            )

            think_filter = _ThinkTagStreamFilter() if strip_thinking else None
            got_first = False
            visible_chars = 0
            for data in _iter_sse_payloads(response):
                raw = _extract_delta_from_sse_payload(
                    data,
                    include_reasoning=not strip_thinking,
                )
                if not raw:
                    continue
                visible = think_filter.feed(raw) if think_filter else raw
                if think_filter:
                    visible = _strip_think_tags(visible)
                if not visible or not visible.strip():
                    continue
                if not got_first:
                    logging.info("开始接收流式输出")
                    got_first = True
                visible_chars += len(visible)
                yield visible

            if think_filter:
                tail = _strip_think_tags(think_filter.flush())
                if tail.strip():
                    if not got_first:
                        logging.info("开始接收流式输出")
                        got_first = True
                    visible_chars += len(tail)
                    yield tail

            if got_first and visible_chars > 0:
                logging.info("流式输出结束，共 %d 个可见字", visible_chars)
                return
            if got_first:
                logging.warning(
                    "流式结束但未产生可见正文（可能全是思考内容），尝试非流式回退"
                )

            logging.warning(
                "流式响应未解析到文本（vLLM 可能在思考阶段无 content），尝试非流式回退"
            )
    except requests.RequestException as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.text[:500]
            except Exception:
                detail = ""
        raise RuntimeError(f"API 请求失败: {exc}. {detail}".strip()) from exc

    text = _non_stream_completion(url, headers, payload, timeouts)
    if text:
        logging.info("非流式回退成功，返回 %d 字", len(text))
        yield text


def chat_completions(
    messages: list[dict[str, str]],
    *,
    provider_key: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 8192,
) -> str:
    """调用 Chat Completions API，返回完整 assistant 文本（非流式）。"""
    parts = list(
        iter_chat_completion_chunks(
            messages,
            provider_key=provider_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )
    if parts:
        return "".join(parts).strip()
    key, meta = _resolve_provider(provider_key)
    prepared = _prepare_messages_for_provider(
        messages, meta, provider_key=key
    )
    url, headers, payload, timeouts = _build_request(
        meta,
        prepared,
        provider_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    logging.info("调用大模型 provider=%s model=%s", provider_key or "active", meta["model"])
    try:
        return _non_stream_completion(url, headers, payload, timeouts)
    except requests.RequestException as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.text[:500]
            except Exception:
                detail = ""
        raise RuntimeError(f"API 请求失败: {exc}. {detail}".strip()) from exc
    except ValueError as exc:
        raise RuntimeError(f"API 返回非 JSON: {exc}") from exc


def message_content_to_plain_text(content: object) -> str:
    """从 Chatbot 消息 content（字符串、gr.HTML 或 HTML 组件 dict）提取纯文本。"""
    if isinstance(content, dict):
        if content.get("component") == "html":
            value = str(content.get("value", ""))
            value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
            value = re.sub(r"</p>\s*<p>", "\n\n", value, flags=re.IGNORECASE)
            value = re.sub(r"</li>\s*", "\n", value, flags=re.IGNORECASE)
            value = re.sub(r"<[^>]+>", "", value)
            return value.strip()
        return str(content.get("value", content)).strip()
    component_value = getattr(content, "value", None)
    if component_value is not None and type(content).__name__ == "HTML":
        return message_content_to_plain_text(
            {"component": "html", "value": str(component_value)}
        )
    return str(content or "").strip()


def _history_to_api_messages(
    history: list[dict[str, str]],
    *,
    provider_key: str | None = None,
) -> list[dict[str, str]]:
    """将聊天历史转为 API messages（跳过空 assistant）。"""
    api_messages: list[dict[str, str]] = [
        {"role": "system", "content": _chat_system_prompt(provider_key)}
    ]
    for item in history:
        role = item.get("role")
        content = message_content_to_plain_text(item.get("content", ""))
        if role in ("user", "assistant") and content:
            api_messages.append({"role": str(role), "content": content})
    return api_messages


def correct_transcript(transcript: str, *, provider_key: str | None = None) -> str:
    """校正 ASR 转写文本（低温度，仅纠错）。"""
    return "".join(correct_transcript_stream(transcript, provider_key=provider_key)).strip()


def correct_transcript_stream(
    transcript: str,
    *,
    provider_key: str | None = None,
) -> Iterator[str]:
    """流式校正 ASR 转写文本。"""
    text = (transcript or "").strip()
    if not text:
        yield "没有可校正的转写内容。"
        return
    messages = [
        {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    yield from iter_chat_completion_chunks(
        messages, provider_key=provider_key, temperature=0.1
    )


def chat_with_history(
    history: list[dict[str, str]],
    *,
    provider_key: str | None = None,
) -> str:
    """基于聊天历史进行多轮对话（非流式）。"""
    return "".join(
        chat_with_history_stream(history, provider_key=provider_key)
    ).strip()


def chat_with_history_stream(
    history: list[dict[str, str]],
    *,
    provider_key: str | None = None,
) -> Iterator[str]:
    """基于聊天历史进行多轮对话（流式）。"""
    api_messages = _history_to_api_messages(history, provider_key=provider_key)
    if len(api_messages) <= 1:
        yield "请先完成转写或输入您的问题。"
        return
    yield from iter_chat_completion_chunks(
        api_messages, provider_key=provider_key, temperature=0.7
    )


def is_correction_request(user_message: str) -> bool:
    """判断用户是否请求 ASR 校正。"""
    return user_message.strip() == CORRECTION_TRIGGER


def original_transcript_from_history(history: list[dict[str, str]]) -> str:
    """取聊天历史中第一条 assistant 消息（原始转写）。"""
    for item in history:
        if item.get("role") == "assistant":
            return message_content_to_plain_text(item.get("content", ""))
    return ""
