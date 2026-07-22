<h1 align="center">🎬 video-crawler-extraction</h1>

<p align="center">
  <b>短视频爬取与智能解析工具</b><br>
  自动化采集 · 语音转写 · 大模型分析 · 一站式内容处理
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/ASR-FunASR-orange" alt="FunASR">
  <img src="https://img.shields.io/badge/LLM-OpenAI%20Compatible-purple" alt="LLM">
</p>

---

## 📖 项目简介

**video-crawler-extraction** 是一款面向中文用户的短视频内容自动采集与智能解析工具。它整合了三大核心能力：

- **🔽 视频下载** — 从抖音、B站、YouTube 等主流平台定向采集公开视频
- **🎤 语音转写（ASR）** — 基于 FunASR 框架，支持多种语音识别模型，自动将语音转为文本
- **🧠 智能分析** — 集成大语言模型（LLM），对转写文本进行校正、关键词提取、情感分析、主题归纳等

项目提供一套美观的 Gradio Web 界面，操作直观，模型可完全离线运行，适合自媒体运营、舆情分析、内容研究等多种场景。

---

## ✨ 核心特性

### 🎯 短视频下载
- 支持 **抖音、B站、YouTube** 等主流短视频平台
- 内置 **抖音登录**（Playwright 浏览器模拟登录），突破有限制的内容
- 可配置画质预设，默认 720p 及以下（为 ASR 转写优先选择较小体积）
- 支持 URL 批量下载

### 🎙️ 多模型语音识别
提供 **4 种 ASR 模型** 可供切换，覆盖不同精度与速度需求：

| 模型 | 识别语言 | 特性 | 适用场景 |
|------|---------|------|---------|
| **Fun-ASR-Nano** | 中英双语 | VAD 分段 + LLM 推理，170x 实时率 | **通用首选**，长音频支持好 |
| **Qwen3-ASR-1.7B** | 中英双语 | 整段识别，精度较高 | 短音频、高质量录音 |
| **Whisper-large-v3-turbo** | 50+ 语种 | VAD 分段 + Whisper decode | 多语种场景（UI 默认隐藏） |
| **GLM-ASR-Nano-1.5B** | 中英双语 | LLM 逐条推理 | 高精度场景（UI 默认隐藏） |

- 内置 **VAD（语音活动检测）** 自动分段，突破单次推理长度限制
- 自动将视频文件（mp4/mkv/avi 等）转为 16kHz 单声道 WAV
- 支持 **bf16 / fp16** 混合精度，GPU 显存高效利用

### 🤖 大模型智能分析
- 集成 **OpenAI 兼容 Chat Completions API**（支持 DeepSeek、本地 vLLM 等）
- **`校正` 一键纠错** — 自动纠正 ASR 常见错别字、同音字、标点错误
- **多轮对话** — 支持提取关键词、总结摘要、情感分析、主题归纳
- **流式输出** — 实时展示推理过程，自动过滤 `<think>` 思考标签

### 🐳 Docker 支持
- 提供 DockerFile 构建 **vLLM 推理服务镜像**
- CUDA 12.4 + Python 3.11 + vLLM 0.7.3
- 国内源配置（阿里云 + 清华镜像），适合中国用户

### 📊 性能基准测试
- `benchmark_vllm.py` 提供 PyTorch vs vLLM **一键性能对比**
- 支持 CER（字错率）自动评估
- 内置 Kaldi 对齐算法，结果准确可靠

---

## 🏗️ 项目架构

```
video-crawler-extraction/
├── gradio_ui.py                # Gradio Web UI 主入口
├── asr_service.py              # ASR 模型加载与语音推理核心
├── llm_service.py              # 大模型对话与 ASR 文本校正
├── llm_config.json             # 大模型 API 配置文件
├── ytdlp_bridge.py             # yt-dlp 下载桥接层
├── model_paths.py              # 模型规格与路径管理
├── download_models.py          # 模型下载脚本（从 ModelScope）
├── benchmark_vllm.py           # vLLM 性能基准测试
├── start.bat                   # Windows 快速启动脚本
├── DockerFile                  # Docker 构建文件（vLLM 服务）
├── requirements.txt            # Python 依赖清单
├── funasr/                     # FunASR 语音识别框架源码
├── fun_text_processing/        # 文本正则化 / 逆文本正则化
├── input/                      # 输入文件目录
├── models/                     # 模型权重目录（自动下载到此）
└── output/                     # 输出结果目录
    ├── uploads/                # 用户上传的音频/视频
    ├── converted/              # ffmpeg 转码后的音频
    └── transcripts/            # 转写结果文本
```

---

## 🚀 快速开始

### 🪟 Windows 一键整合包（推荐）

对于 Windows 用户，我们提供预配置的**一键整合包**，无需手动安装 Python 和依赖，下载解压即可使用：

- **下载地址**：[百度网盘](https://pan.baidu.com/s/1zUXxIsSpbqzmUoeHshUkTw?pwd=2035)（提取码：`2035`）
- **包含内容**：嵌入式 Python 环境 + 全部依赖 + 预下载模型
- **使用方法**：解压后双击 `start.bat`，浏览器访问 `http://localhost:7880`

> 💡 整合包已内置 Fun-ASR-Nano 和 Qwen3-ASR-1.7B 模型，开箱即用。

### 环境要求

| 组件 | 要求 |
|------|------|
| Python | 3.7+（推荐 3.11） |
| GPU | NVIDIA GPU 推荐（仅 CPU 也可运行，速度较慢） |
| 显存 | 至少 4GB（Fun-ASR-Nano）~ 6GB（Qwen3-ASR） |
| 系统 | Windows 10/11、Linux、macOS |
| ffmpeg | 处理视频文件时需要 |

### 安装步骤

#### 1️⃣ 克隆项目

```bash
git clone https://github.com/chenjian120705/video-crawler-extraction.git
cd video-crawler-extraction
```

#### 2️⃣ 安装 PyTorch

```bash
# （推荐）CUDA 12.x 用户
pip install torch==2.12.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130

# 无 NVIDIA GPU（仅 CPU 推理）
pip install torch torchaudio
```

#### 3️⃣ 安装项目依赖

```bash
pip install -r requirements.txt
```

> ⚠️ **注意**：`requirements.txt` 中已排除 `torch` / `torchaudio`，避免覆盖你手动安装的 CUDA 版本。

#### 4️⃣ 安装抖音登录（可选）

```bash
playwright install chromium
```

#### 5️⃣ 下载 ASR 模型

```bash
python download_models.py
```

默认下载 **Fun-ASR-Nano** 和 **Qwen3-ASR-1.7B**，自动从阿里云 ModelScope 镜像拉取。

> 仅下载指定模型（例：只下载 Fun-ASR-Nano）：
> ```bash
> python download_models.py --only "Fun-ASR-Nano"
> ```

> 仅检查本地模型完整性（不下载）：
> ```bash
> python download_models.py --verify-only
> ```

#### 6️⃣ 配置大模型 API（可选）

编辑 `llm_config.json`：

```json
{
  "active_provider": "deepseek",
  "providers": {
    "deepseek": {
      "display_name": "DeepSeek V4 Flash",
      "model": "deepseek-chat",
      "api_key": "your-api-key-here",
      "api_url": "https://api.deepseek.com/v1/chat/completions",
      "disable_thinking": true
    },
    "local_vllm": {
      "display_name": "本地 vLLM 模型",
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "api_key": "sk-no-key-required",
      "api_url": "http://localhost:8000/v1/chat/completions",
      "strict_markdown_output": true,
      "disable_thinking": true
    }
  }
}
```

### 启动 Web UI

```bash
# Windows：双击 start.bat，或
python gradio_ui.py
```

浏览器打开 **http://localhost:7880** 即可进入主界面。

---

## 🎮 使用指南

### 基础流程

```
① 上传文件 / 粘贴链接  →  ② 点击「开始转写」  →  ③ 查看识别结果  →  ④ 大模型对话分析
```

### 详细操作

#### 上传媒体
- **本地文件**：支持 mp3/wav/mp4/mkv/avi/mov/flv 等格式
- **视频链接**：粘贴抖音 / B站 / YouTube 等平台 URL，自动下载

#### 选择模型
- 下拉框选择 **Fun-ASR-Nano** 或 **Qwen3-ASR-1.7B**
- 首次加载模型需等待数分钟（后续切换秒级）

#### 智能对话
在右侧对话面板输入：
- **`校正`** — 触发 ASR 文本自动纠错
- **`提取关键词`** — 自动提取核心关键词
- **`总结`** — 生成内容摘要
- **`情感分析`** — 分析文本情感倾向
- **任意问题** — 与转写内容自由交互

### 输出文件
转写结果自动保存到 `output/transcripts/{文件名}_{模型名}.txt`

---

## 🔬 性能基准测试

```bash
# Fun-ASR-Nano vLLM 基准测试
python benchmark_vllm.py \
    --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --audio-dir /path/to/benchmark_audio \
    --label-json /path/to/benchmark_testset.json

# 仅测试 vLLM（跳过 PyTorch）
python benchmark_vllm.py \
    --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --skip-pytorch \
    --audio-dir /path/to/benchmark_audio \
    --label-json /path/to/benchmark_testset.json

# 快速测试（仅前 20 条）
python benchmark_vllm.py \
    --model FunAudioLLM/Fun-ASR-Nano-2512 \
    --max-files 20 \
    --audio-dir /path/to/benchmark_audio \
    --label-json /path/to/benchmark_testset.json
```

### 性能参考

| 模型 | 推理引擎 | 实时率 (RTFx) | CER 字错率 |
|------|---------|--------------|-----------|
| Fun-ASR-Nano | PyTorch | ~80x | 5~10% |
| Fun-ASR-Nano | vLLM | ~170x | 5~10% |
| Qwen3-ASR-1.7B | PyTorch | ~5x | 4~8% |
| Whisper-large-v3-turbo | PyTorch | ~10x | 4~12%（视语种） |

> RTFx = 音频时长 / 推理时间；数值越大越快。

---

## 🐳 Docker 部署（vLLM 推理服务）

```bash
# 构建镜像
docker build -t vllm-server -f DockerFile .

# 启动容器
docker run -d \
    --gpus all \
    -p 8000:8000 \
    -v /path/to/models:/models \
    --name vllm-server \
    vllm-server \
    python3 -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen/Qwen2.5-7B-Instruct \
    --served-model-name Qwen/Qwen2.5-7B-Instruct
```

然后在 `llm_config.json` 中配置：
```json
"local_vllm": {
  "api_url": "http://localhost:8000/v1/chat/completions"
}
```

---

## 📁 目录说明

| 目录 | 说明 |
|------|------|
| `funasr/` | FunASR 语音识别核心框架（源码） |
| `fun_text_processing/` | 文本正则化与逆文本正则化 |
| `models/` | ASR 模型权重（自动下载到此） |
| `output/uploads/` | 用户上传的媒体文件 |
| `output/converted/` | ffmpeg 转码后的音频 |
| `output/transcripts/` | ASR 转写结果文本 |
| `input/` | 输入文件占位目录 |

---

## 🛠️ 常见问题

**Q: 模型下载失败怎么办？**
> 检查网络连接，脚本默认使用阿里云镜像（`mirrors.aliyun.com/modelscope/`）。可手动设置环境变量：
> ```bash
> export MODELSCOPE_ENDPOINT=https://mirrors.aliyun.com/modelscope/
> ```

**Q: GPU 显存不足？**
> 尝试使用 `Fun-ASR-Nano`（显存需求最小），或在 CPU 上运行。可在 `asr_service.py` 中修改 `resolve_dtype()` 强制使用 fp16。

**Q: 没有 NVIDIA GPU 能运行吗？**
> 可以，CPU 推理完全支持（设备选择 `cpu`），但速度会慢很多。Fun-ASR-Nano 在 CPU 上约 1~2x 实时率。

**Q: 视频转码失败？**
> 确保已安装 **ffmpeg** 并加入系统 PATH。Windows 用户可从 https://ffmpeg.org/download.html 下载。

**Q: 如何让 Whisper 或 GLM-ASR 显示在 UI 中？**
> 在 `model_paths.py` 中将对应规格的 `"ui_visible": false` 改为 `true`，或设置为默认可见。

---

## 📄 开源协议

本项目基于 **MIT 许可证** 开源，详情见 [LICENSE](./LICENSE) 文件。

---

## 🙏 致谢

- [FunASR](https://github.com/modelscope/FunASR) — 达摩院工业级语音识别框架
- [ModelScope](https://modelscope.cn) — 模型下载与社区
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — 视频下载工具
- [Gradio](https://gradio.app) — Web UI 框架
- [Qwen](https://github.com/QwenLM/Qwen) — 通义千问 ASR 模型
- [DeepSeek](https://deepseek.com) — 大模型 API 服务

---

<p align="center">
  <sub>Built with ❤️ for the Chinese-speaking community</sub>
</p>