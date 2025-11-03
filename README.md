# LightTTS

LightTTS 是在 CosyVoice 生态基础上整合 VoxCPM、IndexTTS、Kitten TTS 等模型的多引擎语音合成实验平台，提供统一的脚本、图形界面和批量生产工具，便于在本地快速验证不同模型的推理效果。

## 功能亮点
- 支持 CosyVoice2-0.5B、VoxCPM-0.5B、IndexTTS-2、Kitten TTS Nano 等多种主流零样本语音合成模型
- 内置 CLI 脚本、Gradio Web UI、PySide GUI 以及多进程批量推理范例
- 通过全局变量和 CLI 参数灵活配置模型路径、推理设备、输出目录等
- 预置示例音频 (`asset/zero_shot_prompt.wav`) 及常用文本，开箱即用

## 仓库结构
- `asset/`：示例提示音频与素材
- `BatchGenerate/`：PySide6 批量语音合成与音色管理 GUI
- `cosyvoice/`、`voxcpm/`、`indextts/`、`kittentts/`：各模型的推理与工具代码
- `demo/`：Gradio Web UI 与简单示例脚本
- `playground/`：命令行和并行推理范例
- `models/`：放置已下载的预训练权重
- `requirements.txt`：跨平台依赖列表

## 环境准备
1. 准备 Python 3.10（推荐使用 Conda 环境）
2. 安装 PyTorch 与 torchaudio（根据自身 CUDA 版本参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)）
3. 安装项目依赖

```bash
conda create -n LightTTS python=3.10 -y
conda activate LightTTS
pip install --upgrade pip
# 参考 PyTorch 官网命令安装 torch/torchaudio
pip install -r requirements.txt
```

> **提示**：Linux 环境若提示缺少 `libsndfile`，可执行 `sudo apt install libsndfile1`。

## 模型资源准备
将所需模型按下表放置到 `models/` 目录，或在脚本中修改对应的全局变量（如 `MODEL_DIR`）。

| 模型 | 推荐来源 | 默认目录 | 备注 |
| --- | --- | --- | --- |
| VoxCPM-0.5B | Hugging Face `openbmb/VoxCPM-0.5B` | `models/VoxCPM-0.5B` | 支持本地加载或自动快照下载 |
| CosyVoice2-0.5B | ModelScope `iic/CosyVoice2-0.5B` | `models/CosyVoice2-0.5B` | 需同时准备 Matcha-TTS 依赖目录（仓库已包含） |
| speech_zipenhancer_ans_multiloss_16k_base | ModelScope `iic/speech_zipenhancer_ans_multiloss_16k_base` | `models/speech_zipenhancer_ans_multiloss_16k_base` | VoxCPM 去噪可选 |
| IndexTTS-2 | Hugging Face / ModelScope `IndexTeam/IndexTTS-2` | `models/IndexTTS-2` | 包含配置 `config.yaml` 与权重 |
| Kitten TTS Nano 0.2 | Hugging Face `KittenML/kitten-tts-nano-0.2` | `models/kitten-tts-nano-0.2` | 纯 ONNX，CPU 即可运行 |

### 示例下载命令
```bash
# VoxCPM-0.5B（Hugging Face）
huggingface-cli download openbmb/VoxCPM-0.5B --local-dir models/VoxCPM-0.5B --local-dir-use-symlinks False

# CosyVoice2-0.5B（ModelScope）
modelscope download --model iic/CosyVoice2-0.5B --local_dir models/CosyVoice2-0.5B

# VoxCPM 降噪模型
modelscope download --model iic/speech_zipenhancer_ans_multiloss_16k_base --local_dir models/speech_zipenhancer_ans_multiloss_16k_base

# IndexTTS-2
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir models/IndexTTS-2 --local-dir-use-symlinks False

# Kitten TTS Nano 0.2
huggingface-cli download KittenML/kitten-tts-nano-0.2 --local-dir models/kitten-tts-nano-0.2 --local-dir-use-symlinks False
```

## 配置
各脚本在文件顶部定义了全局变量来管理默认运行参数：
- `DEVICE`: 运行设备（`"cuda"` 或 `"cpu"`）
- `USE_FP16`: 是否使用半精度推理
- `OUTPUT_DIR`: 输出目录路径
- `MODEL_DIR`: 模型目录路径
- 其他模型特定配置（如 `LOAD_JIT`、`CFG_PATH` 等）

运行脚本前请确认路径与本地模型一致。CLI 脚本支持通过 `--model_dir` 和 `--device` 参数覆盖默认配置。例如：
```bash
python playground/voxcpm/infer_voxcpm_cli.py --text "测试" --model_dir models/VoxCPM-0.5B --device cuda
```

## 快速体验
各脚本默认在项目根目录执行，并将结果保存到 `outputs/`（或你在配置中设定的目录）。

### VoxCPM
- 单句示例：`python playground/voxcpm/infer_voxcpm.py`
- 命令行工具：
  ```bash
  python playground/voxcpm/infer_voxcpm_cli.py --text "八百标兵奔北坡，炮兵并排北边跑。" --output outputs/voxcpm.wav
  ```
- 多进程批量推理：`python playground/voxcpm/infer_voxcpm_parallel.py`（根据 GPU 数量调整 `MODEL_COUNT` 和 `CUDA_VISIBLE_DEVICES`）
- Gradio Web UI：`python demo/voxcpm/demo_webui.py`

### CosyVoice2
- 脚本示例：`python playground/cosyvoice/infer_cosyvoice.py`
- CLI：
  ```bash
  python playground/cosyvoice/infer_cosyvoice_cli.py --mode zero_shot --text "收到好友从远方寄来的生日礼物..." --output outputs/cosyvoice.wav
  ```
- Gradio Web UI：`python demo/cosyvoice/demo_webui_cosyvoice2.py`（支持零样本、跨语言、指令模式切换）
- 其它 demo：`demo/cosyvoice/demo_cosyvoice2.py`、`demo/cosyvoice/inference.py`

### IndexTTS-2
- 快速生成：`python playground/indextts/infer_indextts.py`
- CLI：
  ```bash
  python playground/indextts/infer_indextts_cli.py --text "大家好，我是 AI 语音合成系统" --output outputs/indextts.wav
  ```
  支持情感音频、情感向量与文本情感引导参数。

### Kitten TTS Nano
- 查看可用音色：`python demo/kittentts/list_voices.py`
- 合成示例：`python demo/kittentts/demo_kittentts.py`
  （ONNX 推理，可在 CPU 上快速体验。）

### 批量与 GUI 工具
- 音色批处理 GUI：`python BatchGenerate/voice_batch_synthesis_gui.py`
- 随机音色克隆 GUI：`python BatchGenerate/batch_random_clone_gui.py`
- 音色注册管理：`python BatchGenerate/voice_register_manager_gui.py`

## 常见问题
- **缺少 `huggingface-cli` 或 `modelscope`**：分别执行 `pip install huggingface-hub` 或 `pip install modelscope`。
- **提示找不到模型文件**：检查脚本中的 `MODEL_DIR` 等全局变量是否与实际目录匹配，或使用 CLI 参数 `--model_dir` 指定路径。
- **CPU 推理速度慢**：VoxCPM、CosyVoice2、IndexTTS-2 推荐使用 GPU；Kitten TTS 可作为纯 CPU 方案。
- **首次运行耗时长**：模型会在第一次调用时加载到显存/内存，请耐心等待。

本项目整合了以下开源工作，具体协议请参阅各自仓库：
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [VoxCPM](https://github.com/OpenBMB/VoxCPM)
- [IndexTTS](https://github.com/index-tts/index-tts)
- [Kitten TTS](https://github.com/KittenML/KittenTTS)
