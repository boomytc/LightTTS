# CosyVoice2 Flask WebUI

基于 Flask 的 CosyVoice2 语音合成 Web 界面。

## 功能特性

- ✨ 支持三种推理模式：
  - 零样本克隆：基于参考音频和文本克隆声音
  - 跨语言克隆：基于参考音频实现跨语言语音合成
  - 指令控制：通过指令控制语音风格、情感、方言等
- 🎛️ 可调节语速和随机种子
- 🖥️ 支持 CUDA 和 CPU 推理
- 📱 响应式设计，适配桌面和移动端
- 🎨 现代化 UI 设计

## 安装依赖

确保已安装项目的基础依赖，并额外安装 Flask：

```bash
pip install flask
```

## 使用方法

### 启动服务

在项目根目录下运行：

```bash
cd webui/flask/tts_cosyvoice
python app.py
```

服务将在 `http://127.0.0.1:5000` 启动。

### 使用流程

1. **加载模型**
   - 选择运行设备（CUDA/CPU）
   - 点击"加载模型"按钮

2. **配置参数**
   - 选择推理模式
   - 输入待合成文本
   - 上传参考音频（可选，默认使用预设音频）
   - 根据模式填写参考文本或指令文本
   - 调节语速和随机种子

3. **生成语音**
   - 点击"生成语音"按钮
   - 等待生成完成
   - 播放或下载生成的音频

## API 接口

### 加载模型

```
POST /api/load_model
Content-Type: application/json

{
  "device": "cuda"  // 或 "cpu"
}
```

### 生成语音

```
POST /api/generate
Content-Type: multipart/form-data

mode: "zero_shot" | "cross_lingual" | "instruct"
text: 待合成文本
prompt_text: 参考文本（零样本模式必填）
instruct_text: 指令文本（指令模式必填）
speed: 语速 (0.5-2.0)
seed: 随机种子
prompt_audio: 参考音频文件（可选）
```

### 获取默认音频

```
GET /api/default_audio
```

## 配置说明

在 `app.py` 中可以修改以下配置：

```python
DEFAULT_MODEL_DIR = "models/CosyVoice2-0.5B"  # 模型路径
DEFAULT_PROMPT_WAV = "asset/zero_shot_prompt.wav"  # 默认参考音频
USE_FP16 = True  # 是否使用 FP16（仅 CUDA）
LOAD_JIT = False  # 是否加载 JIT 模型
LOAD_TRT = False  # 是否加载 TensorRT 模型
LOAD_VLLM = False  # 是否加载 vLLM 模型
```

## 注意事项

- 参考音频采样率需至少 16000Hz
- 建议使用 CUDA 以获得更好的性能
- 首次加载模型需要较长时间，请耐心等待
- 上传文件大小限制为 50MB

## 技术栈

- **后端**: Flask
- **前端**: HTML5 + CSS3 + JavaScript
- **音频处理**: PyTorch + torchaudio
- **模型**: CosyVoice2
