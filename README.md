# LightTTS

基于 CosyVoice 精简开发的语音合成系统。

## 📖 项目简介

LightTTS 是一个基于 CosyVoice 的轻量级语音合成系统，提供高质量的文本转语音功能。该项目简化了原始 CosyVoice 的复杂性，使其更易于部署和使用。

## ✨ 主要特性

- 🎯 基于 CosyVoice2-0.5B 模型
- 🚀 简化的部署流程
- 🌐 Web UI 界面
- 🔧 支持 macOS、Linux、Windows
- 📦 轻量化设计

## 🛠️ 系统要求

- Python 3.10
- Conda 环境管理器
- macOS、Linux、Windows 操作系统

## 📦 安装说明

### 1. 克隆项目

```bash
git clone https://github.com/boomytc/LightTTS.git
cd LightTTS
```

# 如果使用ubuntu，请运行
```bash
sudo apt install sox libsox-dev
```

### 2. 创建虚拟环境

```bash
conda create -n LightTTS python=3.10 -y
conda activate LightTTS
```

### 3. 安装依赖

# 查看torch和torchaudio的安装命令请访问查看：[PyTorch官网](https://pytorch.org/get-started/locally/)
```bash
# 安装 torch torchaudio 基本命令如下：
pip install torch torchaudio

# 安装 Python 依赖包
# 如使用 Windows，请运行
pip install -r requirements_win.txt
# 如使用 ubuntu，请运行
pip install -r requirements_linux.txt
# 如使用 mac，请运行
pip install -r requirements_mac.txt
```

# 如果是在ubuntu上，可以选择使用ttsfrd来替代WeTextProcessing
```bash
modelscope download --model iic/CosyVoice-ttsfrd --local_dir ./pretrained_models/CosyVoice-ttsfrd

cd pretrained_models/CosyVoice-ttsfrd/

unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

### 4. 下载预训练模型

```bash
modelscope download --model iic/CosyVoice2-0.5B --local_dir ./pretrained_models/CosyVoice2-0.5B
```

## 🚀 快速开始

### 使用脚本启动（推荐）

```bash
# 使用自动化脚本启动
./run_webui.sh
```

### 手动启动

```bash
# 激活环境
conda activate LightTTS

# 启动 Web UI
python webui_cosyvoice2.py
```

### 使用演示脚本

```bash
# 运行演示
python demo_cosyvoice2.py
```

## 📁 项目结构

```
LightTTS/
├── cosyvoice/              # 核心语音合成模块
├── third_party/            # 第三方依赖
│   └── Matcha-TTS/        # Matcha-TTS 集成
├── pretrained_models/      # 预训练模型目录
├── webui_cosyvoice2.py    # Web UI 主程序
├── demo_cosyvoice2.py     # 演示脚本
├── run_webui.sh           # 启动脚本
├── requirements_mac.txt   # 依赖包列表
└── README.md              # 项目说明
```

## 🔧 配置说明

项目会自动设置必要的环境变量：

```bash
export PYTHONPATH=third_party/Matcha-TTS
```

## 📝 使用说明

1. 启动 Web UI 后，在浏览器中访问显示的本地地址
2. 在文本框中输入要合成的文字
3. 选择合适的语音参数
4. 点击生成按钮获取语音文件

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目。

## 📄 许可证

本项目基于开源许可证发布，具体请查看 LICENSE 文件。

## 🙏 致谢

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 原始语音合成模型
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) - 语音合成技术支持