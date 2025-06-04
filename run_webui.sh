#!/bin/bash

# 设置错误时退出
set -e

# 设置PYTHONPATH
export PYTHONPATH=third_party/Matcha-TTS

# 初始化conda（确保conda命令可用）
eval "$(conda shell.bash hook)"

# 检查LightTTS虚拟环境是否存在
if conda env list | grep -q "LightTTS"; then
    echo "LightTTS 虚拟环境已存在"
    conda activate LightTTS
    pip install -r requirements_mac.txt
else
    echo "创建LightTTS虚拟环境"
    conda create -n LightTTS python=3.10 -y
    conda activate LightTTS
    conda install -y -c conda-forge pynini==2.1.5
    pip install -r requirements_mac.txt
fi

# 检查预训练模型是否存在
if [ ! -d "./pretrained_models/CosyVoice2-0.5B" ]; then
    echo "预训练模型不存在，开始下载..."
    modelscope download --model iic/CosyVoice2-0.5B --local_dir ./pretrained_models/CosyVoice2-0.5B
    echo "模型下载完成"
else
    echo "预训练模型已存在"
fi

# 启动WebUI
echo "启动WebUI..."
python webui_cosyvoice2.py