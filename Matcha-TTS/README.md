<div align="center">

# 🍵 Matcha-TTS: 基于条件流匹配的快速 TTS 架构

### [Shivam Mehta](https://www.kth.se/profile/smehta), [Ruibo Tu](https://www.kth.se/profile/ruibo), [Jonas Beskow](https://www.kth.se/profile/beskow), [Éva Székely](https://www.kth.se/profile/szekely), and [Gustav Eje Henter](https://people.kth.se/~ghe/)

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

<p style="text-align: center;">
  <img src="https://shivammehta25.github.io/Matcha-TTS/images/logo.png" height="128"/>
</p>

</div>

> 这是 🍵 Matcha-TTS [ICASSP 2024] 的官方代码实现。

我们提出了 🍵 Matcha-TTS，一种新的非自回归神经 TTS 方法，使用[条件流匹配](https://arxiv.org/abs/2210.02747)（类似于[修正流](https://arxiv.org/abs/2209.03003)）来加速基于 ODE 的语音合成。我们的方法：

- 具有概率性
- 内存占用小
- 声音高度自然
- 合成速度非常快

查看我们的[演示页面](https://shivammehta25.github.io/Matcha-TTS)并阅读[我们的 ICASSP 2024 论文](https://arxiv.org/abs/2309.03199)了解更多详情。

[预训练模型](https://drive.google.com/drive/folders/17C_gYgEHOxI5ZypcfE_k1piKCtyR0isJ?usp=sharing)将通过 CLI 或 gradio 界面自动下载。

你也可以[在 HuggingFace 🤗 spaces 的浏览器中试用 🍵 Matcha-TTS](https://huggingface.co/spaces/shivammehta25/Matcha-TTS)。

## 演示视频

[![观看视频](https://img.youtube.com/vi/xmvJkz3bqw0/hqdefault.jpg)](https://youtu.be/xmvJkz3bqw0)

## 安装

1. 创建环境（建议但可选）

```
conda create -n matcha-tts python=3.10 -y
conda activate matcha-tts
```

2. 使用 pip 或从源码安装 Matcha TTS

```bash
pip install matcha-tts
```

从源码安装

```bash
pip install git+https://github.com/shivammehta25/Matcha-TTS.git
cd Matcha-TTS
pip install -e .
```

3. 运行 CLI / gradio 应用 / jupyter notebook

```bash
# 这将下载所需的模型
matcha-tts --text "<输入文本>"
```

或

```bash
matcha-tts-app
```

或在 jupyter notebook 中打开 `synthesis.ipynb`

### CLI 参数

- 从给定文本合成，运行：

```bash
matcha-tts --text "<输入文本>"
```

- 从文件合成，运行：

```bash
matcha-tts --file <文件路径>
```

- 从文件批量合成，运行：

```bash
matcha-tts --file <文件路径> --batched
```

附加参数

- 语速

```bash
matcha-tts --text "<输入文本>" --speaking_rate 1.0
```

- 采样温度

```bash
matcha-tts --text "<输入文本>" --temperature 0.667
```

- Euler ODE 求解器步数

```bash
matcha-tts --text "<输入文本>" --steps 10
```

## 使用自己的数据集训练

假设我们使用 LJ Speech 进行训练

1. 从[这里](https://keithito.com/LJ-Speech-Dataset/)下载数据集，解压到 `data/LJSpeech-1.1`，并准备文件列表指向解压的数据，参考 [NVIDIA Tacotron 2 仓库设置的第 5 项](https://github.com/NVIDIA/tacotron2#setup)。

2. 克隆并进入 Matcha-TTS 仓库

```bash
git clone https://github.com/shivammehta25/Matcha-TTS.git
cd Matcha-TTS
```

3. 从源码安装包

```bash
pip install -e .
```

4. 进入 `configs/data/ljspeech.yaml` 并修改

```yaml
train_filelist_path: data/filelists/ljs_audio_text_train_filelist.txt
valid_filelist_path: data/filelists/ljs_audio_text_val_filelist.txt
```

5. 使用数据集配置的 yaml 文件生成归一化统计信息

```bash
matcha-data-stats -i ljspeech.yaml
# 输出:
#{'mel_mean': -5.53662231756592, 'mel_std': 2.1161014277038574}
```

在 `configs/data/ljspeech.yaml` 的 `data_statistics` 键下更新这些值。

```bash
data_statistics:  # 为 ljspeech 数据集计算
  mel_mean: -5.536622
  mel_std: 2.116101
```

更新为你的训练和验证文件列表的路径。

6. 运行训练脚本

```bash
make train-ljspeech
```

或

```bash
python matcha/train.py experiment=ljspeech
```

- 最小内存运行

```bash
python matcha/train.py experiment=ljspeech_min_memory
```

- 多 GPU 训练，运行

```bash
python matcha/train.py experiment=ljspeech trainer.devices=[0,1]
```

7. 从自定义训练的模型合成

```bash
matcha-tts --text "<输入文本>" --checkpoint_path <检查点路径>
```

## ONNX 支持

> 特别感谢 [@mush42](https://github.com/mush42) 实现了 ONNX 导出和推理支持。

可以将 Matcha 检查点导出到 [ONNX](https://onnx.ai/)，并在导出的 ONNX 图上运行推理。

### ONNX 导出

要将检查点导出到 ONNX，首先安装 ONNX

```bash
pip install onnx
```

然后运行以下命令：

```bash
python3 -m matcha.onnx.export matcha.ckpt model.onnx --n-timesteps 5
```

可选地，ONNX 导出器接受 **vocoder-name** 和 **vocoder-checkpoint** 参数。这使你能够在导出的图中嵌入声码器，并在一次运行中生成波形（类似于端到端 TTS 系统）。

**注意** `n_timesteps` 被视为超参数而不是模型输入。这意味着你应该在导出时指定它（而不是在推理时）。如果未指定，`n_timesteps` 设置为 **5**。

**重要**：目前导出需要 torch>=2.1.0，因为 `scaled_product_attention` 操作符在旧版本中无法导出。在最终版本发布之前，想要导出模型的用户必须手动安装 torch>=2.1.0 作为预发布版本。

### ONNX 推理

要在导出的模型上运行推理，首先使用以下命令安装 `onnxruntime`

```bash
pip install onnxruntime
pip install onnxruntime-gpu  # 用于 GPU 推理
```

然后使用以下命令：

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs
```

你也可以控制合成参数：

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs --temperature 0.4 --speaking_rate 0.9 --spk 0
```

要在 **GPU** 上运行推理，确保安装 **onnxruntime-gpu** 包，然后在推理命令中传递 `--gpu`：

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs --gpu
```

如果你只将 Matcha 导出到 ONNX，这将把梅尔频谱图作为图形和 `numpy` 数组写入输出目录。
如果你在导出的图中嵌入了声码器，这将把 `.wav` 音频文件写入输出目录。

如果你只将 Matcha 导出到 ONNX，并且想要运行完整的 TTS 管道，你可以传递 `ONNX` 格式的声码器模型路径：

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs --vocoder hifigan.small.onnx
```

这将把 `.wav` 音频文件写入输出目录。

## 引用信息

如果你使用我们的代码或发现这项工作有用，请引用我们的论文：

```text
@inproceedings{mehta2024matcha,
  title={Matcha-{TTS}: A fast {TTS} architecture with conditional flow matching},
  author={Mehta, Shivam and Tu, Ruibo and Beskow, Jonas and Sz{\'e}kely, {\'E}va and Henter, Gustav Eje},
  booktitle={Proc. ICASSP},
  year={2024}
}
```

## 致谢

由于此代码使用了 [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)，你可以使用它附带的所有功能。

我们想要致谢的其他源代码：

- [Coqui-TTS](https://github.com/coqui-ai/TTS/tree/dev)：帮助我弄清楚如何使 cython 二进制文件可通过 pip 安装并给予鼓励
- [Hugging Face Diffusers](https://huggingface.co/)：提供了出色的 diffusers 库及其组件
- [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)：提供了单调对齐搜索源代码
- [torchdyn](https://github.com/DiffEqML/torchdyn)：在研究和开发期间尝试其他 ODE 求解器时很有用
- [labml.ai](https://nn.labml.ai/transformers/rope/index.html)：提供了 RoPE 实现
