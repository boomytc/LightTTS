import os
import sys

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

import soundfile as sf
from voxcpm.core import VoxCPM

# 全局配置变量
DEVICE = "cuda"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models/VoxCPM-0.5B"
SE_MODEL_DIR = "models/speech_zipenhancer_ans_multiloss_16k_base"
LOAD_DENOISER = False
LOCAL_FILES_ONLY = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = VoxCPM.from_pretrained(
    MODEL_DIR,
    load_denoiser=LOAD_DENOISER,
    zipenhancer_model_id=SE_MODEL_DIR,
    local_files_only=LOCAL_FILES_ONLY,
    device=DEVICE,
)

wav = model.generate(
    text="八百标兵奔北坡，炮兵并排北边跑。",
    prompt_wav_path=None,      # 可选：用于声音克隆的提示音频路径
    prompt_text=None,          # 可选：参考文本
    cfg_value=2.0,             # LocDiT上的LM引导，值越高对提示的遵循越好，但质量可能较差
    inference_timesteps=10,   # LocDiT推理时间步数，值越高结果越好，值越低速度越快
    normalize=True,           # 启用外部文本标准化工具
    denoise=True,             # 启用外部降噪工具
    retry_badcase=True,        # 为某些糟糕情况启用重试模式（无法停止）
    retry_badcase_max_times=3,  # 最大重试次数
    retry_badcase_ratio_threshold=6.0, # 糟糕情况检测的最大长度限制（简单但有效），可以为慢语速语音调整
)

output_file = f"{OUTPUT_DIR}/八百标兵奔北坡.wav"
sf.write(output_file, wav, 16000)
print(f"saved: {output_file}")