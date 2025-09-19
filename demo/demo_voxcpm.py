import os
import sys

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import soundfile as sf
from voxcpm.core import VoxCPM

# 全局变量
TTS_MODEL_PATH = "models/VoxCPM-0.5B"
SE_MODEL_PATH = "models/speech_zipenhancer_ans_multiloss_16k_base"
OUTPUT_FILE = "output.wav"
TEST_TEXT = "八百标兵奔北坡，炮兵并排北边跑。"

with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
    model = VoxCPM.from_pretrained(
        TTS_MODEL_PATH,
        zipenhancer_model_id=SE_MODEL_PATH,
        local_files_only=True,
        device="cuda",
    )

wav = model.generate(
    text=TEST_TEXT,
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

sf.write(OUTPUT_FILE, wav, 16000)
print(f"saved: {OUTPUT_FILE}")