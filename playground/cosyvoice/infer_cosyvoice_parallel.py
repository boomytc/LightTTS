import os, sys
import torchaudio
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import onnxruntime as ort
ort.set_default_logger_severity(3)

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 全局配置
MODEL_DIR = 'models/CosyVoice2-0.5B'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_FP16 = True
OUTPUT_DIR = 'outputs'

print(f"使用设备: {DEVICE}")


prompt_audio = load_wav('./asset/zero_shot_prompt.wav', 16000)
prompt_text = '希望你以后能够做的比我还好呦。'
tts_text1 = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
tts_text2 = '在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型
print(f"加载 3 个模型...")
models = []
for i in range(3):
    try:
        model = CosyVoice2(
            model_dir=MODEL_DIR,
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=USE_FP16,
            trt_concurrent=1,
        )
        models.append(model)
        print(f"模型 {i} 加载完成")
    except Exception as e:
        print(f"模型 {i} 加载失败: {e}")

print(f"共加载 {len(models)} 个模型\n")

def inference_worker(model_idx, task_type, *args):
    """推理工作线程"""
    try:
        model = models[model_idx]
        if task_type == 'zero_shot':
            result = model.inference_zero_shot(*args, stream=False, speed=1.0, text_frontend=True)
        elif task_type == 'cross_lingual':
            result = model.inference_cross_lingual(*args, stream=False, speed=1.0, text_frontend=True)
        elif task_type == 'instruct':
            result = model.inference_instruct2(*args, stream=False, speed=1.0, text_frontend=True)
        
        for i, output in enumerate(result):
            filename = f'{OUTPUT_DIR}/{task_type}_model{model_idx}' + (f'_{i}.wav' if i > 0 else '.wav')
            torchaudio.save(filename, output['tts_speech'], model.sample_rate)
            print(f"✓ 模型 {model_idx} - {task_type}: {filename}")
    except Exception as e:
        print(f"✗ 模型 {model_idx} 推理失败: {e}")

# 并行执行三个推理任务
tasks = [
    (0, 'zero_shot', tts_text1, prompt_text, prompt_audio),
    (1, 'cross_lingual', tts_text2, prompt_audio),
    (2, 'instruct', tts_text1, '用四川话说这句话', prompt_audio)
]

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(inference_worker, *task) for task in tasks]
    for future in futures:
        future.result()