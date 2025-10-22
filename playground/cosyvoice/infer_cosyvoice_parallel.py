import os, sys
import torchaudio
import torch
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 全局配置
MODEL_DIR = 'models/CosyVoice2-0.5B'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_FP16 = True
OUTPUT_DIR = 'outputs'
MODEL_COUNT = 3  # 进程数量


# 测试数据
PROMPT_AUDIO_PATH = './asset/zero_shot_prompt.wav'
PROMPT_TEXT = '希望你以后能够做的比我还好呦。'
TTS_TEXT1 = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
TTS_TEXT2 = '在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。'
INSTRUCT_TEXT = '用四川话说这句话'

# 任务列表：(task_type, text, prompt_text, instruct_text)
task_list = [
    ('zero_shot', TTS_TEXT1, PROMPT_TEXT, None),
    ('cross_lingual', TTS_TEXT2, None, None),
    ('instruct', TTS_TEXT1, None, INSTRUCT_TEXT),
]

def inference_worker(process_idx, task_type, text, prompt_text, instruct_text):
    """推理工作进程 - 每个进程独立加载模型"""
    # 每个进程加载自己的模型实例
    model = CosyVoice2(
        model_dir=MODEL_DIR,
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=USE_FP16,
        trt_concurrent=1,
        device=DEVICE,
    )
    print(f"进程 {process_idx} 模型加载完成，任务: {task_type}，设备: {DEVICE}")
    
    # 加载 prompt 音频
    prompt_audio = load_wav(PROMPT_AUDIO_PATH, 16000)
    
    try:
        if task_type == 'zero_shot':
            result = model.inference_zero_shot(text, prompt_text, prompt_audio, stream=False, speed=1.0, text_frontend=True)
        elif task_type == 'cross_lingual':
            result = model.inference_cross_lingual(text, prompt_audio, stream=False, speed=1.0, text_frontend=True)
        elif task_type == 'instruct':
            result = model.inference_instruct2(text, instruct_text, prompt_audio, stream=False, speed=1.0, text_frontend=True)
        
        for i, output in enumerate(result):
            filename = f'{OUTPUT_DIR}/{task_type}_process{process_idx}' + (f'_{i}.wav' if i > 0 else '.wav')
            torchaudio.save(filename, output['tts_speech'], model.sample_rate)
            print(f"✓ 进程 {process_idx} - {task_type}: {filename}")
    except Exception as e:
        print(f"✗ 进程 {process_idx} 推理失败: {e}")

if __name__ == "__main__":
    print(f"使用设备: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"启动 {MODEL_COUNT} 个并行进程...\n")
    
    # 使用 spawn 上下文避免 fork 问题
    with ProcessPoolExecutor(max_workers=MODEL_COUNT, mp_context=get_context('spawn')) as executor:
        futures = []
        for idx, (task_type, text, prompt_text, instruct_text) in enumerate(task_list):
            future = executor.submit(inference_worker, idx, task_type, text, prompt_text, instruct_text)
            futures.append(future)
        
        for future in futures:
            future.result()
    
    print("\n✅ 所有任务完成！")