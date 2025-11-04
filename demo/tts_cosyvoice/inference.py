import os
import sys
import torchaudio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
matcha_path = os.path.join(project_root, 'Matcha-TTS')
sys.path.insert(0, project_root)
sys.path.insert(0, matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 全局配置变量
DEVICE = "cuda"
USE_FP16 = True
OUTPUT_DIR = "outputs"
MODEL_DIR = "models/CosyVoice2-0.5B"
LOAD_JIT = False
LOAD_TRT = False
LOAD_VLLM = False
TRT_CONCURRENT = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

tts_model = CosyVoice2(
    model_dir=MODEL_DIR,
    load_jit=LOAD_JIT,
    load_trt=LOAD_TRT,
    load_vllm=LOAD_VLLM,
    fp16=USE_FP16,
    trt_concurrent=TRT_CONCURRENT,
)

prompt_audio = load_wav('./asset/zero_shot_prompt.wav', 16000)
prompt_text = '希望你以后能够做的比我还好呦。'

tts_text1 = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
tts_text2 = '在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。'

# zero_shot usage
result_zero_shot = tts_model.inference_zero_shot(tts_text1, prompt_text, prompt_audio, stream=False, speed=1.0, text_frontend=True)

# fine_grained_control usage
result_fine_grained_control = tts_model.inference_cross_lingual(tts_text2, prompt_audio, stream=False, speed=1.0, text_frontend=True)

# instruct usage
result_instruct = tts_model.inference_instruct2(tts_text1, '用四川话说这句话', prompt_audio, stream=False, speed=1.0, text_frontend=True)

result_list = [
    ('zero_shot', result_zero_shot),
    ('fine_grained_control', result_fine_grained_control),
    ('instruct', result_instruct)
]

for label, result in result_list:
    for i, j in enumerate(result):
        if (i > 0):
            torchaudio.save('outputs/{}_{}.wav'.format(label, i), j['tts_speech'], tts_model.sample_rate)
        else:
            torchaudio.save('outputs/{}.wav'.format(label), j['tts_speech'], tts_model.sample_rate)