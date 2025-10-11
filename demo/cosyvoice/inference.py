import os, sys, yaml
import torchaudio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
matcha_path = os.path.join(project_root, 'Matcha-TTS')
sys.path.insert(0, project_root)
sys.path.insert(0, matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

with open("config/load.yaml", 'r') as f:
    config = yaml.safe_load(f)

config_default = config['default']
config_cosyvoice = config["models"]["cosyvoice"]

output_dir = config_default["output_dir"]
os.makedirs(output_dir, exist_ok=True)

tts_model = CosyVoice2(
    model_dir=config_cosyvoice["model_dir"],
    load_jit=config_cosyvoice["load_jit"],
    load_trt=config_cosyvoice["load_trt"],
    load_vllm=config_cosyvoice["load_vllm"],
    fp16=config_default["use_fp16"],
    trt_concurrent=config_cosyvoice["trt_concurrent"],
)

prompt_audio = load_wav('./asset/zero_shot_prompt.wav', 16000)

result = tts_model.inference_zero_shot(
    tts_text='收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    prompt_text='希望你以后能够做的比我还好呦。',
    prompt_speech_16k=prompt_audio,
    stream=False,
    speed=1.0,
    text_frontend=True,
)

for i, j in enumerate(result):
    if (i > 0):
        torchaudio.save(os.path.join(output_dir, 'zero_shot_{}.wav'.format(i)), j['tts_speech'], tts_model.sample_rate)
    else:
        torchaudio.save(os.path.join(output_dir, 'zero_shot.wav'), j['tts_speech'], tts_model.sample_rate)