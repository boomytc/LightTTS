import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
matcha_path = os.path.join(project_root, 'Matcha-TTS')
sys.path.insert(0, project_root)
sys.path.insert(0, matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

os.makedirs('outputs', exist_ok=True)

cosyvoice = CosyVoice2('models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
prompt_text = '希望你以后能够做的比我还好呦。'

text1 = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
text2 = '在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。'

# zero_shot usage
result_zero_shot = cosyvoice.inference_zero_shot(text1, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True)

# fine_grained_control usage
result_fine_grained_control = cosyvoice.inference_cross_lingual(text2, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True)

# instruct usage
result_instruct = cosyvoice.inference_instruct2(text1, '用四川话说这句话', prompt_speech_16k, stream=False, speed=1.0, text_frontend=True)

result_list = [
    ('zero_shot', result_zero_shot),
    ('fine_grained_control', result_fine_grained_control),
    ('instruct', result_instruct)
]

for label, result in result_list:
    for i, j in enumerate(result):
        if (i > 0):
            torchaudio.save('outputs/{}_{}.wav'.format(label, i), j['tts_speech'], cosyvoice.sample_rate)
        else:
            torchaudio.save('outputs/{}.wav'.format(label), j['tts_speech'], cosyvoice.sample_rate)
