import sys
import os
# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
# 添加 Matcha-TTS 目录到 Python 路径
matcha_path = os.path.join(project_root, 'Matcha-TTS')
sys.path.insert(0, matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

os.makedirs('outputs', exist_ok=True)

cosyvoice = CosyVoice2('models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# zero_shot usage
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False, speed=1.0, text_frontend=True)):
    torchaudio.save('outputs/zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# 精细控制，支持的控制标记请查看 cosyvoice/tokenizer/tokenizer.py#L248
for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False, speed=1.0, text_frontend=True)):
    torchaudio.save('outputs/fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# instruct usage
for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False, speed=1.0, text_frontend=True)):
    torchaudio.save('outputs/instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
