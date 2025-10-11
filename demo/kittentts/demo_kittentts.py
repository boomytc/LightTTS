import os

# 设置模型下载路径 - 必须在导入任何huggingface相关库之前设置
model_cache_dir = "/Users/boom/Model/TTS/kitten-tts-nano-0.1"
os.makedirs(model_cache_dir, exist_ok=True)
os.environ["HF_HOME"] = model_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = model_cache_dir

# 现在导入其他库
from kittentts import KittenTTS
import soundfile as sf

model = KittenTTS("KittenML/kitten-tts-nano-0.1")

audio = model.generate("fuck! what are you talking about?", voice='expr-voice-2-f' )

# 可用的语音
# available_voices = [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# 保存音频
os.makedirs('output', exist_ok=True)
sf.write('output/output.wav', audio, 24000)