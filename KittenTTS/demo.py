import os
from kittentts import KittenTTS
import soundfile as sf

# 设置模型下载路径
model_cache_dir = "/Users/boom/Model/TTS/kitten-tts-nano-0.1"
os.makedirs(model_cache_dir, exist_ok=True)
os.environ["HF_HOME"] = model_cache_dir

model = KittenTTS("KittenML/kitten-tts-nano-0.1")

audio = model.generate("hello everyone this is kitten tts without gpu", voice='expr-voice-2-f' )

# 可用的语音
# available_voices = [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# 保存音频
sf.write('output.wav', audio, 24000)
