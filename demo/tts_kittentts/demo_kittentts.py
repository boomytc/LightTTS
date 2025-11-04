import os
import sys

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

kittentts_dir = os.path.join(project_root, 'kittentts')
sys.path.insert(0, kittentts_dir)

from kittentts.onnx_model import KittenTTS_Onnx

# 全局配置变量
DEVICE = "cuda"
USE_FP16 = True
OUTPUT_DIR = "outputs"
MODEL_PATH = "models/kitten-tts-nano-0.2/kitten_tts_nano_v0_2.onnx"
VOICES_PATH = "models/kitten-tts-nano-0.2/voices.npz"

model = KittenTTS_Onnx(
    model_path=MODEL_PATH,
    voices_path=VOICES_PATH
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

audio = model.generate_to_file(
    text="fuck! what are you talking about? This high quality TTS model works without a GPU.",
    output_path=f"{OUTPUT_DIR}/output.wav",
    voice='expr-voice-3-m',
    speed=1.2,
    sample_rate=24000
)