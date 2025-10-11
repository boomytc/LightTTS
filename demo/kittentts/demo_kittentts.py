import os
import sys

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

kittentts_dir = os.path.join(project_root, 'kittentts')
sys.path.insert(0, kittentts_dir)

import yaml
from kittentts.onnx_model import KittenTTS_Onnx

with open("config/load.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

config_default = config["default"]
config_kittentts = config["models"]["kittentts"]

model = KittenTTS_Onnx(
    model_path=config_kittentts["model_path"],
    voices_path=config_kittentts["voices_path"]
)

output_dir = config_default["output_dir"]
os.makedirs(output_dir, exist_ok=True)

audio = model.generate_to_file(
    text="fuck! what are you talking about? This high quality TTS model works without a GPU.",
    output_path=f"{output_dir}/output.wav",
    voice='expr-voice-3-m',
    speed=1.2,
    sample_rate=24000
)