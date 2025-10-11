import os
import sys

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

kittentts_dir = os.path.join(project_root, 'kittentts')
sys.path.insert(0, kittentts_dir)

import yaml
from kittentts.onnx_model import KittenTTS_Onnx

with open("config/load.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

config_default = config["default"]
config_kittentts = config["models"]["kittentts"]

model = KittenTTS_Onnx(
    model_path=config_kittentts["model_path"],
    voices_path=config_kittentts["voices_path"]
)

print("可用音色:", model.available_voices)
