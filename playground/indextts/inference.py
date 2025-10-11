import os
import yaml
from indextts.infer_v2 import IndexTTS2

with open("config/load.yaml", "r", encoding = "utf-8") as f:
    config = yaml.safe_load(f)

config_default = config["default"]
config_indextts = config["models"]["indextts"]

os.makedirs(config_default["output_dir"], exist_ok=True)

tts = IndexTTS2(
    cfg_path = config_indextts["cfg_path"],
    model_dir = config_indextts["model_dir"],
    use_fp16 = config_default["use_fp16"],
    device = config_default["device"],
    use_cuda_kernel = config_default["use_cuda_kernel"]
)

prompt_wav = "asset/zero_shot_prompt.wav"

# 单音频推理测试
text_list = [
    ("文字混拼音", "晕 XUAN4 是 一 种 GAN3 觉"),
    ("中混英", "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！"),
    ("英文", "There is a vehicle arriving in dock number 7?"),
    ("中英混引号", '"我爱你！"的英语是"I love you!"'),
    ("中英", "Joseph Gordon-Levitt is an American actor，约瑟夫·高登-莱维特是美国演员"),
    ("蒂姆·库克", "蒂莫西·唐纳德·库克（英文名：Timothy Donald Cook），通称蒂姆·库克（Tim Cook），现任苹果公司首席执行官。")
]

for title, text in text_list:
    tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path=f"outputs/{title}.wav", verbose=True)
