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
    ("蒂姆·库克", "蒂莫西·唐纳德·库克（英文名：Timothy Donald Cook），通称蒂姆·库克（Tim Cook），现任苹果公司首席执行官。"),
    ("长文本《盗梦空间》介绍", "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗·诺兰执导并编剧，莱昂纳多·迪卡普里奥、玛丽昂·歌迪亚、约瑟夫·高登-莱维特、艾利奥特·佩吉、汤姆·哈迪等联袂主演，2010年7月16日在美国上映，2010年9月1日在中国内地上映，2020年8月28日在中国内地重映。影片剧情游走于梦境与现实之间，被定义为“发生在意识结构内的当代动作科幻片”，讲述了由莱昂纳多·迪卡普里奥扮演的造梦师，带领特工团队进入他人梦境，从他人的潜意识中盗取机密，并重塑他人梦境的故事。")
]

for title, text in text_list:
    text = text.replace("\n", "")
    tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path=f"outputs/{title}.wav", verbose=False)