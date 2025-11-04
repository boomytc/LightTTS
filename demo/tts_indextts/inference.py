import os
from indextts.infer_v2 import IndexTTS2

# 全局配置变量
DEVICE = "cuda"
USE_CUDA_KERNEL = True
USE_FP16 = True
OUTPUT_DIR = "outputs"
MODEL_DIR = "models/IndexTTS-2"
CFG_PATH = "models/IndexTTS-2/config.yaml"

os.makedirs(OUTPUT_DIR, exist_ok=True)

tts = IndexTTS2(
    cfg_path=CFG_PATH,
    model_dir=MODEL_DIR,
    use_fp16=USE_FP16,
    device=DEVICE,
    use_cuda_kernel=USE_CUDA_KERNEL
)

prompt_wav = "asset/zero_shot_prompt.wav"

# 单音频推理测试
text_list = [
    ("文字混拼音", "晕 XUAN4 是 一 种 GAN3 觉"),
    ("中混英", "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！"),
    ("英文", "There is a vehicle arriving in dock number 7?"),
    ("中英混引号", '"我爱你！"的英语是"I love you!"'),
    ("中英", "Joseph Gordon-Levitt is an American actor，约瑟夫·高登-莱维特是美国演员"),
    ("鼓励", "亲爱的伙伴们，大家好！每一次的努力都是为了更好的未来，要善于从失败中汲取经验，让我们一起勇敢前行,迈向更加美好的明天！"),
    ("天气", "The weather is really nice today, perfect for studying at home.Thank you!"),
    ("蒂姆·库克", "蒂莫西·唐纳德·库克（英文名：Timothy Donald Cook），通称蒂姆·库克（Tim Cook），现任苹果公司首席执行官。"),
    ("长文本《盗梦空间》介绍", "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗·诺兰执导并编剧，莱昂纳多·迪卡普里奥、玛丽昂·歌迪亚、约瑟夫·高登-莱维特、艾利奥特·佩吉、汤姆·哈迪等联袂主演，2010年7月16日在美国上映，2010年9月1日在中国内地上映，2020年8月28日在中国内地重映。影片剧情游走于梦境与现实之间，被定义为“发生在意识结构内的当代动作科幻片”，讲述了由莱昂纳多·迪卡普里奥扮演的造梦师，带领特工团队进入他人梦境，从他人的潜意识中盗取机密，并重塑他人梦境的故事。"),
    ("长文本博客", '''叶远随口答应一声，一定帮忙云云。
教授看叶远的样子也知道，这事情多半是黄了。
谁得到这样的东西也不会轻易贡献出来，这是很大的一笔财富。
叶远回来后，又自己做了几次试验，发现空间湖水对一些外伤也有很大的帮助。
找来一只断了腿的兔子，喝下空间湖水，一天时间，兔子就完全好了。
还想多做几次试验，可是身边没有试验的对象，就先放到一边，了解空间湖水可以饮用，而且对人有利，这些就足够了。
感谢您的收听，下期再见！
''')
]

for title, text in text_list:
    text = text.replace("\n", "")
    tts.infer(
        spk_audio_prompt=prompt_wav, 
        text=text, 
        output_path=f"{OUTPUT_DIR}/{title}.wav", 
        emo_audio_prompt=None,                      #情感参考音频
        emo_alpha=1,                                #情感参考音频权重(0.0-1.0)     
        # emo_vector=[0, 0, 0, 0, 0, 0, 0.45, 0],     #情感向量([happy, angry, sad, afraid, disgusted, melancholic, surprised, calm])
        # use_emo_text=True,                          #根据text引导情感,建议将 emo_alpha 设置在 0.6以下
        # emo_text="你吓死我了！你是鬼吗？",              #引导情感的文本
        use_random=False,
        interval_silence=200,
        max_text_tokens_per_segment=120,
        verbose=False
    )