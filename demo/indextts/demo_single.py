from indextts.infer_v2 import IndexTTS2

MODEL_DIR = "checkpoints"
CFG_PATH = f"{MODEL_DIR}/config.yaml"
DEVICE = "cpu"  # "cuda" or "cpu"


prompt_wav="demo/examples/诗朗诵_面朝大海春暖花开_一句话.wav"
tts = IndexTTS2(cfg_path=CFG_PATH, model_dir=MODEL_DIR, use_fp16=True, device=DEVICE, use_cuda_kernel=False)


# 单音频推理测试
text="晕 XUAN4 是 一 种 GAN3 觉"
tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)

text='大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！'
tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)

text="There is a vehicle arriving in dock number 7?"
tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)

text = '"我爱你！"的英语是"I love you!"'
tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)

text = "Joseph Gordon-Levitt is an American actor"
tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)

text = "约瑟夫·高登-莱维特是美国演员"
tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)

text = "蒂莫西·唐纳德·库克（英文名：Timothy Donald Cook），通称蒂姆·库克（Tim Cook），现任苹果公司首席执行官。"
tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path="outputs/蒂莫西·唐纳德·库克.wav", verbose=True)