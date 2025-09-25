from indextts.infer_v2 import IndexTTS2

MODEL_DIR = "checkpoints"
CFG_PATH = f"{MODEL_DIR}/config.yaml"
DEVICE = "cpu"  # "cuda" or "cpu"

prompt_wav="demo/examples/诗朗诵_面朝大海春暖花开_一句话.wav"
tts = IndexTTS2(cfg_path=CFG_PATH, model_dir=MODEL_DIR, use_fp16=True, device=DEVICE, use_cuda_kernel=False)

# 并行推理测试
text="亲爱的伙伴们，大家好！每一次的努力都是为了更好的未来，要善于从失败中汲取经验，让我们一起勇敢前行,迈向更加美好的明天！"
tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)

text="The weather is really nice today, perfect for studying at home.Thank you!"
tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)

text='''叶远随口答应一声，一定帮忙云云。
教授看叶远的样子也知道，这事情多半是黄了。
谁得到这样的东西也不会轻易贡献出来，这是很大的一笔财富。
叶远回来后，又自己做了几次试验，发现空间湖水对一些外伤也有很大的帮助。
找来一只断了腿的兔子，喝下空间湖水，一天时间，兔子就完全好了。
还想多做几次试验，可是身边没有试验的对象，就先放到一边，了解空间湖水可以饮用，而且对人有利，这些就足够了。
感谢您的收听，下期再见！
'''.replace("\n", "")
tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path=f"outputs/{text[:20]}.wav", verbose=True)