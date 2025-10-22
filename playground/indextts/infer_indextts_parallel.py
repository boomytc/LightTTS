import os
import torch
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from indextts.infer_v2 import IndexTTS2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 全局配置
MODEL_DIR = 'models/IndexTTS-2'
CFG_PATH = 'models/IndexTTS-2/config.yaml'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# CPU 不支持 FP16 和 CUDA kernel，自动降级
USE_FP16 = True if DEVICE == 'cuda' else False
USE_CUDA_KERNEL = True if DEVICE == 'cuda' else False
OUTPUT_DIR = 'outputs'
PROMPT_WAV = 'asset/zero_shot_prompt.wav'
MODEL_COUNT = 2  # 进程数量

# 推理文本列表
text_list = [
    ("文字混拼音", "晕 XUAN4 是 一 种 GAN3 觉"),
    ("中混英", "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！"),
    ("英文", "There is a vehicle arriving in dock number 7?"),
    ("中英混引号", '"我爱你！"的英语是"I love you!"'),
    ("中英", "Joseph Gordon-Levitt is an American actor，约瑟夫·高登-莱维特是美国演员"),
    ("鼓励", "亲爱的伙伴们，大家好！每一次的努力都是为了更好的未来，要善于从失败中汲取经验，让我们一起勇敢前行,迈向更加美好的明天！"),
    ("天气", "The weather is really nice today, perfect for studying at home.Thank you!"),
    ("蒂姆·库克", "蒂莫西·唐纳德·库克（英文名：Timothy Donald Cook），通称蒂姆·库克（Tim Cook），现任苹果公司首席执行官。"),
    ("长文本《盗梦空间》介绍", "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗·诺兰执导并编剧，莱昂纳多·迪卡普里奥、玛丽昂·歌迪亚、约瑟夫·高登-莱维特、艾利奥特·佩吉、汤姆·哈迪等联袂主演，2010年7月16日在美国上映，2010年9月1日在中国内地上映，2020年8月28日在中国内地重映。"),
]

def inference_worker(process_idx, items):
    """推理工作进程 - 每个进程独立加载模型"""
    # 每个进程加载自己的模型实例
    model = IndexTTS2(
        cfg_path=CFG_PATH,
        model_dir=MODEL_DIR,
        use_fp16=USE_FP16,
        device=DEVICE,
        use_cuda_kernel=USE_CUDA_KERNEL
    )
    print(f"进程 {process_idx} 模型加载完成，任务数: {len(items)}，设备: {DEVICE}，FP16: {USE_FP16}")
    
    for title, raw_text in items:
        text = raw_text.replace("\n", "")
        output_path = f"{OUTPUT_DIR}/{title}.wav"
        try:
            model.infer(
                spk_audio_prompt=PROMPT_WAV,
                text=text,
                output_path=output_path,
                emo_audio_prompt=None,
                emo_alpha=1,
                use_random=False,
                interval_silence=200,
                max_text_tokens_per_segment=120,
                verbose=False,
            )
            print(f"✓ 进程 {process_idx} 生成完成: {output_path}")
        except Exception as e:
            print(f"✗ 进程 {process_idx} 生成失败: {e}")

if __name__ == "__main__":
    print(f"使用设备: {DEVICE}，FP16: {USE_FP16}，CUDA Kernel: {USE_CUDA_KERNEL}")
    if DEVICE == 'cpu':
        print("⚠️  警告：CPU 模式，FP16 和 CUDA Kernel 已自动禁用")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 任务分配：轮询分配任务到各进程
    task_groups = [[] for _ in range(MODEL_COUNT)]
    for idx, task in enumerate(text_list):
        task_groups[idx % MODEL_COUNT].append(task)
    
    print(f"启动 {MODEL_COUNT} 个并行进程...\n")
    
    # 使用 spawn 上下文避免 fork 问题
    with ProcessPoolExecutor(max_workers=MODEL_COUNT, mp_context=get_context('spawn')) as executor:
        futures = [executor.submit(inference_worker, idx, items) for idx, items in enumerate(task_groups) if items]
        for future in futures:
            future.result()
    
    print("\n✅ 所有任务完成！")
