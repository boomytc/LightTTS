import os
import sys
import torch
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from voxcpm.core import VoxCPM

# 全局配置
MODEL_DIR = 'models/VoxCPM-0.5B'
SE_MODEL_DIR = 'models/speech_zipenhancer_ans_multiloss_16k_base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
OUTPUT_DIR = 'outputs'
MODEL_COUNT = 2  # 进程数量

PROMPT_WAV = 'asset/zero_shot_prompt.wav'
PROMPT_TEXT = '希望你以后能够做的比我还好呦。'

# 推理文本列表
text_list = [
    ("中混英", "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！"),
    ("英文", "There is a vehicle arriving in dock number 7?"),
    ("中英混引号", '"我爱你！"的英语是"I love you!"'),
    ("中英", "Joseph Gordon-Levitt is an American actor，约瑟夫·高登-莱维特是美国演员"),
    ("鼓励", "亲爱的伙伴们，大家好！每一次的努力都是为了更好的未来，要善于从失败中汲取经验，让我们一起勇敢前行,迈向更加美好的明天！"),
    ("天气", "The weather is really nice today, perfect for studying at home.Thank you!"),
    ("蒂姆·库克", "蒂莫西·唐纳德·库克（英文名：Timothy Donald Cook），通称蒂姆·库克（Tim Cook），现任苹果公司首席执行官。"),
    ("长文本《盗梦空间》介绍", "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗·诺兰执导并编剧，莱昂纳多·迪卡普里奥、玛丽昂·歌迪亚、约瑟夫·高登-莱维特、艾利奥特·佩吉、汤姆·哈迪等联袂主演，2010年7月16日在美国上映，2010年9月1日在中国内地上映，2020年8月28日在中国内地重映。"),
    ("长文本博客", "叶远随口答应一声，一定帮忙。教授看叶远的样子也知道，这事情多半是黄了。谁得到这样的东西也不会轻易贡献出来，这是很大的一笔财富。叶远回来后，又自己做了几次试验，发现空间湖水对一些外伤也有很大的帮助。找来一只断了腿的兔子，喝下空间湖水，一天时间，兔子就完全好了。还想多做几次试验，可是身边没有试验的对象，就先放到一边，了解空间湖水可以饮用，而且对人有利，这些就足够了。感谢您的收听，下期再见！"),
]

def inference_worker(process_idx, items):
    """推理工作进程 - 每个进程独立加载模型"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 每个进程加载自己的模型实例
    model = VoxCPM.from_pretrained(
        MODEL_DIR,
        load_denoiser=False,
        zipenhancer_model_id=SE_MODEL_DIR,
        local_files_only=True,
        device=DEVICE,
    )
    print(f"进程 {process_idx} 模型加载完成，任务数: {len(items)}")
    
    for title, text in items:
        output_path = f"{OUTPUT_DIR}/{title}.wav"
        try:
            wav = model.generate(
                text=text,
                # prompt_wav_path=PROMPT_WAV,
                # prompt_text=PROMPT_TEXT,
                prompt_wav_path=None,   # 可选：用于声音克隆的提示音频路径
                prompt_text=None,       # 可选：参考文本
                cfg_value=2.0,          # LocDiT上的LM引导，值越高对提示的遵循越好，但质量可能较差
                inference_timesteps=10, # LocDiT推理时间步数，值越高结果越好，值越低速度越快
                normalize=True,         # 启用外部文本标准化工具
                denoise=True,           # 启用外部降噪工具
                retry_badcase=True,     # 为某些糟糕情况启用重试模式（无法停止）
                retry_badcase_max_times=3, # 最大重试次数
                retry_badcase_ratio_threshold=6.0, # 糟糕情况检测的最大长度限制（简单但有效），可以为慢语速语音调整
            )
            sf.write(output_path, wav, 16000)
            print(f"✓ 进程 {process_idx} 生成完成: {output_path}")
        except Exception as e:
            print(f"✗ 进程 {process_idx} 生成失败: {e}")

if __name__ == "__main__":
    print(f"使用设备: {DEVICE}")
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
