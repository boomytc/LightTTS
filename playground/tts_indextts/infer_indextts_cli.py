import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from indextts.infer_v2 import IndexTTS2

# 全局配置变量
DEVICE = "cuda"
USE_CUDA_KERNEL = True
USE_FP16 = True
MODEL_DIR = "models/IndexTTS-2"
CFG_PATH = "models/IndexTTS-2/config.yaml"


def parse_emo_vector(emo_str):
    """解析情感向量字符串
    格式: [0, 0, 0, 0, 0, 0, 0.45, 0]
    顺序: happy,angry,sad,afraid,disgusted,melancholic,surprised,calm
    """
    if not emo_str:
        return None
    # 移除方括号和空格
    emo_str = emo_str.strip().replace('[', '').replace(']', '')
    values = [float(x.strip()) for x in emo_str.split(',')]
    return values


def main():
    parser = argparse.ArgumentParser(
        description='IndexTTS2 语音合成 CLI 工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 基础合成
  %(prog)s --text "大家好，我是AI语音合成系统"
  
  # 指定参考音频
  %(prog)s --text "Hello World" --prompt_wav my_voice.wav --output hello.wav
  
  # 使用情感参考音频
  %(prog)s --text "太棒了！" --emo_audio_prompt happy.wav --emo_alpha 0.8
  
  # 使用情感向量（happy=0.7, surprised=0.3）
  %(prog)s --text "哇！" --emo_vector "[0.7,0,0,0,0,0,0.3,0]"
  
  # 使用情感文本引导
  %(prog)s --text "今天天气真好" --use_emo_text --emo_text "我太开心了！" --emo_alpha 0.5
        '''
    )
    
    # 基础参数
    parser.add_argument('--text', type=str, required=True, help='待合成文本')
    parser.add_argument('--prompt_wav', type=str, default='asset/zero_shot_prompt.wav', 
                        help='说话人参考音频路径')
    parser.add_argument('--output', type=str, default='output.wav', help='输出音频路径')
    
    # 模型配置
    parser.add_argument('--model_dir', type=str, default=None, help='模型路径（覆盖默认配置）')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], 
                        help='运行设备（覆盖默认配置）')
    
    # 情感控制参数
    emo_group = parser.add_argument_group('情感控制选项')
    emo_group.add_argument('--emo_audio_prompt', type=str, default=None, 
                          help='情感参考音频路径')
    emo_group.add_argument('--emo_alpha', type=float, default=1.0, 
                          help='情感参考权重 (0.0-1.0，默认1.0)')
    emo_group.add_argument('--emo_vector', type=parse_emo_vector, default=None,
                          help='情感向量: [happy,angry,sad,afraid,disgusted,melancholic,surprised,calm]，例如 "[0,0,0,0,0,0,0.45,0]"')
    emo_group.add_argument('--use_emo_text', action='store_true', 
                          help='启用情感文本引导（建议emo_alpha设为0.6以下）')
    emo_group.add_argument('--emo_text', type=str, default=None, 
                          help='引导情感的文本（需配合--use_emo_text使用）')
    
    # 生成控制参数
    gen_group = parser.add_argument_group('生成控制选项')
    gen_group.add_argument('--use_random', action='store_true', 
                          help='启用随机性（默认关闭以保证一致性）')
    gen_group.add_argument('--interval_silence', type=int, default=200, 
                          help='句子间静音时长(ms)，默认200')
    gen_group.add_argument('--max_tokens', type=int, default=120, 
                          help='每段最大token数，默认120')
    gen_group.add_argument('--verbose', action='store_true', 
                          help='显示详细信息')
    
    args = parser.parse_args()
    
    # 使用全局变量或命令行参数
    model_dir = args.model_dir if args.model_dir else MODEL_DIR
    device = args.device if args.device else DEVICE
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"正在加载模型: {model_dir}")
    print(f"设备: {device}")
    
    tts = IndexTTS2(
        cfg_path=CFG_PATH,
        model_dir=model_dir,
        use_fp16=USE_FP16,
        device=device,
        use_cuda_kernel=USE_CUDA_KERNEL
    )
    
    # 显示合成信息
    print(f"\n开始生成音频")
    print(f"文本: {args.text}")
    print(f"参考音频: {args.prompt_wav}")
    
    if args.emo_audio_prompt:
        print(f"情感参考: {args.emo_audio_prompt} (权重: {args.emo_alpha})")
    if args.emo_vector:
        emo_names = ['happy', 'angry', 'sad', 'afraid', 'disgusted', 'melancholic', 'surprised', 'calm']
        active_emo = [f"{name}={val:.2f}" for name, val in zip(emo_names, args.emo_vector) if val > 0]
        print(f"情感向量: {', '.join(active_emo)}")
    if args.use_emo_text:
        print(f"情感文本: {args.emo_text}")
    
    # 准备推理参数
    infer_kwargs = {
        'spk_audio_prompt': args.prompt_wav,
        'text': args.text.replace("\n", ""),
        'output_path': args.output,
        'emo_audio_prompt': args.emo_audio_prompt,
        'emo_alpha': args.emo_alpha,
        'use_random': args.use_random,
        'interval_silence': args.interval_silence,
        'max_text_tokens_per_segment': args.max_tokens,
        'verbose': args.verbose
    }
    
    # 添加可选参数
    if args.emo_vector:
        infer_kwargs['emo_vector'] = args.emo_vector
    
    if args.use_emo_text:
        infer_kwargs['use_emo_text'] = True
        infer_kwargs['emo_text'] = args.emo_text
    
    # 生成音频
    tts.infer(**infer_kwargs)
    print(f"\n音频已保存: {args.output}")


if __name__ == '__main__':
    main()
