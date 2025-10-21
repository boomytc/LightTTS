import os
import sys
import argparse
import yaml

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import soundfile as sf
from voxcpm.core import VoxCPM


def main():
    parser = argparse.ArgumentParser(
        description='VoxCPM 语音合成 CLI 工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 基础合成
  %(prog)s --text "八百标兵奔北坡，炮兵并排北边跑。"
  
  # 声音克隆
  %(prog)s --text "你好，世界" --prompt_wav voice.wav --prompt_text "参考文本"
  
  # 调整生成参数
  %(prog)s --text "测试" --cfg_value 3.0 --inference_timesteps 20
        '''
    )
    
    # 基础参数
    parser.add_argument('--text', type=str, required=True, help='待合成文本')
    parser.add_argument('--output', type=str, default='output.wav', help='输出音频路径')
    
    # 声音克隆参数
    parser.add_argument('--prompt_wav', type=str, default=None, help='提示音频路径（用于声音克隆）')
    parser.add_argument('--prompt_text', type=str, default=None, help='提示音频对应的文本')
    
    # 模型配置
    parser.add_argument('--config', type=str, default='config/load.yaml', help='配置文件路径')
    parser.add_argument('--model_dir', type=str, default=None, help='模型路径（覆盖配置文件）')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], 
                        help='运行设备（覆盖配置文件）')
    
    # 生成控制参数
    gen_group = parser.add_argument_group('生成控制选项')
    gen_group.add_argument('--cfg_value', type=float, default=2.0,
                          help='LM引导值，越高对提示遵循越好，默认2.0')
    gen_group.add_argument('--inference_timesteps', type=int, default=10,
                          help='推理时间步数，越高质量越好但速度越慢，默认10')
    gen_group.add_argument('--normalize', action='store_true', default=True,
                          help='启用文本标准化（默认启用）')
    gen_group.add_argument('--no_normalize', dest='normalize', action='store_false',
                          help='禁用文本标准化')
    gen_group.add_argument('--denoise', action='store_true', default=True,
                          help='启用降噪（默认启用）')
    gen_group.add_argument('--no_denoise', dest='denoise', action='store_false',
                          help='禁用降噪')
    gen_group.add_argument('--retry_badcase', action='store_true', default=True,
                          help='启用糟糕情况重试（默认启用）')
    gen_group.add_argument('--no_retry_badcase', dest='retry_badcase', action='store_false',
                          help='禁用糟糕情况重试')
    gen_group.add_argument('--retry_max_times', type=int, default=3,
                          help='最大重试次数，默认3')
    gen_group.add_argument('--retry_ratio_threshold', type=float, default=6.0,
                          help='糟糕情况检测阈值，默认6.0')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config_default = config["default"]
    config_voxcpm = config["models"]["voxcpm"]
    
    # 覆盖配置
    model_dir = args.model_dir if args.model_dir else config_voxcpm["model_dir"]
    device = args.device if args.device else config_default["device"]
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"正在加载模型: {model_dir}")
    print(f"设备: {device}")
    
    model = VoxCPM.from_pretrained(
        model_dir,
        load_denoiser=config_voxcpm["load_denoiser"],
        zipenhancer_model_id=config_voxcpm["se_model_dir"],
        local_files_only=config_voxcpm["local_files_only"],
        device=device,
    )
    
    # 显示合成信息
    print(f"\n开始生成音频")
    print(f"文本: {args.text}")
    if args.prompt_wav:
        print(f"提示音频: {args.prompt_wav}")
        if args.prompt_text:
            print(f"提示文本: {args.prompt_text}")
    print(f"cfg_value: {args.cfg_value}")
    print(f"inference_timesteps: {args.inference_timesteps}")
    
    # 生成音频
    wav = model.generate(
        text=args.text,
        prompt_wav_path=args.prompt_wav,
        prompt_text=args.prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        normalize=args.normalize,
        denoise=args.denoise,
        retry_badcase=args.retry_badcase,
        retry_badcase_max_times=args.retry_max_times,
        retry_badcase_ratio_threshold=args.retry_ratio_threshold,
    )
    
    # 保存音频
    sf.write(args.output, wav, 16000)
    print(f"\n音频已保存: {args.output}")


if __name__ == '__main__':
    main()
