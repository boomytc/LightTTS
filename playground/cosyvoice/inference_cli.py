import sys
import os
import argparse

# 获取项目根目录，添加 Matcha-TTS 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matcha_path = os.path.join(project_root, 'Matcha-TTS')
sys.path.insert(0, project_root)
sys.path.insert(0, matcha_path)

import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed


def main():
    parser = argparse.ArgumentParser(description='CosyVoice2 语音合成 CLI 工具')
    parser.add_argument('--model_dir', type=str, default='models/CosyVoice2-0.5B', help='模型路径')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['zero_shot', 'cross_lingual', 'instruct'],
                        help='推理模式: zero_shot(零样本克隆), cross_lingual(跨语言/精细控制), instruct(指令控制)')
    parser.add_argument('--text', type=str, required=True, help='待合成文本')
    parser.add_argument('--prompt_wav', type=str, default='asset/zero_shot_prompt.wav', help='参考音频路径')
    parser.add_argument('--prompt_text', type=str, default='希望你以后能够做的比我还好呦。', 
                        help='参考音频文本(零样本模式必需)')
    parser.add_argument('--instruct_text', type=str, default='', help='指令文本(指令模式必需)')
    parser.add_argument('--output', type=str, default='output.wav', help='输出音频路径')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--speed', type=float, default=1.0, help='语速(0.5-2.0)')
    
    args = parser.parse_args()
    
    # 检查参数
    if not os.path.exists(args.prompt_wav):
        print(f"错误: 参考音频文件不存在: {args.prompt_wav}")
        return
    
    if torchaudio.info(args.prompt_wav).sample_rate < 16000:
        print("错误: 音频采样率过低，需要至少16000Hz")
        return
    
    if args.mode == 'zero_shot' and not args.prompt_text:
        print("错误: 零样本模式需要提供 --prompt_text")
        return
    
    if args.mode == 'instruct' and not args.instruct_text:
        print("错误: 指令模式需要提供 --instruct_text")
        return
    
    # 加载模型
    print(f"正在加载模型: {args.model_dir}")
    cosyvoice = CosyVoice2(args.model_dir)
    
    # 加载音频
    prompt_speech_16k = load_wav(args.prompt_wav, 16000)
    set_all_random_seed(args.seed)
    
    # 生成音频
    print(f"开始生成音频...")
    print(f"模式: {args.mode}")
    print(f"文本: {args.text}")
    
    if args.mode == 'zero_shot':
        result = cosyvoice.inference_zero_shot(
            args.text, args.prompt_text, prompt_speech_16k, 
            stream=False, speed=args.speed
        )
    elif args.mode == 'cross_lingual':
        result = cosyvoice.inference_cross_lingual(
            args.text, prompt_speech_16k, 
            stream=False, speed=args.speed
        )
    elif args.mode == 'instruct':
        result = cosyvoice.inference_instruct2(
            args.text, args.instruct_text, prompt_speech_16k, 
            stream=False, speed=args.speed
        )
    
    # 保存音频
    for audio_data in result:
        torchaudio.save(args.output, audio_data['tts_speech'], cosyvoice.sample_rate)
        print(f"已保存音频: {args.output}")


if __name__ == '__main__':
    main()