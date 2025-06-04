import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['零样本语音克隆', '跨语言语音合成', '精细控制合成', '指令控制合成']
instruct_dict = {'零样本语音克隆': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
                 '跨语言语音合成': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
                 '精细控制合成': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 在合成文本中使用控制标记如[laughter]\n3. 点击生成音频按钮',
                 '指令控制合成': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入指令文本（如"用四川话说这句话"）\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    
    # 检查prompt音频是否提供
    if prompt_wav is None:
        gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
        yield (cosyvoice.sample_rate, default_data)
        return
    
    # 检查音频采样率
    if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
        gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
        yield (cosyvoice.sample_rate, default_data)
        return
    
    # 根据不同模式进行推理
    if mode_checkbox_group == '零样本语音克隆':
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            yield (cosyvoice.sample_rate, default_data)
            return
        if instruct_text != '':
            gr.Info('您正在使用零样本语音克隆模式，指令文本会被忽略！')
        
        logging.info('get zero_shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    
    elif mode_checkbox_group == '跨语言语音合成':
        if instruct_text != '':
            gr.Info('您正在使用跨语言语音合成模式，指令文本会被忽略！')
        gr.Info('您正在使用跨语言语音合成模式，请确保合成文本和prompt音频为不同语言')
        
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    
    elif mode_checkbox_group == '精细控制合成':
        if instruct_text != '':
            gr.Info('您正在使用精细控制合成模式，指令文本会被忽略！')
        gr.Info('您正在使用精细控制合成模式，请在合成文本中使用控制标记如[laughter]')
        
        logging.info('get fine_grained_control inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    
    elif mode_checkbox_group == '指令控制合成':
        if instruct_text == '':
            gr.Warning('您正在使用指令控制合成模式，请输入指令文本')
            yield (cosyvoice.sample_rate, default_data)
            return
        
        logging.info('get instruct2 inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### CosyVoice2 语音合成系统 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \\\n                    预训练模型 [CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B)")
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")
        gr.Markdown("#### 注意：如果您想要重现官方演示结果，请在推理时添加 text_frontend=False 参数")

        tts_text = gr.Textbox(label="输入合成文本", lines=2, value="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
        prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致...", value='希望你以后能够做的比我还好呦。')
        instruct_text = gr.Textbox(label="输入指令文本", lines=1, placeholder="请输入指令文本，如：用四川话说这句话", value='用四川话说这句话')

        generate_button = gr.Button("生成音频")

        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(inbrowser=True, server_name='127.0.0.1', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8001)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    
    # 只支持CosyVoice2模型
    try:
        cosyvoice = CosyVoice2(args.model_dir)
        logging.info('CosyVoice2 model loaded successfully')
    except Exception as e:
        logging.error('Failed to load CosyVoice2 model: {}'.format(e))
        raise TypeError('Failed to load CosyVoice2 model! Please make sure you are using CosyVoice2-0.5B model.')

    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()