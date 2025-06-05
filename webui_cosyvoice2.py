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
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['零样本语音克隆', '跨语言语音合成', '精细控制合成', '指令控制合成']
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

def generate_audio(tts_text, mode_checkbox_group, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    prompt_wav = prompt_wav_upload or prompt_wav_record
    
    if prompt_wav is None:
        yield (cosyvoice.sample_rate, default_data)
        return
    
    if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
        yield (cosyvoice.sample_rate, default_data)
        return
    
    prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
    set_all_random_seed(seed)
    
    if mode_checkbox_group == '零样本语音克隆':
        if not prompt_text:
            yield (cosyvoice.sample_rate, default_data)
            return
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    
    elif mode_checkbox_group == '跨语言语音合成':
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    
    elif mode_checkbox_group == '精细控制合成':
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    
    elif mode_checkbox_group == '指令控制合成':
        if not instruct_text:
            yield (cosyvoice.sample_rate, default_data)
            return
        for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())

def main():
    with gr.Blocks() as demo:
        gr.Markdown("### CosyVoice2 语音合成系统")
        
        tts_text = gr.Textbox(label="合成文本", lines=2, value="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。")
        
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='推理模式', value=inference_mode_list[0])
            stream = gr.Radio(choices=stream_mode_list, label='流式推理', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="速度", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column():
                seed_button = gr.Button(value="🎲")
                seed = gr.Number(value=0, label="种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='音频文件')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制音频')
        
        prompt_text = gr.Textbox(label="prompt文本", value='希望你以后能够做的比我还好呦。')
        instruct_text = gr.Textbox(label="指令文本", value='用四川话说这句话')
        
        generate_button = gr.Button("生成音频")
        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
    
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(inbrowser=True, server_name='127.0.0.1', server_port=args.port, share=True)

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
    
    cosyvoice = CosyVoice2(args.model_dir)

    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()