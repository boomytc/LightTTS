import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matcha_path = os.path.join(project_root, 'Matcha-TTS')
sys.path.insert(0, project_root)
sys.path.insert(0, matcha_path)

import gradio as gr
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
 
inference_mode_list = ['零样本语音克隆', '跨语言语音合成', '精细控制合成', '指令控制合成']
stream_mode_list = [('否', False), ('是', True)]

def generate_audio(tts_text, mode_checkbox_group, prompt_text, prompt_wav, instruct_text,seed, stream, speed):
    if prompt_wav is None:
        gr.Warning("请先上传或录制prompt音频！", duration=3)
        return
    
    if torchaudio.info(prompt_wav).sample_rate < 16000:
        gr.Warning("音频采样率过低，需要至少16000Hz！", duration=3)
        return
    
    prompt_speech_16k = load_wav(prompt_wav, 16000)
    set_all_random_seed(seed)
    
    if mode_checkbox_group == '零样本语音克隆':
        if not prompt_text:
            gr.Warning("请输入prompt文本！", duration=3)
            return
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())

    elif mode_checkbox_group in ['跨语言语音合成', '精细控制合成']:
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    
    elif mode_checkbox_group == '指令控制合成':
        if not instruct_text:
            gr.Warning("请输入指令文本！", duration=3)
            return
        for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())

def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 语音合成系统")
        
        tts_text = gr.Textbox(label="合成文本", lines=2, value="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。")
        
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='推理模式', value=inference_mode_list[0])
            stream = gr.Radio(choices=stream_mode_list, label='流式推理', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="速度", minimum=0.5, maximum=2.0, step=0.1)
            seed = gr.Number(value=0, label="种子", precision=0, step=1, minimum=0, maximum=1000000)

        with gr.Row():
            prompt_wav = gr.Audio(sources=['upload', 'microphone'], type='filepath', label='音频文件', value="asset/zero_shot_prompt.wav")
        
        prompt_text = gr.Textbox(label="prompt文本", value='希望你以后能够做的比我还好呦。')
        instruct_text = gr.Textbox(label="指令文本", value='用四川话说这句话')
        
        generate_button = gr.Button("生成音频")
        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)

        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, prompt_text, prompt_wav, instruct_text,seed, stream, speed],
                              outputs=[audio_output])
    
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(inbrowser=True, server_name='127.0.0.1', server_port=8001, share=False)

if __name__ == '__main__':
    cosyvoice = CosyVoice2('models/CosyVoice2-0.5B')
    main()