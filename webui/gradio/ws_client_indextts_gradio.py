import os
import sys
import asyncio
import json
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import websockets
import io
import torchaudio
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

SERVER_URI = "ws://127.0.0.1:8770"
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")

EMO_LABELS = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]


async def tts_stream(
    text: str,
    prompt_audio_path: str,
    emo_mode: str,
    emo_audio_path: str,
    emo_alpha: float,
    emo_happy: float,
    emo_angry: float,
    emo_sad: float,
    emo_afraid: float,
    emo_disgusted: float,
    emo_melancholic: float,
    emo_surprised: float,
    emo_calm: float,
    emo_text: str,
    interval_silence: int,
    max_tokens: int,
    use_random: bool,
    server_uri: str,
):
    """通过 WebSocket 流式合成语音"""
    text = (text or "").strip()
    emo_text = (emo_text or "").strip()

    if not text:
        return None, "请输入待合成文本。"

    if emo_mode == "情感参考音频" and not emo_audio_path:
        return None, "情感参考音频模式需要上传情感音频。"

    if emo_mode == "情感文本引导" and not emo_text:
        return None, "情感文本引导模式需要输入引导文本。"

    # 准备请求
    req = {
        "type": "tts_stream",
        "text": text,
        "prompt_audio": prompt_audio_path or DEFAULT_PROMPT_WAV,
        "emo_mode": "none",
        "interval_silence": int(interval_silence),
        "max_tokens": int(max_tokens),
        "use_random": use_random,
    }

    if emo_mode == "情感参考音频":
        req["emo_mode"] = "audio"
        req["emo_audio"] = emo_audio_path
        req["emo_alpha"] = float(emo_alpha)
    elif emo_mode == "情感向量":
        req["emo_mode"] = "vector"
        req["emo_vector"] = [
            emo_happy, emo_angry, emo_sad, emo_afraid,
            emo_disgusted, emo_melancholic, emo_surprised, emo_calm
        ]
    elif emo_mode == "情感文本引导":
        req["emo_mode"] = "text"
        req["emo_text"] = emo_text
        req["emo_alpha"] = float(emo_alpha)

    try:
        audio_segments = []
        sample_rate = 22050  # 默认采样率
        async with websockets.connect(server_uri, max_size=None) as ws:
            # 接收欢迎消息
            welcome = await ws.recv()
            
            # 发送请求
            await ws.send(json.dumps(req, ensure_ascii=False))
            
            # 接收流式音频
            async for msg in ws:
                if isinstance(msg, bytes):
                    # 音频帧
                    buffer = io.BytesIO(msg)
                    waveform, sr = torchaudio.load(buffer)
                    sample_rate = sr
                    audio_segments.append(waveform.squeeze(0).numpy())
                else:
                    # 控制消息
                    data = json.loads(msg)
                    if data.get("type") == "end":
                        break
                    elif data.get("status") == "error":
                        return None, f"❌ 出错：{data.get('message')}"
        
        if not audio_segments:
            return None, "生成结果为空。"
        
        # 合并所有音频片段
        full_audio = np.concatenate(audio_segments)
        return (sample_rate, full_audio), f"✅ 生成完成（{len(audio_segments)} 个片段）"
    
    except Exception as e:
        return None, f"❌ 连接失败：{str(e)}"


def generate_speech(
    text: str,
    prompt_audio_path: str,
    emo_mode: str,
    emo_audio_path: str,
    emo_alpha: float,
    emo_happy: float,
    emo_angry: float,
    emo_sad: float,
    emo_afraid: float,
    emo_disgusted: float,
    emo_melancholic: float,
    emo_surprised: float,
    emo_calm: float,
    emo_text: str,
    interval_silence: int,
    max_tokens: int,
    use_random: bool,
    server_uri: str,
):
    """同步包装器"""
    return asyncio.run(tts_stream(
        text, prompt_audio_path, emo_mode, emo_audio_path, emo_alpha,
        emo_happy, emo_angry, emo_sad, emo_afraid,
        emo_disgusted, emo_melancholic, emo_surprised, emo_calm,
        emo_text, interval_silence, max_tokens, use_random, server_uri
    ))


def stop_generation_message():
    return "生成已停止。"


def update_ui_visibility(emo_mode: str):
    """根据情感模式更新UI可见性"""
    if emo_mode == "无情感控制":
        return (
            gr.update(visible=False),  # emo_audio
            gr.update(visible=False),  # emo_alpha
            gr.update(visible=False),  # emo_vector_group
            gr.update(visible=False),  # emo_text
        )
    elif emo_mode == "情感参考音频":
        return (
            gr.update(visible=True),   # emo_audio
            gr.update(visible=True),   # emo_alpha
            gr.update(visible=False),  # emo_vector_group
            gr.update(visible=False),  # emo_text
        )
    elif emo_mode == "情感向量":
        return (
            gr.update(visible=False),  # emo_audio
            gr.update(visible=False),  # emo_alpha
            gr.update(visible=True),   # emo_vector_group
            gr.update(visible=False),  # emo_text
        )
    elif emo_mode == "情感文本引导":
        return (
            gr.update(visible=False),  # emo_audio
            gr.update(visible=True),   # emo_alpha
            gr.update(visible=False),  # emo_vector_group
            gr.update(visible=True),   # emo_text
        )
    return gr.update(), gr.update(), gr.update(), gr.update()


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="IndexTTS WebSocket 客户端") as demo:
        gr.Markdown(
            """
            # IndexTTS WebSocket 流式语音合成客户端
            连接到 IndexTTS WebSocket 服务器进行实时流式语音合成，支持多种情感控制。
            """
        )

        with gr.Row():
            # 左侧：控制面板
            with gr.Column(scale=1):
                gr.Markdown("### 控制面板")
                
                server_uri = gr.Textbox(
                    label="服务器地址",
                    value=SERVER_URI,
                    info="WebSocket 服务器 URI",
                )

                text = gr.Textbox(
                    label="待合成文本",
                    lines=4,
                    placeholder="请输入需要合成的文本",
                )

                prompt_audio = gr.Audio(
                    label="说话人参考音频",
                    sources=["upload"],
                    type="filepath",
                    value=DEFAULT_PROMPT_WAV if os.path.isfile(DEFAULT_PROMPT_WAV) else None,
                )

                emo_mode = gr.Radio(
                    choices=["无情感控制", "情感参考音频", "情感向量", "情感文本引导"],
                    value="无情感控制",
                    label="情感控制模式",
                )

                emo_audio = gr.Audio(
                    label="情感参考音频（用于控制语音情感）",
                    sources=["upload"],
                    type="filepath",
                    visible=False,
                )

                emo_alpha = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.05,
                    label="情感权重（影响强度 0.0-1.0）",
                    visible=False,
                )

                with gr.Group(visible=False) as emo_vector_group:
                    gr.Markdown("#### 情感向量控制")
                    emo_sliders = []
                    for label in EMO_LABELS:
                        slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                            label=label,
                        )
                        emo_sliders.append(slider)

                emo_text = gr.Textbox(
                    label="情感引导文本（根据文本自动分析情感）",
                    lines=2,
                    placeholder="示例: 我太开心了！",
                    visible=False,
                )

                with gr.Accordion("高级选项", open=False):
                    interval_silence = gr.Slider(
                        minimum=0,
                        maximum=1000,
                        value=200,
                        step=50,
                        label="句子间静音时长 (ms)",
                    )

                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=200,
                        value=120,
                        step=10,
                        label="每段最大 Token 数",
                    )

                    use_random = gr.Checkbox(
                        label="启用随机性（关闭以保证生成一致性）",
                        value=False,
                    )

                with gr.Row():
                    generate_button = gr.Button("生成语音", variant="primary")
                    stop_button = gr.Button("停止生成", variant="stop")

            # 右侧：输出面板
            with gr.Column(scale=1):
                gr.Markdown("### 输出结果")

                status_output = gr.Textbox(
                    label="状态",
                    interactive=False,
                    lines=2,
                )

                audio_output = gr.Audio(
                    label="生成语音",
                    type="numpy",
                    autoplay=False,
                )

        emo_mode.change(
            fn=update_ui_visibility,
            inputs=[emo_mode],
            outputs=[emo_audio, emo_alpha, emo_vector_group, emo_text],
        )

        generate_event = generate_button.click(
            fn=generate_speech,
            inputs=[
                text,
                prompt_audio,
                emo_mode,
                emo_audio,
                emo_alpha,
                *emo_sliders,
                emo_text,
                interval_silence,
                max_tokens,
                use_random,
                server_uri,
            ],
            outputs=[audio_output, status_output],
        )

        stop_button.click(
            fn=stop_generation_message,
            inputs=[],
            outputs=[status_output],
            cancels=[generate_event],
        )

        return demo


demo = build_interface()

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, server_name="0.0.0.0", server_port=7862)
