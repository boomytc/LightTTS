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

SERVER_URI = "ws://127.0.0.1:8771"
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做得比我还好哟。"


async def tts_stream(
    text: str,
    prompt_audio_path: str,
    prompt_text: str,
    cfg_value: float,
    inference_timesteps: int,
    normalize: bool,
    denoise: bool,
    retry_badcase: bool,
    retry_max_times: int,
    retry_ratio_threshold: float,
    server_uri: str,
):
    """通过 WebSocket 流式合成语音"""
    text = (text or "").strip()
    prompt_text = (prompt_text or "").strip()

    if not text:
        return None, "请输入待合成文本。"

    if prompt_audio_path and not prompt_text:
        return None, "使用参考音频时，请提供对应的参考文本。"

    # 准备请求
    req = {
        "type": "tts_stream",
        "text": text,
        "prompt_audio": prompt_audio_path,
        "prompt_text": prompt_text,
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(inference_timesteps),
        "normalize": normalize,
        "denoise": denoise,
        "retry_badcase": retry_badcase,
        "retry_max_times": int(retry_max_times),
        "retry_ratio_threshold": float(retry_ratio_threshold),
    }

    try:
        audio_segments = []
        sample_rate = 16000  # 默认采样率
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
    prompt_text: str,
    cfg_value: float,
    inference_timesteps: int,
    normalize: bool,
    denoise: bool,
    retry_badcase: bool,
    retry_max_times: int,
    retry_ratio_threshold: float,
    server_uri: str,
):
    """同步包装器"""
    return asyncio.run(tts_stream(
        text, prompt_audio_path, prompt_text,
        cfg_value, inference_timesteps, normalize, denoise,
        retry_badcase, retry_max_times, retry_ratio_threshold, server_uri
    ))


async def test_connection(server_uri: str):
    """测试 WebSocket 连接"""
    try:
        async with websockets.connect(server_uri, max_size=None) as ws:
            # 接收欢迎消息
            welcome = await ws.recv()
            data = json.loads(welcome)
            return f"✅ 连接成功：{data.get('message', '服务器已就绪')}", gr.update(interactive=True)
    except Exception as e:
        return f"❌ 连接失败：{str(e)}", gr.update(interactive=False)


def test_connection_sync(server_uri: str):
    """同步包装器"""
    return asyncio.run(test_connection(server_uri))


def stop_generation_message():
    return "生成已停止。"


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="VoxCPM WebSocket 客户端") as demo:
        gr.Markdown(
            """
            # VoxCPM WebSocket 流式语音合成客户端
            连接到 VoxCPM WebSocket 服务器进行实时流式语音合成，支持零样本声音克隆。
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
                    label="参考音频（可选）",
                    sources=["upload"],
                    type="filepath",
                    value=DEFAULT_PROMPT_WAV if os.path.isfile(DEFAULT_PROMPT_WAV) else None,
                )

                prompt_text = gr.Textbox(
                    label="参考文本（使用参考音频时必填）",
                    value=DEFAULT_PROMPT_TEXT,
                    lines=2,
                    info="参考音频对应的文本内容",
                )

                cfg_value = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    label="CFG 值（引导尺度）",
                    info="值越高对提示遵循越好，但质量可能下降",
                )

                inference_timesteps = gr.Slider(
                    minimum=4,
                    maximum=30,
                    value=10,
                    step=1,
                    label="推理时间步数",
                    info="值越高质量越好，但速度越慢",
                )

                normalize = gr.Checkbox(
                    label="启用文本标准化",
                    value=True,
                    info="使用外部工具进行文本标准化",
                )

                denoise = gr.Checkbox(
                    label="启用降噪",
                    value=True,
                    info="使用外部降噪工具增强音质",
                )

                with gr.Accordion("高级选项", open=False):
                    retry_badcase = gr.Checkbox(
                        label="启用糟糕情况重试",
                        value=False,
                        info="自动重试生成质量不佳的情况",
                    )

                    retry_max_times = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="最大重试次数",
                    )

                    retry_ratio_threshold = gr.Slider(
                        minimum=3.0,
                        maximum=10.0,
                        value=6.0,
                        step=0.5,
                        label="糟糕情况检测阈值",
                        info="长度比率阈值，可根据语速调整",
                    )

                with gr.Row():
                    test_button = gr.Button("测试连接", scale=1)
                    generate_button = gr.Button("生成语音", variant="primary", interactive=False, scale=1)
                    stop_button = gr.Button("停止生成", variant="stop", scale=1)

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

        test_button.click(
            fn=test_connection_sync,
            inputs=[server_uri],
            outputs=[status_output, generate_button],
        )

        generate_event = generate_button.click(
            fn=generate_speech,
            inputs=[
                text,
                prompt_audio,
                prompt_text,
                cfg_value,
                inference_timesteps,
                normalize,
                denoise,
                retry_badcase,
                retry_max_times,
                retry_ratio_threshold,
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
    demo.queue().launch(inbrowser=True, server_name="0.0.0.0", server_port=7863)
