import asyncio
import io
import json
import os
import sys

import gradio as gr
import numpy as np
import websockets
import torchaudio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

SERVER_URI = "ws://127.0.0.1:8769"
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好哟。"

MODE_MAPPING = {
    "零样本克隆": "zero_shot",
    "跨语言克隆": "cross_lingual",
    "指令控制": "instruct",
}


async def tts_stream(
    text: str,
    mode: str,
    prompt_audio_path: str,
    prompt_text: str,
    instruct_text: str,
    speed: float,
    seed: int,
    server_uri: str,
):
    """通过 WebSocket 流式合成语音"""
    text = (text or "").strip()
    prompt_text = (prompt_text or "").strip()
    instruct_text = (instruct_text or "").strip()
    mode_key = MODE_MAPPING.get(mode, mode)

    if not text:
        return None, "请输入待合成文本。"

    if mode_key == "zero_shot" and not prompt_text:
        return None, "零样本克隆模式需要提供参考文本。"

    if mode_key == "instruct" and not instruct_text:
        return None, "指令控制模式需要提供指令文本。"

    # 准备请求
    req = {
        "type": "tts_stream",
        "mode": mode_key,
        "text": text,
        "prompt_audio": prompt_audio_path or DEFAULT_PROMPT_WAV,
        "speed": float(speed),
        "seed": int(seed),
    }

    if mode_key == "zero_shot":
        req["prompt_text"] = prompt_text
    elif mode_key == "instruct":
        req["instruct_text"] = instruct_text

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
                        return None, f"出错：{data.get('message')}"
        
        if not audio_segments:
            return None, "生成结果为空。"
        
        # 合并所有音频片段
        full_audio = np.concatenate(audio_segments)
        return (sample_rate, full_audio), f"生成完成（{len(audio_segments)} 个片段）"
    
    except Exception as e:
        return None, f"连接失败：{str(e)}"


def generate_speech(
    text: str,
    mode: str,
    prompt_audio_path: str,
    prompt_text: str,
    instruct_text: str,
    speed: float,
    seed: int,
    server_uri: str,
):
    """同步包装器"""
    return asyncio.run(tts_stream(
        text, mode, prompt_audio_path, prompt_text,
        instruct_text, speed, seed, server_uri
    ))


async def test_connection(server_uri: str):
    """测试 WebSocket 连接"""
    try:
        async with websockets.connect(server_uri, max_size=None) as ws:
            # 接收欢迎消息
            welcome = await ws.recv()
            data = json.loads(welcome)
            return f"连接成功：{data.get('message', '服务器已就绪')}", gr.update(interactive=True)
    except Exception as e:
        return f"连接失败：{str(e)}", gr.update(interactive=False)


def test_connection_sync(server_uri: str):
    """同步包装器"""
    return asyncio.run(test_connection(server_uri))


def stop_generation_message():
    return "生成已停止。"


def update_ui_visibility(mode: str):
    """根据模式更新UI可见性"""
    mode_key = MODE_MAPPING.get(mode, mode)
    
    if mode_key == "zero_shot":
        return (
            gr.update(visible=True),   # prompt_audio
            gr.update(visible=True),   # prompt_text
            gr.update(visible=False),  # instruct_text
        )
    elif mode_key == "cross_lingual":
        return (
            gr.update(visible=True),   # prompt_audio
            gr.update(visible=False),  # prompt_text
            gr.update(visible=False),  # instruct_text
        )
    elif mode_key == "instruct":
        return (
            gr.update(visible=True),   # prompt_audio
            gr.update(visible=False),  # prompt_text
            gr.update(visible=True),   # instruct_text
        )
    return gr.update(), gr.update(), gr.update()


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="CosyVoice WebSocket 客户端") as demo:
        gr.Markdown(
            """
            # CosyVoice WebSocket 流式语音合成客户端
            连接到 CosyVoice WebSocket 服务器进行实时流式语音合成。
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
                
                mode = gr.Radio(
                    choices=["零样本克隆", "跨语言克隆", "指令控制"],
                    value="零样本克隆",
                    label="推理模式",
                )
                
                text = gr.Textbox(
                    label="待合成文本",
                    lines=4,
                    placeholder="请输入需要合成的文本",
                )
                
                prompt_audio = gr.Audio(
                    label="参考音频",
                    sources=["upload"],
                    type="filepath",
                    value=DEFAULT_PROMPT_WAV if os.path.isfile(DEFAULT_PROMPT_WAV) else None,
                    visible=True,
                )
                
                prompt_text = gr.Textbox(
                    label="参考文本",
                    value=DEFAULT_PROMPT_TEXT,
                    lines=2,
                    visible=True,
                    info="零样本克隆模式需要提供参考音频对应的文本",
                )
                
                instruct_text = gr.Textbox(
                    label="指令文本",
                    placeholder="示例: 用四川话说这句话",
                    lines=2,
                    visible=False,
                    info="指令控制模式用于控制语音风格、情感、方言等",
                )
                
                speed = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    label="语速",
                )
                
                seed = gr.Number(
                    value=0,
                    precision=0,
                    label="随机种子",
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

        mode.change(
            fn=update_ui_visibility,
            inputs=[mode],
            outputs=[prompt_audio, prompt_text, instruct_text],
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
                mode,
                prompt_audio,
                prompt_text,
                instruct_text,
                speed,
                seed,
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
    demo.queue().launch(inbrowser=True, server_name="0.0.0.0", server_port=7861)
