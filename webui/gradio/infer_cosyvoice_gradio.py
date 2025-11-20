import os
import sys
from functools import lru_cache

import gradio as gr
import torchaudio
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
matcha_path = os.path.join(project_root, "Matcha-TTS")
sys.path.insert(0, project_root)
sys.path.insert(0, matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

DEFAULT_MODEL_DIR = "models/CosyVoice2-0.5B"
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好哟。"
USE_FP16 = True
LOAD_JIT = False
LOAD_TRT = False
LOAD_VLLM = False
TRT_CONCURRENT = 1

MODE_MAPPING = {
    "零样本克隆": "zero_shot",
    "跨语言克隆": "cross_lingual",
    "指令控制": "instruct",
}

@lru_cache(maxsize=2)
def get_model(model_dir: str = DEFAULT_MODEL_DIR, device: str = "cuda") -> CosyVoice2:
    """根据配置加载并缓存 CosyVoice2 模型。"""
    # 根据设备类型自动配置加载参数
    is_cuda = device == "cuda" and torch.cuda.is_available()
    
    return CosyVoice2(
        model_dir=model_dir,
        load_jit=LOAD_JIT,
        load_trt=LOAD_TRT,
        load_vllm=LOAD_VLLM,
        fp16=is_cuda and USE_FP16,  # 仅 CUDA 启用 FP16
        trt_concurrent=TRT_CONCURRENT,
        device=device,
    )


def prepare_prompt_audio(prompt_audio_path: str) -> torch.Tensor:
    """验证并加载参考提示音频。"""
    if not prompt_audio_path:
        prompt_audio_path = DEFAULT_PROMPT_WAV

    if not os.path.isfile(prompt_audio_path):
        raise FileNotFoundError(f"参考音频不存在: {prompt_audio_path}")

    try:
        sample_rate = torchaudio.info(prompt_audio_path).sample_rate
    except Exception as exc:
        raise RuntimeError(f"无法读取参考音频: {exc}") from exc

    if sample_rate < 16000:
        raise ValueError("参考音频采样率需至少 16000Hz。")

    return load_wav(prompt_audio_path, 16000)


def merge_segments(segments) -> torch.Tensor:
    """将生成的音频片段合并成单一波形。"""
    audio_tensors = []
    for segment in segments:
        tensor = segment.get("tts_speech")
        if tensor is None:
            continue
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        audio_tensors.append(tensor)

    if not audio_tensors:
        raise RuntimeError("生成结果为空。")

    return torch.cat(audio_tensors, dim=-1)


def generate_speech(
    mode: str,
    text: str,
    prompt_audio_path: str,
    prompt_text: str,
    instruct_text: str,
    speed: float,
    seed: int,
    device: str,
    model_loaded: bool,
):
    """运行 CosyVoice 推理并返回音频 + 状态消息。"""
    text = (text or "").strip()
    prompt_text = (prompt_text or "").strip()
    instruct_text = (instruct_text or "").strip()
    
    mode_key = MODE_MAPPING.get(mode, mode)

    if not model_loaded:
        return None, "请先加载模型。"

    if not text:
        return None, "请输入待合成文本。"

    if mode_key == "zero_shot" and not prompt_text:
        return None, "零样本克隆模式需要提供参考文本。"

    if mode_key == "instruct" and not instruct_text:
        return None, "指令控制模式需要提供指令文本。"

    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 0

    try:
        prompt_speech_16k = prepare_prompt_audio(prompt_audio_path)
    except Exception as exc:
        return None, str(exc)

    set_all_random_seed(seed)

    try:
        cosyvoice = get_model(DEFAULT_MODEL_DIR, device)
    except Exception as exc:
        return None, f"模型加载失败: {exc}"

    try:
        if mode_key == "zero_shot":
            result = cosyvoice.inference_zero_shot(
                text,
                prompt_text,
                prompt_speech_16k,
                stream=False,
                speed=speed,
            )
        elif mode_key == "cross_lingual":
            result = cosyvoice.inference_cross_lingual(
                text,
                prompt_speech_16k,
                stream=False,
                speed=speed,
            )
        elif mode_key == "instruct":
            result = cosyvoice.inference_instruct2(
                text,
                instruct_text,
                prompt_speech_16k,
                stream=False,
                speed=speed,
            )
        else:
            return None, f"无效的模式: {mode}"

        audio_tensor = merge_segments(result)
        audio_numpy = audio_tensor.squeeze(0).cpu().numpy()
        return (cosyvoice.sample_rate, audio_numpy), "生成完成"
    except Exception as exc:
        return None, f"推理失败: {exc}"


def load_model(device: str, model_loaded: bool):
    """加载 CosyVoice 模型并为字幕控制更新启用。"""
    if model_loaded:
        return "模型已加载，无需重复加载。", True, gr.update(interactive=True)

    try:
        get_model(DEFAULT_MODEL_DIR, device)
        return "模型加载完成", True, gr.update(interactive=True)
    except Exception as exc:
        return f"模型加载失败: {exc}", False, gr.update(interactive=False)


def stop_generation_message():
    return "生成已停止。"


def update_ui_visibility(mode: str):
    """根据选中模式更新UI组件可见性。"""
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
    with gr.Blocks(title="CosyVoice2 语音合成") as demo:
        gr.Markdown(
            """
            # CosyVoice2 语音合成演示
            选择推理模式，上传或使用默认参考音频，输入文本后点击生成即可试听。
            """
        )

        model_loaded_state = gr.State(False)

        with gr.Row():
            # 左侧：控制面板
            with gr.Column(scale=1):
                gr.Markdown("### 控制面板")
                
                device = gr.Radio(
                    choices=["cuda", "cpu"],
                    value="cuda" if torch.cuda.is_available() else "cpu",
                    label="运行设备",
                    info="选择模型运行的设备（CUDA 会自动启用 FP16）",
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
                    load_button = gr.Button("加载模型", scale=1)
                    generate_button = gr.Button("生成语音", interactive=False, scale=1)
                
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

        mode.change(
            fn=update_ui_visibility,
            inputs=[mode],
            outputs=[prompt_audio, prompt_text, instruct_text],
        )

        load_button.click(
            fn=load_model,
            inputs=[device, model_loaded_state],
            outputs=[status_output, model_loaded_state, generate_button],
        )

        generate_event = generate_button.click(
            fn=generate_speech,
            inputs=[
                mode,
                text,
                prompt_audio,
                prompt_text,
                instruct_text,
                speed,
                seed,
                device,
                model_loaded_state,
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
    demo.queue().launch(inbrowser=True, server_name="127.0.0.1", server_port=7860)
