import os
import sys
import yaml
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

config_path = os.path.join(project_root, "config", "load.yaml")
with open(config_path, "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

config_default = config.get("default", {})
config_cosyvoice = config.get("models", {}).get("cosyvoice", {})

DEFAULT_MODEL_DIR = config_cosyvoice.get("model_dir", "models/CosyVoice2-0.5B")
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"


@lru_cache(maxsize=2)
def get_model(model_dir: str) -> CosyVoice2:
    """Load and cache CosyVoice2 models by directory."""
    return CosyVoice2(
        model_dir=model_dir,
        load_jit=config_cosyvoice.get("load_jit", False),
        load_trt=config_cosyvoice.get("load_trt", False),
        load_vllm=config_cosyvoice.get("load_vllm", False),
        fp16=config_default.get("use_fp16", True),
        trt_concurrent=config_cosyvoice.get("trt_concurrent", 1),
    )


def prepare_prompt_audio(prompt_audio_path: str) -> torch.Tensor:
    """Validate and load the reference prompt audio."""
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
    """Concatenate generated audio segments into a single waveform."""
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
    model_loaded: bool,
):
    """Run CosyVoice inference and return audio + status message."""
    text = (text or "").strip()
    prompt_text = (prompt_text or "").strip()
    instruct_text = (instruct_text or "").strip()

    if not model_loaded:
        return None, "请先加载模型。"

    if not text:
        return None, "请输入待合成文本。"

    if mode == "zero_shot" and not prompt_text:
        return None, "零样本模式需要提供参考文本。"

    if mode == "instruct" and not instruct_text:
        return None, "指令模式需要提供指令文本。"

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
        cosyvoice = get_model(DEFAULT_MODEL_DIR)
    except Exception as exc:
        return None, f"模型加载失败: {exc}"

    try:
        if mode == "zero_shot":
            result = cosyvoice.inference_zero_shot(
                text,
                prompt_text,
                prompt_speech_16k,
                stream=False,
                speed=speed,
            )
        elif mode == "cross_lingual":
            result = cosyvoice.inference_cross_lingual(
                text,
                prompt_speech_16k,
                stream=False,
                speed=speed,
            )
        elif mode == "instruct":
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
        return (cosyvoice.sample_rate, audio_numpy), "生成完成 ✅"
    except Exception as exc:
        return None, f"推理失败: {exc}"


def load_model(model_loaded: bool):
    """Load the CosyVoice model and enable generation once ready."""
    if model_loaded:
        return "模型已加载，无需重复加载。", True, gr.update(interactive=True)

    try:
        get_model(DEFAULT_MODEL_DIR)
        return "模型加载完成 ✅", True, gr.update(interactive=True)
    except Exception as exc:
        return f"模型加载失败: {exc}", False, gr.update(interactive=False)


def stop_generation_message():
    return "生成已停止。"


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
                
                mode = gr.Radio(
                    choices=["zero_shot", "cross_lingual", "instruct"],
                    value="zero_shot",
                    label="推理模式",
                )
                
                text = gr.Textbox(
                    label="待合成文本",
                    lines=4,
                    placeholder="请输入需要合成的文本",
                )
                
                prompt_text = gr.Textbox(
                    label="参考文本 (零样本模式必填)",
                    value=DEFAULT_PROMPT_TEXT,
                    lines=2,
                )
                
                instruct_text = gr.Textbox(
                    label="指令文本 (指令模式必填)",
                    placeholder="示例: 用四川话说这句话",
                    lines=2,
                )
                
                prompt_audio = gr.Audio(
                    label="参考音频",
                    sources=["upload"],
                    type="filepath",
                    value=DEFAULT_PROMPT_WAV if os.path.isfile(DEFAULT_PROMPT_WAV) else None,
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

        load_button.click(
            fn=load_model,
            inputs=[model_loaded_state],
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
    demo.queue().launch()
