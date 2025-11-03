import os
import sys

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import tempfile
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import VoxCPMModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

try:
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True  # type: ignore
except Exception:
    pass


def _disable_optimize(self: VoxCPMModel):
    """禁用在某些 CUDA 设置下失败的 torch.compile 优化。"""
    self.base_lm.forward_step = self.base_lm.forward_step
    self.residual_lm.forward_step = self.residual_lm.forward_step
    self.feat_encoder_step = self.feat_encoder
    self.feat_decoder.estimator = self.feat_decoder.estimator
    return self


VoxCPMModel.optimize = _disable_optimize
MODEL_ID = "models/VoxCPM-0.5B"
ZIPENHANCER_MODEL_ID = "models/speech_zipenhancer_ans_multiloss_16k_base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做得比我还好哟。"

_model: Optional[VoxCPM] = None
_model_config: dict = {}


def get_model(device: str = DEVICE) -> VoxCPM:
    """懒加载并缓存 VoxCPM 管道。"""
    global _model, _model_config
    
    current_config = {"device": device}
    
    # 如果配置变化，重新加载模型
    if _model is None or _model_config != current_config:
        _model = VoxCPM.from_pretrained(
            hf_model_id=MODEL_ID,
            load_denoiser=False,  # 统一使用外部 zipenhancer
            zipenhancer_model_id=ZIPENHANCER_MODEL_ID,
            local_files_only=True,
            device=device,
        )
        _model_config = current_config
    return _model


def _save_prompt_audio(prompt_audio: Tuple[int, np.ndarray]) -> Optional[str]:
    """将上传的提示音频保存到临时 WAV 文件中。"""
    if prompt_audio is None:
        return None

    sample_rate, audio = prompt_audio
    if audio is None:
        return None

    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    audio = audio.astype(np.float32)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        sf.write(tmp_file.name, audio, sample_rate)
        return tmp_file.name


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
    device: str,
    model_loaded: bool,
):
    """运行 VoxCPM 推理并返回音频 + 状态消息。"""
    text = (text or "").strip()
    prompt_text = (prompt_text or "").strip()

    if not model_loaded:
        return None, "请先加载模型。"

    if not text:
        return None, "请输入待合成文本。"

    if prompt_audio_path and not prompt_text:
        return None, "使用参考音频时，请提供对应的参考文本。"

    try:
        voxcpm = get_model(device)
    except Exception as exc:
        return None, f"模型加载失败: {exc}"

    try:
        wav = voxcpm.generate(
            text=text,
            prompt_wav_path=prompt_audio_path or None,  # type: ignore
            prompt_text=prompt_text or None,  # type: ignore
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_max_times,
            retry_badcase_ratio_threshold=retry_ratio_threshold,
        )
        return (16000, wav), "生成完成 ✅"
    except Exception as exc:
        return None, f"推理失败: {exc}"


def load_model(device: str, model_loaded: bool):
    """加载 VoxCPM 模型并在准备就绪后启用生成。"""
    if model_loaded:
        return "模型已加载，无需重复加载。", True, gr.update(interactive=True)

    try:
        get_model(device)
        return "模型加载完成 ✅", True, gr.update(interactive=True)
    except Exception as exc:
        return f"模型加载失败: {exc}", False, gr.update(interactive=False)


def stop_generation_message():
    return "生成已停止。"


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="VoxCPM 语音合成") as demo:
        gr.Markdown(
            """
            # VoxCPM 语音合成演示
            支持零样本声音克隆和高质量语音生成。
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
                    info="选择模型运行的设备（CPU 或 GPU）",
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
                        value=True,
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
            inputs=[device, model_loaded_state],
            outputs=[status_output, model_loaded_state, generate_button],
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
    demo.queue(max_size=4).launch(inbrowser=True, server_name="0.0.0.0", server_port=7860)
