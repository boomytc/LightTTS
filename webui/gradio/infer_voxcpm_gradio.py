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
    dynamo.config.suppress_errors = True
except Exception:
    pass


def _disable_optimize(self: VoxCPMModel):
    """Disable torch.compile optimizations that fail under some CUDA setups."""
    self.base_lm.forward_step = self.base_lm.forward_step
    self.residual_lm.forward_step = self.residual_lm.forward_step
    self.feat_encoder_step = self.feat_encoder
    self.feat_decoder.estimator = self.feat_decoder.estimator
    return self


VoxCPMModel.optimize = _disable_optimize
MODEL_ID = "models/VoxCPM-0.5B"
ZIPENHANCER_MODEL_ID = "models/speech_zipenhancer_ans_multiloss_16k_base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model: Optional[VoxCPM] = None


def get_model() -> VoxCPM:
    """Lazily initialize and cache the VoxCPM pipeline."""
    global _model
    if _model is None:
        _model = VoxCPM.from_pretrained(
            hf_model_id=MODEL_ID,
            load_denoiser=False,
            zipenhancer_model_id=ZIPENHANCER_MODEL_ID,
            local_files_only=True,
            device=DEVICE,
        )
    return _model


def _save_prompt_audio(prompt_audio: Tuple[int, np.ndarray]) -> Optional[str]:
    """Persist uploaded prompt audio to a temporary WAV file."""
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
    model_ready: bool,
    text: str,
    prompt_audio: Optional[Tuple[int, np.ndarray]],
    prompt_text: Optional[str],
    cfg_value: float,
    inference_timesteps: int,
    normalize: bool,
    denoise: bool,
    retry_badcase: bool,
    retry_badcase_max_times: int,
    retry_badcase_ratio_threshold: float,
):
    if not bool(model_ready):
        raise gr.Error("请先点击“加载模型”按钮。")

    text = (text or "").strip()
    if not text:
        raise gr.Error("请输入需要合成的文本。")

    prompt_text = (prompt_text or "").strip()
    prompt_wav_path: Optional[str] = None
    temp_wav_path: Optional[str] = None

    try:
        if prompt_audio is not None:
            if not prompt_text:
                raise gr.Error("提供提示音频时需要同时填写提示文本。")
            prompt_wav_path = _save_prompt_audio(prompt_audio)
            temp_wav_path = prompt_wav_path

        model = get_model()
        sample_rate = getattr(model.tts_model.audio_vae, "sample_rate", 16000)

        wav = model.generate(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text if prompt_wav_path else None,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
        )

        wav = np.asarray(wav, dtype=np.float32)
        return sample_rate, wav

    except Exception as exc:
        raise gr.Error(f"生成语音失败：{exc}") from exc
    finally:
        if temp_wav_path:
            try:
                os.unlink(temp_wav_path)
            except OSError:
                pass


def load_model_action():
    model = get_model()
    sample_rate = getattr(model.tts_model.audio_vae, "sample_rate", 16000)
    return (
        f"**模型状态：已加载 ✅**<br/>采样率：{sample_rate} Hz",
        True,
        gr.update(interactive=False),
        gr.update(interactive=True),
    )


with gr.Blocks(title="VoxCPM WebUI") as demo:
    gr.Markdown(
        "# VoxCPM 简易 WebUI\n"
        "输入文本即可合成语音，可选提示音频实现声音克隆。"
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="待合成文本",
                lines=6,
                placeholder="请输入要朗读的内容。",
            )
            prompt_audio_input = gr.Audio(
                label="提示音频（可选）",
                type="numpy",
            )
            prompt_text_input = gr.Textbox(
                label="提示文本（可选）",
                lines=2,
                placeholder="提示音频对应的文本。",
            )

    with gr.Accordion("高级设置", open=False):
        cfg_slider = gr.Slider(
            minimum=0.5,
            maximum=4.0,
            step=0.1,
            value=2.0,
            label="CFG 系数",
        )
        timestep_slider = gr.Slider(
            minimum=5,
            maximum=30,
            step=1,
            value=10,
            label="推理步数",
        )
        normalize_checkbox = gr.Checkbox(
            value=True,
            label="启用文本标准化",
        )
        denoise_checkbox = gr.Checkbox(
            value=True,
            label="提示音频降噪",
        )
        retry_checkbox = gr.Checkbox(
            value=True,
            label="坏案例自动重试",
        )
        retry_max_slider = gr.Slider(
            minimum=1,
            maximum=5,
            step=1,
            value=3,
            label="最大重试次数",
        )
        ratio_slider = gr.Slider(
            minimum=1.0,
            maximum=10.0,
            step=0.5,
            value=6.0,
            label="音频长度限制（倍）",
        )

    with gr.Row():
        load_button = gr.Button("加载模型", variant="primary")
        generate_button = gr.Button("开始合成", interactive=False)
    model_status = gr.Markdown("**模型状态：未加载**")
    model_loaded_state = gr.State(False)

    load_button.click(
        fn=load_model_action,
        inputs=[],
        outputs=[model_status, model_loaded_state, load_button, generate_button],
    )

    output_audio = gr.Audio(
        label="生成语音",
        type="numpy",
    )

    generate_button.click(
        fn=generate_speech,
        inputs=[
            model_loaded_state,
            text_input,
            prompt_audio_input,
            prompt_text_input,
            cfg_slider,
            timestep_slider,
            normalize_checkbox,
            denoise_checkbox,
            retry_checkbox,
            retry_max_slider,
            ratio_slider,
        ],
        outputs=output_audio,
    )

    gr.Examples(
        examples=[
            ["你好，欢迎使用 VoxCPM 简易 WebUI。"],
            ["八百标兵奔北坡，炮兵并排北边跑。"],
        ],
        inputs=text_input,
        label="示例",
    )


if __name__ == "__main__":
    demo.queue(max_size=4).launch(server_name="0.0.0.0", share=False)
