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
            prompt_wav_path=prompt_wav_path,  # type: ignore
            prompt_text=prompt_text if prompt_wav_path else None,  # type: ignore
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
    try:
        model = get_model()
        sample_rate = getattr(model.tts_model.audio_vae, "sample_rate", 16000)
        device_info = "GPU (CUDA)" if DEVICE == "cuda" else "CPU"
        status_md = (
            f"**🟢 模型状态：已加载**\n\n"
            f"- 设备：{device_info}\n"
            f"- 采样率：{sample_rate} Hz\n"
            f"- 模型：VoxCPM-0.5B\n\n"
            f"✅ 可以开始生成语音了！"
        )
        return (
            status_md,
            True,
            gr.update(interactive=False, value="✅ 已加载"),
            gr.update(interactive=True, variant="primary"),
        )
    except Exception as exc:
        error_md = (
            f"**🔴 模型加载失败**\n\n"
            f"错误信息：{str(exc)}\n\n"
            f"请检查模型文件是否存在于 `{MODEL_ID}` 目录"
        )
        return (
            error_md,
            False,
            gr.update(interactive=True),
            gr.update(interactive=False),
        )


with gr.Blocks(title="VoxCPM 语音合成") as demo:
    gr.Markdown(
        """
        # 🎙️ VoxCPM 语音合成演示
        
        VoxCPM 是一个端到端的零样本语音合成模型，支持高质量的声音克隆和多语言合成。
        """
    )

    model_loaded_state = gr.State(False)

    # 快速入门提示
    with gr.Accordion("📖 快速入门", open=False, elem_id="tips-accordion"):
        gr.Markdown(
            """
            ### 使用步骤
            1. **加载模型** - 点击"加载模型"按钮，等待模型初始化完成
            2. **输入文本** - 在文本框中输入需要合成的内容
            3. **（可选）声音克隆** - 上传参考音频和对应文本，实现声音克隆
            4. **生成语音** - 点击"生成语音"按钮开始合成
            
            ### 💡 参数说明
            - **CFG 值**：控制对提示音频的遵循程度（1.0-4.0），值越高越接近参考音色
            - **推理步数**：影响生成质量和速度（5-30步），步数越多质量越好但速度越慢
            - **文本标准化**：自动处理数字、符号等特殊文本，关闭后支持音素输入
            - **音频降噪**：对参考音频进行降噪处理，提升克隆效果
            """
        )

    with gr.Row():
        # 左侧：输入控制区
        with gr.Column(scale=1):
            gr.Markdown("### 📝 文本输入")
            
            text_input = gr.Textbox(
                label="待合成文本",
                lines=6,
                placeholder="请输入需要合成的文本内容...",
                info="支持中英文混合输入，可使用标点符号控制停顿",
            )

            gr.Markdown("### 🎤 声音克隆（可选）")
            
            prompt_audio_input = gr.Audio(
                label="参考音频",
                type="numpy",
                sources=["upload", "microphone"],
            )
            
            prompt_text_input = gr.Textbox(
                label="参考文本",
                lines=2,
                placeholder="请输入参考音频对应的文本内容...",
                info="参考音频中说话人说的内容",
            )

            gr.Markdown("### ⚙️ 生成参数")
            
            with gr.Row():
                cfg_slider = gr.Slider(
                    minimum=0.5,
                    maximum=4.0,
                    step=0.1,
                    value=2.0,
                    label="CFG 引导值",
                    info="推荐值：2.0-2.5",
                )
                timestep_slider = gr.Slider(
                    minimum=5,
                    maximum=30,
                    step=1,
                    value=10,
                    label="推理步数",
                    info="推荐值：10-15",
                )

            with gr.Row():
                normalize_checkbox = gr.Checkbox(
                    value=True,
                    label="文本标准化",
                    info="处理数字、符号等",
                )
                denoise_checkbox = gr.Checkbox(
                    value=True,
                    label="音频降噪",
                    info="提升克隆效果",
                )

            # 高级选项
            with gr.Accordion("🔧 高级选项", open=False):
                retry_checkbox = gr.Checkbox(
                    value=True,
                    label="启用自动重试",
                    info="检测到生成异常时自动重试",
                )
                
                with gr.Row():
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
                        label="音频长度阈值（倍）",
                        info="音频/文本长度比例上限",
                    )

            # 操作按钮
            with gr.Row():
                load_button = gr.Button(
                    "🚀 加载模型",
                    variant="primary",
                    scale=1,
                )
                generate_button = gr.Button(
                    "🎵 生成语音",
                    interactive=False,
                    variant="secondary",
                    scale=1,
                )

        # 右侧：输出展示区
        with gr.Column(scale=1):
            gr.Markdown("### 📊 状态信息")
            
            model_status = gr.Markdown(
                "**🔴 模型状态：未加载** - 请先点击加载模型按钮",
            )

            gr.Markdown("### 🔊 生成结果")
            
            output_audio = gr.Audio(
                label="合成音频",
                type="numpy",
                show_download_button=True,
                autoplay=False,
            )

            # 示例文本
            gr.Markdown("### 📚 示例文本")
            gr.Examples(
                examples=[
                    ["你好，欢迎使用 VoxCPM 语音合成系统。"],
                    ["八百标兵奔北坡，炮兵并排北边跑。"],
                    ["VoxCPM 是一个强大的端到端语音合成模型，支持零样本声音克隆。"],
                    ["春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"],
                ],
                inputs=text_input,
                label=None,
            )

            # 使用提示
            with gr.Accordion("💬 使用建议", open=False):
                gr.Markdown(
                    """
                    **声音克隆技巧**
                    - 参考音频建议时长：3-10秒
                    - 参考音频质量：清晰、无背景噪音
                    - 参考文本要准确对应音频内容
                    
                    **参数调优建议**
                    - CFG 值过高：音质可能下降，声音过于夸张
                    - CFG 值过低：可能偏离参考音色
                    - 推理步数增加：质量提升但速度变慢
                    
                    **文本输入提示**
                    - 支持中英文混合
                    - 使用标点符号控制语气和停顿
                    - 关闭标准化可输入音素（如 {ni3}{hao3}）
                    """
                )

    # 事件绑定
    load_button.click(
        fn=load_model_action,
        inputs=[],
        outputs=[model_status, model_loaded_state, load_button, generate_button],
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


if __name__ == "__main__":
    demo.queue(max_size=4).launch(server_name="0.0.0.0", share=False)
