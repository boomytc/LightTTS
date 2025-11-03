import os
import sys
from functools import lru_cache

import gradio as gr

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from indextts.infer_v2 import IndexTTS2

DEFAULT_MODEL_DIR = "models/IndexTTS-2"
DEFAULT_CFG_PATH = "models/IndexTTS-2/config.yaml"
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
USE_FP16 = True
DEVICE = "cuda"
USE_CUDA_KERNEL = False

EMO_LABELS = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]


@lru_cache(maxsize=1)
def get_model() -> IndexTTS2:
    """Load and cache IndexTTS2 model."""
    return IndexTTS2(
        cfg_path=DEFAULT_CFG_PATH,
        model_dir=DEFAULT_MODEL_DIR,
        use_fp16=USE_FP16,
        device=DEVICE,
        use_cuda_kernel=USE_CUDA_KERNEL,
    )


def update_ui_visibility(emo_mode: str):
    """Update UI component visibility based on emotion control mode."""
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
    model_loaded: bool,
):
    """Run IndexTTS2 inference and return audio + status message."""
    text = (text or "").strip()
    emo_text = (emo_text or "").strip()

    if not model_loaded:
        return None, "请先加载模型。"

    if not text:
        return None, "请输入待合成文本。"

    if emo_mode == "情感参考音频" and not emo_audio_path:
        return None, "情感参考音频模式需要上传情感音频。"

    if emo_mode == "情感文本引导" and not emo_text:
        return None, "情感文本引导模式需要输入引导文本。"

    try:
        indextts = get_model()
    except Exception as exc:
        return None, f"模型加载失败: {exc}"

    try:
        kwargs = {
            "spk_audio_prompt": prompt_audio_path or DEFAULT_PROMPT_WAV,
            "text": text,
            "output_path": None,
            "use_random": use_random,
            "interval_silence": interval_silence,
            "max_text_tokens_per_segment": max_tokens,
            "verbose": False,
        }

        if emo_mode == "情感参考音频":
            kwargs["emo_audio_prompt"] = emo_audio_path
            kwargs["emo_alpha"] = emo_alpha
        elif emo_mode == "情感向量":
            emo_vector = [
                emo_happy,
                emo_angry,
                emo_sad,
                emo_afraid,
                emo_disgusted,
                emo_melancholic,
                emo_surprised,
                emo_calm,
            ]
            kwargs["emo_vector"] = emo_vector
            kwargs["emo_alpha"] = 1.0
        elif emo_mode == "情感文本引导":
            kwargs["use_emo_text"] = True
            kwargs["emo_text"] = emo_text
            kwargs["emo_alpha"] = emo_alpha

        result = indextts.infer(**kwargs)
        return result, "生成完成 ✅"
    except Exception as exc:
        return None, f"推理失败: {exc}"


def load_model(model_loaded: bool):
    """Load the IndexTTS2 model and enable generation once ready."""
    if model_loaded:
        return "模型已加载，无需重复加载。", True, gr.update(interactive=True)

    try:
        get_model()
        return "模型加载完成 ✅", True, gr.update(interactive=True)
    except Exception as exc:
        return f"模型加载失败: {exc}", False, gr.update(interactive=False)


def stop_generation_message():
    return "生成已停止。"


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="IndexTTS2 语音合成") as demo:
        gr.Markdown(
            """
            # IndexTTS2 语音合成演示
            支持多种情感控制方式：情感参考音频、情感向量、情感文本引导。
            """
        )

        model_loaded_state = gr.State(False)

        with gr.Row():
            # 左侧：控制面板
            with gr.Column(scale=1):
                gr.Markdown("### 控制面板")

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

        emo_mode.change(
            fn=update_ui_visibility,
            inputs=[emo_mode],
            outputs=[emo_audio, emo_alpha, emo_vector_group, emo_text],
        )

        load_button.click(
            fn=load_model,
            inputs=[model_loaded_state],
            outputs=[status_output, model_loaded_state, generate_button],
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
