import os
import sys
import numpy as np
import torch
import gradio as gr  
from typing import Optional, Tuple
from pathlib import Path

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from voxcpm.core import VoxCPM


class VoxCPMDemo:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Running on device: {self.device}")

        # TTS 模型（延迟初始化）
        self.voxcpm_model: Optional[VoxCPM] = None
        self.tts_model_path = "models/VoxCPM-0.5B"

    # ---------- 模型辅助方法 ----------

    def get_or_load_voxcpm(self) -> VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        print("模型未加载，正在初始化...")
        print(f"使用模型路径: {self.tts_model_path}")
        self.voxcpm_model = VoxCPM.from_pretrained(
            self.tts_model_path,
            local_files_only=True,
            device=self.device,
        )
        print("模型加载成功。")
        return self.voxcpm_model

    # ---------- 功能接口 ----------
    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        # 移除自动识别功能，返回空字符串
        return ""

    def generate_tts_audio(
        self,
        text_input: str,
        prompt_wav_path_input: Optional[str] = None,
        prompt_text_input: Optional[str] = None,
        cfg_value_input: float = 2.0,
        inference_timesteps_input: int = 10,
        do_normalize: bool = True,
        denoise: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """
        使用 VoxCPM 从文本生成语音；可选参考音频用于语音风格指导。
        返回 (采样率, 波形数组)
        """
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("请输入要合成的文本。")

        prompt_wav_path = prompt_wav_path_input if prompt_wav_path_input else None
        prompt_text = prompt_text_input if prompt_text_input else None

        print(f"正在为文本生成音频: '{text[:60]}...'")
        wav = current_model.generate(
            text=text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=int(inference_timesteps_input),
            normalize=do_normalize,
            denoise=denoise,
        )
        return (16000, wav)


# ---------- UI 构建器 ----------

def create_demo_interface(demo: VoxCPMDemo):
    """构建 VoxCPM 演示的 Gradio UI 界面。"""
    # 静态资源（logo 路径）
    gr.set_static_paths(paths=[Path.cwd().absolute()/"assets"])

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
        ),
        css="""
        .logo-container {
            text-align: center;
            margin: 0.5rem 0 1rem 0;
        }
        .logo-container img {
            height: 80px;
            width: auto;
            max-width: 200px;
            display: inline-block;
        }
        /* Bold accordion labels */
        #acc_quick details > summary,
        #acc_tips details > summary {
            font-weight: 600 !important;
            font-size: 1.1em !important;
        }
        /* Bold labels for specific checkboxes */
        #chk_denoise label,
        #chk_denoise span,
        #chk_normalize label,
        #chk_normalize span {
            font-weight: 600;
        }
        """
    ) as interface:
        # 页头 logo
        gr.HTML('<div class="logo-container"><img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo"></div>')

        # 快速入门
        with gr.Accordion("📋 快速入门", open=False, elem_id="acc_quick"):
            gr.Markdown("""
            ### 使用说明
            1. **（可选）提供参考声音** - 上传或录制一段音频，为声音合成提供音色、语调和情感等个性化特征
            2. **（可选）输入参考文本** - 如果提供了参考语音，请输入其对应的文本内容
            3. **输入目标文本** - 输入您希望模型朗读的文字内容
            4. **生成语音** - 点击"生成"按钮，即可创造出音频
            """)

        # 使用建议
        with gr.Accordion("💡 使用建议", open=False, elem_id="acc_tips"):
            gr.Markdown("""
            ### 参考语音降噪
            - **启用**：通过 ZipEnhancer 组件消除背景噪音，获得更好的音质
            - **禁用**：保留原始音频的背景环境声，如果想复刻相应声学环境

            ### 文本正则化
            - **启用**：使用 WeTextProcessing 组件，可处理常见文本
            - **禁用**：将使用 VoxCPM 内置的文本理解能力。支持音素输入（如 {da4}{jia1}好）和公式符号合成

            ### CFG 值
            - **调低**：如果提示语音听起来不自然或过于夸张
            - **调高**：为更好地贴合提示音频的风格或输入文本

            ### 推理时间步
            - **调低**：合成速度更快
            - **调高**：合成质量更佳
            """)

        # Main controls
        with gr.Row():
            with gr.Column():
                prompt_wav = gr.Audio(
                    sources=["upload", 'microphone'],
                    type="filepath",
                    label="参考语音（可选，或让 VoxCPM 自由发挥）",
                    value="./examples/example.wav",
                )
                DoDenoisePromptAudio = gr.Checkbox(
                    value=False,
                    label="参考语音增强",
                    elem_id="chk_denoise",
                    info="我们使用 ZipEnhancer 模型对参考音频进行降噪。"
                )
                with gr.Row():
                    prompt_text = gr.Textbox(
                        value="每天只需听几分钟，你就能通过调节心态来消除负面想法，让思维变得更积极。",
                        label="参考文本",
                        placeholder="请输入参考文本。支持自动识别，您可以自行修正结果..."
                    )
                run_btn = gr.Button("生成语音", variant="primary")
                gr.Markdown("**注意**：自动语音识别已被禁用，请手动输入与音频对应的参考文本")

            with gr.Column():
                cfg_value = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    label="CFG 值（引导尺度）",
                    info="较高的值增加对提示的遵循，较低的值允许更多创造性"
                )
                inference_timesteps = gr.Slider(
                    minimum=4,
                    maximum=30,
                    value=10,
                    step=1,
                    label="推理时间步",
                    info="生成的推理时间步数（较高的值可能提高质量但速度较慢）"
                )
                with gr.Row():
                    text = gr.Textbox(
                        value="VoxCPM 是来自 ModelBest 的创新端到端 TTS 模型，旨在生成高度逼真的语音。",
                        label="目标文本",
                    )
                with gr.Row():
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label="文本标准化",
                        elem_id="chk_normalize",
                        info="我们使用 wetext 库对输入文本进行标准化。"
                    )
                audio_output = gr.Audio(label="输出音频")

        # 组件连接
        run_btn.click(
            fn=demo.generate_tts_audio,
            inputs=[text, prompt_wav, prompt_text, cfg_value, inference_timesteps, DoNormalizeText, DoDenoisePromptAudio],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
        )

    return interface


def run_demo(server_name: str = "localhost", server_port: int = 7860, show_error: bool = True):
    demo = VoxCPMDemo()
    interface = create_demo_interface(demo)
    interface.queue(max_size=10).launch(server_name=server_name, server_port=server_port, show_error=show_error)


if __name__ == "__main__":
    run_demo()