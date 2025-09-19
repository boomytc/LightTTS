import os
import re
import tempfile
from huggingface_hub import snapshot_download
from voxcpm.model.voxcpm import VoxCPMModel

class VoxCPM:
    def __init__(self,
            voxcpm_model_path : str,
            zipenhancer_model_path : str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
            enable_denoiser : bool = True,
        ):
        """初始化 VoxCPM TTS 流水线。

        Args:
            voxcpm_model_path: VoxCPM 模型资源的本地文件系统路径
                (权重、配置等)。通常是先前下载步骤返回的目录。
            zipenhancer_model_path: ModelScope 声学噪声抑制模型
                ID 或本地路径。如果为 None，则不会初始化去噪器。
            enable_denoiser: 是否初始化去噪器流水线。
        """
        print(f"voxcpm_model_path: {voxcpm_model_path}, zipenhancer_model_path: {zipenhancer_model_path}, enable_denoiser: {enable_denoiser}")
        self.tts_model = VoxCPMModel.from_local(voxcpm_model_path)
        self.text_normalizer = None
        if enable_denoiser and zipenhancer_model_path is not None:
            from voxcpm.zipenhancer import ZipEnhancer
            self.denoiser = ZipEnhancer(zipenhancer_model_path)
        else:
            self.denoiser = None
        print("Warm up VoxCPMModel...")
        self.tts_model.generate(
            target_text="Hello, this is the first test sentence.",
            max_len=10,
        )

    @classmethod
    def from_pretrained(cls,
            hf_model_id: str = "openbmb/VoxCPM-0.5B",
            load_denoiser: bool = True,
            zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
            cache_dir: str = None,
            local_files_only: bool = False,
        ):
        """从 Hugging Face Hub 快照实例化 ``VoxCPM``。

        Args:
            hf_model_id: 明确的 Hugging Face 仓库 ID (例如 "org/repo") 或本地路径。
            load_denoiser: 是否初始化去噪器流水线。
            zipenhancer_model_id: ModelScope 声学噪声抑制的去噪器模型 ID 或路径。
            cache_dir: 快照的自定义缓存目录。
            local_files_only: 如果为 True，则仅使用本地文件，不尝试下载。

        Returns:
            VoxCPM: 初始化的实例，其 ``voxcpm_model_path`` 指向下载的快照目录。

        Raises:
            ValueError: 如果未提供有效的 ``hf_model_id`` 或可解析的 ``hf_model_id``。
        """
        repo_id = hf_model_id
        if not repo_id:
            raise ValueError("You must provide hf_model_id")
        
        # Load from local path if provided
        if os.path.isdir(repo_id):
            local_path = repo_id
        else:
            # Otherwise, try from_pretrained (Hub); exit on failure
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        return cls(
            voxcpm_model_path=local_path,
            zipenhancer_model_path=zipenhancer_model_id if load_denoiser else None,
            enable_denoiser=load_denoiser,
        )

    def generate(self, 
            text : str,
            prompt_wav_path : str = None,
            prompt_text : str = None,
            cfg_value : float = 2.0,    
            inference_timesteps : int = 10,
            max_length : int = 4096,
            normalize : bool = True,
            denoise : bool = True,
            retry_badcase : bool = True,
            retry_badcase_max_times : int = 3,
            retry_badcase_ratio_threshold : float = 6.0,
        ):
        """为给定文本合成语音并返回单一波形。

        此方法可选择性地构建和重用提示缓存。如果提供了外部提示
        (``prompt_wav_path`` + ``prompt_text``)，则将用于所有子句子。
        否则，提示缓存将从第一次生成结果构建，并在剩余的文本块中重用。

        Args:
            text: 输入文本。可以包含换行符；每个非空行被视为一个子句子。
            prompt_wav_path: 用于提示的参考音频文件路径。
            prompt_text: 与提示音频对应的文本内容。
            cfg_value: 生成模型的指导比例。
            inference_timesteps: 推理步数。
            max_length: 生成过程中的最大 token 长度。
            normalize: 是否在生成前运行文本标准化。
            denoise: 如果有去噪器可用，是否对提示音频进行去噪。
            retry_badcase: 是否重试错误案例。
            retry_badcase_max_times: 重试错误案例的最大次数。
            retry_badcase_ratio_threshold: 音频到文本比例的阈值。
        Returns:
            numpy.ndarray: CPU 上的 1D 波形数组 (float32)。
        """
        if not text.strip() or not isinstance(text, str):
            raise ValueError("target text must be a non-empty string")
        
        if prompt_wav_path is not None:
            if not os.path.exists(prompt_wav_path):
                raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")
        
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")
        
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        temp_prompt_wav_path = None
        
        try:
            if prompt_wav_path is not None and prompt_text is not None:
                if denoise and self.denoiser is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        temp_prompt_wav_path = tmp_file.name
                    self.denoiser.enhance(prompt_wav_path, output_path=temp_prompt_wav_path)
                    prompt_wav_path = temp_prompt_wav_path
                fixed_prompt_cache = self.tts_model.build_prompt_cache(
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text
                )
            else:
                fixed_prompt_cache = None  # 将会在第一次生成时构建
            
            if normalize:
                if self.text_normalizer is None:
                    from voxcpm.utils.text_normalize import TextNormalizer
                    self.text_normalizer = TextNormalizer()
                text = self.text_normalizer.normalize(text)
            
            wav, target_text_token, generated_audio_feat = self.tts_model.generate_with_prompt_cache(
                            target_text=text,
                            prompt_cache=fixed_prompt_cache,
                            min_len=2,
                            max_len=max_length,
                            inference_timesteps=inference_timesteps,
                            cfg_value=cfg_value,
                            retry_badcase=retry_badcase,
                            retry_badcase_max_times=retry_badcase_max_times,
                            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                        )
        
            return wav.squeeze(0).cpu().numpy()
        
        finally:
            if temp_prompt_wav_path and os.path.exists(temp_prompt_wav_path):
                try:
                    os.unlink(temp_prompt_wav_path)
                except OSError:
                    pass
