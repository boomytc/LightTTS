"""
ZipEnhancer 模块 - 音频降噪增强器

提供按需导入的 ZipEnhancer 功能用于音频降噪处理。
仅在需要降噪功能时才导入相关依赖。
"""

import os
import tempfile
from typing import Optional
import torchaudio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class ZipEnhancer:
    """ZipEnhancer 音频降噪增强器"""
    def __init__(self, model_path: str = "iic/speech_zipenhancer_ans_multiloss_16k_base"):
        """
        初始化 ZipEnhancer
        参数:
            model_path: ModelScope 模型路径或本地路径
        """
        self.model_path = model_path
        self._pipeline = pipeline(
                Tasks.acoustic_noise_suppression,
                model=self.model_path
            )
        
    def _normalize_loudness(self, wav_path: str):
        """
        音频响度归一化

        参数:
            wav_path: 音频文件路径
        """
        audio, sr = torchaudio.load(wav_path)
        loudness = torchaudio.functional.loudness(audio, sr)
        normalized_audio = torchaudio.functional.gain(audio, -20-loudness)
        torchaudio.save(wav_path, normalized_audio, sr)
    
    def enhance(self, input_path: str, output_path: Optional[str] = None, 
                normalize_loudness: bool = True) -> str:
        """
        音频降噪增强
        参数:
            input_path: 输入音频文件路径
            output_path: 输出音频文件路径（可选，默认创建临时文件）
            normalize_loudness: 是否执行响度归一化
        返回:
            str: 输出音频文件路径
        异常:
            RuntimeError: 如果 pipeline 未初始化或处理失败
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file does not exist: {input_path}")
        # 如果未指定输出路径，则创建临时文件
        if output_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                output_path = tmp_file.name
        try:
            # 执行降噪处理
            self._pipeline(input_path, output_path=output_path)
            # 响度归一化
            if normalize_loudness:
                self._normalize_loudness(output_path)
            return output_path
        except Exception as e:
            # 清理可能创建的临时文件
            if output_path and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
            raise RuntimeError(f"Audio denoising processing failed: {e}")