#!/usr/bin/env python3
"""
VoxCPM WebSocket 语音合成服务器（流式）
- 客户端发送 JSON：{"type":"tts_stream", "text":"...", "prompt_text":"...", ...}
- 服务端实时生成音频并逐段以"二进制帧"发回客户端
- 合成结束后，发送 JSON 文本消息：{"type":"end"}
"""

import asyncio
import json
import os
import sys
import io
import websockets
import torch
import soundfile as sf
import numpy as np

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

try:
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True  # type: ignore
except Exception:
    pass

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import VoxCPMModel


def _disable_optimize(self: VoxCPMModel):
    """禁用在某些 CUDA 设置下失败的 torch.compile 优化。"""
    self.base_lm.forward_step = self.base_lm.forward_step
    self.residual_lm.forward_step = self.residual_lm.forward_step
    self.feat_encoder_step = self.feat_encoder
    self.feat_decoder.estimator = self.feat_decoder.estimator
    return self


VoxCPMModel.optimize = _disable_optimize


class VoxCPMWebSocketServer:
    def __init__(
        self,
        model_dir: str = "models/VoxCPM-0.5B",
        zipenhancer_model_id: str = "models/speech_zipenhancer_ans_multiloss_16k_base",
        device: str = "cuda",
        host: str = "0.0.0.0",
        port: int = 8771,
    ):
        self.model_dir = model_dir
        self.zipenhancer_model_id = zipenhancer_model_id
        self.device = device
        self.host = host
        self.port = port
        self.model = None

    def load_model(self):
        """懒加载模型"""
        if self.model is None:
            self.model = VoxCPM.from_pretrained(
                hf_model_id=self.model_dir,
                load_denoiser=False,  # 统一使用外部 zipenhancer
                zipenhancer_model_id=self.zipenhancer_model_id,
                local_files_only=True,
                device=self.device,
            )
        return self.model

    async def websocket_handler(self, websocket):
        """处理 WebSocket 连接"""
        # 发送欢迎消息
        welcome_msg = {
            "status": "success",
            "message": "已连接到 VoxCPM WebSocket（流式）",
            "usage": "发送 {\"type\":\"tts_stream\", \"text\":\"...\", \"prompt_text\":\"...\", ...} 开始流式合成",
        }
        await websocket.send(json.dumps(welcome_msg, ensure_ascii=False))

        try:
            # 接收客户端请求
            raw_msg = await websocket.recv()
            try:
                data = json.loads(raw_msg)
            except Exception:
                await websocket.send(
                    json.dumps({"status": "error", "message": "无效的JSON"}, ensure_ascii=False)
                )
                return

            if data.get("type") != "tts_stream":
                await websocket.send(
                    json.dumps({"status": "error", "message": "仅支持 tts_stream"}, ensure_ascii=False)
                )
                return

            # 解析参数
            text = data.get("text")
            prompt_wav_path = data.get("prompt_audio")
            prompt_text = data.get("prompt_text", "")
            cfg_value = data.get("cfg_value", 2.0)
            inference_timesteps = data.get("inference_timesteps", 10)
            normalize = data.get("normalize", True)
            denoise = data.get("denoise", True)
            retry_badcase = data.get("retry_badcase", False)
            retry_max_times = data.get("retry_max_times", 3)
            retry_ratio_threshold = data.get("retry_ratio_threshold", 6.0)

            # 参数验证
            if not text or not isinstance(text, str):
                await websocket.send(
                    json.dumps({"status": "error", "message": "缺少有效的 text"}, ensure_ascii=False)
                )
                return

            if prompt_wav_path and not prompt_text:
                await websocket.send(
                    json.dumps({"status": "error", "message": "使用参考音频时，请提供对应的 prompt_text"}, ensure_ascii=False)
                )
                return

            # 加载模型
            try:
                model = self.load_model()
            except Exception as e:
                await websocket.send(
                    json.dumps({"status": "error", "message": f"模型加载失败: {str(e)}"}, ensure_ascii=False)
                )
                return

            # 执行推理
            try:
                wav = model.generate(
                    text=text,
                    prompt_wav_path=prompt_wav_path,  # type: ignore
                    prompt_text=prompt_text,  # type: ignore
                    cfg_value=float(cfg_value),
                    inference_timesteps=int(inference_timesteps),
                    normalize=normalize,
                    denoise=denoise,
                    retry_badcase=retry_badcase,
                    retry_badcase_max_times=int(retry_max_times),
                    retry_badcase_ratio_threshold=float(retry_ratio_threshold),
                )
            except Exception as e:
                await websocket.send(
                    json.dumps({"status": "error", "message": f"推理失败: {str(e)}"}, ensure_ascii=False)
                )
                return

            # 将音频转换为 WAV 格式的字节流
            buffer = io.BytesIO()
            sf.write(buffer, wav, 16000, format="wav")
            buffer.seek(0)
            audio_bytes = buffer.read()

            # 发送二进制音频数据
            await websocket.send(audio_bytes)

            # 合成完成，发送结束标记
            await websocket.send(json.dumps({"type": "end"}, ensure_ascii=False))

        except websockets.exceptions.ConnectionClosed:
            # 客户端中断连接
            return
        except Exception as e:
            # 发生错误时返回错误消息
            try:
                await websocket.send(
                    json.dumps({"status": "error", "message": f"服务器错误: {str(e)}"}, ensure_ascii=False)
                )
            except Exception:
                pass

    async def start_server(self):
        """启动 WebSocket 服务器"""
        print(f"启动 VoxCPM 流式 WebSocket 服务器: ws://{self.host}:{self.port}")
        async with websockets.serve(self.websocket_handler, self.host, self.port, max_size=None):
            await asyncio.Future()  # 运行直到手动停止


async def main():
    server = VoxCPMWebSocketServer(
        model_dir="models/VoxCPM-0.5B",
        zipenhancer_model_id="models/speech_zipenhancer_ans_multiloss_16k_base",
        device="cuda" if torch.cuda.is_available() else "cpu",
        host="0.0.0.0",
        port=8771,
    )
    await server.start_server()


if __name__ == "__main__":
    asyncio.run(main())
