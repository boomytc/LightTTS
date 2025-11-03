#!/usr/bin/env python3
"""
CosyVoice WebSocket 语音合成服务器（流式）
- 客户端发送 JSON：{"type":"tts_stream", "text":"...", "mode":"zero_shot/cross_lingual/instruct", ...}
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
import torchaudio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
matcha_path = os.path.join(project_root, "Matcha-TTS")
sys.path.insert(0, project_root)
sys.path.insert(0, matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed


class CosyVoiceWebSocketServer:
    def __init__(
        self,
        model_dir: str = "models/CosyVoice2-0.5B",
        device: str = "cuda",
        host: str = "0.0.0.0",
        port: int = 8769,
    ):
        self.model_dir = model_dir
        self.device = device
        self.host = host
        self.port = port
        self.model = None

    def load_model(self):
        """懒加载模型"""
        if self.model is None:
            is_cuda = self.device == "cuda" and torch.cuda.is_available()
            self.model = CosyVoice2(
                model_dir=self.model_dir,
                load_jit=False,
                load_trt=False,
                load_vllm=False,
                fp16=is_cuda,
                trt_concurrent=1,
                device=self.device,
            )
        return self.model

    def load_prompt_audio(self, prompt_audio_path: str) -> torch.Tensor:
        """加载参考音频"""
        if not os.path.isfile(prompt_audio_path):
            raise FileNotFoundError(f"参考音频不存在: {prompt_audio_path}")
        return load_wav(prompt_audio_path, 16000)

    async def websocket_handler(self, websocket):
        """处理 WebSocket 连接"""
        # 发送欢迎消息
        welcome_msg = {
            "status": "success",
            "message": "已连接到 CosyVoice WebSocket（流式）",
            "usage": "发送 {\"type\":\"tts_stream\", \"text\":\"...\", \"mode\":\"zero_shot/cross_lingual/instruct\", ...} 开始流式合成",
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
            mode = data.get("mode", "zero_shot")  # zero_shot / cross_lingual / instruct
            prompt_audio_path = data.get("prompt_audio", os.path.join(project_root, "asset", "zero_shot_prompt.wav"))
            prompt_text = data.get("prompt_text", "希望你以后能够做的比我还好呀。")
            instruct_text = data.get("instruct_text", "")
            speed = data.get("speed", 1.0)
            seed = data.get("seed", 0)

            # 参数验证
            if not text or not isinstance(text, str):
                await websocket.send(
                    json.dumps({"status": "error", "message": "缺少有效的 text"}, ensure_ascii=False)
                )
                return

            if mode not in ["zero_shot", "cross_lingual", "instruct"]:
                await websocket.send(
                    json.dumps({"status": "error", "message": "mode 必须是 zero_shot/cross_lingual/instruct"}, ensure_ascii=False)
                )
                return

            if mode == "zero_shot" and not prompt_text:
                await websocket.send(
                    json.dumps({"status": "error", "message": "零样本克隆模式需要提供 prompt_text"}, ensure_ascii=False)
                )
                return

            if mode == "instruct" and not instruct_text:
                await websocket.send(
                    json.dumps({"status": "error", "message": "指令控制模式需要提供 instruct_text"}, ensure_ascii=False)
                )
                return

            # 加载模型和参考音频
            try:
                model = self.load_model()
                prompt_audio = self.load_prompt_audio(prompt_audio_path)
            except Exception as e:
                await websocket.send(
                    json.dumps({"status": "error", "message": f"模型或音频加载失败: {str(e)}"}, ensure_ascii=False)
                )
                return

            # 设置随机种子
            set_all_random_seed(int(seed))

            # 根据模式调用推理
            try:
                if mode == "zero_shot":
                    result_generator = model.inference_zero_shot(
                        text,
                        prompt_text,
                        prompt_audio,
                        stream=True,  # 流式生成
                        speed=float(speed),
                    )
                elif mode == "cross_lingual":
                    result_generator = model.inference_cross_lingual(
                        text,
                        prompt_audio,
                        stream=True,
                        speed=float(speed),
                    )
                elif mode == "instruct":
                    result_generator = model.inference_instruct2(
                        text,
                        instruct_text,
                        prompt_audio,
                        stream=True,
                        speed=float(speed),
                    )
            except Exception as e:
                await websocket.send(
                    json.dumps({"status": "error", "message": f"推理失败: {str(e)}"}, ensure_ascii=False)
                )
                return

            # 流式发送音频片段
            for segment in result_generator:
                audio_tensor = segment.get("tts_speech")
                if audio_tensor is None:
                    continue

                # 将音频转换为 WAV 格式的字节流
                audio_numpy = audio_tensor.squeeze(0).cpu().numpy()
                buffer = io.BytesIO()
                torchaudio.save(
                    buffer,
                    torch.from_numpy(audio_numpy).unsqueeze(0),
                    model.sample_rate,
                    format="wav"
                )
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
        print(f"启动 CosyVoice 流式 WebSocket 服务器: ws://{self.host}:{self.port}")
        async with websockets.serve(self.websocket_handler, self.host, self.port, max_size=None):
            await asyncio.Future()  # 运行直到手动停止


async def main():
    server = CosyVoiceWebSocketServer(
        model_dir="models/CosyVoice2-0.5B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        host="0.0.0.0",
        port=8769,
    )
    await server.start_server()


if __name__ == "__main__":
    asyncio.run(main())
