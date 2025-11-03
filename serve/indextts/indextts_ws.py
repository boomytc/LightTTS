#!/usr/bin/env python3
"""
IndexTTS WebSocket 语音合成服务器（流式）
- 客户端发送 JSON：{"type":"tts_stream", "text":"...", "emo_mode":"...", ...}
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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from indextts.infer_v2 import IndexTTS2


class IndexTTSWebSocketServer:
    def __init__(
        self,
        model_dir: str = "models/IndexTTS-2",
        cfg_path: str = "models/IndexTTS-2/config.yaml",
        device: str = "cuda",
        host: str = "0.0.0.0",
        port: int = 8770,
    ):
        self.model_dir = model_dir
        self.cfg_path = cfg_path
        self.device = device
        self.host = host
        self.port = port
        self.model = None

    def load_model(self):
        """加载模型"""
        if self.model is None:
            print("正在加载 IndexTTS 模型...")
            is_cuda = self.device.startswith("cuda")
            self.model = IndexTTS2(
                cfg_path=self.cfg_path,
                model_dir=self.model_dir,
                use_fp16=is_cuda,
                device=self.device,
                use_cuda_kernel=is_cuda,
            )
            print(f"✅ IndexTTS 模型加载完成 [设备: {self.device}]")
        return self.model

    async def websocket_handler(self, websocket):
        """处理 WebSocket 连接"""
        # 发送欢迎消息
        welcome_msg = {
            "status": "success",
            "message": "已连接到 IndexTTS WebSocket（流式）",
            "usage": "发送 {\"type\":\"tts_stream\", \"text\":\"...\", \"emo_mode\":\"none/audio/vector/text\", ...} 开始流式合成",
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
            prompt_audio_path = data.get("prompt_audio", os.path.join(project_root, "asset", "zero_shot_prompt.wav"))
            emo_mode = data.get("emo_mode", "none")  # none / audio / vector / text
            emo_audio_path = data.get("emo_audio")
            emo_alpha = data.get("emo_alpha", 1.0)
            emo_vector = data.get("emo_vector")  # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emo_text = data.get("emo_text", "")
            interval_silence = data.get("interval_silence", 200)
            max_tokens = data.get("max_tokens", 120)
            use_random = data.get("use_random", False)

            # 参数验证
            if not text or not isinstance(text, str):
                await websocket.send(
                    json.dumps({"status": "error", "message": "缺少有效的 text"}, ensure_ascii=False)
                )
                return

            if emo_mode not in ["none", "audio", "vector", "text"]:
                await websocket.send(
                    json.dumps({"status": "error", "message": "emo_mode 必须是 none/audio/vector/text"}, ensure_ascii=False)
                )
                return

            if emo_mode == "audio" and not emo_audio_path:
                await websocket.send(
                    json.dumps({"status": "error", "message": "情感参考音频模式需要提供 emo_audio"}, ensure_ascii=False)
                )
                return

            if emo_mode == "vector" and not emo_vector:
                await websocket.send(
                    json.dumps({"status": "error", "message": "情感向量模式需要提供 emo_vector"}, ensure_ascii=False)
                )
                return

            if emo_mode == "text" and not emo_text:
                await websocket.send(
                    json.dumps({"status": "error", "message": "情感文本引导模式需要提供 emo_text"}, ensure_ascii=False)
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

            # 构建推理参数
            kwargs = {
                "spk_audio_prompt": prompt_audio_path,
                "text": text,
                "output_path": None,
                "use_random": use_random,
                "interval_silence": int(interval_silence),
                "max_text_tokens_per_segment": int(max_tokens),
                "verbose": False,
            }

            if emo_mode == "audio":
                kwargs["emo_audio_prompt"] = emo_audio_path
                kwargs["emo_alpha"] = float(emo_alpha)
            elif emo_mode == "vector":
                kwargs["emo_vector"] = emo_vector
                kwargs["emo_alpha"] = 1.0
            elif emo_mode == "text":
                kwargs["use_emo_text"] = True
                kwargs["emo_text"] = emo_text
                kwargs["emo_alpha"] = float(emo_alpha)

            # 执行推理
            try:
                result = model.infer(**kwargs)
            except Exception as e:
                await websocket.send(
                    json.dumps({"status": "error", "message": f"推理失败: {str(e)}"}, ensure_ascii=False)
                )
                return

            # IndexTTS 返回的是 (sample_rate, audio_numpy)
            if result is None or len(result) != 2:
                await websocket.send(
                    json.dumps({"status": "error", "message": "生成结果为空"}, ensure_ascii=False)
                )
                return

            sample_rate, audio_numpy = result

            # 将音频转换为 WAV 格式的字节流
            # 确保 audio_numpy 是 1D 或 2D numpy 数组
            if isinstance(audio_numpy, np.ndarray):
                if audio_numpy.ndim > 1:
                    audio_numpy = audio_numpy.squeeze()  # 移除多余维度
            else:
                audio_numpy = np.array(audio_numpy)
            
            buffer = io.BytesIO()
            sf.write(buffer, audio_numpy, sample_rate, format="wav")
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
        print(f"启动 IndexTTS 流式 WebSocket 服务器: ws://{self.host}:{self.port}")
        
        # 预加载模型
        self.load_model()
        
        print(f"\n🚀 服务器已就绪，等待客户端连接...")
        async with websockets.serve(self.websocket_handler, self.host, self.port, max_size=None):
            await asyncio.Future()  # 运行直到手动停止


async def main():
    server = IndexTTSWebSocketServer(
        model_dir="models/IndexTTS-2",
        cfg_path="models/IndexTTS-2/config.yaml",
        device="cuda" if torch.cuda.is_available() else "cpu",
        host="0.0.0.0",
        port=8770,
    )
    await server.start_server()


if __name__ == "__main__":
    asyncio.run(main())
