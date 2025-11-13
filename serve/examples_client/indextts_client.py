#!/usr/bin/env python3
"""
IndexTTS WebSocket 客户端（命令行）
用法：
    python indextts_client.py "大家好，我现在正在体验 AI 科技！"
"""

import asyncio
import sys
import json
import websockets
import sounddevice as sd
import io
import torchaudio


SERVER_URI = "ws://127.0.0.1:8770"  # 如果在远程服务器运行，请改成服务器地址


async def play_audio_from_ws(text: str):
    async with websockets.connect(SERVER_URI, max_size=None) as ws:
        # 发送 TTS 请求
        req = {
            "type": "tts_stream",
            "text": text,
            "prompt_audio": "asset/zero_shot_prompt.wav",
            "emo_mode": "none",  # none / audio / vector / text
            "interval_silence": 200,
            "max_tokens": 120,
            "use_random": False,
        }
        await ws.send(json.dumps(req, ensure_ascii=False))

        print("已发送请求，等待语音流...")
        async for msg in ws:
            if isinstance(msg, bytes):
                # 音频帧（二进制）
                buffer = io.BytesIO(msg)
                waveform, sample_rate = torchaudio.load(buffer)
                sd.play(waveform.squeeze(0).numpy(), samplerate=sample_rate)
                sd.wait()
            else:
                data = json.loads(msg)
                if data.get("type") == "end":
                    print("合成完成")
                    break
                elif data.get("status") == "error":
                    print("出错：", data.get("message"))
                    break
                else:
                    print("服务器消息：", data)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python indextts_client.py '你要合成的文本'")
        sys.exit(1)

    text = sys.argv[1]
    asyncio.run(play_audio_from_ws(text))
