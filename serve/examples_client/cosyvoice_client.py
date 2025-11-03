#!/usr/bin/env python3
"""
CosyVoice WebSocket 客户端（命令行）
支持实时流式播放和持久连接
用法：
    python cosyvoice_client.py "你好，欢迎使用 CosyVoice！"
    python cosyvoice_client.py "测试文本" --mode cross_lingual
    python cosyvoice_client.py "用四川话说" --mode instruct --instruct "用四川话说这句话"
"""

import asyncio
import sys
import json
import argparse
import websockets
import sounddevice as sd
import io
import torchaudio
import torch
import time


SERVER_URI = "ws://127.0.0.1:8769"  # 如果在远程服务器运行，请改成服务器地址


async def play_audio_from_ws(text: str, mode: str = "zero_shot", instruct_text: str = ""):
    """连接到 WebSocket 服务器并实时流式播放音频"""
    start_time = time.time()
    first_audio_time = None
    total_segments = 0
    
    async with websockets.connect(SERVER_URI, max_size=None) as ws:
        # 接收欢迎消息
        welcome = await ws.recv()
        print("📡", json.loads(welcome).get("message"))
        
        # 发送 TTS 请求
        req = {
            "type": "tts_stream",
            "mode": mode,
            "text": text,
            "prompt_audio": "asset/zero_shot_prompt.wav",
        }
        
        if mode == "zero_shot":
            req["prompt_text"] = "希望你以后能够做的比我还好呀。"
        elif mode == "instruct":
            req["instruct_text"] = instruct_text or "用温柔的语气说"
        
        await ws.send(json.dumps(req, ensure_ascii=False))
        print(f"\n🎯 发送请求 [模式: {mode}]")
        print(f"📝 文本: {text}")
        print("\n⏳ 等待流式音频...\n")
        
        async for msg in ws:
            if isinstance(msg, bytes):
                # 音频帧（二进制）- 实时播放
                if first_audio_time is None:
                    first_audio_time = time.time()
                    latency = (first_audio_time - start_time) * 1000
                    print(f"⚡ 首包延迟: {latency:.0f}ms\n")
                
                buffer = io.BytesIO(msg)
                waveform, sample_rate = torchaudio.load(buffer)
                total_segments += 1
                
                # 实时播放音频片段
                print(f"🔊 播放片段 #{total_segments}")
                sd.play(waveform.squeeze(0).numpy(), samplerate=sample_rate)
                sd.wait()
            else:
                # 文本消息（控制信息）
                data = json.loads(msg)
                msg_type = data.get("type")
                
                if msg_type == "start":
                    print("▶️ ", data.get("message"))
                elif msg_type == "end":
                    total_time = (time.time() - start_time) * 1000
                    print(f"\n✅ 合成完成")
                    print(f"📊 总片段数: {data.get('segments', total_segments)}")
                    print(f"⏱️  总耗时: {total_time:.0f}ms")
                    if first_audio_time:
                        print(f"⚡ 首包延迟: {(first_audio_time - start_time) * 1000:.0f}ms")
                    break
                elif data.get("status") == "error":
                    print("❌ 出错：", data.get("message"))
                    break
                else:
                    print("📨 服务器消息：", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CosyVoice WebSocket 流式语音合成客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python cosyvoice_client.py "你好，欢迎使用 CosyVoice！"
  python cosyvoice_client.py "Hello, welcome!" --mode cross_lingual
  python cosyvoice_client.py "测试文本" --mode instruct --instruct "用四川话说这句话"
        """
    )
    
    parser.add_argument("text", help="待合成的文本")
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "cross_lingual", "instruct"],
        default="zero_shot",
        help="合成模式 (默认: zero_shot)"
    )
    parser.add_argument(
        "--instruct",
        default="",
        help="指令文本 (仅在 instruct 模式下使用)"
    )
    parser.add_argument(
        "--server",
        default=SERVER_URI,
        help=f"服务器地址 (默认: {SERVER_URI})"
    )
    
    args = parser.parse_args()
    
    # 更新服务器地址
    SERVER_URI = args.server
    
    # 运行客户端
    try:
        asyncio.run(play_audio_from_ws(args.text, args.mode, args.instruct))
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)
