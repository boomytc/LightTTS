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
import uuid
import websockets
import torch
import torchaudio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
matcha_path = os.path.join(project_root, "Matcha-TTS")
sys.path.insert(0, project_root)
sys.path.insert(0, matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """任务对象"""
    task_id: str
    session_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
        }


@dataclass
class Session:
    """会话对象"""
    session_id: str
    websocket: object
    connected_at: datetime = field(default_factory=datetime.now)
    active_tasks: list = field(default_factory=list)


class TaskManager:
    """任务管理器"""
    def __init__(self, cleanup_interval: int = 300):
        self.tasks: Dict[str, Task] = {}
        self.lock = asyncio.Lock()
        self.cleanup_interval = cleanup_interval  # 清理间隔（秒）

    async def create_task(self, session_id: str) -> Task:
        """创建新任务（自动生成task_id）"""
        async with self.lock:
            task_id = str(uuid.uuid4())
            task = Task(task_id=task_id, session_id=session_id)
            self.tasks[task_id] = task
            print(f"[会话 {session_id[:8]}] 创建任务: {task_id[:8]}")
            return task

    async def update_status(
        self, 
        task_id: str, 
        status: TaskStatus, 
        error: Optional[str] = None
    ):
        """更新任务状态"""
        async with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = status
                if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task.completed_at = datetime.now()
                if error:
                    task.error_message = error
                print(f"[任务 {task_id[:8]}] 状态更新: {status.value}")

    async def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务信息"""
        async with self.lock:
            return self.tasks.get(task_id)

    async def cleanup_old_tasks(self):
        """清理过期任务"""
        async with self.lock:
            now = datetime.now()
            expired_tasks = [
                task_id for task_id, task in self.tasks.items()
                if task.completed_at and (now - task.completed_at).total_seconds() > self.cleanup_interval
            ]
            for task_id in expired_tasks:
                del self.tasks[task_id]
            if expired_tasks:
                print(f"清理了 {len(expired_tasks)} 个过期任务")

    async def cleanup_session_tasks(self, session_id: str):
        """清理指定会话的所有任务"""
        async with self.lock:
            session_tasks = [
                task_id for task_id, task in self.tasks.items()
                if task.session_id == session_id
            ]
            for task_id in session_tasks:
                del self.tasks[task_id]
            if session_tasks:
                print(f"[会话 {session_id[:8]}] 清理了 {len(session_tasks)} 个任务")

    async def start_cleanup_loop(self):
        """启动后台清理任务"""
        while True:
            await asyncio.sleep(self.cleanup_interval)
            await self.cleanup_old_tasks()


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
        self.task_manager = TaskManager(cleanup_interval=300)  # 5分钟清理一次

    def load_model(self):
        """加载模型（自动适配设备）"""
        if self.model is None:
            # 设备自适应：如果指定 cuda 但不可用，自动降级到 cpu
            if self.device == "cuda" and not torch.cuda.is_available():
                print("⚠️  CUDA 不可用，自动切换到 CPU")
                self.device = "cpu"
            
            is_cuda = self.device == "cuda"
            
            print(f"正在加载 CosyVoice 模型 [设备: {self.device}]...")
            if is_cuda:
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            self.model = CosyVoice2(
                model_dir=self.model_dir,
                load_jit=False,
                load_trt=False,
                load_vllm=False,
                fp16=is_cuda,  # 仅 CUDA 启用 FP16
                trt_concurrent=1,
                device=self.device,
            )
            print(f"✅ CosyVoice 模型加载完成 [设备: {self.device}, FP16: {is_cuda}]")
        return self.model

    def load_prompt_audio(self, prompt_audio_path: str) -> torch.Tensor:
        """加载参考音频"""
        if not os.path.isfile(prompt_audio_path):
            raise FileNotFoundError(f"参考音频不存在: {prompt_audio_path}")
        return load_wav(prompt_audio_path, 16000)

    async def websocket_handler(self, websocket):
        """处理 WebSocket 连接"""
        # 创建会话
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id, websocket=websocket)
        print(f"\n[会话 {session_id[:8]}] 新连接建立")
        
        # 发送欢迎消息
        welcome_msg = {
            "status": "success",
            "session_id": session_id,
            "message": "已连接到 CosyVoice WebSocket（流式）",
            "usage": "发送 {\"type\":\"tts_stream\", \"text\":\"...\", \"mode\":\"zero_shot/cross_lingual/instruct\", ...} 开始流式合成",
        }
        await websocket.send(json.dumps(welcome_msg, ensure_ascii=False))

        try:
            # 支持持久连接，可以处理多个请求
            async for raw_msg in websocket:
                try:
                    data = json.loads(raw_msg)
                except Exception:
                    await websocket.send(
                        json.dumps({
                            "status": "error",
                            "session_id": session_id,
                            "message": "无效的JSON"
                        }, ensure_ascii=False)
                    )
                    continue

                msg_type = data.get("type")
                
                # 处理任务状态查询
                if msg_type == "task_status":
                    task_id = data.get("task_id")
                    if not task_id:
                        await websocket.send(
                            json.dumps({
                                "status": "error",
                                "session_id": session_id,
                                "message": "缺少 task_id"
                            }, ensure_ascii=False)
                        )
                        continue
                    
                    task = await self.task_manager.get_task(task_id)
                    if task:
                        await websocket.send(
                            json.dumps({
                                "type": "task_status",
                                "session_id": session_id,
                                **task.to_dict()
                            }, ensure_ascii=False)
                        )
                    else:
                        await websocket.send(
                            json.dumps({
                                "status": "error",
                                "session_id": session_id,
                                "message": f"任务不存在: {task_id}"
                            }, ensure_ascii=False)
                        )
                    continue

                if msg_type != "tts_stream":
                    await websocket.send(
                        json.dumps({
                            "status": "error",
                            "session_id": session_id,
                            "message": f"不支持的消息类型: {msg_type}"
                        }, ensure_ascii=False)
                    )
                    continue

                # 创建任务（服务端自动生成task_id）
                task = await self.task_manager.create_task(session_id)
                task_id = task.task_id
                session.active_tasks.append(task_id)
                
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
                    await self.task_manager.update_status(task_id, TaskStatus.FAILED, "缺少有效的 text")
                    await websocket.send(
                        json.dumps({
                            "status": "error",
                            "task_id": task_id,
                            "session_id": session_id,
                            "message": "缺少有效的 text"
                        }, ensure_ascii=False)
                    )
                    continue

                if mode not in ["zero_shot", "cross_lingual", "instruct"]:
                    await self.task_manager.update_status(task_id, TaskStatus.FAILED, "无效的 mode")
                    await websocket.send(
                        json.dumps({
                            "status": "error",
                            "task_id": task_id,
                            "session_id": session_id,
                            "message": "mode 必须是 zero_shot/cross_lingual/instruct"
                        }, ensure_ascii=False)
                    )
                    continue

                if mode == "zero_shot" and not prompt_text:
                    await self.task_manager.update_status(task_id, TaskStatus.FAILED, "缺少 prompt_text")
                    await websocket.send(
                        json.dumps({
                            "status": "error",
                            "task_id": task_id,
                            "session_id": session_id,
                            "message": "零样本克隆模式需要提供 prompt_text"
                        }, ensure_ascii=False)
                    )
                    continue

                if mode == "instruct" and not instruct_text:
                    await self.task_manager.update_status(task_id, TaskStatus.FAILED, "缺少 instruct_text")
                    await websocket.send(
                        json.dumps({
                            "status": "error",
                            "task_id": task_id,
                            "session_id": session_id,
                            "message": "指令控制模式需要提供 instruct_text"
                        }, ensure_ascii=False)
                    )
                    continue

                # 加载模型和参考音频
                try:
                    model = self.load_model()
                    prompt_audio = self.load_prompt_audio(prompt_audio_path)
                except Exception as e:
                    error_msg = f"模型或音频加载失败: {str(e)}"
                    await self.task_manager.update_status(task_id, TaskStatus.FAILED, error_msg)
                    await websocket.send(
                        json.dumps({
                            "status": "error",
                            "task_id": task_id,
                            "session_id": session_id,
                            "message": error_msg
                        }, ensure_ascii=False)
                    )
                    continue

                # 设置随机种子
                set_all_random_seed(int(seed))

                # 更新任务状态为运行中
                await self.task_manager.update_status(task_id, TaskStatus.RUNNING)

                # 发送开始标记
                await websocket.send(json.dumps({
                    "type": "start",
                    "task_id": task_id,
                    "session_id": session_id,
                    "message": "开始生成音频"
                }, ensure_ascii=False))

                # 根据模式调用推理并实时流式发送
                result_generator = None
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

                    # 实时流式发送音频片段
                    if result_generator is not None:
                        segment_count = 0
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

                            # 立即发送二进制音频数据（边生成边发送）
                            await websocket.send(audio_bytes)
                            segment_count += 1

                        # 合成完成，更新任务状态
                        await self.task_manager.update_status(task_id, TaskStatus.COMPLETED)
                        
                        # 发送结束标记
                        await websocket.send(json.dumps({
                            "type": "end",
                            "task_id": task_id,
                            "session_id": session_id,
                            "message": "生成完成",
                            "segments": segment_count
                        }, ensure_ascii=False))
                    else:
                        error_msg = "无效的模式"
                        await self.task_manager.update_status(task_id, TaskStatus.FAILED, error_msg)
                        await websocket.send(
                            json.dumps({
                                "status": "error",
                                "task_id": task_id,
                                "session_id": session_id,
                                "message": error_msg
                            }, ensure_ascii=False)
                        )
                except Exception as e:
                    error_msg = f"推理失败: {str(e)}"
                    await self.task_manager.update_status(task_id, TaskStatus.FAILED, error_msg)
                    await websocket.send(
                        json.dumps({
                            "status": "error",
                            "task_id": task_id,
                            "session_id": session_id,
                            "message": error_msg
                        }, ensure_ascii=False)
                    )
                    continue

        except websockets.exceptions.ConnectionClosed:
            # 客户端中断连接
            print(f"[会话 {session_id[:8]}] 连接已关闭")
        except Exception as e:
            # 发生错误时返回错误消息
            print(f"[会话 {session_id[:8]}] 服务器错误: {str(e)}")
            try:
                await websocket.send(
                    json.dumps({
                        "status": "error",
                        "session_id": session_id,
                        "message": f"服务器错误: {str(e)}"
                    }, ensure_ascii=False)
                )
            except Exception:
                pass
        finally:
            # 清理会话相关的所有任务
            await self.task_manager.cleanup_session_tasks(session_id)
            print(f"[会话 {session_id[:8]}] 会话已清理")

    async def start_server(self):
        """启动 WebSocket 服务器"""
        print(f"启动 CosyVoice 流式 WebSocket 服务器: ws://{self.host}:{self.port}")
        
        # 预加载模型
        self.load_model()
        
        # 启动后台清理任务
        asyncio.create_task(self.task_manager.start_cleanup_loop())
        
        print(f"\n🚀 服务器已就绪，等待客户端连接...")
        print(f"   - 会话管理: 已启用")
        print(f"   - 任务追踪: 已启用")
        print(f"   - 自动清理: 每 {self.task_manager.cleanup_interval} 秒")
        async with websockets.serve(self.websocket_handler, self.host, self.port, max_size=None):
            await asyncio.Future()  # 运行直到手动停止


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CosyVoice WebSocket 流式语音合成服务",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/CosyVoice2-0.5B",
        help="CosyVoice 模型目录路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="运行设备（cuda 或 cpu，如果 cuda 不可用会自动降级到 cpu）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器绑定地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8769,
        help="服务器端口号"
    )
    
    args = parser.parse_args()
    
    server = CosyVoiceWebSocketServer(
        model_dir=args.model_dir,
        device=args.device,
        host=args.host,
        port=args.port,
    )
    await server.start_server()


if __name__ == "__main__":
    asyncio.run(main())
