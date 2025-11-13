#!/usr/bin/env python3
"""
IndexTTS WebSocket 语音合成服务器
- 客户端发送 JSON：{"type":"tts_stream", "text":"...", "emo_mode":"...", ...}
- 服务端生成音频并以"二进制帧"发回客户端
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
import soundfile as sf
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from indextts.infer_v2 import IndexTTS2


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
        self.cleanup_interval = cleanup_interval

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
        self.task_manager = TaskManager(cleanup_interval=300)
        self.inference_semaphore = asyncio.Semaphore(1)

    def load_model(self):
        """加载模型（自动适配设备）"""
        if self.model is None:
            # 设备自适应
            if self.device == "cuda" and not torch.cuda.is_available():
                print("警告: CUDA 不可用，自动切换到 CPU")
                self.device = "cpu"
            
            is_cuda = self.device.startswith("cuda")
            
            print(f"正在加载 IndexTTS 模型 [设备: {self.device}]...")
            if is_cuda:
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            self.model = IndexTTS2(
                cfg_path=self.cfg_path,
                model_dir=self.model_dir,
                use_fp16=is_cuda,
                device=self.device,
                use_cuda_kernel=is_cuda,
            )
            print(f"IndexTTS 模型加载完成 [设备: {self.device}, FP16: {is_cuda}]")
        return self.model

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
            "message": "已连接到 IndexTTS WebSocket",
            "usage": "发送 {\"type\":\"tts_stream\", \"text\":\"...\", \"emo_mode\":\"none/audio/vector/text\", ...} 开始合成",
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

                # 创建任务
                task = await self.task_manager.create_task(session_id)
                task_id = task.task_id
                session.active_tasks.append(task_id)
                
                # 解析参数
                text = data.get("text")
                prompt_audio_path = data.get("prompt_audio", os.path.join(project_root, "asset", "zero_shot_prompt.wav"))
                emo_mode = data.get("emo_mode", "none")
                emo_audio_path = data.get("emo_audio")
                emo_alpha = data.get("emo_alpha", 1.0)
                emo_vector = data.get("emo_vector")
                emo_text = data.get("emo_text", "")
                interval_silence = data.get("interval_silence", 200)
                max_tokens = data.get("max_tokens", 120)
                use_random = data.get("use_random", False)

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

                if emo_mode not in ["none", "audio", "vector", "text"]:
                    await self.task_manager.update_status(task_id, TaskStatus.FAILED, "无效的 emo_mode")
                    await websocket.send(
                        json.dumps({
                            "status": "error",
                            "task_id": task_id,
                            "session_id": session_id,
                            "message": "emo_mode 必须是 none/audio/vector/text"
                        }, ensure_ascii=False)
                    )
                    continue

                if emo_mode == "audio" and not emo_audio_path:
                    await self.task_manager.update_status(task_id, TaskStatus.FAILED, "缺少 emo_audio")
                    await websocket.send(
                        json.dumps({
                            "status": "error",
                            "task_id": task_id,
                            "session_id": session_id,
                            "message": "情感参考音频模式需要提供 emo_audio"
                        }, ensure_ascii=False)
                    )
                    continue

                if emo_mode == "vector" and not emo_vector:
                    await self.task_manager.update_status(task_id, TaskStatus.FAILED, "缺少 emo_vector")
                    await websocket.send(
                        json.dumps({
                            "status": "error",
                            "task_id": task_id,
                            "session_id": session_id,
                            "message": "情感向量模式需要提供 emo_vector"
                        }, ensure_ascii=False)
                    )
                    continue

                if emo_mode == "text" and not emo_text:
                    await self.task_manager.update_status(task_id, TaskStatus.FAILED, "缺少 emo_text")
                    await websocket.send(
                        json.dumps({
                            "status": "error",
                            "task_id": task_id,
                            "session_id": session_id,
                            "message": "情感文本引导模式需要提供 emo_text"
                        }, ensure_ascii=False)
                    )
                    continue

                # 加载模型
                try:
                    model = self.load_model()
                except Exception as e:
                    error_msg = f"模型加载失败: {str(e)}"
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

                # 发送排队消息
                await websocket.send(json.dumps({
                    "type": "queued",
                    "task_id": task_id,
                    "session_id": session_id,
                    "message": "任务已加入队列，等待处理"
                }, ensure_ascii=False))

                # 进入推理队列
                async with self.inference_semaphore:
                    # 更新任务状态为运行中
                    await self.task_manager.update_status(task_id, TaskStatus.RUNNING)

                    # 发送开始标记
                    await websocket.send(json.dumps({
                        "type": "start",
                        "task_id": task_id,
                        "session_id": session_id,
                        "message": "开始生成音频"
                    }, ensure_ascii=False))

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
                        
                        # IndexTTS 返回的是 (sample_rate, audio_numpy)
                        if result is None or len(result) != 2:
                            error_msg = "生成结果为空"
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

                        sample_rate, audio_numpy = result

                        # 将音频转换为 WAV 格式的字节流
                        if isinstance(audio_numpy, np.ndarray):
                            if audio_numpy.ndim > 1:
                                audio_numpy = audio_numpy.squeeze()
                        else:
                            audio_numpy = np.array(audio_numpy)
                        
                        buffer = io.BytesIO()
                        sf.write(buffer, audio_numpy, sample_rate, format="wav")
                        buffer.seek(0)
                        audio_bytes = buffer.read()

                        # 发送二进制音频数据
                        await websocket.send(audio_bytes)

                        # 合成完成，更新任务状态
                        await self.task_manager.update_status(task_id, TaskStatus.COMPLETED)

                        # 发送结束标记
                        await websocket.send(json.dumps({
                            "type": "end",
                            "task_id": task_id,
                            "session_id": session_id,
                            "message": "生成完成"
                        }, ensure_ascii=False))
                        
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
            print(f"[会话 {session_id[:8]}] 连接已关闭")
        except Exception as e:
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
        print(f"启动 IndexTTS WebSocket 服务器: ws://{self.host}:{self.port}")
        
        # 预加载模型
        self.load_model()
        
        # 启动后台清理任务
        asyncio.create_task(self.task_manager.start_cleanup_loop())
        
        print(f"\n服务器已就绪，等待客户端连接...")
        print(f"   - 会话管理: 已启用")
        print(f"   - 任务追踪: 已启用")
        print(f"   - 推理队列: 已启用（同时处理 1 个请求）")
        print(f"   - 自动清理: 每 {self.task_manager.cleanup_interval} 秒")
        async with websockets.serve(self.websocket_handler, self.host, self.port, max_size=None):
            await asyncio.Future()


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="IndexTTS WebSocket 流式语音合成服务",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/IndexTTS-2",
        help="IndexTTS 模型目录路径"
    )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="models/IndexTTS-2/config.yaml",
        help="IndexTTS 配置文件路径"
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
        default=8770,
        help="服务器端口号"
    )
    
    args = parser.parse_args()
    
    server = IndexTTSWebSocketServer(
        model_dir=args.model_dir,
        cfg_path=args.cfg_path,
        device=args.device,
        host=args.host,
        port=args.port,
    )
    await server.start_server()


if __name__ == "__main__":
    asyncio.run(main())
