#!/usr/bin/env python3
"""
VoxCPM WebSocket 语音合成服务器
- 客户端发送 JSON：{"type":"tts_stream", "text":"...", "prompt_text":"...", ...}
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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict

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
        self.task_manager = TaskManager(cleanup_interval=300)
        self.inference_semaphore = asyncio.Semaphore(1)

    def load_model(self):
        """加载模型（自动适配设备）"""
        if self.model is None:
            # 设备自适应
            if self.device == "cuda" and not torch.cuda.is_available():
                print("警告: CUDA 不可用，自动切换到 CPU")
                self.device = "cpu"
            
            is_cuda = self.device == "cuda"
            
            print(f"正在加载 VoxCPM 模型 [设备: {self.device}]...")
            if is_cuda:
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            self.model = VoxCPM.from_pretrained(
                hf_model_id=self.model_dir,
                load_denoiser=False,
                zipenhancer_model_id=self.zipenhancer_model_id,
                local_files_only=True,
                device=self.device,
            )
            print(f"VoxCPM 模型加载完成 [设备: {self.device}]")
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
            "message": "已连接到 VoxCPM WebSocket",
            "usage": "发送 {\"type\":\"tts_stream\", \"text\":\"...\", \"prompt_text\":\"...\", ...} 开始合成",
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

                if prompt_wav_path and not prompt_text:
                    await self.task_manager.update_status(task_id, TaskStatus.FAILED, "缺少 prompt_text")
                    await websocket.send(
                        json.dumps({
                            "status": "error",
                            "task_id": task_id,
                            "session_id": session_id,
                            "message": "使用参考音频时，请提供对应的 prompt_text"
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

                        # 将音频转换为 WAV 格式的字节流
                        buffer = io.BytesIO()
                        sf.write(buffer, wav, 16000, format="wav")
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
        print(f"启动 VoxCPM WebSocket 服务器: ws://{self.host}:{self.port}")
        
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
        description="VoxCPM WebSocket 流式语音合成服务",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/VoxCPM-0.5B",
        help="VoxCPM 模型目录路径"
    )
    parser.add_argument(
        "--zipenhancer-model-id",
        type=str,
        default="models/speech_zipenhancer_ans_multiloss_16k_base",
        help="ZipEnhancer 降噪模型路径"
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
        default=8771,
        help="服务器端口号"
    )
    
    args = parser.parse_args()
    
    server = VoxCPMWebSocketServer(
        model_dir=args.model_dir,
        zipenhancer_model_id=args.zipenhancer_model_id,
        device=args.device,
        host=args.host,
        port=args.port,
    )
    await server.start_server()


if __name__ == "__main__":
    asyncio.run(main())
