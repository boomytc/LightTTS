import os
import sys
import io
import base64
from functools import lru_cache

import torch
import torchaudio
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
matcha_path = os.path.join(project_root, "Matcha-TTS")
sys.path.insert(0, project_root)
sys.path.insert(0, matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

app = FastAPI(title="CosyVoice2 语音合成")

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

DEFAULT_MODEL_DIR = os.path.join(project_root, "models", "CosyVoice2-0.5B")
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呀。"
USE_FP16 = True
LOAD_JIT = False
LOAD_TRT = False
LOAD_VLLM = False
TRT_CONCURRENT = 1

# 检测 CUDA 是否可用
CUDA_AVAILABLE = torch.cuda.is_available()
DEFAULT_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

MODE_MAPPING = {
    "zero_shot": "零样本克隆",
    "cross_lingual": "跨语言克隆",
    "instruct": "指令控制",
}

model_instance = None
current_device = None


@lru_cache(maxsize=2)
def get_model(model_dir: str, device: str) -> CosyVoice2:
    """根据配置加载并缓存 CosyVoice2 模型。"""
    is_cuda = device == "cuda" and torch.cuda.is_available()
    
    return CosyVoice2(
        model_dir=model_dir,
        load_jit=LOAD_JIT,
        load_trt=LOAD_TRT,
        load_vllm=LOAD_VLLM,
        fp16=is_cuda and USE_FP16,
        trt_concurrent=TRT_CONCURRENT,
        device=device,
    )


def prepare_prompt_audio(prompt_audio_path: str) -> torch.Tensor:
    """验证并加载参考提示音频。"""
    if not prompt_audio_path:
        prompt_audio_path = DEFAULT_PROMPT_WAV

    if not os.path.isfile(prompt_audio_path):
        raise FileNotFoundError(f"参考音频不存在: {prompt_audio_path}")

    try:
        sample_rate = torchaudio.info(prompt_audio_path).sample_rate
    except Exception as exc:
        raise RuntimeError(f"无法读取参考音频: {exc}") from exc

    if sample_rate < 16000:
        raise ValueError("参考音频采样率需至少 16000Hz。")

    return load_wav(prompt_audio_path, 16000)


def merge_segments(segments) -> torch.Tensor:
    """将生成的音频片段合并成单一波形。"""
    audio_tensors = []
    for segment in segments:
        tensor = segment.get("tts_speech")
        if tensor is None:
            continue
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        audio_tensors.append(tensor)

    if not audio_tensors:
        raise RuntimeError("生成结果为空。")

    return torch.cat(audio_tensors, dim=-1)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页面"""
    has_default_audio = os.path.isfile(DEFAULT_PROMPT_WAV)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_prompt_text": DEFAULT_PROMPT_TEXT,
            "has_default_audio": has_default_audio,
            "cuda_available": CUDA_AVAILABLE,
            "default_device": DEFAULT_DEVICE
        }
    )


@app.post("/api/load_model")
async def load_model(request: Request):
    """加载模型API"""
    global model_instance, current_device
    
    data = await request.json()
    device = data.get("device", DEFAULT_DEVICE)
    
    # 强制使用检测到的设备
    if not CUDA_AVAILABLE:
        device = "cpu"
    
    if model_instance is not None and current_device == device:
        return JSONResponse({
            "status": "success",
            "message": "模型已加载，无需重复加载。"
        })
    
    try:
        model_instance = get_model(DEFAULT_MODEL_DIR, device)
        current_device = device
        return JSONResponse({
            "status": "success",
            "message": f"模型加载完成 ✅ (设备: {device.upper()})"
        })
    except Exception as exc:
        return JSONResponse(
            {
                "status": "error",
                "message": f"模型加载失败: {exc}"
            },
            status_code=500
        )


@app.post("/api/generate")
async def generate(
    mode: str = Form(...),
    text: str = Form(...),
    prompt_text: str = Form(""),
    instruct_text: str = Form(""),
    speed: float = Form(1.0),
    seed: int = Form(0),
    prompt_audio: UploadFile = File(None),
):
    """生成语音API"""
    global model_instance
    
    if model_instance is None:
        return JSONResponse(
            {
                "status": "error",
                "message": "请先加载模型。"
            },
            status_code=400
        )
    
    try:
        text = text.strip()
        prompt_text = prompt_text.strip()
        instruct_text = instruct_text.strip()
        
        if not text:
            return JSONResponse(
                {
                    "status": "error",
                    "message": "请输入待合成文本。"
                },
                status_code=400
            )
        
        if mode == "zero_shot" and not prompt_text:
            return JSONResponse(
                {
                    "status": "error",
                    "message": "零样本克隆模式需要提供参考文本。"
                },
                status_code=400
            )
        
        if mode == "instruct" and not instruct_text:
            return JSONResponse(
                {
                    "status": "error",
                    "message": "指令控制模式需要提供指令文本。"
                },
                status_code=400
            )
        
        prompt_audio_path = None
        if prompt_audio and prompt_audio.filename:
            temp_path = os.path.join("/tmp", f"prompt_{os.getpid()}.wav")
            with open(temp_path, "wb") as f:
                f.write(await prompt_audio.read())
            prompt_audio_path = temp_path
        
        if not prompt_audio_path:
            prompt_audio_path = DEFAULT_PROMPT_WAV
        
        try:
            prompt_speech_16k = prepare_prompt_audio(prompt_audio_path)
        except Exception as exc:
            return JSONResponse(
                {
                    "status": "error",
                    "message": str(exc)
                },
                status_code=400
            )
        finally:
            if prompt_audio_path and prompt_audio_path.startswith("/tmp"):
                try:
                    os.remove(prompt_audio_path)
                except:
                    pass
        
        set_all_random_seed(seed)
        
        if mode == "zero_shot":
            result = model_instance.inference_zero_shot(
                text,
                prompt_text,
                prompt_speech_16k,
                stream=False,
                speed=speed,
            )
        elif mode == "cross_lingual":
            result = model_instance.inference_cross_lingual(
                text,
                prompt_speech_16k,
                stream=False,
                speed=speed,
            )
        elif mode == "instruct":
            result = model_instance.inference_instruct2(
                text,
                instruct_text,
                prompt_speech_16k,
                stream=False,
                speed=speed,
            )
        else:
            return JSONResponse(
                {
                    "status": "error",
                    "message": f"无效的模式: {mode}"
                },
                status_code=400
            )
        
        audio_tensor = merge_segments(result)
        audio_numpy = audio_tensor.squeeze(0).cpu().numpy()
        
        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            torch.from_numpy(audio_numpy).unsqueeze(0),
            model_instance.sample_rate,
            format="wav"
        )
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        
        return JSONResponse({
            "status": "success",
            "message": "生成完成 ✅",
            "audio": audio_base64,
            "sample_rate": model_instance.sample_rate
        })
        
    except Exception as exc:
        return JSONResponse(
            {
                "status": "error",
                "message": f"推理失败: {exc}"
            },
            status_code=500
        )


@app.get("/api/default_audio")
async def get_default_audio():
    """获取默认参考音频"""
    if os.path.isfile(DEFAULT_PROMPT_WAV):
        return FileResponse(DEFAULT_PROMPT_WAV, media_type="audio/wav")
    return JSONResponse(
        {
            "status": "error",
            "message": "默认音频文件不存在"
        },
        status_code=404
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
