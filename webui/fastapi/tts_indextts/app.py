import os
import sys
import io
import json
import base64
import tempfile
from functools import lru_cache

import torch
import soundfile as sf
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from indextts.infer_v2 import IndexTTS2

app = FastAPI(title="IndexTTS2 语音合成")

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

DEFAULT_MODEL_DIR = os.path.join(project_root, "models", "IndexTTS-2")
DEFAULT_CFG_PATH = os.path.join(project_root, "models", "IndexTTS-2", "config.yaml")
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
USE_FP16 = True
USE_CUDA_KERNEL = False

# 检测 CUDA 是否可用
CUDA_AVAILABLE = torch.cuda.is_available()
DEFAULT_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

EMO_LABELS = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]

model_instance = None
current_device = None


@lru_cache(maxsize=2)
def get_model(model_dir: str, cfg_path: str, device: str) -> IndexTTS2:
    """加载并缓存 IndexTTS2 模型。"""
    is_cuda = device == "cuda" and torch.cuda.is_available()
    
    return IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        use_fp16=is_cuda and USE_FP16,
        device=device,
        use_cuda_kernel=is_cuda and USE_CUDA_KERNEL,
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页面"""
    has_default_audio = os.path.isfile(DEFAULT_PROMPT_WAV)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
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
        model_instance = get_model(DEFAULT_MODEL_DIR, DEFAULT_CFG_PATH, device)
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
    text: str = Form(...),
    emo_mode: str = Form("none"),
    emo_alpha: float = Form(1.0),
    emo_text: str = Form(""),
    interval_silence: int = Form(200),
    max_tokens: int = Form(120),
    use_random: bool = Form(False),
    prompt_audio: UploadFile = File(None),
    emo_audio: UploadFile = File(None),
    emo_vector: str = Form("[]"),
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
        
        if not text:
            return JSONResponse(
                {
                    "status": "error",
                    "message": "请输入待合成文本。"
                },
                status_code=400
            )
        
        if emo_mode == "audio" and (not emo_audio or not emo_audio.filename):
            return JSONResponse(
                {
                    "status": "error",
                    "message": "情感参考音频模式需要上传情感音频。"
                },
                status_code=400
            )
        
        if emo_mode == "text" and not emo_text.strip():
            return JSONResponse(
                {
                    "status": "error",
                    "message": "情感文本引导模式需要输入引导文本。"
                },
                status_code=400
            )
        
        prompt_temp_file = None
        emo_temp_file = None
        prompt_audio_path = None
        emo_audio_path = None
        
        try:
            if prompt_audio and prompt_audio.filename:
                prompt_temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                prompt_temp_file.write(await prompt_audio.read())
                prompt_temp_file.flush()
                prompt_audio_path = prompt_temp_file.name
            
            if not prompt_audio_path:
                prompt_audio_path = DEFAULT_PROMPT_WAV
            
            if emo_mode == "audio" and emo_audio and emo_audio.filename:
                emo_temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                emo_temp_file.write(await emo_audio.read())
                emo_temp_file.flush()
                emo_audio_path = emo_temp_file.name
            
            kwargs = {
                "spk_audio_prompt": prompt_audio_path,
                "text": text,
                "output_path": None,
                "use_random": use_random,
                "interval_silence": interval_silence,
                "max_text_tokens_per_segment": max_tokens,
                "verbose": False,
            }
            
            if emo_mode == "audio":
                kwargs["emo_audio_prompt"] = emo_audio_path
                kwargs["emo_alpha"] = emo_alpha
            elif emo_mode == "vector":
                emo_vector_list = json.loads(emo_vector)
                if len(emo_vector_list) != 8:
                    return JSONResponse(
                        {
                            "status": "error",
                            "message": "情感向量必须包含8个值。"
                        },
                        status_code=400
                    )
                kwargs["emo_vector"] = emo_vector_list
                kwargs["emo_alpha"] = 1.0
            elif emo_mode == "text":
                kwargs["use_emo_text"] = True
                kwargs["emo_text"] = emo_text.strip()
                kwargs["emo_alpha"] = emo_alpha
            
            result = model_instance.infer(**kwargs)
            
            if result is None:
                return JSONResponse(
                    {
                        "status": "error",
                        "message": "生成结果为空。"
                    },
                    status_code=500
                )
            
            if isinstance(result, tuple):
                sample_rate = result[0]
                audio_numpy = result[1]
            else:
                audio_numpy = result
                sample_rate = 16000
            
            buffer = io.BytesIO()
            
            if isinstance(audio_numpy, torch.Tensor):
                audio_tensor = audio_numpy
            else:
                audio_tensor = torch.from_numpy(audio_numpy)
            
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.ndim > 2:
                audio_tensor = audio_tensor.squeeze()
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
            
            audio_numpy_final = audio_tensor.squeeze().cpu().numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor.squeeze()
            sf.write(buffer, audio_numpy_final, sample_rate, format='wav')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            
            return JSONResponse({
                "status": "success",
                "message": "生成完成 ✅",
                "audio": audio_base64,
                "sample_rate": sample_rate
            })
            
        finally:
            if prompt_temp_file is not None:
                try:
                    prompt_temp_file.close()
                    os.unlink(prompt_temp_file.name)
                except OSError:
                    pass
            if emo_temp_file is not None:
                try:
                    emo_temp_file.close()
                    os.unlink(emo_temp_file.name)
                except OSError:
                    pass
        
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
    uvicorn.run(app, host="127.0.0.1", port=8001)
