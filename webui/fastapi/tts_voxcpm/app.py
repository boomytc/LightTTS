import os
import sys
import io
import base64
import tempfile

import torch
import soundfile as sf
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import VoxCPMModel

app = FastAPI(title="VoxCPM 语音合成")

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

MODEL_ID = os.path.join(project_root, "models", "VoxCPM-0.5B")
ZIPENHANCER_MODEL_ID = os.path.join(project_root, "models", "speech_zipenhancer_ans_multiloss_16k_base")
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做得比我还好哟。"

# 检测 CUDA 是否可用
CUDA_AVAILABLE = torch.cuda.is_available()
DEFAULT_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

try:
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True
except Exception:
    pass


def disable_optimize(self: VoxCPMModel):
    """禁用在某些 CUDA 设置下失败的 torch.compile 优化。"""
    self.base_lm.forward_step = self.base_lm.forward_step
    self.residual_lm.forward_step = self.residual_lm.forward_step
    self.feat_encoder_step = self.feat_encoder
    self.feat_decoder.estimator = self.feat_decoder.estimator
    return self


VoxCPMModel.optimize = disable_optimize

model_instance = None
current_device = None


def get_model(device: str) -> VoxCPM:
    """根据配置加载并缓存 VoxCPM 模型。"""
    global model_instance, current_device
    
    if model_instance is not None and current_device == device:
        return model_instance
    
    model_instance = VoxCPM.from_pretrained(
        hf_model_id=MODEL_ID,
        load_denoiser=False,
        zipenhancer_model_id=ZIPENHANCER_MODEL_ID,
        local_files_only=True,
        device=device,
    )
    current_device = device
    return model_instance


def validate_prompt_audio(prompt_audio_path: str) -> str:
    """验证参考提示音频并返回有效路径。"""
    if not prompt_audio_path:
        prompt_audio_path = DEFAULT_PROMPT_WAV

    if not os.path.isfile(prompt_audio_path):
        raise FileNotFoundError(f"参考音频不存在: {prompt_audio_path}")

    try:
        info = sf.info(prompt_audio_path)
        if info.samplerate < 16000:
            raise ValueError("参考音频采样率需至少 16000Hz。")
    except Exception as exc:
        raise RuntimeError(f"无法读取参考音频: {exc}") from exc
    
    return prompt_audio_path


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
        get_model(device)
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
    prompt_text: str = Form(""),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
    normalize: bool = Form(True),
    denoise: bool = Form(True),
    retry_badcase: bool = Form(True),
    retry_max_times: int = Form(3),
    retry_ratio_threshold: float = Form(6.0),
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
    
    temp_file_obj = None
    
    try:
        text = text.strip()
        prompt_text = prompt_text.strip()
        
        if not text:
            return JSONResponse(
                {
                    "status": "error",
                    "message": "请输入待合成文本。"
                },
                status_code=400
            )
        
        prompt_audio_path = None
        if prompt_audio and prompt_audio.filename:
            temp_file_obj = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file_obj.write(await prompt_audio.read())
            temp_file_obj.flush()
            prompt_audio_path = temp_file_obj.name
        
        if not prompt_audio_path:
            prompt_audio_path = DEFAULT_PROMPT_WAV
        
        if prompt_audio_path != DEFAULT_PROMPT_WAV and not prompt_text:
            return JSONResponse(
                {
                    "status": "error",
                    "message": "使用参考音频时，请提供对应的参考文本。"
                },
                status_code=400
            )
        
        validated_path = validate_prompt_audio(prompt_audio_path)
        
        wav = model_instance.generate(
            text=text,
            prompt_wav_path=validated_path,
            prompt_text=prompt_text or None,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_max_times,
            retry_badcase_ratio_threshold=retry_ratio_threshold,
        )
        
        if wav.ndim > 1:
            wav = wav.squeeze()
        
        buffer = io.BytesIO()
        sf.write(buffer, wav, 16000, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        
        return JSONResponse({
            "status": "success",
            "message": "生成完成 ✅",
            "audio": audio_base64,
            "sample_rate": 16000
        })
        
    except Exception as exc:
        return JSONResponse(
            {
                "status": "error",
                "message": f"推理失败: {exc}"
            },
            status_code=500
        )
    finally:
        if temp_file_obj is not None:
            try:
                temp_file_obj.close()
                os.unlink(temp_file_obj.name)
            except OSError:
                pass


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
    uvicorn.run(app, host="127.0.0.1", port=8002)
