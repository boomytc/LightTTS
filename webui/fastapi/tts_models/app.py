import os
import sys
import io
import gc
import json
import base64
import time
from typing import Optional

import torch
import soundfile as sf
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
from indextts.infer_v2 import IndexTTS2
from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import VoxCPMModel

app = FastAPI(title="TTS 模型统一演示平台")

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呀。"

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

current_model_type = None
model_instance = None
current_device = None


def clear_model_cache():
    """清除当前模型并释放内存"""
    global model_instance, current_model_type, current_device
    
    if model_instance is not None:
        del model_instance
        model_instance = None
        current_model_type = None
        current_device = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def load_cosyvoice_model(device: str):
    """加载 CosyVoice2 模型"""
    model_dir = os.path.join(project_root, "models", "CosyVoice2-0.5B")
    is_cuda = device == "cuda" and torch.cuda.is_available()
    
    return CosyVoice2(
        model_dir=model_dir,
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=is_cuda,
        trt_concurrent=1,
        device=device,
    )


def load_indextts_model(device: str):
    """加载 IndexTTS2 模型"""
    model_dir = os.path.join(project_root, "models", "IndexTTS-2")
    cfg_path = os.path.join(model_dir, "config.yaml")
    is_cuda = device == "cuda" and torch.cuda.is_available()
    
    return IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        use_fp16=is_cuda,
        device=device,
        use_cuda_kernel=False,
    )


def load_voxcpm_model(device: str):
    """加载 VoxCPM 模型"""
    model_id = os.path.join(project_root, "models", "VoxCPM-0.5B")
    zipenhancer_model_id = os.path.join(project_root, "models", "speech_zipenhancer_ans_multiloss_16k_base")
    
    return VoxCPM.from_pretrained(
        hf_model_id=model_id,
        load_denoiser=False,
        zipenhancer_model_id=zipenhancer_model_id,
        local_files_only=True,
        device=device,
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页面"""
    has_default_audio = os.path.isfile(DEFAULT_PROMPT_WAV)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_prompt_text": DEFAULT_PROMPT_TEXT,
            "has_default_audio": has_default_audio
        }
    )


@app.post("/api/load_model")
async def load_model(request: Request):
    """加载模型API"""
    global model_instance, current_model_type, current_device
    
    data = await request.json()
    model_type = data.get("model_type")
    device = data.get("device", "cuda")
    
    if not model_type:
        return JSONResponse(
            {
                "status": "error",
                "message": "请选择模型类型。"
            },
            status_code=400
        )
    
    if current_model_type == model_type and current_device == device and model_instance is not None:
        return JSONResponse({
            "status": "success",
            "message": f"{model_type.upper()} 模型已加载，无需重复加载。"
        })
    
    try:
        clear_model_cache()
        
        if model_type == "cosyvoice":
            model_instance = load_cosyvoice_model(device)
        elif model_type == "indextts":
            model_instance = load_indextts_model(device)
        elif model_type == "voxcpm":
            model_instance = load_voxcpm_model(device)
        else:
            return JSONResponse(
                {
                    "status": "error",
                    "message": f"未知的模型类型: {model_type}"
                },
                status_code=400
            )
        
        current_model_type = model_type
        current_device = device
        
        return JSONResponse({
            "status": "success",
            "message": f"{model_type.upper()} 模型加载完成 ✅"
        })
    except Exception as exc:
        clear_model_cache()
        return JSONResponse(
            {
                "status": "error",
                "message": f"模型加载失败: {exc}"
            },
            status_code=500
        )


@app.post("/api/generate")
async def generate(request: Request):
    """生成语音API - 根据当前模型类型调用不同的生成逻辑"""
    global model_instance, current_model_type
    
    if model_instance is None or current_model_type is None:
        return JSONResponse(
            {
                "status": "error",
                "message": "请先加载模型。"
            },
            status_code=400
        )
    
    try:
        if current_model_type == "cosyvoice":
            return await generate_cosyvoice(request)
        elif current_model_type == "indextts":
            return await generate_indextts(request)
        elif current_model_type == "voxcpm":
            return await generate_voxcpm(request)
        else:
            return JSONResponse(
                {
                    "status": "error",
                    "message": f"未知的模型类型: {current_model_type}"
                },
                status_code=400
            )
    except Exception as exc:
        return JSONResponse(
            {
                "status": "error",
                "message": f"推理失败: {exc}"
            },
            status_code=500
        )


async def generate_cosyvoice(request: Request):
    """CosyVoice 生成逻辑"""
    form = await request.form()
    
    mode = form.get("mode")
    text = form.get("text", "").strip()
    prompt_text = form.get("prompt_text", "").strip()
    instruct_text = form.get("instruct_text", "").strip()
    speed = float(form.get("speed", 1.0))
    seed = int(form.get("seed", 0))
    prompt_audio = form.get("prompt_audio")
    
    if not text:
        return JSONResponse(
            {"status": "error", "message": "请输入待合成文本。"},
            status_code=400
        )
    
    if mode == "zero_shot" and not prompt_text:
        return JSONResponse(
            {"status": "error", "message": "零样本克隆模式需要提供参考文本。"},
            status_code=400
        )
    
    if mode == "instruct" and not instruct_text:
        return JSONResponse(
            {"status": "error", "message": "指令控制模式需要提供指令文本。"},
            status_code=400
        )
    
    prompt_audio_path = None
    if prompt_audio and hasattr(prompt_audio, 'filename') and prompt_audio.filename:
        temp_path = os.path.join("/tmp", f"prompt_{os.getpid()}.wav")
        with open(temp_path, "wb") as f:
            f.write(await prompt_audio.read())
        prompt_audio_path = temp_path
    
    if not prompt_audio_path:
        prompt_audio_path = DEFAULT_PROMPT_WAV
    
    try:
        prompt_speech_16k = load_wav(prompt_audio_path, 16000)
        
        set_all_random_seed(seed)
        
        if mode == "zero_shot":
            result = model_instance.inference_zero_shot(
                text, prompt_text, prompt_speech_16k, stream=False, speed=speed
            )
        elif mode == "cross_lingual":
            result = model_instance.inference_cross_lingual(
                text, prompt_speech_16k, stream=False, speed=speed
            )
        elif mode == "instruct":
            result = model_instance.inference_instruct2(
                text, instruct_text, prompt_speech_16k, stream=False, speed=speed
            )
        else:
            return JSONResponse(
                {"status": "error", "message": f"无效的模式: {mode}"},
                status_code=400
            )
        
        audio_tensors = []
        for segment in result:
            tensor = segment.get("tts_speech")
            if tensor is not None:
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                audio_tensors.append(tensor)
        
        if not audio_tensors:
            raise RuntimeError("生成结果为空。")
        
        audio_tensor = torch.cat(audio_tensors, dim=-1)
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
    finally:
        if prompt_audio_path and prompt_audio_path.startswith("/tmp"):
            try:
                os.remove(prompt_audio_path)
            except:
                pass


async def generate_indextts(request: Request):
    """IndexTTS 生成逻辑"""
    form = await request.form()
    
    text = form.get("text", "").strip()
    emo_mode = form.get("emo_mode", "none")
    emo_alpha = float(form.get("emo_alpha", 1.0))
    emo_text = form.get("emo_text", "")
    interval_silence = int(form.get("interval_silence", 200))
    max_tokens = int(form.get("max_tokens", 120))
    use_random = form.get("use_random") == "true"
    prompt_audio = form.get("prompt_audio")
    emo_audio = form.get("emo_audio")
    emo_vector = form.get("emo_vector", "[]")
    
    if not text:
        return JSONResponse(
            {"status": "error", "message": "请输入待合成文本。"},
            status_code=400
        )
    
    if emo_mode == "audio" and (not emo_audio or not hasattr(emo_audio, 'filename') or not emo_audio.filename):
        return JSONResponse(
            {"status": "error", "message": "情感参考音频模式需要上传情感音频。"},
            status_code=400
        )
    
    if emo_mode == "text" and not emo_text.strip():
        return JSONResponse(
            {"status": "error", "message": "情感文本引导模式需要输入引导文本。"},
            status_code=400
        )
    
    prompt_audio_path = None
    emo_audio_path = None
    
    try:
        if prompt_audio and hasattr(prompt_audio, 'filename') and prompt_audio.filename:
            temp_path = os.path.join("/tmp", f"prompt_{os.getpid()}_{int(time.time())}.wav")
            with open(temp_path, "wb") as f:
                f.write(await prompt_audio.read())
            prompt_audio_path = temp_path
        
        if not prompt_audio_path:
            prompt_audio_path = DEFAULT_PROMPT_WAV
        
        if emo_mode == "audio" and emo_audio and hasattr(emo_audio, 'filename') and emo_audio.filename:
            temp_path = os.path.join("/tmp", f"emo_{os.getpid()}_{int(time.time())}.wav")
            with open(temp_path, "wb") as f:
                f.write(await emo_audio.read())
            emo_audio_path = temp_path
        
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
                    {"status": "error", "message": "情感向量必须包含8个值。"},
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
                {"status": "error", "message": "生成结果为空。"},
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
        if prompt_audio_path and prompt_audio_path.startswith("/tmp"):
            try:
                os.remove(prompt_audio_path)
            except:
                pass
        if emo_audio_path:
            try:
                os.remove(emo_audio_path)
            except:
                pass


async def generate_voxcpm(request: Request):
    """VoxCPM 生成逻辑"""
    form = await request.form()
    
    text = form.get("text", "").strip()
    prompt_text = form.get("prompt_text", "").strip()
    cfg_value = float(form.get("cfg_value", 2.0))
    inference_timesteps = int(form.get("inference_timesteps", 10))
    normalize = form.get("normalize") == "true"
    denoise = form.get("denoise") == "true"
    retry_badcase = form.get("retry_badcase") == "true"
    retry_max_times = int(form.get("retry_max_times", 3))
    retry_ratio_threshold = float(form.get("retry_ratio_threshold", 6.0))
    prompt_audio = form.get("prompt_audio")
    
    if not text:
        return JSONResponse(
            {"status": "error", "message": "请输入待合成文本。"},
            status_code=400
        )
    
    prompt_audio_path = None
    if prompt_audio and hasattr(prompt_audio, 'filename') and prompt_audio.filename:
        temp_path = os.path.join("/tmp", f"prompt_{os.getpid()}.wav")
        with open(temp_path, "wb") as f:
            f.write(await prompt_audio.read())
        prompt_audio_path = temp_path
    
    if not prompt_audio_path:
        prompt_audio_path = DEFAULT_PROMPT_WAV
    
    if prompt_audio_path != DEFAULT_PROMPT_WAV and not prompt_text:
        if prompt_audio_path and prompt_audio_path.startswith("/tmp"):
            try:
                os.remove(prompt_audio_path)
            except:
                pass
        return JSONResponse(
            {"status": "error", "message": "使用参考音频时，请提供对应的参考文本。"},
            status_code=400
        )
    
    try:
        wav = model_instance.generate(
            text=text,
            prompt_wav_path=prompt_audio_path,
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
    finally:
        if prompt_audio_path and prompt_audio_path.startswith("/tmp"):
            try:
                os.remove(prompt_audio_path)
            except:
                pass


@app.get("/api/default_audio")
async def get_default_audio():
    """获取默认参考音频"""
    if os.path.isfile(DEFAULT_PROMPT_WAV):
        return FileResponse(DEFAULT_PROMPT_WAV, media_type="audio/wav")
    return JSONResponse(
        {"status": "error", "message": "默认音频文件不存在"},
        status_code=404
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
