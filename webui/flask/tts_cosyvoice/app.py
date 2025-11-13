import os
import sys
import io
import base64
from functools import lru_cache

import torch
import torchaudio
from flask import Flask, render_template, request, jsonify, send_file

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
matcha_path = os.path.join(project_root, "Matcha-TTS")
sys.path.insert(0, project_root)
sys.path.insert(0, matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

DEFAULT_MODEL_DIR = os.path.join(project_root, "models", "CosyVoice2-0.5B")
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呀。"
USE_FP16 = True
LOAD_JIT = False
LOAD_TRT = False
LOAD_VLLM = False
TRT_CONCURRENT = 1

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


@app.route('/')
def index():
    """主页面"""
    has_default_audio = os.path.isfile(DEFAULT_PROMPT_WAV)
    return render_template(
        'index.html',
        default_prompt_text=DEFAULT_PROMPT_TEXT,
        has_default_audio=has_default_audio
    )


@app.route('/api/load_model', methods=['POST'])
def load_model():
    """加载模型API"""
    global model_instance, current_device
    
    data = request.json
    device = data.get('device', 'cuda')
    
    if model_instance is not None and current_device == device:
        return jsonify({
            'status': 'success',
            'message': '模型已加载，无需重复加载。'
        })
    
    try:
        model_instance = get_model(DEFAULT_MODEL_DIR, device)
        current_device = device
        return jsonify({
            'status': 'success',
            'message': '模型加载完成'
        })
    except Exception as exc:
        return jsonify({
            'status': 'error',
            'message': f'模型加载失败: {exc}'
        }), 500


@app.route('/api/generate', methods=['POST'])
def generate():
    """生成语音API"""
    global model_instance
    
    if model_instance is None:
        return jsonify({
            'status': 'error',
            'message': '请先加载模型。'
        }), 400
    
    try:
        mode = request.form.get('mode', 'zero_shot')
        text = request.form.get('text', '').strip()
        prompt_text = request.form.get('prompt_text', '').strip()
        instruct_text = request.form.get('instruct_text', '').strip()
        speed = float(request.form.get('speed', 1.0))
        seed = int(request.form.get('seed', 0))
        
        if not text:
            return jsonify({
                'status': 'error',
                'message': '请输入待合成文本。'
            }), 400
        
        if mode == 'zero_shot' and not prompt_text:
            return jsonify({
                'status': 'error',
                'message': '零样本克隆模式需要提供参考文本。'
            }), 400
        
        if mode == 'instruct' and not instruct_text:
            return jsonify({
                'status': 'error',
                'message': '指令控制模式需要提供指令文本。'
            }), 400
        
        # 处理参考音频
        prompt_audio_path = None
        if 'prompt_audio' in request.files:
            audio_file = request.files['prompt_audio']
            if audio_file.filename:
                temp_path = os.path.join('/tmp', f'prompt_{os.getpid()}.wav')
                audio_file.save(temp_path)
                prompt_audio_path = temp_path
        
        if not prompt_audio_path:
            prompt_audio_path = DEFAULT_PROMPT_WAV
        
        try:
            prompt_speech_16k = prepare_prompt_audio(prompt_audio_path)
        except Exception as exc:
            return jsonify({
                'status': 'error',
                'message': str(exc)
            }), 400
        finally:
            if prompt_audio_path and prompt_audio_path.startswith('/tmp'):
                try:
                    os.remove(prompt_audio_path)
                except:
                    pass
        
        set_all_random_seed(seed)
        
        # 执行推理
        if mode == 'zero_shot':
            result = model_instance.inference_zero_shot(
                text,
                prompt_text,
                prompt_speech_16k,
                stream=False,
                speed=speed,
            )
        elif mode == 'cross_lingual':
            result = model_instance.inference_cross_lingual(
                text,
                prompt_speech_16k,
                stream=False,
                speed=speed,
            )
        elif mode == 'instruct':
            result = model_instance.inference_instruct2(
                text,
                instruct_text,
                prompt_speech_16k,
                stream=False,
                speed=speed,
            )
        else:
            return jsonify({
                'status': 'error',
                'message': f'无效的模式: {mode}'
            }), 400
        
        audio_tensor = merge_segments(result)
        audio_numpy = audio_tensor.squeeze(0).cpu().numpy()
        
        # 将音频转换为WAV格式并编码为base64
        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            torch.from_numpy(audio_numpy).unsqueeze(0),
            model_instance.sample_rate,
            format='wav'
        )
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'message': '生成完成',
            'audio': audio_base64,
            'sample_rate': model_instance.sample_rate
        })
        
    except Exception as exc:
        return jsonify({
            'status': 'error',
            'message': f'推理失败: {exc}'
        }), 500


@app.route('/api/default_audio')
def get_default_audio():
    """获取默认参考音频"""
    if os.path.isfile(DEFAULT_PROMPT_WAV):
        return send_file(DEFAULT_PROMPT_WAV, mimetype='audio/wav')
    return jsonify({
        'status': 'error',
        'message': '默认音频文件不存在'
    }), 404


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
