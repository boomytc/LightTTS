import os
import sys
import io
import json
import base64
import time
from functools import lru_cache

import torch
import soundfile as sf
from flask import Flask, render_template, request, jsonify, send_file

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from indextts.infer_v2 import IndexTTS2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

DEFAULT_MODEL_DIR = os.path.join(project_root, "models", "IndexTTS-2")
DEFAULT_CFG_PATH = os.path.join(project_root, "models", "IndexTTS-2", "config.yaml")
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
USE_FP16 = True
USE_CUDA_KERNEL = False

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


@app.route('/')
def index():
    """主页面"""
    has_default_audio = os.path.isfile(DEFAULT_PROMPT_WAV)
    return render_template('index.html', has_default_audio=has_default_audio)


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
        model_instance = get_model(DEFAULT_MODEL_DIR, DEFAULT_CFG_PATH, device)
        current_device = device
        return jsonify({
            'status': 'success',
            'message': '模型加载完成 ✅'
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
        text = request.form.get('text', '').strip()
        emo_mode = request.form.get('emo_mode', 'none')
        emo_alpha = float(request.form.get('emo_alpha', 1.0))
        emo_text = request.form.get('emo_text', '').strip()
        interval_silence = int(request.form.get('interval_silence', 200))
        max_tokens = int(request.form.get('max_tokens', 120))
        use_random = request.form.get('use_random', 'false').lower() == 'true'
        
        if not text:
            return jsonify({
                'status': 'error',
                'message': '请输入待合成文本。'
            }), 400
        
        if emo_mode == 'audio' and 'emo_audio' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '情感参考音频模式需要上传情感音频。'
            }), 400
        
        if emo_mode == 'text' and not emo_text:
            return jsonify({
                'status': 'error',
                'message': '情感文本引导模式需要输入引导文本。'
            }), 400
        
        prompt_audio_path = None
        emo_audio_path = None
        
        try:
            if 'prompt_audio' in request.files:
                audio_file = request.files['prompt_audio']
                if audio_file.filename:
                    temp_path = os.path.join('/tmp', f'prompt_{os.getpid()}_{int(time.time())}.wav')
                    audio_file.save(temp_path)
                    prompt_audio_path = temp_path
            
            if not prompt_audio_path:
                prompt_audio_path = DEFAULT_PROMPT_WAV
            
            if emo_mode == 'audio' and 'emo_audio' in request.files:
                emo_file = request.files['emo_audio']
                if emo_file.filename:
                    temp_path = os.path.join('/tmp', f'emo_{os.getpid()}_{int(time.time())}.wav')
                    emo_file.save(temp_path)
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
            
            if emo_mode == 'audio':
                kwargs["emo_audio_prompt"] = emo_audio_path
                kwargs["emo_alpha"] = emo_alpha
            elif emo_mode == 'vector':
                emo_vector_str = request.form.get('emo_vector', '[]')
                emo_vector = json.loads(emo_vector_str)
                if len(emo_vector) != 8:
                    return jsonify({
                        'status': 'error',
                        'message': '情感向量必须包含8个值。'
                    }), 400
                kwargs["emo_vector"] = emo_vector
                kwargs["emo_alpha"] = 1.0
            elif emo_mode == 'text':
                kwargs["use_emo_text"] = True
                kwargs["emo_text"] = emo_text
                kwargs["emo_alpha"] = emo_alpha
            
            result = model_instance.infer(**kwargs)
            
            if result is None:
                return jsonify({
                    'status': 'error',
                    'message': '生成结果为空。'
                }), 500
            
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
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            return jsonify({
                'status': 'success',
                'message': '生成完成 ✅',
                'audio': audio_base64,
                'sample_rate': sample_rate
            })
            
        finally:
            if prompt_audio_path and prompt_audio_path.startswith('/tmp'):
                try:
                    os.remove(prompt_audio_path)
                except:
                    pass
            if emo_audio_path:
                try:
                    os.remove(emo_audio_path)
                except:
                    pass
        
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
    app.run(host='127.0.0.1', port=5001, debug=True)
