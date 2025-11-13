import os
import sys
import io
import base64

import torch
import soundfile as sf
from flask import Flask, render_template, request, jsonify, send_file

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import VoxCPMModel

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

MODEL_ID = os.path.join(project_root, "models", "VoxCPM-0.5B")
ZIPENHANCER_MODEL_ID = os.path.join(project_root, "models", "speech_zipenhancer_ans_multiloss_16k_base")
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做得比我还好哟。"

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
        get_model(device)
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
        text = request.form.get('text', '').strip()
        prompt_text = request.form.get('prompt_text', '').strip()
        cfg_value = float(request.form.get('cfg_value', 2.0))
        inference_timesteps = int(request.form.get('inference_timesteps', 10))
        normalize = request.form.get('normalize', 'true').lower() == 'true'
        denoise = request.form.get('denoise', 'true').lower() == 'true'
        retry_badcase = request.form.get('retry_badcase', 'true').lower() == 'true'
        retry_max_times = int(request.form.get('retry_max_times', 3))
        retry_ratio_threshold = float(request.form.get('retry_ratio_threshold', 6.0))
        
        if not text:
            return jsonify({
                'status': 'error',
                'message': '请输入待合成文本。'
            }), 400
        
        prompt_audio_path = None
        if 'prompt_audio' in request.files:
            audio_file = request.files['prompt_audio']
            if audio_file.filename:
                temp_path = os.path.join('/tmp', f'prompt_{os.getpid()}.wav')
                audio_file.save(temp_path)
                prompt_audio_path = temp_path
        
        if not prompt_audio_path:
            prompt_audio_path = DEFAULT_PROMPT_WAV
        
        if prompt_audio_path != DEFAULT_PROMPT_WAV and not prompt_text:
            if prompt_audio_path and prompt_audio_path.startswith('/tmp'):
                try:
                    os.remove(prompt_audio_path)
                except:
                    pass
            return jsonify({
                'status': 'error',
                'message': '使用参考音频时，请提供对应的参考文本。'
            }), 400
        
        try:
            validated_path = validate_prompt_audio(prompt_audio_path)
        except Exception as exc:
            if prompt_audio_path and prompt_audio_path.startswith('/tmp'):
                try:
                    os.remove(prompt_audio_path)
                except:
                    pass
            return jsonify({
                'status': 'error',
                'message': str(exc)
            }), 400
        
        try:
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
        finally:
            if prompt_audio_path and prompt_audio_path.startswith('/tmp'):
                try:
                    os.remove(prompt_audio_path)
                except:
                    pass
        
        # 确保音频数据是正确的格式
        if wav.ndim > 1:
            wav = wav.squeeze()
        
        # 使用 soundfile 写入内存缓冲区
        buffer = io.BytesIO()
        sf.write(buffer, wav, 16000, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'message': '生成完成',
            'audio': audio_base64,
            'sample_rate': 16000
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
    app.run(host='127.0.0.1', port=5002, debug=True)
