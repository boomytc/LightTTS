let currentModel = null;
let isModelLoaded = false;
let isGenerating = false;

const elements = {
    modelRadios: document.querySelectorAll('input[name="model"]'),
    deviceRadios: document.querySelectorAll('input[name="device"]'),
    loadBtn: document.getElementById('load-btn'),
    generateBtn: document.getElementById('generate-btn'),
    stopBtn: document.getElementById('stop-btn'),
    status: document.getElementById('status'),
    audioOutput: document.getElementById('audio-output'),
    audioPlaceholder: document.getElementById('audio-placeholder'),
    text: document.getElementById('text'),
    
    configCosyvoice: document.getElementById('config-cosyvoice'),
    configIndextts: document.getElementById('config-indextts'),
    configVoxcpm: document.getElementById('config-voxcpm'),
};

function updateStatus(message, type = 'info') {
    elements.status.textContent = message;
    elements.status.style.background = type === 'error' ? '#fed7d7' : 
                                       type === 'success' ? '#c6f6d5' : '#f7fafc';
    elements.status.style.borderColor = type === 'error' ? '#fc8181' : 
                                        type === 'success' ? '#68d391' : '#e0e0e0';
}

function switchModelConfig(model) {
    elements.configCosyvoice.style.display = 'none';
    elements.configIndextts.style.display = 'none';
    elements.configVoxcpm.style.display = 'none';
    
    if (model === 'cosyvoice') {
        elements.configCosyvoice.style.display = 'block';
    } else if (model === 'indextts') {
        elements.configIndextts.style.display = 'block';
    } else if (model === 'voxcpm') {
        elements.configVoxcpm.style.display = 'block';
    }
}

elements.modelRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
        currentModel = e.target.value;
        elements.loadBtn.disabled = false;
        isModelLoaded = false;
        elements.generateBtn.disabled = true;
        elements.text.placeholder = `请先加载 ${currentModel.toUpperCase()} 模型`;
        updateStatus(`已选择 ${currentModel.toUpperCase()} 模型，请点击"加载模型"按钮`);
        switchModelConfig(currentModel);
    });
});

elements.loadBtn.addEventListener('click', async () => {
    if (!currentModel) {
        updateStatus('请先选择模型', 'error');
        return;
    }
    
    const device = document.querySelector('input[name="device"]:checked').value;
    
    elements.loadBtn.disabled = true;
    elements.loadBtn.textContent = '加载中...';
    updateStatus(`正在加载 ${currentModel.toUpperCase()} 模型...`);
    
    try {
        const response = await fetch('/api/load_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_type: currentModel, device })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            isModelLoaded = true;
            elements.generateBtn.disabled = false;
            elements.text.placeholder = '请输入需要合成的文本';
            updateStatus(data.message, 'success');
        } else {
            updateStatus(data.message, 'error');
        }
    } catch (error) {
        updateStatus(`加载失败: ${error.message}`, 'error');
    } finally {
        elements.loadBtn.disabled = false;
        elements.loadBtn.textContent = '加载模型';
    }
});

elements.generateBtn.addEventListener('click', async () => {
    if (!isModelLoaded) {
        updateStatus('请先加载模型', 'error');
        return;
    }
    
    if (isGenerating) {
        return;
    }
    
    const text = elements.text.value.trim();
    if (!text) {
        updateStatus('请输入待合成文本', 'error');
        return;
    }
    
    isGenerating = true;
    elements.generateBtn.disabled = true;
    elements.generateBtn.textContent = '生成中...';
    updateStatus('正在生成语音...');
    
    try {
        const formData = new FormData();
        
        if (currentModel === 'cosyvoice') {
            await generateCosyvoice(formData);
        } else if (currentModel === 'indextts') {
            await generateIndextts(formData);
        } else if (currentModel === 'voxcpm') {
            await generateVoxcpm(formData);
        }
    } catch (error) {
        updateStatus(`生成失败: ${error.message}`, 'error');
    } finally {
        isGenerating = false;
        elements.generateBtn.disabled = false;
        elements.generateBtn.textContent = '生成语音';
    }
});

async function generateCosyvoice(formData) {
    formData.append('text', elements.text.value.trim());
    formData.append('mode', document.querySelector('input[name="mode"]:checked').value);
    formData.append('prompt_text', document.getElementById('prompt-text').value.trim());
    formData.append('instruct_text', document.getElementById('instruct-text').value.trim());
    formData.append('speed', document.getElementById('speed').value);
    formData.append('seed', document.getElementById('seed').value);
    
    const promptAudioInput = document.getElementById('prompt-audio-cosyvoice');
    if (promptAudioInput.files.length > 0) {
        formData.append('prompt_audio', promptAudioInput.files[0]);
    }
    
    await sendGenerateRequest(formData);
}

async function generateIndextts(formData) {
    formData.append('text', elements.text.value.trim());
    formData.append('emo_mode', document.querySelector('input[name="emo-mode"]:checked').value);
    formData.append('emo_alpha', document.getElementById('emo-alpha').value);
    formData.append('emo_text', document.getElementById('emo-text').value.trim());
    formData.append('interval_silence', document.getElementById('interval-silence').value);
    formData.append('max_tokens', document.getElementById('max-tokens').value);
    formData.append('use_random', document.getElementById('use-random').checked);
    
    const promptAudioInput = document.getElementById('prompt-audio-indextts');
    if (promptAudioInput.files.length > 0) {
        formData.append('prompt_audio', promptAudioInput.files[0]);
    }
    
    const emoAudioInput = document.getElementById('emo-audio');
    if (emoAudioInput.files.length > 0) {
        formData.append('emo_audio', emoAudioInput.files[0]);
    }
    
    const emoVector = [
        parseFloat(document.getElementById('emo-happy').value),
        parseFloat(document.getElementById('emo-angry').value),
        parseFloat(document.getElementById('emo-sad').value),
        parseFloat(document.getElementById('emo-afraid').value),
        parseFloat(document.getElementById('emo-disgusted').value),
        parseFloat(document.getElementById('emo-melancholic').value),
        parseFloat(document.getElementById('emo-surprised').value),
        parseFloat(document.getElementById('emo-calm').value),
    ];
    formData.append('emo_vector', JSON.stringify(emoVector));
    
    await sendGenerateRequest(formData);
}

async function generateVoxcpm(formData) {
    formData.append('text', elements.text.value.trim());
    formData.append('prompt_text', document.getElementById('prompt-text-voxcpm').value.trim());
    formData.append('cfg_value', document.getElementById('cfg-value').value);
    formData.append('inference_timesteps', document.getElementById('inference-timesteps').value);
    formData.append('normalize', document.getElementById('normalize').checked);
    formData.append('denoise', document.getElementById('denoise').checked);
    formData.append('retry_badcase', document.getElementById('retry-badcase').checked);
    formData.append('retry_max_times', document.getElementById('retry-max-times').value);
    formData.append('retry_ratio_threshold', document.getElementById('retry-ratio-threshold').value);
    
    const promptAudioInput = document.getElementById('prompt-audio-voxcpm');
    if (promptAudioInput.files.length > 0) {
        formData.append('prompt_audio', promptAudioInput.files[0]);
    }
    
    await sendGenerateRequest(formData);
}

async function sendGenerateRequest(formData) {
    const response = await fetch('/api/generate', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    
    if (data.status === 'success') {
        const audioBlob = base64ToBlob(data.audio, 'audio/wav');
        const audioUrl = URL.createObjectURL(audioBlob);
        
        elements.audioOutput.src = audioUrl;
        elements.audioOutput.style.display = 'block';
        elements.audioPlaceholder.style.display = 'none';
        
        updateStatus(data.message, 'success');
    } else {
        updateStatus(data.message, 'error');
    }
}

function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

elements.stopBtn.addEventListener('click', () => {
    if (isGenerating) {
        isGenerating = false;
        elements.generateBtn.disabled = false;
        elements.generateBtn.textContent = '生成语音';
        updateStatus('生成已停止', 'error');
    }
});

document.getElementById('speed')?.addEventListener('input', (e) => {
    document.getElementById('speed-value').textContent = parseFloat(e.target.value).toFixed(2);
});

document.getElementById('emo-alpha')?.addEventListener('input', (e) => {
    document.getElementById('emo-alpha-value').textContent = parseFloat(e.target.value).toFixed(2);
});

document.getElementById('interval-silence')?.addEventListener('input', (e) => {
    document.getElementById('interval-silence-value').textContent = e.target.value;
});

document.getElementById('max-tokens')?.addEventListener('input', (e) => {
    document.getElementById('max-tokens-value').textContent = e.target.value;
});

document.getElementById('cfg-value')?.addEventListener('input', (e) => {
    document.getElementById('cfg-value-display').textContent = parseFloat(e.target.value).toFixed(1);
});

document.getElementById('inference-timesteps')?.addEventListener('input', (e) => {
    document.getElementById('timesteps-display').textContent = e.target.value;
});

document.getElementById('retry-max-times')?.addEventListener('input', (e) => {
    document.getElementById('retry-times-display').textContent = e.target.value;
});

document.getElementById('retry-ratio-threshold')?.addEventListener('input', (e) => {
    document.getElementById('retry-threshold-display').textContent = parseFloat(e.target.value).toFixed(1);
});

document.querySelectorAll('.emo-slider').forEach(slider => {
    slider.addEventListener('input', (e) => {
        const valueSpan = document.getElementById(`${e.target.id}-value`);
        if (valueSpan) {
            valueSpan.textContent = parseFloat(e.target.value).toFixed(2);
        }
    });
});

document.querySelectorAll('input[name="mode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        const promptTextGroup = document.getElementById('prompt-text-group');
        const instructTextGroup = document.getElementById('instruct-text-group');
        
        if (e.target.value === 'zero_shot') {
            promptTextGroup.classList.remove('hidden');
            instructTextGroup.classList.add('hidden');
        } else if (e.target.value === 'cross_lingual') {
            promptTextGroup.classList.add('hidden');
            instructTextGroup.classList.add('hidden');
        } else if (e.target.value === 'instruct') {
            promptTextGroup.classList.remove('hidden');
            instructTextGroup.classList.remove('hidden');
        }
    });
});

document.querySelectorAll('input[name="emo-mode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        const emoAudioGroup = document.getElementById('emo-audio-group');
        const emoAlphaGroup = document.getElementById('emo-alpha-group');
        const emoVectorGroup = document.getElementById('emo-vector-group');
        const emoTextGroup = document.getElementById('emo-text-group');
        
        emoAudioGroup.classList.add('hidden');
        emoAlphaGroup.classList.add('hidden');
        emoVectorGroup.classList.add('hidden');
        emoTextGroup.classList.add('hidden');
        
        if (e.target.value === 'audio') {
            emoAudioGroup.classList.remove('hidden');
            emoAlphaGroup.classList.remove('hidden');
        } else if (e.target.value === 'vector') {
            emoVectorGroup.classList.remove('hidden');
        } else if (e.target.value === 'text') {
            emoTextGroup.classList.remove('hidden');
            emoAlphaGroup.classList.remove('hidden');
        }
    });
});

document.querySelectorAll('.upload-audio-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const parent = btn.closest('.form-group');
        const fileInput = parent.querySelector('input[type="file"]');
        fileInput?.click();
    });
});

document.querySelectorAll('.use-default-audio-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const parent = btn.closest('.form-group');
        const fileInput = parent.querySelector('input[type="file"]');
        const preview = parent.querySelector('.prompt-audio-preview');
        const fileName = parent.querySelector('.audio-file-name');
        
        if (fileInput) {
            fileInput.value = '';
        }
        
        if (preview) {
            preview.src = '/api/default_audio';
        }
        
        if (fileName) {
            fileName.textContent = '当前使用：默认参考音频';
        }
    });
});

document.querySelectorAll('input[type="file"][accept="audio/*"]').forEach(input => {
    input.addEventListener('change', (e) => {
        const parent = e.target.closest('.form-group');
        const preview = parent?.querySelector('.prompt-audio-preview');
        const fileName = parent?.querySelector('.audio-file-name');
        
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            
            if (preview) {
                const url = URL.createObjectURL(file);
                preview.src = url;
            }
            
            if (fileName) {
                fileName.textContent = `当前文件：${file.name}`;
            }
        }
    });
});

window.addEventListener('load', () => {
    updateStatus('请选择模型并加载...');
});
