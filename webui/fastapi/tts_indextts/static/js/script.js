let currentGenerateRequest = null;

const elements = {
    promptAudio: document.getElementById('prompt-audio'),
    promptAudioPreview: null,
    uploadAudioBtn: null,
    useDefaultAudioBtn: null,
    audioFileName: null,
};

document.addEventListener('DOMContentLoaded', function() {
    elements.promptAudioPreview = document.getElementById('prompt-audio-preview');
    elements.uploadAudioBtn = document.getElementById('upload-audio-btn');
    elements.useDefaultAudioBtn = document.getElementById('use-default-audio-btn');
    elements.audioFileName = document.getElementById('audio-file-name');
    
    initializeEventListeners();
    updateSliderValues();
    loadDefaultAudio();
});

function initializeEventListeners() {
    const loadBtn = document.getElementById('load-btn');
    const generateBtn = document.getElementById('generate-btn');
    const stopBtn = document.getElementById('stop-btn');
    const emoModeRadios = document.querySelectorAll('input[name="emo-mode"]');
    
    loadBtn.addEventListener('click', loadModel);
    generateBtn.addEventListener('click', generateSpeech);
    stopBtn.addEventListener('click', stopGeneration);
    
    emoModeRadios.forEach(radio => {
        radio.addEventListener('change', updateEmoModeVisibility);
    });
    
    elements.uploadAudioBtn.addEventListener('click', () => {
        elements.promptAudio.click();
    });
    
    elements.promptAudio.addEventListener('change', handleAudioUpload);
    elements.useDefaultAudioBtn.addEventListener('click', useDefaultAudio);
    
    const emoAlpha = document.getElementById('emo-alpha');
    emoAlpha.addEventListener('input', function() {
        document.getElementById('emo-alpha-value').textContent = parseFloat(this.value).toFixed(2);
    });
    
    const intervalSilence = document.getElementById('interval-silence');
    intervalSilence.addEventListener('input', function() {
        document.getElementById('interval-silence-value').textContent = this.value;
    });
    
    const maxTokens = document.getElementById('max-tokens');
    maxTokens.addEventListener('input', function() {
        document.getElementById('max-tokens-value').textContent = this.value;
    });
    
    const emoSliders = document.querySelectorAll('.emo-slider');
    emoSliders.forEach(slider => {
        slider.addEventListener('input', function() {
            const valueSpan = document.getElementById(this.id + '-value');
            valueSpan.textContent = parseFloat(this.value).toFixed(2);
        });
    });
}

function updateSliderValues() {
    document.getElementById('emo-alpha-value').textContent = '1.00';
    document.getElementById('interval-silence-value').textContent = '200';
    document.getElementById('max-tokens-value').textContent = '120';
    
    const emoSliders = document.querySelectorAll('.emo-slider');
    emoSliders.forEach(slider => {
        const valueSpan = document.getElementById(slider.id + '-value');
        valueSpan.textContent = '0.00';
    });
}

function updateEmoModeVisibility() {
    const emoMode = document.querySelector('input[name="emo-mode"]:checked').value;
    
    const emoAudioGroup = document.getElementById('emo-audio-group');
    const emoAlphaGroup = document.getElementById('emo-alpha-group');
    const emoVectorGroup = document.getElementById('emo-vector-group');
    const emoTextGroup = document.getElementById('emo-text-group');
    
    emoAudioGroup.classList.add('hidden');
    emoAlphaGroup.classList.add('hidden');
    emoVectorGroup.classList.add('hidden');
    emoTextGroup.classList.add('hidden');
    
    if (emoMode === 'audio') {
        emoAudioGroup.classList.remove('hidden');
        emoAlphaGroup.classList.remove('hidden');
    } else if (emoMode === 'vector') {
        emoVectorGroup.classList.remove('hidden');
    } else if (emoMode === 'text') {
        emoAlphaGroup.classList.remove('hidden');
        emoTextGroup.classList.remove('hidden');
    }
}

async function loadModel() {
    const loadBtn = document.getElementById('load-btn');
    const generateBtn = document.getElementById('generate-btn');
    const statusBox = document.getElementById('status');
    const device = document.querySelector('input[name="device"]:checked').value;
    
    loadBtn.disabled = true;
    loadBtn.innerHTML = '<span class="spinner"></span> 加载中...';
    updateStatus('模型加载中...', 'loading');
    
    try {
        const response = await fetch('/api/load_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ device: device })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            updateStatus(data.message, 'success');
            generateBtn.disabled = false;
        } else {
            updateStatus(data.message, 'error');
        }
    } catch (error) {
        updateStatus('网络错误: ' + error.message, 'error');
    } finally {
        loadBtn.disabled = false;
        loadBtn.textContent = '加载模型';
    }
}

async function generateSpeech() {
    const text = document.getElementById('text').value.trim();
    const emoMode = document.querySelector('input[name="emo-mode"]:checked').value;
    
    if (!text) {
        updateStatus('请输入待合成文本。', 'error');
        return;
    }
    
    if (emoMode === 'audio') {
        const emoAudioFile = document.getElementById('emo-audio').files[0];
        if (!emoAudioFile) {
            updateStatus('情感参考音频模式需要上传情感音频。', 'error');
            return;
        }
    }
    
    if (emoMode === 'text') {
        const emoText = document.getElementById('emo-text').value.trim();
        if (!emoText) {
            updateStatus('情感文本引导模式需要输入引导文本。', 'error');
            return;
        }
    }
    
    const generateBtn = document.getElementById('generate-btn');
    generateBtn.disabled = true;
    generateBtn.innerHTML = '<span class="spinner"></span> 生成中...';
    updateStatus('语音生成中...', 'loading');
    
    const formData = new FormData();
    formData.append('text', text);
    formData.append('emo_mode', emoMode);
    
    const promptAudioFile = document.getElementById('prompt-audio').files[0];
    if (promptAudioFile) {
        formData.append('prompt_audio', promptAudioFile);
    }
    
    if (emoMode === 'audio') {
        const emoAudioFile = document.getElementById('emo-audio').files[0];
        formData.append('emo_audio', emoAudioFile);
        const emoAlpha = document.getElementById('emo-alpha').value;
        formData.append('emo_alpha', emoAlpha);
    } else if (emoMode === 'vector') {
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
    } else if (emoMode === 'text') {
        const emoText = document.getElementById('emo-text').value.trim();
        formData.append('emo_text', emoText);
        const emoAlpha = document.getElementById('emo-alpha').value;
        formData.append('emo_alpha', emoAlpha);
    }
    
    const intervalSilence = document.getElementById('interval-silence').value;
    const maxTokens = document.getElementById('max-tokens').value;
    const useRandom = document.getElementById('use-random').checked;
    
    formData.append('interval_silence', intervalSilence);
    formData.append('max_tokens', maxTokens);
    formData.append('use_random', useRandom);
    
    try {
        const controller = new AbortController();
        currentGenerateRequest = controller;
        
        const response = await fetch('/api/generate', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            updateStatus(data.message, 'success');
            displayAudio(data.audio);
        } else {
            updateStatus(data.message, 'error');
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            updateStatus('生成已停止。', 'error');
        } else {
            updateStatus('网络错误: ' + error.message, 'error');
        }
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = '生成语音';
        currentGenerateRequest = null;
    }
}

function stopGeneration() {
    if (currentGenerateRequest) {
        currentGenerateRequest.abort();
        currentGenerateRequest = null;
    }
}

function updateStatus(message, type) {
    const statusBox = document.getElementById('status');
    statusBox.textContent = message;
    statusBox.className = 'status-box';
    
    if (type === 'loading') {
        statusBox.classList.add('loading');
    } else if (type === 'success') {
        statusBox.classList.add('success');
    } else if (type === 'error') {
        statusBox.classList.add('error');
    }
}

function displayAudio(audioBase64) {
    const audioPlayer = document.getElementById('audio-output');
    const audioPlaceholder = document.getElementById('audio-placeholder');
    
    const audioBlob = base64ToBlob(audioBase64, 'audio/wav');
    const audioUrl = URL.createObjectURL(audioBlob);
    
    audioPlayer.src = audioUrl;
    audioPlayer.style.display = 'block';
    audioPlaceholder.style.display = 'none';
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

async function loadDefaultAudio() {
    try {
        const response = await fetch('/api/default_audio');
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            elements.promptAudioPreview.src = url;
            elements.promptAudioPreview.style.display = 'block';
            elements.audioFileName.textContent = '当前使用：默认参考音频';
        }
    } catch (error) {
        console.error('加载默认音频失败:', error);
    }
}

function handleAudioUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const url = URL.createObjectURL(file);
        elements.promptAudioPreview.src = url;
        elements.promptAudioPreview.style.display = 'block';
        elements.audioFileName.textContent = `当前使用：${file.name}`;
    }
}

function useDefaultAudio() {
    elements.promptAudio.value = '';
    loadDefaultAudio();
}
