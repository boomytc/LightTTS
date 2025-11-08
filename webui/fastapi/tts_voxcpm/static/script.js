let modelLoaded = false;
let isGenerating = false;
let currentAbortController = null;

const elements = {
    device: () => document.querySelector('input[name="device"]:checked'),
    text: document.getElementById('text'),
    promptAudio: document.getElementById('prompt-audio'),
    promptAudioPreview: document.getElementById('prompt-audio-preview'),
    uploadAudioBtn: document.getElementById('upload-audio-btn'),
    useDefaultAudioBtn: document.getElementById('use-default-audio-btn'),
    audioFileName: document.getElementById('audio-file-name'),
    promptText: document.getElementById('prompt-text'),
    cfgValue: document.getElementById('cfg-value'),
    cfgValueDisplay: document.getElementById('cfg-value-display'),
    inferenceTimesteps: document.getElementById('inference-timesteps'),
    timestepsDisplay: document.getElementById('timesteps-display'),
    normalize: document.getElementById('normalize'),
    denoise: document.getElementById('denoise'),
    retryBadcase: document.getElementById('retry-badcase'),
    retryMaxTimes: document.getElementById('retry-max-times'),
    retryTimesDisplay: document.getElementById('retry-times-display'),
    retryRatioThreshold: document.getElementById('retry-ratio-threshold'),
    retryThresholdDisplay: document.getElementById('retry-threshold-display'),
    loadBtn: document.getElementById('load-btn'),
    generateBtn: document.getElementById('generate-btn'),
    stopBtn: document.getElementById('stop-btn'),
    status: document.getElementById('status'),
    audioOutput: document.getElementById('audio-output'),
    audioPlaceholder: document.getElementById('audio-placeholder'),
};


function updateStatus(message, type = 'info') {
    elements.status.textContent = message;
    elements.status.className = 'status-box';
    
    if (type === 'success') {
        elements.status.classList.add('success');
    } else if (type === 'error') {
        elements.status.classList.add('error');
    } else if (type === 'loading') {
        elements.status.classList.add('loading');
    }
}


async function loadModel() {
    const device = elements.device().value;
    
    elements.loadBtn.disabled = true;
    elements.loadBtn.innerHTML = '<span class="loading-spinner"></span>加载中...';
    updateStatus('正在加载模型...', 'loading');
    
    try {
        const response = await fetch('/api/load_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ device }),
        });
        
        const data = await response.json();
        
        if (response.ok) {
            modelLoaded = true;
            elements.generateBtn.disabled = false;
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
}


async function generateSpeech() {
    if (!modelLoaded) {
        updateStatus('请先加载模型。', 'error');
        return;
    }
    
    if (isGenerating) {
        updateStatus('正在生成中，请等待...', 'error');
        return;
    }
    
    const text = elements.text.value.trim();
    const promptText = elements.promptText.value.trim();
    
    if (!text) {
        updateStatus('请输入待合成文本。', 'error');
        return;
    }
    
    if (elements.promptAudio.files.length > 0 && !promptText) {
        updateStatus('使用参考音频时，请提供对应的参考文本。', 'error');
        return;
    }
    
    isGenerating = true;
    currentAbortController = new AbortController();
    
    elements.generateBtn.disabled = true;
    elements.generateBtn.innerHTML = '<span class="loading-spinner"></span>生成中...';
    updateStatus('正在生成语音...', 'loading');
    
    try {
        const formData = new FormData();
        formData.append('text', text);
        formData.append('prompt_text', promptText);
        formData.append('cfg_value', elements.cfgValue.value);
        formData.append('inference_timesteps', elements.inferenceTimesteps.value);
        formData.append('normalize', elements.normalize.checked);
        formData.append('denoise', elements.denoise.checked);
        formData.append('retry_badcase', elements.retryBadcase.checked);
        formData.append('retry_max_times', elements.retryMaxTimes.value);
        formData.append('retry_ratio_threshold', elements.retryRatioThreshold.value);
        
        if (elements.promptAudio.files.length > 0) {
            formData.append('prompt_audio', elements.promptAudio.files[0]);
        }
        
        const response = await fetch('/api/generate', {
            method: 'POST',
            body: formData,
            signal: currentAbortController.signal,
        });
        
        const data = await response.json();
        
        if (response.ok) {
            const audioBlob = base64ToBlob(data.audio, 'audio/wav');
            const audioUrl = URL.createObjectURL(audioBlob);
            
            elements.audioOutput.src = audioUrl;
            elements.audioOutput.classList.add('active');
            elements.audioPlaceholder.style.display = 'none';
            
            updateStatus(data.message, 'success');
        } else {
            updateStatus(data.message, 'error');
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            updateStatus('生成已停止。', 'info');
        } else {
            updateStatus(`生成失败: ${error.message}`, 'error');
        }
    } finally {
        isGenerating = false;
        currentAbortController = null;
        elements.generateBtn.disabled = false;
        elements.generateBtn.textContent = '生成语音';
    }
}


function stopGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
        updateStatus('正在停止生成...', 'loading');
    } else {
        updateStatus('当前没有正在进行的生成任务。', 'info');
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


document.addEventListener('DOMContentLoaded', () => {
    elements.cfgValue.addEventListener('input', (e) => {
        elements.cfgValueDisplay.textContent = parseFloat(e.target.value).toFixed(1);
    });
    
    elements.inferenceTimesteps.addEventListener('input', (e) => {
        elements.timestepsDisplay.textContent = e.target.value;
    });
    
    elements.retryMaxTimes.addEventListener('input', (e) => {
        elements.retryTimesDisplay.textContent = e.target.value;
    });
    
    elements.retryRatioThreshold.addEventListener('input', (e) => {
        elements.retryThresholdDisplay.textContent = parseFloat(e.target.value).toFixed(1);
    });
    
    elements.uploadAudioBtn.addEventListener('click', () => {
        elements.promptAudio.click();
    });
    
    elements.promptAudio.addEventListener('change', handleAudioUpload);
    elements.useDefaultAudioBtn.addEventListener('click', useDefaultAudio);
    
    elements.loadBtn.addEventListener('click', loadModel);
    elements.generateBtn.addEventListener('click', generateSpeech);
    elements.stopBtn.addEventListener('click', stopGeneration);
    
    loadDefaultAudio();
});
