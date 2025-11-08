// 全局状态
let modelLoaded = false;
let isGenerating = false;
let currentAbortController = null;

// DOM 元素（使用函数延迟获取，确保 DOM 已加载）
const elements = {
    device: () => document.querySelector('input[name="device"]:checked'),
    mode: () => document.querySelector('input[name="mode"]:checked'),
    get text() { return document.getElementById('text'); },
    get promptAudio() { return document.getElementById('prompt-audio'); },
    get promptAudioPreview() { return document.getElementById('prompt-audio-preview'); },
    get uploadAudioBtn() { return document.getElementById('upload-audio-btn'); },
    get useDefaultAudioBtn() { return document.getElementById('use-default-audio-btn'); },
    get audioFileName() { return document.getElementById('audio-file-name'); },
    get promptText() { return document.getElementById('prompt-text'); },
    get instructText() { return document.getElementById('instruct-text'); },
    get speed() { return document.getElementById('speed'); },
    get speedValue() { return document.getElementById('speed-value'); },
    get seed() { return document.getElementById('seed'); },
    get loadBtn() { return document.getElementById('load-btn'); },
    get generateBtn() { return document.getElementById('generate-btn'); },
    get stopBtn() { return document.getElementById('stop-btn'); },
    get status() { return document.getElementById('status'); },
    get audioOutput() { return document.getElementById('audio-output'); },
    get audioPlaceholder() { return document.getElementById('audio-placeholder'); },
    get promptAudioGroup() { return document.getElementById('prompt-audio-group'); },
    get promptTextGroup() { return document.getElementById('prompt-text-group'); },
    get instructTextGroup() { return document.getElementById('instruct-text-group'); },
};

// 更新状态显示
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

// 更新 UI 可见性
function updateUIVisibility() {
    const mode = elements.mode().value;
    
    // 所有模式都显示参考音频
    elements.promptAudioGroup.classList.remove('hidden');
    
    if (mode === 'zero_shot') {
        elements.promptTextGroup.classList.remove('hidden');
        elements.instructTextGroup.classList.add('hidden');
    } else if (mode === 'cross_lingual') {
        elements.promptTextGroup.classList.add('hidden');
        elements.instructTextGroup.classList.add('hidden');
    } else if (mode === 'instruct') {
        elements.promptTextGroup.classList.add('hidden');
        elements.instructTextGroup.classList.remove('hidden');
    }
}

// 加载模型
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

// 生成语音
async function generateSpeech() {
    if (!modelLoaded) {
        updateStatus('请先加载模型。', 'error');
        return;
    }
    
    if (isGenerating) {
        updateStatus('正在生成中，请等待...', 'error');
        return;
    }
    
    const mode = elements.mode().value;
    const text = elements.text.value.trim();
    const promptText = elements.promptText.value.trim();
    const instructText = elements.instructText.value.trim();
    const speed = parseFloat(elements.speed.value);
    const seed = parseInt(elements.seed.value);
    
    // 验证输入
    if (!text) {
        updateStatus('请输入待合成文本。', 'error');
        return;
    }
    
    if (mode === 'zero_shot' && !promptText) {
        updateStatus('零样本克隆模式需要提供参考文本。', 'error');
        return;
    }
    
    if (mode === 'instruct' && !instructText) {
        updateStatus('指令控制模式需要提供指令文本。', 'error');
        return;
    }
    
    isGenerating = true;
    currentAbortController = new AbortController();
    
    elements.generateBtn.disabled = true;
    elements.generateBtn.innerHTML = '<span class="loading-spinner"></span>生成中...';
    updateStatus('正在生成语音...', 'loading');
    
    try {
        const formData = new FormData();
        formData.append('mode', mode);
        formData.append('text', text);
        formData.append('prompt_text', promptText);
        formData.append('instruct_text', instructText);
        formData.append('speed', speed);
        formData.append('seed', seed);
        
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
            // 显示生成的音频
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

// 停止生成
function stopGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
        updateStatus('正在停止生成...', 'loading');
    } else {
        updateStatus('当前没有正在进行的生成任务。', 'info');
    }
}

// Base64 转 Blob
function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

// 加载默认参考音频
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

// 处理音频文件上传
function handleAudioUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const url = URL.createObjectURL(file);
        elements.promptAudioPreview.src = url;
        elements.promptAudioPreview.style.display = 'block';
        elements.audioFileName.textContent = `当前使用：${file.name}`;
    }
}

// 使用默认音频
function useDefaultAudio() {
    elements.promptAudio.value = '';
    loadDefaultAudio();
}

// 事件监听
document.addEventListener('DOMContentLoaded', () => {
    // 语速滑块
    elements.speed.addEventListener('input', (e) => {
        elements.speedValue.textContent = parseFloat(e.target.value).toFixed(2);
    });
    
    // 模式切换
    document.querySelectorAll('input[name="mode"]').forEach(radio => {
        radio.addEventListener('change', updateUIVisibility);
    });
    
    // 音频上传相关
    elements.uploadAudioBtn.addEventListener('click', () => {
        elements.promptAudio.click();
    });
    
    elements.promptAudio.addEventListener('change', handleAudioUpload);
    elements.useDefaultAudioBtn.addEventListener('click', useDefaultAudio);
    
    // 按钮事件 - 确保所有按钮都能正常工作
    const loadBtn = document.getElementById('load-btn');
    const generateBtn = document.getElementById('generate-btn');
    const stopBtn = document.getElementById('stop-btn');
    
    if (loadBtn) {
        loadBtn.addEventListener('click', loadModel);
    }
    if (generateBtn) {
        generateBtn.addEventListener('click', generateSpeech);
    }
    if (stopBtn) {
        stopBtn.addEventListener('click', stopGeneration);
    }
    
    // 初始化 UI
    updateUIVisibility();
    
    // 加载默认参考音频
    loadDefaultAudio();
    
    // 自动检测 CUDA 可用性
    if (!navigator.gpu && !navigator.userAgent.includes('CUDA')) {
        const cpuRadio = document.getElementById('device-cpu');
        if (cpuRadio) cpuRadio.checked = true;
    }
});
