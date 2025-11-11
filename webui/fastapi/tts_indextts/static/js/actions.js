import elements from './dom.js';
import { updateStatus, showButtonLoading, resetButton, displayAudio } from './ui.js';
import { setGenerateRequest, getGenerateRequest } from './state.js';

function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i += 1) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

function collectEmoVector() {
    const ids = [
        'emo-happy',
        'emo-angry',
        'emo-sad',
        'emo-afraid',
        'emo-disgusted',
        'emo-melancholic',
        'emo-surprised',
        'emo-calm',
    ];
    return ids.map((id) => parseFloat(document.getElementById(id)?.value || '0'));
}

export async function loadModelAction() {
    const loadBtn = elements.loadBtn;
    const generateBtn = elements.generateBtn;
    const selectedDevice = document.querySelector('input[name="device"]:checked');

    if (!loadBtn || !selectedDevice) {
        return;
    }

    showButtonLoading(loadBtn, '加载中...');
    updateStatus('模型加载中...', 'loading');

    try {
        const response = await fetch('/api/load_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ device: selectedDevice.value }),
        });
        const data = await response.json();
        if (data.status === 'success') {
            updateStatus(data.message, 'success');
            if (generateBtn) {
                generateBtn.disabled = false;
            }
        } else {
            updateStatus(data.message, 'error');
        }
    } catch (error) {
        updateStatus(`网络错误: ${error.message}`, 'error');
    } finally {
        resetButton(loadBtn, '加载模型');
    }
}

export async function generateSpeechAction() {
    const text = document.getElementById('text')?.value.trim();
    const emoMode = document.querySelector('input[name="emo-mode"]:checked')?.value || 'none';

    if (!text) {
        updateStatus('请输入待合成文本。', 'error');
        return;
    }

    if (emoMode === 'audio' && !document.getElementById('emo-audio')?.files[0]) {
        updateStatus('情感参考音频模式需要上传情感音频。', 'error');
        return;
    }

    if (emoMode === 'text' && !document.getElementById('emo-text')?.value.trim()) {
        updateStatus('情感文本引导模式需要输入引导文本。', 'error');
        return;
    }

    const generateBtn = elements.generateBtn;
    showButtonLoading(generateBtn, '生成中...');
    updateStatus('语音生成中...', 'loading');

    const formData = new FormData();
    formData.append('text', text);
    formData.append('emo_mode', emoMode);

    const promptAudioFile = elements.promptAudio?.files[0];
    if (promptAudioFile) {
        formData.append('prompt_audio', promptAudioFile);
    }

    if (emoMode === 'audio') {
        const emoAudioFile = document.getElementById('emo-audio')?.files[0];
        const emoAlpha = document.getElementById('emo-alpha')?.value || '1.0';
        if (emoAudioFile) {
            formData.append('emo_audio', emoAudioFile);
        }
        formData.append('emo_alpha', emoAlpha);
    } else if (emoMode === 'vector') {
        formData.append('emo_vector', JSON.stringify(collectEmoVector()));
    } else if (emoMode === 'text') {
        const emoText = document.getElementById('emo-text')?.value.trim() || '';
        const emoAlpha = document.getElementById('emo-alpha')?.value || '1.0';
        formData.append('emo_text', emoText);
        formData.append('emo_alpha', emoAlpha);
    }

    formData.append('interval_silence', document.getElementById('interval-silence')?.value || '200');
    formData.append('max_tokens', document.getElementById('max-tokens')?.value || '120');
    formData.append('use_random', document.getElementById('use-random')?.checked || false);

    try {
        const controller = new AbortController();
        setGenerateRequest(controller);

        const response = await fetch('/api/generate', {
            method: 'POST',
            body: formData,
            signal: controller.signal,
        });

        const data = await response.json();
        if (data.status === 'success') {
            updateStatus(data.message, 'success');
            const audioBlob = base64ToBlob(data.audio, 'audio/wav');
            const audioUrl = URL.createObjectURL(audioBlob);
            displayAudio(audioUrl);
        } else {
            updateStatus(data.message, 'error');
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            updateStatus('生成已停止。', 'error');
        } else {
            updateStatus(`网络错误: ${error.message}`, 'error');
        }
    } finally {
        resetButton(generateBtn, '生成语音');
        setGenerateRequest(null);
    }
}

export function stopGenerationAction() {
    const controller = getGenerateRequest();
    if (controller) {
        controller.abort();
        setGenerateRequest(null);
    }
}
