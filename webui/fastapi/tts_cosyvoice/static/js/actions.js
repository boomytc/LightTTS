import elements from './dom.js';
import { updateStatus, showButtonLoading, resetButton, showGeneratedAudio } from './ui.js';
import {
    setModelLoaded,
    isModelLoaded,
    setGenerating,
    isGenerating,
    setAbortController,
    getAbortController,
} from './state.js';

function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i += 1) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

function buildFormData(mode, text, promptText, instructText, speed, seed) {
    const formData = new FormData();
    formData.append('mode', mode);
    formData.append('text', text);
    formData.append('prompt_text', promptText);
    formData.append('instruct_text', instructText);
    formData.append('speed', speed);
    formData.append('seed', seed);

    if (elements.promptAudio && elements.promptAudio.files.length > 0) {
        formData.append('prompt_audio', elements.promptAudio.files[0]);
    }

    return formData;
}

export async function loadModelAction() {
    const selectedDevice = elements.device();
    if (!selectedDevice) {
        updateStatus('请选择运行设备。', 'error');
        return;
    }

    showButtonLoading(elements.loadBtn, '加载中...');
    updateStatus('正在加载模型...', 'loading');

    try {
        const response = await fetch('/api/load_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ device: selectedDevice.value }),
        });
        const data = await response.json();
        if (response.ok) {
            setModelLoaded(true);
            if (elements.generateBtn) {
                elements.generateBtn.disabled = false;
            }
            updateStatus(data.message, 'success');
        } else {
            updateStatus(data.message, 'error');
        }
    } catch (error) {
        updateStatus(`加载失败: ${error.message}`, 'error');
    } finally {
        resetButton(elements.loadBtn, '加载模型');
    }
}

export async function generateSpeechAction() {
    if (!isModelLoaded()) {
        updateStatus('请先加载模型。', 'error');
        return;
    }

    if (isGenerating()) {
        updateStatus('正在生成中，请等待...', 'error');
        return;
    }

    const modeValue = elements.mode()?.value || 'zero_shot';
    const text = elements.text?.value.trim() || '';
    const promptText = elements.promptText?.value.trim() || '';
    const instructText = elements.instructText?.value.trim() || '';
    const speed = elements.speed?.value || '1.0';
    const seed = elements.seed?.value || '0';

    if (!text) {
        updateStatus('请输入待合成文本。', 'error');
        return;
    }

    if (modeValue === 'zero_shot' && !promptText) {
        updateStatus('零样本克隆模式需要提供参考文本。', 'error');
        return;
    }

    if (modeValue === 'instruct' && !instructText) {
        updateStatus('指令控制模式需要提供指令文本。', 'error');
        return;
    }

    setGenerating(true);
    const controller = new AbortController();
    setAbortController(controller);

    showButtonLoading(elements.generateBtn, '生成中...');
    updateStatus('正在生成语音...', 'loading');

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            body: buildFormData(modeValue, text, promptText, instructText, speed, seed),
            signal: controller.signal,
        });
        const data = await response.json();

        if (response.ok) {
            const audioBlob = base64ToBlob(data.audio, 'audio/wav');
            const audioUrl = URL.createObjectURL(audioBlob);
            showGeneratedAudio(audioUrl);
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
        setGenerating(false);
        setAbortController(null);
        resetButton(elements.generateBtn, '生成语音');
    }
}

export function stopGenerationAction() {
    const controller = getAbortController();
    if (controller) {
        controller.abort();
        updateStatus('正在停止生成...', 'loading');
    } else {
        updateStatus('当前没有正在进行的生成任务。', 'info');
    }
}
