import elements from './dom.js';

export async function loadDefaultAudio() {
    if (!elements.promptAudioPreview) {
        return;
    }

    try {
        const response = await fetch('/api/default_audio');
        if (!response.ok) {
            return;
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        elements.promptAudioPreview.src = url;
        elements.promptAudioPreview.style.display = 'block';
        if (elements.audioFileName) {
            elements.audioFileName.textContent = '当前使用：默认参考音频';
        }
    } catch (error) {
        console.error('加载默认音频失败:', error);
    }
}

export function handleAudioUpload(event) {
    const file = event.target.files[0];
    if (!file || !elements.promptAudioPreview) {
        return;
    }

    const url = URL.createObjectURL(file);
    elements.promptAudioPreview.src = url;
    elements.promptAudioPreview.style.display = 'block';
    if (elements.audioFileName) {
        elements.audioFileName.textContent = `当前使用：${file.name}`;
    }
}

export function useDefaultAudio() {
    if (elements.promptAudio) {
        elements.promptAudio.value = '';
    }
    loadDefaultAudio();
}
