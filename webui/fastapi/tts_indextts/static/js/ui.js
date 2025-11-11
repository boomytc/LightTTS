import elements from './dom.js';

export function updateStatus(message, type = 'info') {
    if (!elements.status) {
        return;
    }
    elements.status.textContent = message;
    elements.status.className = 'status-box';

    if (type === 'loading') {
        elements.status.classList.add('loading');
    } else if (type === 'success') {
        elements.status.classList.add('success');
    } else if (type === 'error') {
        elements.status.classList.add('error');
    }
}

export function showButtonLoading(button, text) {
    if (!button) {
        return;
    }
    button.disabled = true;
    button.innerHTML = `<span class="spinner"></span> ${text}`;
}

export function resetButton(button, text) {
    if (!button) {
        return;
    }
    button.disabled = false;
    button.textContent = text;
}

export function displayAudio(audioUrl) {
    if (!elements.audioPlayer || !elements.audioPlaceholder) {
        return;
    }
    elements.audioPlayer.src = audioUrl;
    elements.audioPlayer.style.display = 'block';
    elements.audioPlaceholder.style.display = 'none';
}

export function updateEmoModeVisibility(modeValue) {
    const mode = modeValue || document.querySelector('input[name="emo-mode"]:checked')?.value || 'none';
    const emoAudioGroup = document.getElementById('emo-audio-group');
    const emoAlphaGroup = document.getElementById('emo-alpha-group');
    const emoVectorGroup = document.getElementById('emo-vector-group');
    const emoTextGroup = document.getElementById('emo-text-group');

    [emoAudioGroup, emoAlphaGroup, emoVectorGroup, emoTextGroup].forEach((group) => {
        group?.classList.add('hidden');
    });

    if (mode === 'audio') {
        emoAudioGroup?.classList.remove('hidden');
        emoAlphaGroup?.classList.remove('hidden');
    } else if (mode === 'vector') {
        emoVectorGroup?.classList.remove('hidden');
    } else if (mode === 'text') {
        emoAlphaGroup?.classList.remove('hidden');
        emoTextGroup?.classList.remove('hidden');
    }
}

export function updateSliderValue(elementId, value) {
    const target = document.getElementById(elementId);
    if (target) {
        target.textContent = value;
    }
}
