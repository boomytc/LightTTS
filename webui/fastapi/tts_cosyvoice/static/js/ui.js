import elements from './dom.js';

export function updateStatus(message, type = 'info') {
    if (!elements.status) {
        return;
    }

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

export function showButtonLoading(button, text) {
    if (!button) {
        return;
    }
    button.disabled = true;
    button.innerHTML = `<span class="loading-spinner"></span>${text}`;
}

export function resetButton(button, text) {
    if (!button) {
        return;
    }
    button.disabled = false;
    button.textContent = text;
}

export function showGeneratedAudio(audioUrl) {
    if (!elements.audioOutput || !elements.audioPlaceholder) {
        return;
    }
    elements.audioOutput.src = audioUrl;
    elements.audioOutput.classList.add('active');
    elements.audioPlaceholder.style.display = 'none';
}

export function updateModeVisibility(modeValue) {
    const mode = modeValue || elements.mode()?.value || 'zero_shot';

    if (elements.promptAudioGroup) {
        elements.promptAudioGroup.classList.remove('hidden');
    }

    if (elements.promptTextGroup) {
        elements.promptTextGroup.classList.toggle('hidden', mode !== 'zero_shot');
    }

    if (elements.instructTextGroup) {
        elements.instructTextGroup.classList.toggle('hidden', mode !== 'instruct');
    }
}
