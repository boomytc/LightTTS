import elements from './dom.js';
import { loadModelAction, generateSpeechAction, stopGenerationAction } from './actions.js';
import { handleAudioUpload, useDefaultAudio, loadDefaultAudio } from './audio.js';
import { updateModeVisibility } from './ui.js';

function on(element, event, handler) {
    if (element) {
        element.addEventListener(event, handler);
    }
}

function bindRangeDisplay(rangeInput, displayElement, formatter = (value) => value) {
    if (!rangeInput || !displayElement) {
        return;
    }
    displayElement.textContent = formatter(rangeInput.value);
    rangeInput.addEventListener('input', (event) => {
        displayElement.textContent = formatter(event.target.value);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    bindRangeDisplay(elements.speed, elements.speedValue, (value) => parseFloat(value).toFixed(2));

    document.querySelectorAll('input[name="mode"]').forEach((radio) => {
        radio.addEventListener('change', (event) => {
            updateModeVisibility(event.target.value);
        });
    });

    on(elements.uploadAudioBtn, 'click', () => {
        if (elements.promptAudio) {
            elements.promptAudio.click();
        }
    });
    on(elements.promptAudio, 'change', handleAudioUpload);
    on(elements.useDefaultAudioBtn, 'click', useDefaultAudio);

    on(elements.loadBtn, 'click', loadModelAction);
    on(elements.generateBtn, 'click', generateSpeechAction);
    on(elements.stopBtn, 'click', stopGenerationAction);

    updateModeVisibility();
    loadDefaultAudio();
});
