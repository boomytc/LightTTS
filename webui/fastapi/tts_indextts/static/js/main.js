import elements from './dom.js';
import { loadModelAction, generateSpeechAction, stopGenerationAction } from './actions.js';
import { handleAudioUpload, useDefaultAudio, loadDefaultAudio } from './audio.js';
import { updateEmoModeVisibility, updateSliderValue } from './ui.js';

function on(element, event, handler) {
    if (element) {
        element.addEventListener(event, handler);
    }
}

function initSliderValue(sliderId, valueElementId, formatter = (value) => value) {
    const slider = document.getElementById(sliderId);
    if (!slider) {
        return;
    }
    const update = (value) => updateSliderValue(valueElementId, formatter(value));
    update(slider.value);
    slider.addEventListener('input', (event) => {
        update(event.target.value);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    on(elements.loadBtn, 'click', loadModelAction);
    on(elements.generateBtn, 'click', generateSpeechAction);
    on(elements.stopBtn, 'click', stopGenerationAction);

    on(elements.uploadAudioBtn, 'click', () => elements.promptAudio?.click());
    on(elements.promptAudio, 'change', handleAudioUpload);
    on(elements.useDefaultAudioBtn, 'click', useDefaultAudio);

    document.querySelectorAll('input[name="emo-mode"]').forEach((radio) => {
        radio.addEventListener('change', (event) => updateEmoModeVisibility(event.target.value));
    });

    initSliderValue('emo-alpha', 'emo-alpha-value', (value) => parseFloat(value).toFixed(2));
    initSliderValue('interval-silence', 'interval-silence-value');
    initSliderValue('max-tokens', 'max-tokens-value');

    document.querySelectorAll('.emo-slider').forEach((slider) => {
        const valueId = `${slider.id}-value`;
        initSliderValue(slider.id, valueId, (value) => parseFloat(value).toFixed(2));
    });

    updateEmoModeVisibility();
    loadDefaultAudio();
});
