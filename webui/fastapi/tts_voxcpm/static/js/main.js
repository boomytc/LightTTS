import elements from './dom.js';
import { loadModelAction, generateSpeechAction, stopGenerationAction } from './actions.js';
import { handleAudioUpload, useDefaultAudio, loadDefaultAudio } from './audio.js';

function on(element, event, handler) {
    if (element) {
        element.addEventListener(event, handler);
    }
}

function bindRangeDisplay(rangeInput, displayElement, formatter = (value) => value) {
    if (!rangeInput || !displayElement) {
        return;
    }
    rangeInput.addEventListener('input', (event) => {
        displayElement.textContent = formatter(event.target.value);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    bindRangeDisplay(elements.cfgValue, elements.cfgValueDisplay, (value) => parseFloat(value).toFixed(1));
    bindRangeDisplay(elements.inferenceTimesteps, elements.timestepsDisplay);
    bindRangeDisplay(elements.retryMaxTimes, elements.retryTimesDisplay);
    bindRangeDisplay(elements.retryRatioThreshold, elements.retryThresholdDisplay, (value) => parseFloat(value).toFixed(1));

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

    loadDefaultAudio();
});
