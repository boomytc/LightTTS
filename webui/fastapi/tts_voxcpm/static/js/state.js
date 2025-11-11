let modelLoaded = false;
let isGenerating = false;
let currentAbortController = null;

export function setModelLoaded(value) {
    modelLoaded = value;
}

export function isModelLoaded() {
    return modelLoaded;
}

export function setGenerating(value) {
    isGenerating = value;
}

export function isGeneratingAudio() {
    return isGenerating;
}

export function setAbortController(controller) {
    currentAbortController = controller;
}

export function getAbortController() {
    return currentAbortController;
}
