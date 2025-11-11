let modelLoaded = false;
let generating = false;
let abortController = null;

export function setModelLoaded(value) {
    modelLoaded = value;
}

export function isModelLoaded() {
    return modelLoaded;
}

export function setGenerating(value) {
    generating = value;
}

export function isGenerating() {
    return generating;
}

export function setAbortController(controller) {
    abortController = controller;
}

export function getAbortController() {
    return abortController;
}
