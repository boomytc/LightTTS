let currentGenerateRequest = null;

export function setGenerateRequest(controller) {
    currentGenerateRequest = controller;
}

export function getGenerateRequest() {
    return currentGenerateRequest;
}
