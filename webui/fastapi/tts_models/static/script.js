window.addEventListener('DOMContentLoaded', () => {
    let currentModel = null;
    let isModelLoaded = false;
    let isGenerating = false;
    let currentAbortController = null;
    let currentAudioUrl = null;

    const bodyDataset = document.body && document.body.dataset ? document.body.dataset : {};
    const hasDefaultAudio = bodyDataset.hasDefaultAudio === 'true';
    const defaultPromptTextFromServer = (bodyDataset.defaultPromptText || '').trim();
    const defaultAudioUrl = '/api/default_audio';

    const elements = {
        modelRadios: Array.from(document.querySelectorAll('input[name="model"]')),
        deviceRadios: Array.from(document.querySelectorAll('input[name="device"]')),
        loadBtn: document.getElementById('load-btn'),
        generateBtn: document.getElementById('generate-btn'),
        stopBtn: document.getElementById('stop-btn'),
        status: document.getElementById('status'),
        audioOutput: document.getElementById('audio-output'),
        audioPlaceholder: document.getElementById('audio-placeholder'),
        text: document.getElementById('text'),
        configCosyvoice: document.getElementById('config-cosyvoice'),
        configIndextts: document.getElementById('config-indextts'),
        configVoxcpm: document.getElementById('config-voxcpm'),
    };

    const promptTextCosyvoice = document.getElementById('prompt-text');
    const promptTextVoxcpm = document.getElementById('prompt-text-voxcpm');
    const fallbackPromptText =
        defaultPromptTextFromServer ||
        (promptTextCosyvoice && promptTextCosyvoice.defaultValue && promptTextCosyvoice.defaultValue.trim()) ||
        (promptTextVoxcpm && promptTextVoxcpm.defaultValue && promptTextVoxcpm.defaultValue.trim()) ||
        '';

    function updateStatus(message, type) {
        const statusEl = elements.status;
        if (!statusEl) {
            return;
        }
        statusEl.textContent = message;
        let background = '#f7fafc';
        let border = '#e0e0e0';
        if (type === 'error') {
            background = '#fed7d7';
            border = '#fc8181';
        } else if (type === 'success') {
            background = '#c6f6d5';
            border = '#68d391';
        }
        statusEl.style.background = background;
        statusEl.style.borderColor = border;
    }

    function switchModelConfig(model) {
        if (elements.configCosyvoice) {
            elements.configCosyvoice.style.display = model === 'cosyvoice' ? 'block' : 'none';
        }
        if (elements.configIndextts) {
            elements.configIndextts.style.display = model === 'indextts' ? 'block' : 'none';
        }
        if (elements.configVoxcpm) {
            elements.configVoxcpm.style.display = model === 'voxcpm' ? 'block' : 'none';
        }
    }

    function getCheckedValue(name) {
        const checked = document.querySelector(`input[name="${name}"]:checked`);
        return checked ? checked.value : null;
    }

    function setModelLoadedState(loaded) {
        isModelLoaded = loaded;
        if (elements.generateBtn) {
            elements.generateBtn.disabled = !loaded;
        }
    }

    function attachModelSelectionHandlers() {
        if (!elements.modelRadios.length) {
            return;
        }

        const onChange = (event) => {
            currentModel = event.target.value;
            setModelLoadedState(false);
            if (elements.loadBtn) {
                elements.loadBtn.disabled = false;
            }
            if (elements.text) {
                elements.text.placeholder = `请先加载 ${currentModel.toUpperCase()} 模型`;
            }
            updateStatus(`已选择 ${currentModel.toUpperCase()} 模型，请点击"加载模型"按钮`);
            switchModelConfig(currentModel);
        };

        elements.modelRadios.forEach((radio) => {
            radio.addEventListener('change', onChange);
        });

        const initiallyChecked = elements.modelRadios.find((radio) => radio.checked);
        if (initiallyChecked) {
            initiallyChecked.dispatchEvent(new Event('change'));
        }
    }

    async function handleLoadModel() {
        if (!currentModel) {
            updateStatus('请先选择模型', 'error');
            return;
        }

        const device = getCheckedValue('device') || 'cuda';

        if (elements.loadBtn) {
            elements.loadBtn.disabled = true;
            elements.loadBtn.textContent = '加载中...';
        }
        updateStatus(`正在加载 ${currentModel.toUpperCase()} 模型...`);

        try {
            const response = await fetch('/api/load_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_type: currentModel, device }),
            });
            const data = await response.json();

            if (data.status === 'success') {
                setModelLoadedState(true);
                if (elements.text) {
                    elements.text.placeholder = '请输入需要合成的文本';
                }
                updateStatus(data.message, 'success');
            } else {
                setModelLoadedState(false);
                updateStatus(data.message || '模型加载失败', 'error');
            }
        } catch (error) {
            setModelLoadedState(false);
            updateStatus(`加载失败: ${error.message}`, 'error');
        } finally {
            if (elements.loadBtn) {
                elements.loadBtn.disabled = false;
                elements.loadBtn.textContent = '加载模型';
            }
        }
    }

    function appendFileField(formData, inputId, fieldName) {
        const input = document.getElementById(inputId);
        if (input && input.files && input.files.length > 0) {
            formData.append(fieldName, input.files[0]);
            return true;
        }
        return false;
    }

    function appendCommonCosyvoiceFields(formData) {
        formData.append('text', elements.text ? elements.text.value.trim() : '');
        formData.append('mode', getCheckedValue('mode') || 'zero_shot');
        formData.append('prompt_text', promptTextCosyvoice ? promptTextCosyvoice.value.trim() : '');
        const instructField = document.getElementById('instruct-text');
        formData.append('instruct_text', instructField ? instructField.value.trim() : '');
        const speedInput = document.getElementById('speed');
        formData.append('speed', speedInput ? speedInput.value : '1.0');
        const seedInput = document.getElementById('seed');
        formData.append('seed', seedInput ? seedInput.value : '0');
        appendFileField(formData, 'prompt-audio-cosyvoice', 'prompt_audio');
    }

    function appendCommonIndexFields(formData) {
        formData.append('text', elements.text ? elements.text.value.trim() : '');
        const emoMode = getCheckedValue('emo-mode') || 'none';
        formData.append('emo_mode', emoMode);
        const emoAlpha = document.getElementById('emo-alpha');
        formData.append('emo_alpha', emoAlpha ? emoAlpha.value : '1.0');
        const emoText = document.getElementById('emo-text');
        formData.append('emo_text', emoText ? emoText.value.trim() : '');
        const intervalSilence = document.getElementById('interval-silence');
        formData.append('interval_silence', intervalSilence ? intervalSilence.value : '200');
        const maxTokens = document.getElementById('max-tokens');
        formData.append('max_tokens', maxTokens ? maxTokens.value : '120');
        const useRandom = document.getElementById('use-random');
        formData.append('use_random', useRandom && useRandom.checked ? 'true' : 'false');

        appendFileField(formData, 'prompt-audio-indextts', 'prompt_audio');
        appendFileField(formData, 'emo-audio', 'emo_audio');

        const emoVector = [
            'emo-happy',
            'emo-angry',
            'emo-sad',
            'emo-afraid',
            'emo-disgusted',
            'emo-melancholic',
            'emo-surprised',
            'emo-calm',
        ].map((id) => {
            const slider = document.getElementById(id);
            return slider ? parseFloat(slider.value || '0') : 0;
        });
        formData.append('emo_vector', JSON.stringify(emoVector));
    }

    function appendCommonVoxcpmFields(formData) {
        formData.append('text', elements.text ? elements.text.value.trim() : '');
        formData.append('prompt_text', promptTextVoxcpm ? promptTextVoxcpm.value.trim() : '');

        const cfgValue = document.getElementById('cfg-value');
        formData.append('cfg_value', cfgValue ? cfgValue.value : '2.0');
        const timesteps = document.getElementById('inference-timesteps');
        formData.append('inference_timesteps', timesteps ? timesteps.value : '10');
        const normalize = document.getElementById('normalize');
        formData.append('normalize', normalize && normalize.checked ? 'true' : 'false');
        const denoise = document.getElementById('denoise');
        formData.append('denoise', denoise && denoise.checked ? 'true' : 'false');
        const retryBadcase = document.getElementById('retry-badcase');
        formData.append('retry_badcase', retryBadcase && retryBadcase.checked ? 'true' : 'false');
        const retryMaxTimes = document.getElementById('retry-max-times');
        formData.append('retry_max_times', retryMaxTimes ? retryMaxTimes.value : '3');
        const retryThreshold = document.getElementById('retry-ratio-threshold');
        formData.append('retry_ratio_threshold', retryThreshold ? retryThreshold.value : '6.0');

        appendFileField(formData, 'prompt-audio-voxcpm', 'prompt_audio');
    }

    async function sendGenerateRequest(formData) {
        if (currentAbortController) {
            currentAbortController.abort();
        }
        const abortController = new AbortController();
        currentAbortController = abortController;

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                body: formData,
                signal: abortController.signal,
            });
            const data = await response.json();

            if (data.status === 'success') {
                const audioBlob = base64ToBlob(data.audio, 'audio/wav');
                if (currentAudioUrl) {
                    URL.revokeObjectURL(currentAudioUrl);
                }
                currentAudioUrl = URL.createObjectURL(audioBlob);

                if (elements.audioOutput) {
                    elements.audioOutput.src = currentAudioUrl;
                    elements.audioOutput.style.display = 'block';
                }
                if (elements.audioPlaceholder) {
                    elements.audioPlaceholder.style.display = 'none';
                }
                updateStatus(data.message || '生成完成', 'success');
            } else {
                const message = data.message || '生成失败';
                updateStatus(message, 'error');
            }
        } finally {
            if (currentAbortController === abortController) {
                currentAbortController = null;
            }
        }
    }

    async function handleGenerate() {
        if (!isModelLoaded) {
            updateStatus('请先加载模型', 'error');
            return;
        }

        if (isGenerating) {
            return;
        }

        if (!elements.text) {
            updateStatus('页面未正确加载文本输入框', 'error');
            return;
        }

        const textValue = elements.text.value.trim();
        if (!textValue) {
            updateStatus('请输入待合成文本', 'error');
            return;
        }

        if (!currentModel) {
            updateStatus('请先选择模型', 'error');
            return;
        }

        isGenerating = true;
        if (elements.generateBtn) {
            elements.generateBtn.disabled = true;
            elements.generateBtn.textContent = '生成中...';
        }
        updateStatus('正在生成语音...');

        const formData = new FormData();
        try {
            if (currentModel === 'cosyvoice') {
                appendCommonCosyvoiceFields(formData);
            } else if (currentModel === 'indextts') {
                appendCommonIndexFields(formData);
            } else if (currentModel === 'voxcpm') {
                appendCommonVoxcpmFields(formData);
            } else {
                updateStatus(`未知的模型类型: ${currentModel}`, 'error');
                return;
            }

            await sendGenerateRequest(formData);
        } catch (error) {
            if (error.name === 'AbortError') {
                updateStatus('生成已停止', 'error');
            } else {
                updateStatus(`生成失败: ${error.message}`, 'error');
            }
        } finally {
            isGenerating = false;
            if (elements.generateBtn) {
                elements.generateBtn.disabled = false;
                elements.generateBtn.textContent = '生成语音';
            }
        }
    }

    function stopGeneration() {
        if (currentAbortController) {
            currentAbortController.abort();
        }
    }

    function base64ToBlob(base64, mimeType) {
        const byteCharacters = window.atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i += 1) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
    }

    function setupRangeDisplay(inputId, displayId, formatter) {
        const input = document.getElementById(inputId);
        const display = document.getElementById(displayId);
        if (!input || !display) {
            return;
        }
        const update = () => {
            const value = input.value;
            display.textContent = formatter ? formatter(value) : value;
        };
        input.addEventListener('input', update);
        update();
    }

    function attachCosyvoiceModeHandlers() {
        const modeRadios = Array.from(document.querySelectorAll('input[name="mode"]'));
        if (!modeRadios.length) {
            return;
        }
        const promptTextGroup = document.getElementById('prompt-text-group');
        const instructTextGroup = document.getElementById('instruct-text-group');

        const updateVisibility = (value) => {
            if (promptTextGroup) {
                promptTextGroup.classList.toggle('hidden', value === 'cross_lingual');
            }
            if (instructTextGroup) {
                instructTextGroup.classList.toggle('hidden', value !== 'instruct');
            }
        };

        modeRadios.forEach((radio) => {
            radio.addEventListener('change', () => updateVisibility(radio.value));
        });

        const checked = modeRadios.find((radio) => radio.checked);
        updateVisibility(checked ? checked.value : 'zero_shot');
    }

    function attachEmotionModeHandlers() {
        const emoRadios = Array.from(document.querySelectorAll('input[name="emo-mode"]'));
        if (!emoRadios.length) {
            return;
        }

        const emoAudioGroup = document.getElementById('emo-audio-group');
        const emoAlphaGroup = document.getElementById('emo-alpha-group');
        const emoVectorGroup = document.getElementById('emo-vector-group');
        const emoTextGroup = document.getElementById('emo-text-group');

        const setHidden = (element, hidden) => {
            if (element) {
                element.classList.toggle('hidden', hidden);
            }
        };

        const updateVisibility = (value) => {
            setHidden(emoAudioGroup, value !== 'audio');
            setHidden(emoAlphaGroup, value !== 'audio' && value !== 'text');
            setHidden(emoVectorGroup, value !== 'vector');
            setHidden(emoTextGroup, value !== 'text');
        };

        emoRadios.forEach((radio) => {
            radio.addEventListener('change', () => updateVisibility(radio.value));
        });

        const checked = emoRadios.find((radio) => radio.checked);
        updateVisibility(checked ? checked.value : 'none');
    }

    function initAudioUploadControls() {
        const uploadButtons = document.querySelectorAll('.upload-audio-btn');
        uploadButtons.forEach((btn) => {
            btn.addEventListener('click', () => {
                const parent = btn.closest('.form-group');
                if (!parent) {
                    return;
                }
                const fileInput = parent.querySelector('input[type="file"]');
                if (fileInput) {
                    fileInput.click();
                }
            });
        });

        const defaultButtons = document.querySelectorAll('.use-default-audio-btn');
        defaultButtons.forEach((btn) => {
            if (!hasDefaultAudio) {
                btn.disabled = true;
                btn.title = '默认参考音频不存在';
                return;
            }
            btn.addEventListener('click', () => {
                const parent = btn.closest('.form-group');
                if (!parent) {
                    return;
                }
                const fileInput = parent.querySelector('input[type="file"]');
                const preview = parent.querySelector('.prompt-audio-preview');
                const fileName = parent.querySelector('.audio-file-name');

                if (fileInput) {
                    fileInput.value = '';
                }
                if (preview) {
                    preview.src = defaultAudioUrl;
                    preview.load();
                }
                if (fileName) {
                    fileName.textContent = '当前使用：默认参考音频';
                }

                if (promptTextCosyvoice && parent.closest('#config-cosyvoice')) {
                    promptTextCosyvoice.value = fallbackPromptText;
                }
                if (promptTextVoxcpm && parent.closest('#config-voxcpm')) {
                    promptTextVoxcpm.value = fallbackPromptText;
                }
            });
        });

        const audioInputs = document.querySelectorAll('input[type="file"][accept="audio/*"]');
        audioInputs.forEach((input) => {
            input.addEventListener('change', (event) => {
                const parent = event.target.closest('.form-group');
                if (!parent) {
                    return;
                }
                const preview = parent.querySelector('.prompt-audio-preview');
                const fileName = parent.querySelector('.audio-file-name');
                const { files } = event.target;
                if (files && files.length > 0) {
                    const file = files[0];
                    if (preview) {
                        const url = URL.createObjectURL(file);
                        preview.src = url;
                        preview.load();
                    }
                    if (fileName) {
                        fileName.textContent = `当前文件：${file.name}`;
                    }
                }
            });
        });

        if (!hasDefaultAudio) {
            const fileLabels = document.querySelectorAll('.audio-file-name');
            fileLabels.forEach((label) => {
                label.textContent = '未找到默认参考音频，请上传文件';
            });
            const previews = document.querySelectorAll('.prompt-audio-preview');
            previews.forEach((preview) => {
                preview.removeAttribute('src');
                preview.load();
            });
        }
    }

    function initSliderDisplays() {
        setupRangeDisplay('speed', 'speed-value', (value) => parseFloat(value || '1').toFixed(2));
        setupRangeDisplay('emo-alpha', 'emo-alpha-value', (value) => parseFloat(value || '1').toFixed(2));
        setupRangeDisplay('interval-silence', 'interval-silence-value');
        setupRangeDisplay('max-tokens', 'max-tokens-value');
        setupRangeDisplay('cfg-value', 'cfg-value-display', (value) => parseFloat(value || '2').toFixed(1));
        setupRangeDisplay('inference-timesteps', 'timesteps-display');
        setupRangeDisplay('retry-max-times', 'retry-times-display');
        setupRangeDisplay('retry-ratio-threshold', 'retry-threshold-display', (value) => parseFloat(value || '6').toFixed(1));

        const emotionSliders = document.querySelectorAll('.emo-slider');
        emotionSliders.forEach((slider) => {
            slider.addEventListener('input', (event) => {
                const valueSpan = document.getElementById(`${event.target.id}-value`);
                if (valueSpan) {
                    valueSpan.textContent = parseFloat(event.target.value || '0').toFixed(2);
                }
            });
            const initialSpan = document.getElementById(`${slider.id}-value`);
            if (initialSpan) {
                initialSpan.textContent = parseFloat(slider.value || '0').toFixed(2);
            }
        });
    }

    if (elements.loadBtn) {
        elements.loadBtn.addEventListener('click', handleLoadModel);
    }
    if (elements.generateBtn) {
        elements.generateBtn.addEventListener('click', handleGenerate);
    }
    if (elements.stopBtn) {
        elements.stopBtn.addEventListener('click', stopGeneration);
    }

    attachModelSelectionHandlers();
    attachCosyvoiceModeHandlers();
    attachEmotionModeHandlers();
    initAudioUploadControls();
    initSliderDisplays();

    updateStatus('请选择模型并加载...');
});
