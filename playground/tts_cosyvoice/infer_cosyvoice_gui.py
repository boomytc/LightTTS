import os
import sys
from pathlib import Path

# 路径设置，确保本地 cosyvoice 包优先导入
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
matcha_path = os.path.join(project_root, 'Matcha-TTS')
sys.path.insert(0, project_root)
sys.path.insert(0, matcha_path)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import onnxruntime as ort
ort.set_default_logger_severity(3)

import torch
import torchaudio

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QProgressBar,
    QGroupBox, QMessageBox, QDoubleSpinBox, QSpinBox, QComboBox,
    QFormLayout, QHBoxLayout
)
from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtGui import QFont
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

# ============ 常量配置 ============
DEFAULT_MODEL_DIR = "models/CosyVoice2-0.5B"
DEFAULT_OUTPUT_DIR = "playground/tts_cosyvoice/voice_output"
PROMPT_SAMPLE_RATE = 16000
DEFAULT_SPEED = 1.0
DEFAULT_SEED = -1
MAX_VAL = 0.95

# 运行配置（参考 webui/gradio 逻辑）
USE_FP16 = True
LOAD_JIT = False
LOAD_TRT = False
LOAD_VLLM = False
TRT_CONCURRENT = 1

# 默认参考音频与文本
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呀。"

# 模式映射
MODE_MAPPING = {
    "零样本克隆": "zero_shot",
    "跨语言克隆": "cross_lingual",
    "指令控制": "instruct",
}


def prepare_prompt_audio(prompt_audio_path: str) -> torch.Tensor:
    """验证并加载参考提示音频。"""
    if not prompt_audio_path:
        prompt_audio_path = DEFAULT_PROMPT_WAV

    if not os.path.isfile(prompt_audio_path):
        raise FileNotFoundError(f"参考音频不存在: {prompt_audio_path}")

    sample_rate = torchaudio.info(prompt_audio_path).sample_rate
    if sample_rate < PROMPT_SAMPLE_RATE:
        raise ValueError(f"参考音频采样率需至少 {PROMPT_SAMPLE_RATE}Hz。")

    return load_wav(prompt_audio_path, PROMPT_SAMPLE_RATE)


def merge_segments(segments) -> torch.Tensor:
    """将生成的音频片段合并成单一波形。"""
    audio_tensors = []
    for segment in segments:
        tensor = segment.get("tts_speech")
        if tensor is None:
            continue
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        audio_tensors.append(tensor)

    if not audio_tensors:
        raise RuntimeError("生成结果为空。")

    return torch.cat(audio_tensors, dim=-1)


class ModelLoadWorker(QObject):
    """模型加载线程"""
    status_updated = Signal(str)
    log_updated = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, model_dir: str, device: str):
        super().__init__()
        self.model_dir = model_dir
        self.device = device

    def load_model(self):
        try:
            self.status_updated.emit("正在加载模型...")
            self.log_updated.emit(f"加载 CosyVoice2 模型: {self.model_dir}")
            is_cuda = self.device == "cuda" and torch.cuda.is_available()
            cosyvoice = CosyVoice2(
                model_dir=self.model_dir,
                load_jit=LOAD_JIT,
                load_trt=LOAD_TRT,
                load_vllm=LOAD_VLLM,
                fp16=is_cuda and USE_FP16,
                trt_concurrent=TRT_CONCURRENT,
                device=self.device,
            )
            self.status_updated.emit("模型加载完成")
            self.log_updated.emit("模型加载完成")
            self.finished.emit(cosyvoice)
        except Exception as e:
            msg = f"模型加载失败: {e}"
            self.status_updated.emit(msg)
            self.log_updated.emit(msg)
            self.error.emit(msg)


class SingleSynthesisWorker(QObject):
    """单音频合成线程"""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    log_updated = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        cosyvoice: CosyVoice2,
        mode: str,
        prompt_audio_path: str,
        prompt_text: str,
        instruct_text: str,
        tts_text: str,
        speed: float,
        seed: int | None,
        output_path: str,
    ):
        super().__init__()
        self.cosyvoice = cosyvoice
        self.mode = mode
        self.prompt_audio_path = prompt_audio_path
        self.prompt_text = prompt_text
        self.instruct_text = instruct_text
        self.tts_text = tts_text
        self.speed = speed
        self.seed = seed
        self.output_path = output_path

    def _normalize(self, speech: torch.Tensor) -> torch.Tensor:
        if speech.abs().max() > MAX_VAL:
            speech = speech / speech.abs().max() * MAX_VAL
        return speech

    def run(self):
        try:
            self.progress_updated.emit(0)
            self.status_updated.emit("准备音色...")
            self.log_updated.emit(f"音色音频: {self.prompt_audio_path}")

            if not Path(self.prompt_audio_path).exists():
                raise FileNotFoundError(f"音频文件不存在: {self.prompt_audio_path}")

            audio_info = torchaudio.info(self.prompt_audio_path)
            if audio_info.sample_rate < PROMPT_SAMPLE_RATE:
                raise RuntimeError(
                    f"音频采样率过低 ({audio_info.sample_rate} Hz)，需要至少 {PROMPT_SAMPLE_RATE} Hz"
                )

            prompt_speech_16k = load_wav(self.prompt_audio_path, PROMPT_SAMPLE_RATE)
            prompt_speech_16k = self._normalize(prompt_speech_16k)

            self.status_updated.emit("开始合成...")
            self.progress_updated.emit(30)

            if self.seed is not None:
                set_all_random_seed(self.seed)

            mode_key = MODE_MAPPING.get(self.mode, self.mode)
            if mode_key == "zero_shot":
                result_iter = self.cosyvoice.inference_zero_shot(
                    self.tts_text,
                    self.prompt_text,
                    prompt_speech_16k,
                    stream=False,
                    speed=self.speed,
                )
            elif mode_key == "cross_lingual":
                result_iter = self.cosyvoice.inference_cross_lingual(
                    self.tts_text,
                    prompt_speech_16k,
                    stream=False,
                    speed=self.speed,
                )
            elif mode_key == "instruct":
                result_iter = self.cosyvoice.inference_instruct2(
                    self.tts_text,
                    self.instruct_text,
                    prompt_speech_16k,
                    stream=False,
                    speed=self.speed,
                )
            else:
                raise RuntimeError(f"无效的模式: {self.mode}")

            audio_tensor = merge_segments(result_iter)

            Path(os.path.dirname(self.output_path)).mkdir(parents=True, exist_ok=True)
            self.progress_updated.emit(80)
            torchaudio.save(self.output_path, audio_tensor, self.cosyvoice.sample_rate)

            self.progress_updated.emit(100)
            self.status_updated.emit("合成完成")
            self.log_updated.emit(f"输出: {self.output_path}")
            self.finished.emit(self.output_path)
        except Exception as e:
            msg = f"合成失败: {e}"
            self.status_updated.emit(msg)
            self.log_updated.emit(msg)
            self.error.emit(msg)


class SingleSynthesisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_load_thread = None
        self.model_load_worker = None
        self.synth_thread = None
        self.synth_worker = None
        self.cosyvoice_model: CosyVoice2 | None = None
        self.media_player = None
        self.audio_output = None
        self.last_output_path: str | None = None

        self.init_ui()
        self.init_audio_player()

    def init_audio_player(self):
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

    def init_ui(self):
        self.setWindowTitle("LightTTS 单音频语音合成")
        self.setGeometry(120, 120, 900, 480)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(10)
        root_layout.setContentsMargins(15, 15, 15, 15)

        # 模型设置
        model_group = QGroupBox("模型设置")
        model_layout = QGridLayout()
        model_layout.setSpacing(8)
        model_layout.setColumnStretch(1, 1)
        model_group.setLayout(model_layout)

        model_layout.addWidget(QLabel("模型路径:"), 0, 0)
        self.model_dir_edit = QLineEdit(DEFAULT_MODEL_DIR)
        model_layout.addWidget(self.model_dir_edit, 0, 1)
        self.model_browse_btn = QPushButton("浏览")
        self.model_browse_btn.setFixedWidth(70)
        self.model_browse_btn.clicked.connect(self.select_model_dir)
        model_layout.addWidget(self.model_browse_btn, 0, 2)

        model_layout.addWidget(QLabel("运行设备:"), 1, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        self.device_combo.setCurrentText("cuda" if torch.cuda.is_available() else "cpu")
        device_row = QHBoxLayout()
        device_row.setContentsMargins(0, 0, 0, 0)
        device_row.addWidget(self.device_combo)
        device_row.addStretch()
        model_layout.addLayout(device_row, 1, 1)

        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.setFixedWidth(100)
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn, 1, 2)

        self.model_status_label = QLabel("未加载")
        self.model_status_label.setStyleSheet("color: #dc3545;")
        model_layout.addWidget(self.model_status_label, 2, 0, 1, 3)

        root_layout.addWidget(model_group)

        # 参数输入
        io_group = QGroupBox("参数输入")
        io_layout = QFormLayout()
        io_layout.setSpacing(12)
        io_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        io_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        io_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        io_layout.setContentsMargins(12, 12, 12, 12)
        io_group.setLayout(io_layout)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(MODE_MAPPING.keys()))
        self.mode_combo.setCurrentText("零样本克隆")
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)

        self.speed_spinbox = QDoubleSpinBox()
        self.speed_spinbox.setRange(0.5, 2.0)
        self.speed_spinbox.setSingleStep(0.05)
        self.speed_spinbox.setValue(DEFAULT_SPEED)
        self.speed_spinbox.setFixedWidth(90)

        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(DEFAULT_SEED, 999999999)
        self.seed_spinbox.setValue(DEFAULT_SEED)
        self.seed_spinbox.setSpecialValueText("随机")
        self.seed_spinbox.setFixedWidth(120)

        mode_row_widget = QWidget()
        mode_row_layout = QHBoxLayout(mode_row_widget)
        mode_row_layout.setContentsMargins(0, 0, 0, 0)
        mode_row_layout.setSpacing(12)
        mode_row_layout.addWidget(self.mode_combo)
        mode_row_layout.addSpacing(12)
        mode_row_layout.addWidget(QLabel("语速:"))
        mode_row_layout.addWidget(self.speed_spinbox)
        mode_row_layout.addSpacing(12)
        mode_row_layout.addWidget(QLabel("随机种子:"))
        mode_row_layout.addWidget(self.seed_spinbox)
        mode_row_layout.addStretch()
        io_layout.addRow("推理模式:", mode_row_widget)

        self.prompt_audio_edit = QLineEdit()
        self.prompt_audio_edit.setText(DEFAULT_PROMPT_WAV if os.path.isfile(DEFAULT_PROMPT_WAV) else "")
        self.prompt_audio_btn = QPushButton("选择")
        self.prompt_audio_btn.setFixedWidth(70)
        self.prompt_audio_btn.clicked.connect(self.select_prompt_audio)
        prompt_audio_row = QWidget()
        prompt_audio_layout = QHBoxLayout(prompt_audio_row)
        prompt_audio_layout.setContentsMargins(0, 0, 0, 0)
        prompt_audio_layout.setSpacing(8)
        prompt_audio_layout.addWidget(self.prompt_audio_edit)
        prompt_audio_layout.addWidget(self.prompt_audio_btn)
        io_layout.addRow("音色音频:", prompt_audio_row)

        self.prompt_text_label = QLabel("音色文本:")
        self.prompt_text_edit = QLineEdit()
        self.prompt_text_edit.setText(DEFAULT_PROMPT_TEXT)
        io_layout.addRow(self.prompt_text_label, self.prompt_text_edit)

        self.instruct_text_label = QLabel("指令文本:")
        self.instruct_text_edit = QLineEdit()
        io_layout.addRow(self.instruct_text_label, self.instruct_text_edit)

        self.tts_text_edit = QLineEdit()
        io_layout.addRow("合成文本:", self.tts_text_edit)

        self.output_name_edit = QLineEdit("single.wav")
        io_layout.addRow("输出文件:", self.output_name_edit)

        self.output_dir_edit = QLineEdit(DEFAULT_OUTPUT_DIR)
        self.output_browse_btn = QPushButton("浏览")
        self.output_browse_btn.setFixedWidth(70)
        self.output_browse_btn.clicked.connect(self.select_output_dir)
        output_dir_row = QWidget()
        output_dir_layout = QHBoxLayout(output_dir_row)
        output_dir_layout.setContentsMargins(0, 0, 0, 0)
        output_dir_layout.setSpacing(8)
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_browse_btn)
        io_layout.addRow("输出目录:", output_dir_row)

        root_layout.addWidget(io_group)

        # 合成控制
        ctrl_group = QGroupBox("合成控制")
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setSpacing(8)
        ctrl_layout.setContentsMargins(12, 12, 12, 12)
        ctrl_group.setLayout(ctrl_layout)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666;")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        ctrl_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedHeight(22)
        ctrl_layout.addWidget(self.progress_bar)

        # 按钮行：开始合成 + 播放输出
        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(12)

        self.start_btn = QPushButton("开始合成")
        self.start_btn.setFixedHeight(35)
        self.start_btn.clicked.connect(self.start_synthesis)
        self.start_btn.setEnabled(False)
        button_row.addWidget(self.start_btn)

        self.play_btn = QPushButton("播放输出")
        self.play_btn.setFixedHeight(35)
        self.play_btn.clicked.connect(self.play_output)
        self.play_btn.setEnabled(False)
        button_row.addWidget(self.play_btn)
        button_row.addStretch()

        ctrl_layout.addLayout(button_row)

        root_layout.addWidget(ctrl_group)
        self.on_mode_changed(self.mode_combo.currentText())

    # ===== 交互方法 =====
    def select_model_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if dir_path:
            self.model_dir_edit.setText(dir_path)

    def select_prompt_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择音色音频", filter="音频文件 (*.wav *.mp3 *.flac)")
        if file_path:
            self.prompt_audio_edit.setText(file_path)

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def load_model(self):
        if not Path(self.model_dir_edit.text()).exists():
            QMessageBox.warning(self, "错误", "模型路径不存在！")
            return

        self.load_model_btn.setEnabled(False)
        self.model_status_label.setText("加载中...")
        self.model_status_label.setStyleSheet("color: #ffc107;")

        self.model_load_thread = QThread()
        self.model_load_worker = ModelLoadWorker(self.model_dir_edit.text(), self.device_combo.currentText())

        self.model_load_worker.status_updated.connect(self.status_label.setText)
        self.model_load_worker.log_updated.connect(lambda s: None)
        self.model_load_worker.finished.connect(self.on_model_loaded)
        self.model_load_worker.error.connect(self.on_model_load_error)

        self.model_load_worker.moveToThread(self.model_load_thread)
        self.model_load_thread.started.connect(self.model_load_worker.load_model)
        self.model_load_thread.start()

    def on_model_loaded(self, model: CosyVoice2):
        self.cosyvoice_model = model
        self.model_status_label.setText("已加载")
        self.model_status_label.setStyleSheet("color: #28a745;")
        self.start_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        if self.model_load_thread:
            self.model_load_thread.quit()
            self.model_load_thread.wait()

    def on_model_load_error(self, msg: str):
        self.model_status_label.setText("加载失败")
        self.model_status_label.setStyleSheet("color: #dc3545;")
        self.load_model_btn.setEnabled(True)
        QMessageBox.critical(self, "错误", msg)
        if self.model_load_thread:
            self.model_load_thread.quit()
            self.model_load_thread.wait()

    def start_synthesis(self):
        if self.cosyvoice_model is None:
            QMessageBox.warning(self, "错误", "请先加载模型！")
            return

        mode = self.mode_combo.currentText()
        prompt_audio = self.prompt_audio_edit.text().strip()
        tts_text = self.tts_text_edit.text().strip()
        prompt_text = self.prompt_text_edit.text().strip()
        instruct_text = self.instruct_text_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip() or DEFAULT_OUTPUT_DIR
        output_name = self.output_name_edit.text().strip() or "single.wav"
        output_path = str(Path(output_dir) / output_name)

        if not prompt_audio:
            QMessageBox.warning(self, "错误", "请先选择音色音频！")
            return
        if not tts_text:
            QMessageBox.warning(self, "错误", "请输入合成文本！")
            return

        seed = None if self.seed_spinbox.value() == -1 else self.seed_spinbox.value()

        self.start_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在启动合成...")

        self.synth_thread = QThread()
        self.synth_worker = SingleSynthesisWorker(
            cosyvoice=self.cosyvoice_model,
            mode=mode,
            prompt_audio_path=prompt_audio,
            prompt_text=prompt_text,
            instruct_text=instruct_text,
            tts_text=tts_text,
            speed=self.speed_spinbox.value(),
            seed=seed,
            output_path=output_path,
        )

        self.synth_worker.progress_updated.connect(self.progress_bar.setValue)
        self.synth_worker.status_updated.connect(self.status_label.setText)
        self.synth_worker.log_updated.connect(lambda s: None)
        self.synth_worker.finished.connect(self.on_synth_finished)
        self.synth_worker.error.connect(self.on_synth_error)

        self.synth_worker.moveToThread(self.synth_thread)
        self.synth_thread.started.connect(self.synth_worker.run)
        self.synth_thread.start()

    def on_synth_finished(self, output_path: str):
        self.last_output_path = output_path
        self.start_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.status_label.setText("合成完成")
        if self.synth_thread:
            self.synth_thread.quit()
            self.synth_thread.wait()
        QMessageBox.information(self, "完成", f"输出文件: {output_path}")

    def on_synth_error(self, msg: str):
        self.start_btn.setEnabled(True)
        self.play_btn.setEnabled(False)
        if self.synth_thread:
            self.synth_thread.quit()
            self.synth_thread.wait()
        QMessageBox.critical(self, "错误", msg)

    def play_output(self):
        if self.last_output_path and Path(self.last_output_path).exists():
            from PySide6.QtCore import QUrl
            self.media_player.stop()
            self.media_player.setSource(QUrl.fromLocalFile(self.last_output_path))
            self.media_player.play()
        else:
            QMessageBox.warning(self, "错误", "没有可播放的输出文件")

    def on_mode_changed(self, text: str):
        mode_key = MODE_MAPPING.get(text, text)
        # 控制字段可见性
        if mode_key == "zero_shot":
            self.prompt_text_label.setVisible(True)
            self.prompt_text_edit.setVisible(True)
            self.instruct_text_label.setVisible(False)
            self.instruct_text_edit.setVisible(False)
        elif mode_key == "cross_lingual":
            self.prompt_text_label.setVisible(False)
            self.prompt_text_edit.setVisible(False)
            self.instruct_text_label.setVisible(False)
            self.instruct_text_edit.setVisible(False)
        elif mode_key == "instruct":
            self.prompt_text_label.setVisible(False)
            self.prompt_text_edit.setVisible(False)
            self.instruct_text_label.setVisible(True)
            self.instruct_text_edit.setVisible(True)
        else:
            self.prompt_text_label.setVisible(True)
            self.prompt_text_edit.setVisible(True)
            self.instruct_text_label.setVisible(False)
            self.instruct_text_edit.setVisible(False)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SingleSynthesisGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
