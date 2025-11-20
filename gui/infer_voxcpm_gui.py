import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import soundfile as sf

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QProgressBar,
    QGroupBox, QMessageBox, QDoubleSpinBox, QSpinBox, QComboBox,
    QFormLayout, QHBoxLayout, QCheckBox, QPlainTextEdit
)
from PySide6.QtCore import Qt, QThread, QObject, Signal, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import VoxCPMModel

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

try:
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True
except Exception:
    pass


def _disable_optimize(self: VoxCPMModel):
    """禁用在某些 CUDA 设置下失败的 torch.compile 优化。"""
    self.base_lm.forward_step = self.base_lm.forward_step
    self.residual_lm.forward_step = self.residual_lm.forward_step
    self.feat_encoder_step = self.feat_encoder
    self.feat_decoder.estimator = self.feat_decoder.estimator
    return self


VoxCPMModel.optimize = _disable_optimize

DEFAULT_MODEL_DIR = "models/VoxCPM-0.5B"
DEFAULT_ZIPENHANCER_MODEL_ID = "models/speech_zipenhancer_ans_multiloss_16k_base"
DEFAULT_OUTPUT_DIR = "playground/tts_voxcpm/voice_output"
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做得比我还好哟。"


class ModelLoadWorker(QObject):
    """模型加载线程"""
    status_updated = Signal(str)
    log_updated = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, model_dir: str, zipenhancer_model_id: str, device: str):
        super().__init__()
        self.model_dir = model_dir
        self.zipenhancer_model_id = zipenhancer_model_id
        self.device = device

    def load_model(self):
        try:
            self.status_updated.emit("正在加载模型...")
            self.log_updated.emit(f"加载 VoxCPM 模型: {self.model_dir}")
            voxcpm = VoxCPM.from_pretrained(
                hf_model_id=self.model_dir,
                load_denoiser=False,
                zipenhancer_model_id=self.zipenhancer_model_id,
                local_files_only=True,
                device=self.device,
            )
            self.status_updated.emit("模型加载完成")
            self.log_updated.emit("模型加载完成")
            self.finished.emit(voxcpm)
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
        voxcpm: VoxCPM,
        text: str,
        prompt_audio_path: str,
        prompt_text: str,
        cfg_value: float,
        inference_timesteps: int,
        normalize: bool,
        denoise: bool,
        retry_badcase: bool,
        retry_max_times: int,
        retry_ratio_threshold: float,
        output_path: str,
    ):
        super().__init__()
        self.voxcpm = voxcpm
        self.text = text
        self.prompt_audio_path = prompt_audio_path
        self.prompt_text = prompt_text
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.normalize = normalize
        self.denoise = denoise
        self.retry_badcase = retry_badcase
        self.retry_max_times = retry_max_times
        self.retry_ratio_threshold = retry_ratio_threshold
        self.output_path = output_path

    def run(self):
        try:
            self.progress_updated.emit(0)
            self.status_updated.emit("准备合成...")
            self.log_updated.emit(f"待合成文本: {self.text}")

            if self.prompt_audio_path and not Path(self.prompt_audio_path).exists():
                raise FileNotFoundError(f"音频文件不存在: {self.prompt_audio_path}")

            self.status_updated.emit("开始合成...")
            self.progress_updated.emit(30)

            wav = self.voxcpm.generate(
                text=self.text,
                prompt_wav_path=self.prompt_audio_path or None,
                prompt_text=self.prompt_text or None,
                cfg_value=self.cfg_value,
                inference_timesteps=self.inference_timesteps,
                normalize=self.normalize,
                denoise=self.denoise,
                retry_badcase=self.retry_badcase,
                retry_badcase_max_times=self.retry_max_times,
                retry_badcase_ratio_threshold=self.retry_ratio_threshold,
            )

            Path(os.path.dirname(self.output_path)).mkdir(parents=True, exist_ok=True)
            self.progress_updated.emit(80)
            
            # 保存音频
            sf.write(self.output_path, wav, 16000)

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
        self.voxcpm_model: VoxCPM | None = None
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
        self.setWindowTitle("LightTTS VoxCPM 语音合成")
        self.setGeometry(120, 120, 900, 600)

        # 加载样式表
        style_path = os.path.join(os.path.dirname(__file__), 'style', 'style.qss')
        if os.path.exists(style_path):
            with open(style_path, 'r', encoding='utf-8') as f:
                self.setStyleSheet(f.read())

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(15)
        root_layout.setContentsMargins(20, 20, 20, 20)

        # 模型设置
        model_group = QGroupBox("模型设置")
        model_layout = QGridLayout()
        model_layout.setSpacing(10)
        model_layout.setContentsMargins(15, 20, 15, 15)
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
        self.model_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
        model_layout.addWidget(self.model_status_label, 2, 0, 1, 3)

        root_layout.addWidget(model_group)

        # 参数输入
        io_group = QGroupBox("参数输入")
        io_layout = QFormLayout()
        io_layout.setSpacing(12)
        io_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        io_layout.setContentsMargins(15, 20, 15, 15)
        io_group.setLayout(io_layout)

        self.tts_text_edit = QPlainTextEdit()
        self.tts_text_edit.setPlaceholderText("请输入要合成的文本...")
        self.tts_text_edit.setMinimumHeight(80)
        io_layout.addRow("合成文本:", self.tts_text_edit)

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
        io_layout.addRow("参考音频:", prompt_audio_row)

        self.prompt_text_edit = QLineEdit()
        self.prompt_text_edit.setText(DEFAULT_PROMPT_TEXT)
        io_layout.addRow("参考文本:", self.prompt_text_edit)

        # 推理参数
        params_row_widget = QWidget()
        params_row_layout = QHBoxLayout(params_row_widget)
        params_row_layout.setContentsMargins(0, 0, 0, 0)
        params_row_layout.setSpacing(12)

        params_row_layout.addWidget(QLabel("CFG值:"))
        self.cfg_spinbox = QDoubleSpinBox()
        self.cfg_spinbox.setRange(1.0, 3.0)
        self.cfg_spinbox.setSingleStep(0.1)
        self.cfg_spinbox.setValue(2.0)
        self.cfg_spinbox.setFixedWidth(90)
        params_row_layout.addWidget(self.cfg_spinbox)

        params_row_layout.addSpacing(12)
        params_row_layout.addWidget(QLabel("推理步数:"))
        self.timesteps_spinbox = QSpinBox()
        self.timesteps_spinbox.setRange(4, 30)
        self.timesteps_spinbox.setSingleStep(1)
        self.timesteps_spinbox.setValue(10)
        self.timesteps_spinbox.setFixedWidth(90)
        params_row_layout.addWidget(self.timesteps_spinbox)
        params_row_layout.addStretch()

        io_layout.addRow("推理参数:", params_row_widget)

        # 选项
        options_row_widget = QWidget()
        options_row_layout = QHBoxLayout(options_row_widget)
        options_row_layout.setContentsMargins(0, 0, 0, 0)
        options_row_layout.setSpacing(12)

        self.normalize_checkbox = QCheckBox("文本标准化")
        self.normalize_checkbox.setChecked(True)
        options_row_layout.addWidget(self.normalize_checkbox)

        self.denoise_checkbox = QCheckBox("启用降噪")
        self.denoise_checkbox.setChecked(True)
        options_row_layout.addWidget(self.denoise_checkbox)

        self.retry_checkbox = QCheckBox("重试糟糕情况")
        self.retry_checkbox.setChecked(True)
        options_row_layout.addWidget(self.retry_checkbox)
        options_row_layout.addStretch()

        io_layout.addRow("选项:", options_row_widget)

        # 高级选项
        adv_row_widget = QWidget()
        adv_row_layout = QHBoxLayout(adv_row_widget)
        adv_row_layout.setContentsMargins(0, 0, 0, 0)
        adv_row_layout.setSpacing(12)

        adv_row_layout.addWidget(QLabel("最大重试:"))
        self.retry_max_spinbox = QSpinBox()
        self.retry_max_spinbox.setRange(1, 10)
        self.retry_max_spinbox.setValue(3)
        self.retry_max_spinbox.setFixedWidth(90)
        adv_row_layout.addWidget(self.retry_max_spinbox)

        adv_row_layout.addSpacing(12)
        adv_row_layout.addWidget(QLabel("检测阈值:"))
        self.retry_threshold_spinbox = QDoubleSpinBox()
        self.retry_threshold_spinbox.setRange(3.0, 10.0)
        self.retry_threshold_spinbox.setSingleStep(0.5)
        self.retry_threshold_spinbox.setValue(6.0)
        self.retry_threshold_spinbox.setFixedWidth(90)
        adv_row_layout.addWidget(self.retry_threshold_spinbox)
        adv_row_layout.addStretch()

        io_layout.addRow("高级选项:", adv_row_widget)

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
        ctrl_layout.setSpacing(15)
        ctrl_layout.setContentsMargins(15, 20, 15, 15)
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

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(15)

        self.start_btn = QPushButton("开始合成")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.setFixedHeight(45)
        self.start_btn.clicked.connect(self.start_synthesis)
        self.start_btn.setEnabled(False)
        button_row.addWidget(self.start_btn)

        self.play_btn = QPushButton("播放输出")
        self.play_btn.setFixedHeight(45)
        self.play_btn.clicked.connect(self.play_output)
        self.play_btn.setEnabled(False)
        button_row.addWidget(self.play_btn)
        button_row.addStretch()

        ctrl_layout.addLayout(button_row)

        root_layout.addWidget(ctrl_group)

    def select_model_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if dir_path:
            self.model_dir_edit.setText(dir_path)

    def select_prompt_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择参考音频", filter="音频文件 (*.wav *.mp3 *.flac)")
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
        self.model_load_worker = ModelLoadWorker(
            self.model_dir_edit.text(),
            DEFAULT_ZIPENHANCER_MODEL_ID,
            self.device_combo.currentText()
        )

        self.model_load_worker.status_updated.connect(self.status_label.setText)
        self.model_load_worker.log_updated.connect(lambda s: None)
        self.model_load_worker.finished.connect(self.on_model_loaded)
        self.model_load_worker.error.connect(self.on_model_load_error)

        self.model_load_worker.moveToThread(self.model_load_thread)
        self.model_load_thread.started.connect(self.model_load_worker.load_model)
        self.model_load_thread.start()

    def on_model_loaded(self, model: VoxCPM):
        self.voxcpm_model = model
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
        if self.voxcpm_model is None:
            QMessageBox.warning(self, "错误", "请先加载模型！")
            return

        tts_text = self.tts_text_edit.toPlainText().strip()
        prompt_audio = self.prompt_audio_edit.text().strip()
        prompt_text = self.prompt_text_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip() or DEFAULT_OUTPUT_DIR
        output_name = self.output_name_edit.text().strip() or "single.wav"
        output_path = str(Path(output_dir) / output_name)

        if not tts_text:
            QMessageBox.warning(self, "错误", "请输入合成文本！")
            return

        if prompt_audio and not prompt_text:
            QMessageBox.warning(self, "错误", "使用参考音频时，请提供对应的参考文本！")
            return

        self.start_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在启动合成...")

        self.synth_thread = QThread()
        self.synth_worker = SingleSynthesisWorker(
            voxcpm=self.voxcpm_model,
            text=tts_text,
            prompt_audio_path=prompt_audio,
            prompt_text=prompt_text,
            cfg_value=self.cfg_spinbox.value(),
            inference_timesteps=self.timesteps_spinbox.value(),
            normalize=self.normalize_checkbox.isChecked(),
            denoise=self.denoise_checkbox.isChecked(),
            retry_badcase=self.retry_checkbox.isChecked(),
            retry_max_times=self.retry_max_spinbox.value(),
            retry_ratio_threshold=self.retry_threshold_spinbox.value(),
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
        # 清除播放器缓存，确保下次播放时加载新文件
        self.media_player.stop()
        self.media_player.setSource(QUrl())

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
            # 停止当前播放
            self.media_player.stop()
            # 清除旧的音频源，强制重新加载文件
            self.media_player.setSource(QUrl())
            # 等待清除完成
            QApplication.processEvents()
            # 设置新的音频源
            self.media_player.setSource(QUrl.fromLocalFile(self.last_output_path))
            # 开始播放
            self.media_player.play()
        else:
            QMessageBox.warning(self, "错误", "没有可播放的输出文件")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SingleSynthesisGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
