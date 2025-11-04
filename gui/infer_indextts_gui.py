import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

os.environ['HF_HUB_CACHE'] = os.path.join(project_root, 'models/IndexTTS-2/hf_cache')

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import onnxruntime as ort
ort.set_default_logger_severity(3)

import torch
import torchaudio

from indextts.infer_v2 import IndexTTS2

DEFAULT_MODEL_DIR = "models/IndexTTS-2"
DEFAULT_CFG_PATH = "models/IndexTTS-2/config.yaml"
DEFAULT_OUTPUT_DIR = "playground/tts_indextts/voice_output"
DEFAULT_PROMPT_WAV = os.path.join(project_root, "asset", "zero_shot_prompt.wav")

EMO_LABELS = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]

# 自动检测设备并设置参数
if torch.cuda.is_available():
    DEVICE = "cuda"
    USE_FP16 = True
    USE_CUDA_KERNEL = True
    print(">> 检测到 CUDA，使用 GPU 加速")
else:
    DEVICE = "cpu"
    USE_FP16 = False
    USE_CUDA_KERNEL = False
    print(">> 未检测到 CUDA，使用 CPU 模式")

# 在导入 PySide6 之前加载模型
print(">> 正在加载 IndexTTS2 模型...")
try:
    INDEXTTS_MODEL = IndexTTS2(
        cfg_path=DEFAULT_CFG_PATH,
        model_dir=DEFAULT_MODEL_DIR,
        use_fp16=USE_FP16,
        device=DEVICE,
        use_cuda_kernel=USE_CUDA_KERNEL,
    )
    print(">> 模型加载完成")
except Exception as e:
    print(f">> 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 模型加载完成后再导入 PySide6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QProgressBar,
    QGroupBox, QMessageBox, QDoubleSpinBox, QSpinBox, QComboBox,
    QFormLayout, QHBoxLayout, QCheckBox
)
from PySide6.QtCore import Qt, QObject, Signal, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput


class SynthesisWorker(QObject):
    """合成工作器（在主线程中执行）"""
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        indextts: IndexTTS2,
        text: str,
        prompt_audio_path: str,
        emo_mode: str,
        emo_audio_path: str,
        emo_alpha: float,
        emo_vector: list,
        emo_text: str,
        interval_silence: int,
        max_tokens: int,
        use_random: bool,
        output_path: str,
    ):
        super().__init__()
        self.indextts = indextts
        self.text = text
        self.prompt_audio_path = prompt_audio_path
        self.emo_mode = emo_mode
        self.emo_audio_path = emo_audio_path
        self.emo_alpha = emo_alpha
        self.emo_vector = emo_vector
        self.emo_text = emo_text
        self.interval_silence = interval_silence
        self.max_tokens = max_tokens
        self.use_random = use_random
        self.output_path = output_path

    def run(self):
        try:
            self.progress.emit(0)
            self.status.emit("准备合成...")

            if not Path(self.prompt_audio_path).exists():
                raise FileNotFoundError(f"音频文件不存在: {self.prompt_audio_path}")

            self.status.emit("开始合成...")
            self.progress.emit(30)

            kwargs = {
                "spk_audio_prompt": self.prompt_audio_path,
                "text": self.text,
                "output_path": self.output_path,
                "use_random": self.use_random,
                "interval_silence": self.interval_silence,
                "max_text_tokens_per_segment": self.max_tokens,
                "verbose": False,
            }

            if self.emo_mode == "情感参考音频":
                kwargs["emo_audio_prompt"] = self.emo_audio_path
                kwargs["emo_alpha"] = self.emo_alpha
            elif self.emo_mode == "情感向量":
                kwargs["emo_vector"] = self.emo_vector
                kwargs["emo_alpha"] = 1.0
            elif self.emo_mode == "情感文本引导":
                kwargs["use_emo_text"] = True
                kwargs["emo_text"] = self.emo_text
                kwargs["emo_alpha"] = self.emo_alpha

            self.progress.emit(60)
            self.indextts.infer(**kwargs)

            self.progress.emit(100)
            self.status.emit("合成完成")
            self.finished.emit(self.output_path)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.status.emit(f"合成失败: {e}")
            self.error.emit(str(e))


class SingleSynthesisGUI(QMainWindow):
    def __init__(self, indextts_model: IndexTTS2):
        super().__init__()
        self.indextts_model = indextts_model
        self.media_player = None
        self.audio_output = None
        self.last_output_path: str | None = None
        self.emo_sliders = []
        self.is_processing = False

        self.init_ui()
        self.init_audio_player()

    def init_audio_player(self):
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

    def init_ui(self):
        self.setWindowTitle("LightTTS IndexTTS2 语音合成")
        self.setGeometry(120, 120, 900, 600)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(10)
        root_layout.setContentsMargins(15, 15, 15, 15)

        model_group = QGroupBox("模型信息")
        model_layout = QGridLayout()
        model_layout.setSpacing(8)
        model_group.setLayout(model_layout)

        model_layout.addWidget(QLabel("模型路径:"), 0, 0)
        model_layout.addWidget(QLabel(DEFAULT_MODEL_DIR), 0, 1)

        model_layout.addWidget(QLabel("运行设备:"), 1, 0)
        model_layout.addWidget(QLabel(DEVICE.upper()), 1, 1)

        model_layout.addWidget(QLabel("模型状态:"), 2, 0)
        self.model_status_label = QLabel("已加载 ✓")
        self.model_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        model_layout.addWidget(self.model_status_label, 2, 1)

        root_layout.addWidget(model_group)

        io_group = QGroupBox("参数输入")
        io_layout = QFormLayout()
        io_layout.setSpacing(12)
        io_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        io_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        io_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        io_layout.setContentsMargins(12, 12, 12, 12)
        io_group.setLayout(io_layout)

        self.tts_text_edit = QLineEdit()
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
        io_layout.addRow("说话人音频:", prompt_audio_row)

        self.emo_mode_combo = QComboBox()
        self.emo_mode_combo.addItems(["无情感控制", "情感参考音频", "情感向量", "情感文本引导"])
        self.emo_mode_combo.setCurrentText("无情感控制")
        self.emo_mode_combo.currentTextChanged.connect(self.on_emo_mode_changed)
        io_layout.addRow("情感控制模式:", self.emo_mode_combo)

        self.emo_audio_label = QLabel("情感音频:")
        self.emo_audio_edit = QLineEdit()
        self.emo_audio_btn = QPushButton("选择")
        self.emo_audio_btn.setFixedWidth(70)
        self.emo_audio_btn.clicked.connect(self.select_emo_audio)
        emo_audio_row = QWidget()
        emo_audio_layout = QHBoxLayout(emo_audio_row)
        emo_audio_layout.setContentsMargins(0, 0, 0, 0)
        emo_audio_layout.setSpacing(8)
        emo_audio_layout.addWidget(self.emo_audio_edit)
        emo_audio_layout.addWidget(self.emo_audio_btn)
        io_layout.addRow(self.emo_audio_label, emo_audio_row)

        self.emo_alpha_label = QLabel("情感权重:")
        self.emo_alpha_spinbox = QDoubleSpinBox()
        self.emo_alpha_spinbox.setRange(0.0, 1.0)
        self.emo_alpha_spinbox.setSingleStep(0.05)
        self.emo_alpha_spinbox.setValue(1.0)
        self.emo_alpha_spinbox.setFixedWidth(90)
        io_layout.addRow(self.emo_alpha_label, self.emo_alpha_spinbox)

        self.emo_vector_group = QGroupBox("情感向量控制")
        emo_vector_layout = QGridLayout()
        emo_vector_layout.setSpacing(8)
        self.emo_vector_group.setLayout(emo_vector_layout)
        
        for i, label in enumerate(EMO_LABELS):
            row = i // 2
            col = (i % 2) * 2
            emo_vector_layout.addWidget(QLabel(f"{label}:"), row, col)
            slider = QDoubleSpinBox()
            slider.setRange(0.0, 1.0)
            slider.setSingleStep(0.05)
            slider.setValue(0.0)
            slider.setFixedWidth(90)
            emo_vector_layout.addWidget(slider, row, col + 1)
            self.emo_sliders.append(slider)
        
        io_layout.addRow(self.emo_vector_group)

        self.emo_text_label = QLabel("情感引导文本:")
        self.emo_text_edit = QLineEdit()
        io_layout.addRow(self.emo_text_label, self.emo_text_edit)

        adv_row_widget = QWidget()
        adv_row_layout = QHBoxLayout(adv_row_widget)
        adv_row_layout.setContentsMargins(0, 0, 0, 0)
        adv_row_layout.setSpacing(12)
        
        adv_row_layout.addWidget(QLabel("句间静音(ms):"))
        self.interval_silence_spinbox = QSpinBox()
        self.interval_silence_spinbox.setRange(0, 1000)
        self.interval_silence_spinbox.setSingleStep(50)
        self.interval_silence_spinbox.setValue(200)
        self.interval_silence_spinbox.setFixedWidth(90)
        adv_row_layout.addWidget(self.interval_silence_spinbox)
        
        adv_row_layout.addSpacing(12)
        adv_row_layout.addWidget(QLabel("最大Token:"))
        self.max_tokens_spinbox = QSpinBox()
        self.max_tokens_spinbox.setRange(50, 200)
        self.max_tokens_spinbox.setSingleStep(10)
        self.max_tokens_spinbox.setValue(120)
        self.max_tokens_spinbox.setFixedWidth(90)
        adv_row_layout.addWidget(self.max_tokens_spinbox)
        
        adv_row_layout.addSpacing(12)
        self.use_random_checkbox = QCheckBox("启用随机性")
        self.use_random_checkbox.setChecked(False)
        adv_row_layout.addWidget(self.use_random_checkbox)
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
        self.on_emo_mode_changed(self.emo_mode_combo.currentText())
        
        # 启用合成按钮（模型已加载）
        self.start_btn.setEnabled(True)

    def select_prompt_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择说话人音频", filter="音频文件 (*.wav *.mp3 *.flac)")
        if file_path:
            self.prompt_audio_edit.setText(file_path)

    def select_emo_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择情感音频", filter="音频文件 (*.wav *.mp3 *.flac)")
        if file_path:
            self.emo_audio_edit.setText(file_path)

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def start_synthesis(self):
        if self.is_processing:
            QMessageBox.warning(self, "提示", "正在处理中，请稍候...")
            return

        emo_mode = self.emo_mode_combo.currentText()
        prompt_audio = self.prompt_audio_edit.text().strip()
        tts_text = self.tts_text_edit.text().strip()
        emo_audio = self.emo_audio_edit.text().strip()
        emo_text = self.emo_text_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip() or DEFAULT_OUTPUT_DIR
        output_name = self.output_name_edit.text().strip() or "single.wav"
        output_path = str(Path(output_dir) / output_name)

        if not prompt_audio:
            QMessageBox.warning(self, "错误", "请先选择说话人音频！")
            return
        if not tts_text:
            QMessageBox.warning(self, "错误", "请输入合成文本！")
            return
        if emo_mode == "情感参考音频" and not emo_audio:
            QMessageBox.warning(self, "错误", "情感参考音频模式需要上传情感音频！")
            return
        if emo_mode == "情感文本引导" and not emo_text:
            QMessageBox.warning(self, "错误", "情感文本引导模式需要输入引导文本！")
            return

        emo_vector = [slider.value() for slider in self.emo_sliders]

        self.is_processing = True
        self.start_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        worker = SynthesisWorker(
            indextts=self.indextts_model,
            text=tts_text,
            prompt_audio_path=prompt_audio,
            emo_mode=emo_mode,
            emo_audio_path=emo_audio,
            emo_alpha=self.emo_alpha_spinbox.value(),
            emo_vector=emo_vector,
            emo_text=emo_text,
            interval_silence=self.interval_silence_spinbox.value(),
            max_tokens=self.max_tokens_spinbox.value(),
            use_random=self.use_random_checkbox.isChecked(),
            output_path=output_path,
        )

        worker.progress.connect(self.progress_bar.setValue)
        worker.status.connect(self.status_label.setText)
        worker.finished.connect(self.on_synth_finished)
        worker.error.connect(self.on_synth_error)

        QApplication.processEvents()
        worker.run()

    def on_synth_finished(self, output_path: str):
        # 清除播放器缓存，确保下次播放时加载新文件
        self.media_player.stop()
        self.media_player.setSource(QUrl())
        
        # 更新输出路径
        self.last_output_path = output_path
        self.start_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.is_processing = False
        QMessageBox.information(self, "完成", f"输出文件: {output_path}")

    def on_synth_error(self, msg: str):
        self.start_btn.setEnabled(True)
        self.play_btn.setEnabled(False)
        self.is_processing = False
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

    def on_emo_mode_changed(self, text: str):
        if text == "无情感控制":
            self.emo_audio_label.setVisible(False)
            self.emo_audio_edit.setVisible(False)
            self.emo_audio_btn.setVisible(False)
            self.emo_alpha_label.setVisible(False)
            self.emo_alpha_spinbox.setVisible(False)
            self.emo_vector_group.setVisible(False)
            self.emo_text_label.setVisible(False)
            self.emo_text_edit.setVisible(False)
        elif text == "情感参考音频":
            self.emo_audio_label.setVisible(True)
            self.emo_audio_edit.setVisible(True)
            self.emo_audio_btn.setVisible(True)
            self.emo_alpha_label.setVisible(True)
            self.emo_alpha_spinbox.setVisible(True)
            self.emo_vector_group.setVisible(False)
            self.emo_text_label.setVisible(False)
            self.emo_text_edit.setVisible(False)
        elif text == "情感向量":
            self.emo_audio_label.setVisible(False)
            self.emo_audio_edit.setVisible(False)
            self.emo_audio_btn.setVisible(False)
            self.emo_alpha_label.setVisible(False)
            self.emo_alpha_spinbox.setVisible(False)
            self.emo_vector_group.setVisible(True)
            self.emo_text_label.setVisible(False)
            self.emo_text_edit.setVisible(False)
        elif text == "情感文本引导":
            self.emo_audio_label.setVisible(False)
            self.emo_audio_edit.setVisible(False)
            self.emo_audio_btn.setVisible(False)
            self.emo_alpha_label.setVisible(True)
            self.emo_alpha_spinbox.setVisible(True)
            self.emo_vector_group.setVisible(False)
            self.emo_text_label.setVisible(True)
            self.emo_text_edit.setVisible(True)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SingleSynthesisGUI(INDEXTTS_MODEL)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
