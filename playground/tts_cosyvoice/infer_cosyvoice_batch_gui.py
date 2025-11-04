import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
matcha_path = os.path.join(project_root, 'Matcha-TTS')
sys.path.insert(0, project_root)
sys.path.insert(0, matcha_path)

playground_dir = Path(project_root) / 'playground'

import torch
import torchaudio
import librosa
import json

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                               QPushButton, QFileDialog, QProgressBar, QTextEdit,
                               QSpinBox, QDoubleSpinBox, QGroupBox, QMessageBox,
                               QSplitter, QComboBox)
from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer, QUrl
from PySide6.QtGui import QFont
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

# 禁用警告
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import onnxruntime as ort
ort.set_default_logger_severity(3)

# ============ 模型配置 ============
DEFAULT_MODEL_DIR = "models/CosyVoice2-0.5B"

# ============ 目录配置 ============
DEFAULT_INPUT_DIR = "playground/tts_cosyvoice/texts"
DEFAULT_OUTPUT_DIR = "playground/tts_cosyvoice/voice_output"
DB_CLONE_DIR_NAME = "DB_clone"
DB_CLONE_JSONL_NAME = "db_clone.jsonl"
VOICE_MANAGER_SCRIPT = "playground/voice_manager_gui.py"

# 根据 project_root 设置 playground_dir
project_root_path = Path(project_root)

# ============ 音频参数 ============
MAX_VAL = 0.8
PROMPT_SAMPLE_RATE = 16000
DEFAULT_OUTPUT_SAMPLE_RATE = 22050
AUDIO_SILENCE_DURATION = 0.2

# ============ 文件扩展名 ============
TEXT_EXTENSIONS = ['.txt']

# ============ 合成参数 ============
DEFAULT_SPEED = 1.0
DEFAULT_SEED = -1

class ModelLoadWorker(QObject):
    """模型加载工作线程"""
    status_updated = Signal(str)
    log_updated = Signal(str)
    finished = Signal(object)
    error = Signal(str)
    
    def __init__(self, model_dir):
        super().__init__()
        self.model_dir = model_dir
    
    def load_model(self):
        """加载模型"""
        try:
            self.status_updated.emit("正在加载模型...")
            self.log_updated.emit(f"正在加载CosyVoice2模型: {self.model_dir}")
            cosyvoice = CosyVoice2(self.model_dir)
            self.log_updated.emit("模型加载完成")
            self.status_updated.emit("模型加载完成")
            self.finished.emit(cosyvoice)
        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            self.log_updated.emit(error_msg)
            self.status_updated.emit(error_msg)
            self.error.emit(error_msg)

class VoiceSynthesisWorker(QObject):
    """音色批量合成工作线程"""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    log_updated = Signal(str)
    finished = Signal(int, int)
    current_file_updated = Signal(str)
    progress_count_updated = Signal(int, int)
    
    def __init__(self, cosyvoice, voice_data, input_dir, output_dir, 
                 speed=DEFAULT_SPEED, seed=None, sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE):
        super().__init__()
        self.cosyvoice = cosyvoice
        self.voice_data = voice_data
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.speed = speed
        self.seed = seed
        self.sample_rate = sample_rate
        self.is_running = True
        self.prompt_speech_16k = None
    
    def stop(self):
        self.is_running = False
    
    def postprocess(self, speech):
        """音频后处理"""
        speech, _ = librosa.effects.trim(
            speech, top_db=60,
            frame_length=440,
            hop_length=220
        )
        if speech.abs().max() > MAX_VAL:
            speech = speech / speech.abs().max() * MAX_VAL
        speech = torch.concat([speech, torch.zeros(1, int(PROMPT_SAMPLE_RATE * AUDIO_SILENCE_DURATION))], dim=1)
        return speech
    
    def read_text_file(self, text_file):
        """读取文本文件内容"""
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except:
            try:
                with open(text_file, 'r', encoding='gbk') as f:
                    return f.read().strip()
            except:
                return ""
    
    def get_input_text_files(self, input_dir):
        """获取输入目录中的所有文本文件"""
        text_files = []
        input_path = Path(input_dir)
        
        for text_ext in TEXT_EXTENSIONS:
            text_files.extend([str(f) for f in input_path.glob(f"*{text_ext}")])
        
        return text_files
    
    def load_voice(self):
        """加载音色"""
        try:
            prompt_audio_path = self.voice_data.get('source', '')
            if not Path(prompt_audio_path).exists():
                self.log_updated.emit(f"错误: 音频文件不存在: {prompt_audio_path}")
                return False
            
            audio_info = torchaudio.info(prompt_audio_path)
            if audio_info.sample_rate < PROMPT_SAMPLE_RATE:
                self.log_updated.emit(f"错误: 音频采样率过低 ({audio_info.sample_rate} Hz)，需要至少 {PROMPT_SAMPLE_RATE} Hz")
                return False
            
            self.status_updated.emit("正在加载音色...")
            self.log_updated.emit(f"正在处理音色: {self.voice_data.get('key', 'Unknown')}")
            self.log_updated.emit(f"音频采样率: {audio_info.sample_rate} Hz")
            self.log_updated.emit(f"音频时长: {audio_info.num_frames / audio_info.sample_rate:.2f} 秒")
            self.log_updated.emit(f"Prompt文本: {self.voice_data.get('target', '')}")
            
            self.prompt_speech_16k = self.postprocess(load_wav(prompt_audio_path, PROMPT_SAMPLE_RATE))
            
            self.log_updated.emit("音色加载完成！")
            return True
            
        except Exception as e:
            self.log_updated.emit(f"音色加载失败: {str(e)}")
            return False
    
    def synthesize_audio(self, tts_text):
        """使用注册的音色合成音频"""
        if self.seed is not None:
            set_all_random_seed(self.seed)
        
        try:
            result = None
            prompt_text = self.voice_data.get('target', '')
            for i in self.cosyvoice.inference_zero_shot(tts_text, prompt_text, self.prompt_speech_16k, stream=False, speed=self.speed):
                result = i['tts_speech'].numpy().flatten()
            return result
        except Exception as e:
            self.log_updated.emit(f"合成失败: {e}")
            return None
    
    def run_synthesis(self):
        """执行批量合成"""
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            
            if not self.load_voice():
                self.status_updated.emit("音色加载失败")
                return
            
            self.status_updated.emit("获取文本文件列表...")
            input_text_files = self.get_input_text_files(self.input_dir)
            if not input_text_files:
                self.log_updated.emit(f"错误: 在 {self.input_dir} 中未找到任何文本文件")
                return
            
            self.log_updated.emit(f"找到 {len(input_text_files)} 个待合成的文本文件")
            
            success_count = 0
            total_count = len(input_text_files)
            
            self.status_updated.emit("开始批量合成...")
            
            for i, input_text_file in enumerate(input_text_files):
                if not self.is_running:
                    break
                
                filename = Path(input_text_file).name
                self.current_file_updated.emit(filename)
                self.progress_count_updated.emit(i + 1, total_count)
                self.status_updated.emit(f"正在合成: {filename}")
                self.progress_updated.emit(int((i / total_count) * 100))
                
                tts_text = self.read_text_file(input_text_file)
                if not tts_text:
                    self.log_updated.emit(f"⚠ 跳过: {filename} (无法读取)")
                    continue
                
                synthesized_audio = self.synthesize_audio(tts_text)
                
                if synthesized_audio is not None:
                    input_basename = Path(input_text_file).stem
                    output_audio_path = str(Path(self.output_dir) / f"{input_basename}.wav")
                    
                    torchaudio.save(
                        output_audio_path,
                        torch.from_numpy(synthesized_audio).unsqueeze(0),
                        self.sample_rate
                    )
                    
                    self.log_updated.emit(f"✓ {filename}")
                    success_count += 1
                else:
                    self.log_updated.emit(f"✗ {filename} (合成失败)")
                    success_count += 1
            
            self.progress_updated.emit(100)
            self.status_updated.emit("批量合成完成!")
            self.log_updated.emit(f"\n=== 批量合成完成 ===")
            self.log_updated.emit(f"成功: {success_count}/{total_count}")
            self.log_updated.emit(f"输出目录: {self.output_dir}")
            
            self.finished.emit(success_count, total_count)
            
        except Exception as e:
            self.log_updated.emit(f"发生错误: {str(e)}")
            self.status_updated.emit(f"发生错误: {str(e)}")

class VoiceBatchSynthesisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker_thread = None
        self.worker = None
        self.model_load_thread = None
        self.model_load_worker = None
        self.cosyvoice_model = None
        self.media_player = None
        self.audio_output = None
        self.init_ui()
        self.init_audio_player()
        
    def init_audio_player(self):
        """初始化音频播放器"""
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.errorOccurred.connect(self.on_player_error)
        
    def init_ui(self):
        self.setWindowTitle("LightTTS 音色选择与批量语音合成系统")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal) 
        main_layout.addWidget(splitter)
        
        main_panel = self.create_main_panel()
        splitter.addWidget(main_panel)
        
        log_panel = self.create_log_panel()
        splitter.addWidget(log_panel)
        
        splitter.setSizes([900, 300])
        
        QTimer.singleShot(100, self.refresh_voice_combo)
    
    def create_main_panel(self):
        """创建主面板"""
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        
        title_label = QLabel("音色选择与批量语音合成")
        title_label.setFont(QFont("Arial", 16, QFont.Bold)) 
        title_label.setAlignment(Qt.AlignCenter)   
        layout.addWidget(title_label)
        
        model_group = QGroupBox("模型设置")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("模型路径:"), 0, 0)
        self.model_dir_edit = QLineEdit(DEFAULT_MODEL_DIR)
        model_layout.addWidget(self.model_dir_edit, 0, 1)
        model_dir_btn = QPushButton("浏览")
        model_dir_btn.clicked.connect(self.select_model_dir)
        model_layout.addWidget(model_dir_btn, 0, 2)
        
        layout.addWidget(model_group)
        
        voice_select_group = QGroupBox("音色选择")
        voice_select_layout = QGridLayout(voice_select_group)
        
        voice_select_layout.addWidget(QLabel("选择音色:"), 0, 0)
        self.voice_combo = QComboBox()
        voice_select_layout.addWidget(self.voice_combo, 0, 1)
        
        refresh_combo_btn = QPushButton("🔄 刷新")
        refresh_combo_btn.clicked.connect(self.refresh_voice_combo)
        voice_select_layout.addWidget(refresh_combo_btn, 0, 2)
        
        self.play_voice_btn = QPushButton("🔊 播放音色")
        self.play_voice_btn.clicked.connect(self.play_selected_voice)
        self.play_voice_btn.setEnabled(False)
        voice_select_layout.addWidget(self.play_voice_btn, 0, 3)
        
        voice_select_layout.addWidget(QLabel("音色信息:"), 1, 0)
        self.combo_voice_info_label = QLabel("请选择音色")
        self.combo_voice_info_label.setWordWrap(True)
        self.combo_voice_info_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        voice_select_layout.addWidget(self.combo_voice_info_label, 1, 1, 1, 3)
        
        layout.addWidget(voice_select_group)
        
        batch_group = QGroupBox("批量合成设置")
        batch_layout = QGridLayout(batch_group)
        
        batch_layout.addWidget(QLabel("输入文本文件夹:"), 0, 0)
        self.input_dir_edit = QLineEdit(DEFAULT_INPUT_DIR)
        batch_layout.addWidget(self.input_dir_edit, 0, 1)
        input_dir_btn = QPushButton("浏览")
        input_dir_btn.clicked.connect(self.select_input_dir)
        batch_layout.addWidget(input_dir_btn, 0, 2)
        
        batch_layout.addWidget(QLabel("输出文件夹:"), 1, 0)
        self.output_dir_edit = QLineEdit(DEFAULT_OUTPUT_DIR)
        batch_layout.addWidget(self.output_dir_edit, 1, 1)
        output_dir_btn = QPushButton("浏览")
        output_dir_btn.clicked.connect(self.select_output_dir)
        batch_layout.addWidget(output_dir_btn, 1, 2)
        
        layout.addWidget(batch_group)
        
        params_group = QGroupBox("参数设置")
        params_layout = QGridLayout(params_group)
        
        params_layout.addWidget(QLabel("语音速度:"), 0, 0)
        self.speed_spinbox = QDoubleSpinBox()
        self.speed_spinbox.setRange(0.5, 2.0)
        self.speed_spinbox.setSingleStep(0.1)
        self.speed_spinbox.setValue(DEFAULT_SPEED)
        params_layout.addWidget(self.speed_spinbox, 0, 1)
        
        params_layout.addWidget(QLabel("随机种子:"), 1, 0)
        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(DEFAULT_SEED, 999999999)
        self.seed_spinbox.setValue(DEFAULT_SEED)
        self.seed_spinbox.setSpecialValueText("随机")
        params_layout.addWidget(self.seed_spinbox, 1, 1)
        
        params_layout.addWidget(QLabel("采样率:"), 2, 0)
        self.sample_rate_spinbox = QSpinBox()
        self.sample_rate_spinbox.setRange(16000, 48000)
        self.sample_rate_spinbox.setSingleStep(1000)
        self.sample_rate_spinbox.setValue(DEFAULT_OUTPUT_SAMPLE_RATE)
        params_layout.addWidget(self.sample_rate_spinbox, 2, 1)
        
        layout.addWidget(params_group)
        
        model_status_panel = self.create_model_status_panel()
        layout.addWidget(model_status_panel)
        
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("🚀 开始批量合成")
        self.start_btn.clicked.connect(self.start_synthesis)
        self.start_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹️ 停止合成")
        self.stop_btn.clicked.connect(self.stop_synthesis)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        layout.addLayout(btn_layout)
        
        progress_group = QGroupBox("合成进度")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                font-size: 14px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #28a745;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        self.current_file_label = QLabel("当前文件: -")
        self.current_file_label.setFont(QFont("Arial", 10))
        progress_layout.addWidget(self.current_file_label)
        
        self.progress_count_label = QLabel("进度: 0/0")
        self.progress_count_label.setFont(QFont("Arial", 10))
        progress_layout.addWidget(self.progress_count_label)
        
        self.status_label = QLabel("就绪")
        self.status_label.setFont(QFont("Arial", 10))
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        layout.addStretch()
        return main_widget
    
    def create_model_status_panel(self):
        """创建模型状态面板"""
        model_widget = QGroupBox("模型状态")
        layout = QHBoxLayout(model_widget)
        layout.setSpacing(15)
        
        self.model_status_label = QLabel("模型状态: 未加载")
        self.model_status_label.setFont(QFont("Arial", 11))
        self.model_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
        layout.addWidget(self.model_status_label)
        
        layout.addStretch()
        
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_model_btn.setFixedSize(120, 35)
        self.load_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        layout.addWidget(self.load_model_btn)
        
        return model_widget
    
    def create_log_panel(self):
        """创建日志面板"""
        log_widget = QWidget()
        layout = QVBoxLayout(log_widget)
        
        log_header = QHBoxLayout()
        log_label = QLabel("运行日志")
        log_label.setFont(QFont("Arial", 11, QFont.Bold)) 
        log_header.addWidget(log_label)
        log_header.addStretch()
        layout.addLayout(log_header)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)
        
        btn_layout = QHBoxLayout()
        
        clear_btn = QPushButton("清除日志")
        clear_btn.clicked.connect(self.clear_log)
        btn_layout.addWidget(clear_btn)
        
        view_db_btn = QPushButton("查看音色数据库")
        view_db_btn.clicked.connect(self.view_voice_database)
        btn_layout.addWidget(view_db_btn)
        
        manage_btn = QPushButton("打开音色管理器")
        manage_btn.clicked.connect(self.open_voice_manager)
        btn_layout.addWidget(manage_btn)
        
        layout.addLayout(btn_layout)
        
        return log_widget
    
    def select_model_dir(self):
        """选择模型目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if dir_path:
            self.model_dir_edit.setText(dir_path)
    
    def select_input_dir(self):
        """选择输入文本文件夹"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入文本文件夹")
        if dir_path:
            self.input_dir_edit.setText(dir_path)
    
    def select_output_dir(self):
        """选择输出文件夹"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def load_model(self):
        """加载模型"""
        if not Path(self.model_dir_edit.text()).exists():
            QMessageBox.warning(self, "错误", "模型路径不存在！")
            return
        
        self.load_model_btn.setEnabled(False)
        self.model_status_label.setText("模型状态: 加载中...")
        self.model_status_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        
        self.model_load_thread = QThread()
        self.model_load_worker = ModelLoadWorker(self.model_dir_edit.text())
        
        self.model_load_worker.status_updated.connect(self.status_label.setText)
        self.model_load_worker.log_updated.connect(self.log_text.append)
        self.model_load_worker.finished.connect(self.on_model_loaded)
        self.model_load_worker.error.connect(self.on_model_load_error)
        
        self.model_load_worker.moveToThread(self.model_load_thread)
        self.model_load_thread.started.connect(self.model_load_worker.load_model)
        self.model_load_thread.start()
    
    def on_model_loaded(self, model):
        """模型加载完成"""
        self.cosyvoice_model = model
        self.model_status_label.setText("模型状态: 已加载")
        self.model_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        self.start_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        
        if self.model_load_thread:
            self.model_load_thread.quit()
            self.model_load_thread.wait()
    
    def on_model_load_error(self, error_msg):
        """模型加载失败"""
        self.model_status_label.setText("模型状态: 加载失败")
        self.model_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
        self.load_model_btn.setEnabled(True)
        QMessageBox.critical(self, "错误", error_msg)
        
        if self.model_load_thread:
            self.model_load_thread.quit()
            self.model_load_thread.wait()
    
    def clear_log(self):
        """清除日志"""
        self.log_text.clear()
    
    def view_voice_database(self):
        """查看音色数据库"""
        db_clone_dir = project_root_path / "playground" / DB_CLONE_DIR_NAME
        db_clone_jsonl = db_clone_dir / DB_CLONE_JSONL_NAME
        
        self.log_text.append("=== 音色数据库内容 ===")
        
        if not db_clone_jsonl.exists():
            self.log_text.append("数据库文件不存在，还没有注册任何音色")
            return
        
        try:
            count = 0
            with open(db_clone_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        count += 1
                        self.log_text.append(f"{count}. 键名: {entry.get('key', 'N/A')}")
                        self.log_text.append(f"   音频: {entry.get('source', 'N/A')}")
                        self.log_text.append(f"   文本: {entry.get('target', 'N/A')}")
                        self.log_text.append(f"   时长: {entry.get('source_len', 0)}ms")
                        self.log_text.append(f"   创建时间: {entry.get('created_time', 'N/A')}")
                        self.log_text.append("")
            
            if count == 0:
                self.log_text.append("数据库为空")
            else:
                self.log_text.append(f"共找到 {count} 个注册的音色")
                
        except Exception as e:
            self.log_text.append(f"读取数据库失败: {str(e)}")
        
        self.log_text.append("=== 数据库查看完毕 ===\n")
    
    def open_voice_manager(self):
        """打开音色管理器"""
        try:
            import subprocess
            
            script_path = project_root_path / VOICE_MANAGER_SCRIPT
            
            if not script_path.exists():
                self.log_text.append(f"错误: 脚本不存在: {script_path}")
                return
            
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(project_root_path)
            )
            self.log_text.append(f"已启动音色管理器 (PID: {process.pid})")
            
        except Exception as e:
            self.log_text.append(f"启动失败: {e}")
            self.log_text.append(f"请手动运行: python {VOICE_MANAGER_SCRIPT}")
    
    def start_synthesis(self):
        """开始批量合成"""
        if self.cosyvoice_model is None:
            QMessageBox.warning(self, "错误", "请先加载模型！")
            return
        
        selected_voice = self.get_selected_voice_for_synthesis()
        if not selected_voice:
            QMessageBox.warning(self, "错误", "请先选择一个音色！")
            return
        
        if not Path(self.input_dir_edit.text()).exists():
            QMessageBox.warning(self, "错误", "输入文本文件夹不存在！")
            return
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.progress_bar.setValue(0)
        
        seed = self.seed_spinbox.value() if self.seed_spinbox.value() != -1 else None
        
        self.worker_thread = QThread()
        self.worker = VoiceSynthesisWorker(
            cosyvoice=self.cosyvoice_model,
            voice_data=selected_voice,
            input_dir=self.input_dir_edit.text(),
            output_dir=self.output_dir_edit.text(),
            speed=self.speed_spinbox.value(),
            seed=seed,
            sample_rate=self.sample_rate_spinbox.value()
        )
        
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.log_updated.connect(self.log_text.append)
        self.worker.finished.connect(self.synthesis_finished)
        self.worker.current_file_updated.connect(self.update_current_file)
        self.worker.progress_count_updated.connect(self.update_progress_count)
        
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run_synthesis)
        self.worker_thread.start()
    
    def get_selected_voice_for_synthesis(self):
        """获取批量合成选择的音色"""
        current_index = self.voice_combo.currentIndex()
        if current_index > 0:
            return self.voice_combo.itemData(current_index)
        return None
    
    def stop_synthesis(self):
        """停止合成"""
        if self.worker:
            self.worker.stop()
        self.synthesis_finished(0, 0)
        self.log_text.append("用户手动停止合成")
    
    def update_current_file(self, filename):
        """更新当前文件显示"""
        self.current_file_label.setText(f"当前文件: {filename}")
    
    def update_progress_count(self, current, total):
        """更新进度计数"""
        self.progress_count_label.setText(f"进度: {current}/{total}")
    
    def synthesis_finished(self, success_count, total_count):
        """合成完成"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.current_file_label.setText("当前文件: -")
        
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        
        if success_count > 0 or total_count > 0:
            QMessageBox.information(self, "完成", 
                                  f"批量合成完成！\n成功: {success_count}/{total_count}")
    
    def refresh_voice_combo(self):
        """刷新音色下拉框"""
        current_text = self.voice_combo.currentText()
        
        try:
            self.voice_combo.currentTextChanged.disconnect()
        except:
            pass
        
        self.voice_combo.clear()
        
        db_clone_dir = project_root_path / "playground" / DB_CLONE_DIR_NAME
        db_clone_jsonl = db_clone_dir / DB_CLONE_JSONL_NAME
        
        if not db_clone_jsonl.exists():
            self.voice_combo.addItem("请选择音色...")
            self.combo_voice_info_label.setText("数据库为空，请先注册音色")
            self.voice_combo.currentTextChanged.connect(self.on_voice_combo_changed)
            return
        
        try:
            voices = []
            with open(db_clone_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        voices.append(entry)
            
            self.voice_combo.addItem("请选择音色...")
            for voice in voices:
                voice_key = voice.get('key', 'Unknown')
                self.voice_combo.addItem(voice_key)
                index = self.voice_combo.count() - 1
                self.voice_combo.setItemData(index, voice)
            
            if current_text:
                index = self.voice_combo.findText(current_text)
                if index >= 0:
                    self.voice_combo.setCurrentIndex(index)
            
            self.log_text.append(f"刷新音色列表完成，共 {len(voices)} 个音色")
            
        except Exception as e:
            self.log_text.append(f"刷新音色下拉框失败: {str(e)}")
        
        self.voice_combo.currentTextChanged.connect(self.on_voice_combo_changed)
    
    def on_voice_combo_changed(self, text):
        """音色下拉框选择改变"""
        if text == "请选择音色..." or not text:
            self.combo_voice_info_label.setText("请选择音色")
            self.play_voice_btn.setEnabled(False)
            return
        
        current_index = self.voice_combo.currentIndex()
        if current_index > 0:
            voice_data = self.voice_combo.itemData(current_index)
            if voice_data:
                info_text = f"音色: {voice_data.get('key', 'N/A')}\n"
                info_text += f"文本: {voice_data.get('target', 'N/A')}\n"
                info_text += f"时长: {voice_data.get('source_len', 0)}ms"
                self.combo_voice_info_label.setText(info_text)
                self.play_voice_btn.setEnabled(True)
    
    def play_selected_voice(self):
        """播放选择的音色"""
        selected_voice = self.get_selected_voice_for_synthesis()
        if selected_voice:
            source_path = selected_voice.get('source', '')
            if Path(source_path).exists():
                self.media_player.stop()
                self.media_player.setSource(QUrl.fromLocalFile(source_path)) 
                self.media_player.play() 
                self.log_text.append(f"播放: {selected_voice.get('key', '')}")
            else:
                QMessageBox.warning(self, "错误", "音频文件不存在！")
    
    def on_player_error(self, error):
        """处理播放器错误"""
        self.log_text.append(f"播放器错误: {self.media_player.errorString()}")
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.media_player:
            self.media_player.stop()
        
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(3000)
        
        if self.model_load_thread and self.model_load_thread.isRunning():
            self.model_load_thread.quit()
            self.model_load_thread.wait(3000)
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    app.setStyle('Fusion')
    
    window = VoiceBatchSynthesisGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
