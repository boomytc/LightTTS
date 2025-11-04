import sys
import os

current_script_absolute_path = os.path.abspath(__file__)
batch_generate_dir = os.path.dirname(current_script_absolute_path)
project_root = os.path.dirname(batch_generate_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

matcha_tts_path = os.path.join(project_root, 'Matcha-TTS')
if os.path.isdir(matcha_tts_path) and matcha_tts_path not in sys.path:
    sys.path.insert(1, matcha_tts_path)

import random
import glob
import torch
import torchaudio
import librosa

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                               QPushButton, QFileDialog, QProgressBar, QTextEdit,
                               QSpinBox, QDoubleSpinBox, QGroupBox, QMessageBox,
                               QComboBox)
from PySide6.QtCore import Qt, QThread, QObject, Signal
from PySide6.QtGui import QFont

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

# 禁用警告
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 全局常量配置
MAX_VAL = 0.8
DEFAULT_MODEL_DIR = "models/CosyVoice2-0.5B"
DEFAULT_CLONE_SRC_DIR = "BatchGenerate/clone_src_wav"
DEFAULT_INPUT_DIR = "BatchGenerate/texts"
DEFAULT_OUTPUT_DIR = "BatchGenerate/output"

# 音频处理参数
PROMPT_SAMPLE_RATE = 16000
DEFAULT_OUTPUT_SAMPLE_RATE = 22050
MIN_SAMPLE_RATE = 16000
MAX_SAMPLE_RATE = 48000
SAMPLE_RATE_STEP = 1000

# 音频后处理参数
AUDIO_TRIM_TOP_DB = 60
AUDIO_HOP_LENGTH = 220
AUDIO_WIN_LENGTH = 440
AUDIO_SILENCE_DURATION = 0.2

# 支持的文件扩展名
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
TEXT_EXTENSIONS = ['.txt']

# 默认合成参数
DEFAULT_SPEED = 1.0
DEFAULT_SEED = -1
MIN_SPEED = 0.5
MAX_SPEED = 2.0
SPEED_STEP = 0.1

class SynthesisWorker(QObject):
    """语音合成工作线程"""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    log_updated = Signal(str)
    stats_updated = Signal(int, int, int)  # total, completed, failed
    finished = Signal(int, int)  # success_count, total_count
    
    def __init__(self, model_dir, clone_src_dir, input_dir, output_dir, 
                 speed=DEFAULT_SPEED, seed=None, sample_rate=DEFAULT_OUTPUT_SAMPLE_RATE):
        super().__init__()
        self.model_dir = model_dir
        self.clone_src_dir = clone_src_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.speed = speed
        self.seed = seed
        self.sample_rate = sample_rate
        self.is_running = True
    
    def stop(self):
        self.is_running = False
    
    def postprocess(self, speech, top_db=AUDIO_TRIM_TOP_DB, hop_length=AUDIO_HOP_LENGTH, win_length=AUDIO_WIN_LENGTH):
        """音频后处理"""
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > MAX_VAL:
            speech = speech / speech.abs().max() * MAX_VAL 
        speech = torch.concat([speech, torch.zeros(1, int(PROMPT_SAMPLE_RATE * AUDIO_SILENCE_DURATION))], dim=1) 
        return speech
    
    def get_prompt_pairs(self, clone_src_dir):
        """获取clone_src_dir中的音频文本对"""
        pairs = []
        
        for audio_ext in AUDIO_EXTENSIONS:
            audio_files = glob.glob(os.path.join(clone_src_dir, f"*{audio_ext}"))
            for audio_file in audio_files:
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                
                for text_ext in TEXT_EXTENSIONS:
                    text_file = os.path.join(clone_src_dir, f"{base_name}{text_ext}")
                    if os.path.exists(text_file):
                        pairs.append((audio_file, text_file))
                        break
        
        return pairs
    
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
        
        for text_ext in TEXT_EXTENSIONS:
            text_files.extend(glob.glob(os.path.join(input_dir, f"*{text_ext}")))
        
        return text_files
    
    def synthesize_audio(self, cosyvoice, tts_text, prompt_text, prompt_wav_path, prompt_sr=PROMPT_SAMPLE_RATE):
        """语音合成函数"""
        if self.seed is not None:
            set_all_random_seed(self.seed)
        
        # 检查音频采样率
        if torchaudio.info(prompt_wav_path).sample_rate < prompt_sr:
            self.log_updated.emit(f"警告: 音频 {prompt_wav_path} 采样率过低，跳过")
            return None
        
        # 加载并处理prompt音频
        prompt_speech_16k = self.postprocess(load_wav(prompt_wav_path, prompt_sr))
        
        # 进行零样本语音克隆
        try:
            result = None
            for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, 
                                                   stream=False, speed=self.speed):
                result = i['tts_speech'].numpy().flatten()
            return result
        except Exception as e:
            self.log_updated.emit(f"合成失败: {e}")
            return None
    
    def run_synthesis(self):
        """执行批量合成"""
        try:
            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 初始化CosyVoice2模型
            self.status_updated.emit("正在加载模型...")
            self.log_updated.emit(f"正在加载CosyVoice2模型: {self.model_dir}")
            cosyvoice = CosyVoice2(self.model_dir)
            self.log_updated.emit("模型加载完成")
            
            # 获取prompt音频文本对
            self.status_updated.emit("获取prompt音频文本对...")
            prompt_pairs = self.get_prompt_pairs(self.clone_src_dir)
            if not prompt_pairs:
                self.log_updated.emit(f"错误: 在 {self.clone_src_dir} 中未找到任何音频文本对")
                return
            
            self.log_updated.emit(f"找到 {len(prompt_pairs)} 个prompt音频文本对")
            
            # 获取输入文本文件
            input_text_files = self.get_input_text_files(self.input_dir)
            if not input_text_files:
                self.log_updated.emit(f"错误: 在 {self.input_dir} 中未找到任何文本文件")
                return
            
            self.log_updated.emit(f"找到 {len(input_text_files)} 个待合成的文本文件")
            
            # 批量合成
            success_count = 0
            failed_count = 0
            total_count = len(input_text_files)
            
            # 发送初始统计信息
            self.stats_updated.emit(total_count, 0, 0)
            
            for i, input_text_file in enumerate(input_text_files):
                if not self.is_running:
                    break
                
                input_basename = os.path.splitext(os.path.basename(input_text_file))[0]
                self.status_updated.emit(f"正在处理: {input_basename} ({i+1}/{total_count})")
                self.progress_updated.emit(int((i / total_count) * 100))
                
                # 读取待合成的文本
                tts_text = self.read_text_file(input_text_file)
                if not tts_text:
                    self.log_updated.emit(f"警告: 无法读取文本文件 {input_text_file}，跳过")
                    failed_count += 1
                    self.stats_updated.emit(total_count, success_count, failed_count)
                    continue
                
                self.log_updated.emit(f"待合成文本: {tts_text[:50]}...")
                
                # 随机选择一个prompt音频文本对
                prompt_audio_path, prompt_text_path = random.choice(prompt_pairs)
                prompt_text = self.read_text_file(prompt_text_path)
                
                if not prompt_text:
                    self.log_updated.emit(f"警告: 无法读取prompt文本文件 {prompt_text_path}，跳过")
                    failed_count += 1
                    self.stats_updated.emit(total_count, success_count, failed_count)
                    continue
                
                self.log_updated.emit(f"使用prompt音频: {os.path.basename(prompt_audio_path)}")
                self.log_updated.emit(f"prompt文本: {prompt_text[:50]}...")
                
                # 合成音频
                synthesized_audio = self.synthesize_audio(
                    cosyvoice=cosyvoice,
                    tts_text=tts_text,
                    prompt_text=prompt_text,
                    prompt_wav_path=prompt_audio_path
                )
                
                if synthesized_audio is not None:
                    # 保存合成的音频
                    output_audio_path = os.path.join(self.output_dir, f"{input_basename}.wav")
                    
                    # 保存为wav文件
                    torchaudio.save(
                        output_audio_path,
                        torch.from_numpy(synthesized_audio).unsqueeze(0),
                        self.sample_rate
                    )
                    
                    self.log_updated.emit(f"✓ 合成成功: {output_audio_path}")
                    success_count += 1
                else:
                    self.log_updated.emit(f"✗ 合成失败: {input_text_file}")
                    failed_count += 1
                
                # 更新统计信息
                self.stats_updated.emit(total_count, success_count, failed_count)
            
            self.progress_updated.emit(100)
            self.status_updated.emit("批量合成完成!")
            self.log_updated.emit(f"\n批量合成完成!")
            self.log_updated.emit(f"成功: {success_count}/{total_count}")
            self.log_updated.emit(f"输出目录: {self.output_dir}")
            
            self.finished.emit(success_count, total_count)
            
        except Exception as e:
            self.log_updated.emit(f"发生错误: {str(e)}")
            self.status_updated.emit(f"发生错误: {str(e)}")

class BatchCloneGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker_thread = None
        self.worker = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("LightTTS 批量克隆语音合成系统")
        self.setGeometry(100, 100, 800, 650)
        
        # 创建中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # 控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # 进度信息面板
        progress_panel = self.create_progress_panel()
        main_layout.addWidget(progress_panel)
        
        # 控制按钮
        button_layout = self.create_button_layout()
        main_layout.addLayout(button_layout)
    
    def create_control_panel(self):
        """创建控制面板"""
        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)
        layout.setSpacing(10)
        
        # 文件夹设置组
        folders_group = QGroupBox("文件夹设置")
        folders_layout = QGridLayout(folders_group)
        folders_layout.setColumnStretch(1, 1)
        folders_layout.setHorizontalSpacing(10)
        folders_layout.setVerticalSpacing(8)
        
        # 模型路径
        model_label = QLabel("模型路径:")
        model_label.setMinimumWidth(110)
        folders_layout.addWidget(model_label, 0, 0)
        self.model_dir_edit = QLineEdit(DEFAULT_MODEL_DIR)
        folders_layout.addWidget(self.model_dir_edit, 0, 1)
        model_dir_btn = QPushButton("浏览")
        model_dir_btn.setFixedWidth(70)
        model_dir_btn.clicked.connect(self.select_model_dir)
        folders_layout.addWidget(model_dir_btn, 0, 2)
        
        # Clone源文件夹
        clone_label = QLabel("Clone源文件夹:")
        clone_label.setMinimumWidth(110)
        folders_layout.addWidget(clone_label, 1, 0)
        self.clone_src_edit = QLineEdit(DEFAULT_CLONE_SRC_DIR)
        folders_layout.addWidget(self.clone_src_edit, 1, 1)
        clone_src_btn = QPushButton("浏览")
        clone_src_btn.setFixedWidth(70)
        clone_src_btn.clicked.connect(self.select_clone_src_dir)
        folders_layout.addWidget(clone_src_btn, 1, 2)
        
        # 输入文本文件夹
        input_label = QLabel("输入文本文件夹:")
        input_label.setMinimumWidth(110)
        folders_layout.addWidget(input_label, 2, 0)
        self.input_dir_edit = QLineEdit(DEFAULT_INPUT_DIR)
        folders_layout.addWidget(self.input_dir_edit, 2, 1)
        input_dir_btn = QPushButton("浏览")
        input_dir_btn.setFixedWidth(70)
        input_dir_btn.clicked.connect(self.select_input_dir)
        folders_layout.addWidget(input_dir_btn, 2, 2)
        
        # 输出文件夹
        output_label = QLabel("输出文件夹:")
        output_label.setMinimumWidth(110)
        folders_layout.addWidget(output_label, 3, 0)
        self.output_dir_edit = QLineEdit(DEFAULT_OUTPUT_DIR)
        folders_layout.addWidget(self.output_dir_edit, 3, 1)
        output_dir_btn = QPushButton("浏览")
        output_dir_btn.setFixedWidth(70)
        output_dir_btn.clicked.connect(self.select_output_dir)
        folders_layout.addWidget(output_dir_btn, 3, 2)
        
        layout.addWidget(folders_group)
        
        # 参数设置组
        params_group = QGroupBox("参数设置")
        params_layout = QHBoxLayout(params_group)
        params_layout.setSpacing(15)
        
        # 语音速度
        params_layout.addWidget(QLabel("语音速度:"))
        self.speed_spinbox = QDoubleSpinBox()
        self.speed_spinbox.setRange(MIN_SPEED, MAX_SPEED)
        self.speed_spinbox.setSingleStep(SPEED_STEP)
        self.speed_spinbox.setValue(DEFAULT_SPEED)
        self.speed_spinbox.setSuffix("x")
        self.speed_spinbox.setFixedWidth(80)
        params_layout.addWidget(self.speed_spinbox)
        
        params_layout.addSpacing(20)
        
        # 随机种子
        params_layout.addWidget(QLabel("随机种子:"))
        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(DEFAULT_SEED, 999999999)
        self.seed_spinbox.setValue(DEFAULT_SEED)
        self.seed_spinbox.setSpecialValueText("随机")
        self.seed_spinbox.setFixedWidth(100)
        params_layout.addWidget(self.seed_spinbox)
        
        params_layout.addSpacing(20)
        
        # 采样率
        params_layout.addWidget(QLabel("采样率:"))
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["16000 Hz", "22050 Hz"])
        self.sample_rate_combo.setCurrentText("22050 Hz")
        self.sample_rate_combo.setFixedWidth(110)
        params_layout.addWidget(self.sample_rate_combo)
        
        params_layout.addStretch()
        
        layout.addWidget(params_group)
        
        return control_widget
    
    def create_progress_panel(self):
        """创建进度信息面板"""
        progress_widget = QGroupBox("任务进度")
        layout = QVBoxLayout(progress_widget)
        layout.setSpacing(10)
        
        # 进度统计信息
        stats_layout = QHBoxLayout()
        
        # 总任务数
        self.total_label = QLabel("总任务: 0")
        self.total_label.setFont(QFont("Arial", 11))
        stats_layout.addWidget(self.total_label)
        
        stats_layout.addSpacing(30)
        
        # 已完成
        self.completed_label = QLabel("已完成: 0")
        self.completed_label.setFont(QFont("Arial", 11))
        self.completed_label.setStyleSheet("color: #28a745;")
        stats_layout.addWidget(self.completed_label)
        
        stats_layout.addSpacing(30)
        
        # 失败数
        self.failed_label = QLabel("失败: 0")
        self.failed_label.setFont(QFont("Arial", 11))
        self.failed_label.setStyleSheet("color: #dc3545;")
        stats_layout.addWidget(self.failed_label)
        
        stats_layout.addStretch()
        
        layout.addLayout(stats_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMinimumHeight(30)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ccc;
                border-radius: 5px;
                text-align: center;
                font-size: 14px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # 当前状态
        self.status_label = QLabel("就绪")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setStyleSheet("""
            padding: 8px;
            background-color: #e9ecef;
            border-radius: 4px;
            color: #495057;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # 详细日志（可折叠，默认隐藏）
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMaximumHeight(120)
        self.log_text.setVisible(False)
        layout.addWidget(self.log_text)
        
        # 显示/隐藏日志按钮
        self.toggle_log_btn = QPushButton("显示详细日志")
        self.toggle_log_btn.clicked.connect(self.toggle_log)
        self.toggle_log_btn.setFixedHeight(25)
        layout.addWidget(self.toggle_log_btn)
        
        return progress_widget
    
    def create_button_layout(self):
        """创建按钮布局"""
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        # 打开输出文件夹按钮
        self.open_output_btn = QPushButton("打开输出文件夹")
        self.open_output_btn.clicked.connect(self.open_output_folder)
        self.open_output_btn.setFixedHeight(35)
        btn_layout.addWidget(self.open_output_btn)
        
        btn_layout.addStretch()
        
        # 开始合成按钮
        self.start_btn = QPushButton("开始合成")
        self.start_btn.clicked.connect(self.start_synthesis)
        self.start_btn.setFixedSize(120, 40)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        btn_layout.addWidget(self.start_btn)
        
        # 停止合成按钮
        self.stop_btn = QPushButton("停止合成")
        self.stop_btn.clicked.connect(self.stop_synthesis)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setFixedSize(120, 40)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        btn_layout.addWidget(self.stop_btn)
        
        return btn_layout
    
    def toggle_log(self):
        """切换日志显示"""
        if self.log_text.isVisible():
            self.log_text.setVisible(False)
            self.toggle_log_btn.setText("显示详细日志")
        else:
            self.log_text.setVisible(True)
            self.toggle_log_btn.setText("隐藏详细日志")
    

    
    def select_model_dir(self):
        """选择模型目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if dir_path:
            self.model_dir_edit.setText(dir_path)
    
    def select_clone_src_dir(self):
        """选择Clone源文件夹"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择Clone源文件夹")
        if dir_path:
            self.clone_src_edit.setText(dir_path)
    
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
    

    
    def start_synthesis(self):
        """开始合成"""
        # 验证输入
        if not os.path.exists(self.model_dir_edit.text()):
            QMessageBox.warning(self, "错误", "模型路径不存在！")
            return
        
        if not os.path.exists(self.clone_src_edit.text()):
            QMessageBox.warning(self, "错误", "Clone源文件夹不存在！")
            return
        
        if not os.path.exists(self.input_dir_edit.text()):
            QMessageBox.warning(self, "错误", "输入文本文件夹不存在！")
            return
        
        # 禁用开始按钮，启用停止按钮
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # 重置进度条
        self.progress_bar.setValue(0)
        
        # 获取参数
        seed = self.seed_spinbox.value() if self.seed_spinbox.value() != -1 else None
        sample_rate = int(self.sample_rate_combo.currentText().split()[0])
        
        # 创建工作线程
        self.worker_thread = QThread()
        self.worker = SynthesisWorker(
            model_dir=self.model_dir_edit.text(),
            clone_src_dir=self.clone_src_edit.text(),
            input_dir=self.input_dir_edit.text(),
            output_dir=self.output_dir_edit.text(),
            speed=self.speed_spinbox.value(),
            seed=seed,
            sample_rate=sample_rate
        )
        
        # 连接信号
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.log_updated.connect(self.log_text.append)
        self.worker.stats_updated.connect(self.update_stats)
        self.worker.finished.connect(self.synthesis_finished)
        
        # 移动到线程并启动
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run_synthesis)
        self.worker_thread.start()
    
    def stop_synthesis(self):
        """停止合成"""
        if self.worker:
            self.worker.stop()
        self.synthesis_finished(0, 0)
        self.log_text.append("用户手动停止合成")
    
    def update_stats(self, total, completed, failed):
        """更新统计信息"""
        self.total_label.setText(f"总任务: {total}")
        self.completed_label.setText(f"已完成: {completed}")
        self.failed_label.setText(f"失败: {failed}")
    
    def open_output_folder(self):
        """打开输出文件夹"""
        output_dir = self.output_dir_edit.text()
        if os.path.exists(output_dir):
            os.system(f'xdg-open "{output_dir}"')
        else:
            QMessageBox.warning(self, "提示", "输出文件夹不存在！")
    
    def synthesis_finished(self, success_count, total_count):
        """合成完成"""
        # 重新启用按钮
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # 清理线程
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        
        if success_count > 0 or total_count > 0:
            QMessageBox.information(self, "完成", 
                                  f"批量合成完成！\n成功: {success_count}/{total_count}")

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    window = BatchCloneGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 