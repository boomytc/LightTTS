import sys
import os

_current_script_absolute_path = os.path.abspath(__file__)
_batch_generate_dir = os.path.dirname(_current_script_absolute_path)
_project_root = os.path.dirname(_batch_generate_dir)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_matcha_tts_path = os.path.join(_project_root, 'Matcha-TTS')
if os.path.isdir(_matcha_tts_path) and _matcha_tts_path not in sys.path:
    sys.path.insert(1, _matcha_tts_path)

import random
import glob
import logging
import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from threading import Thread
import time

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                               QPushButton, QFileDialog, QProgressBar, QTextEdit,
                               QSpinBox, QDoubleSpinBox, QGroupBox, QMessageBox,
                               QSplitter, QFrame)
from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer, QUrl
from PySide6.QtGui import QFont, QIcon
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

# 禁用警告
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

max_val = 0.8

class VoiceRegisterWorker(QObject):
    """音色注册和批量合成工作线程"""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    log_updated = Signal(str)
    finished = Signal(int, int)  # success_count, total_count
    voice_registered = Signal(str)  # 音色注册完成信号
    
    def __init__(self, model_dir, prompt_audio_path, prompt_text, input_dir, output_dir, 
                 speed=1.0, seed=None, sample_rate=22050):
        super().__init__()
        self.model_dir = model_dir
        self.prompt_audio_path = prompt_audio_path
        self.prompt_text = prompt_text
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.speed = speed
        self.seed = seed
        self.sample_rate = sample_rate
        self.is_running = True
        self.cosyvoice = None
        self.prompt_speech_16k = None
    
    def stop(self):
        self.is_running = False
    
    def postprocess(self, speech, top_db=60, hop_length=220, win_length=440):
        """音频后处理"""
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > max_val:
            speech = speech / speech.abs().max() * max_val
        speech = torch.concat([speech, torch.zeros(1, int(16000 * 0.2))], dim=1)
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
        text_exts = ['.txt']
        
        for text_ext in text_exts:
            text_files.extend(glob.glob(os.path.join(input_dir, f"*{text_ext}")))
        
        return text_files
    
    def register_voice(self):
        """注册音色"""
        try:
            # 初始化CosyVoice2模型
            self.status_updated.emit("正在加载模型...")
            self.log_updated.emit(f"正在加载CosyVoice2模型: {self.model_dir}")
            self.cosyvoice = CosyVoice2(self.model_dir)
            self.log_updated.emit("模型加载完成")
            
            # 验证prompt音频
            if not os.path.exists(self.prompt_audio_path):
                self.log_updated.emit(f"错误: 音频文件不存在: {self.prompt_audio_path}")
                return False
            
            # 检查音频采样率
            audio_info = torchaudio.info(self.prompt_audio_path)
            if audio_info.sample_rate < 16000:
                self.log_updated.emit(f"警告: 音频 {self.prompt_audio_path} 采样率过低")
                return False
            
            # 加载并处理prompt音频
            self.status_updated.emit("正在注册音色...")
            self.log_updated.emit(f"正在处理音频: {os.path.basename(self.prompt_audio_path)}")
            self.log_updated.emit(f"音频采样率: {audio_info.sample_rate} Hz")
            self.log_updated.emit(f"音频时长: {audio_info.num_frames / audio_info.sample_rate:.2f} 秒")
            
            self.prompt_speech_16k = self.postprocess(load_wav(self.prompt_audio_path, 16000))
            
            self.log_updated.emit(f"使用prompt文本: {self.prompt_text}")
            self.log_updated.emit("音色注册完成！")
            self.voice_registered.emit(os.path.basename(self.prompt_audio_path))
            
            return True
            
        except Exception as e:
            self.log_updated.emit(f"音色注册失败: {str(e)}")
            return False
    
    def synthesize_audio(self, tts_text):
        """使用注册的音色合成音频"""
        if self.seed is not None:
            set_all_random_seed(self.seed)
        
        try:
            result = None
            for i in self.cosyvoice.inference_zero_shot(tts_text, self.prompt_text, self.prompt_speech_16k, 
                                                       stream=False, speed=self.speed):
                result = i['tts_speech'].numpy().flatten()
            return result
        except Exception as e:
            self.log_updated.emit(f"合成失败: {e}")
            return None
    
    def run_synthesis(self):
        """执行音色注册和批量合成"""
        try:
            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 注册音色
            if not self.register_voice():
                self.status_updated.emit("音色注册失败")
                return
            
            # 获取输入文本文件
            self.status_updated.emit("获取文本文件列表...")
            input_text_files = self.get_input_text_files(self.input_dir)
            if not input_text_files:
                self.log_updated.emit(f"错误: 在 {self.input_dir} 中未找到任何文本文件")
                return
            
            self.log_updated.emit(f"找到 {len(input_text_files)} 个待合成的文本文件")
            
            # 批量合成
            success_count = 0
            total_count = len(input_text_files)
            
            self.status_updated.emit("开始批量合成...")
            
            for i, input_text_file in enumerate(input_text_files):
                if not self.is_running:
                    break
                
                self.status_updated.emit(f"处理进度: {i+1}/{total_count}")
                self.progress_updated.emit(int((i / total_count) * 100))
                
                # 读取待合成的文本
                tts_text = self.read_text_file(input_text_file)
                if not tts_text:
                    self.log_updated.emit(f"警告: 无法读取文本文件 {input_text_file}，跳过")
                    continue
                
                filename = os.path.basename(input_text_file)
                self.log_updated.emit(f"正在合成: {filename}")
                self.log_updated.emit(f"文本内容: {tts_text[:50]}...")
                
                # 合成音频
                synthesized_audio = self.synthesize_audio(tts_text)
                
                if synthesized_audio is not None:
                    # 保存合成的音频
                    input_basename = os.path.splitext(os.path.basename(input_text_file))[0]
                    output_audio_path = os.path.join(self.output_dir, f"{input_basename}.wav")
                    
                    # 保存为wav文件
                    torchaudio.save(
                        output_audio_path,
                        torch.from_numpy(synthesized_audio).unsqueeze(0),
                        self.sample_rate
                    )
                    
                    self.log_updated.emit(f"✓ 合成成功: {os.path.basename(output_audio_path)}")
                    success_count += 1
                else:
                    self.log_updated.emit(f"✗ 合成失败: {filename}")
            
            self.progress_updated.emit(100)
            self.status_updated.emit("批量合成完成!")
            self.log_updated.emit(f"\n=== 批量合成完成 ===")
            self.log_updated.emit(f"成功: {success_count}/{total_count}")
            self.log_updated.emit(f"输出目录: {self.output_dir}")
            
            self.finished.emit(success_count, total_count)
            
        except Exception as e:
            self.log_updated.emit(f"发生错误: {str(e)}")
            self.status_updated.emit(f"发生错误: {str(e)}")

class VoiceRegisterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker_thread = None
        self.worker = None
        self.media_player = None
        self.audio_output = None
        self.init_ui()
        self.init_audio_player()
        
    def init_audio_player(self):
        """初始化音频播放器"""
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
    def init_ui(self):
        self.setWindowTitle("LightTTS 音色注册与批量合成系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # 右侧日志面板
        log_panel = self.create_log_panel()
        splitter.addWidget(log_panel)
        
        # 设置分割器比例
        splitter.setSizes([500, 700])
    
    def create_control_panel(self):
        """创建控制面板"""
        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)
        
        # 标题
        title_label = QLabel("音色注册与批量合成")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 模型设置组
        model_group = QGroupBox("模型设置")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("模型路径:"), 0, 0)
        self.model_dir_edit = QLineEdit("pretrained_models/CosyVoice2-0.5B")
        model_layout.addWidget(self.model_dir_edit, 0, 1)
        model_dir_btn = QPushButton("浏览")
        model_dir_btn.clicked.connect(self.select_model_dir)
        model_layout.addWidget(model_dir_btn, 0, 2)
        
        layout.addWidget(model_group)
        
        # 音色注册组
        voice_group = QGroupBox("音色注册")
        voice_layout = QGridLayout(voice_group)
        
        # 音频文件选择
        voice_layout.addWidget(QLabel("音频文件:"), 0, 0)
        self.audio_file_edit = QLineEdit()
        voice_layout.addWidget(self.audio_file_edit, 0, 1)
        audio_file_btn = QPushButton("选择")
        audio_file_btn.clicked.connect(self.select_audio_file)
        voice_layout.addWidget(audio_file_btn, 0, 2)
        
        # 音频播放按钮
        self.play_audio_btn = QPushButton("播放音频")
        self.play_audio_btn.clicked.connect(self.play_audio)
        self.play_audio_btn.setEnabled(False)
        voice_layout.addWidget(self.play_audio_btn, 0, 3)
        
        # Prompt文本
        voice_layout.addWidget(QLabel("Prompt文本:"), 1, 0)
        self.prompt_text_edit = QTextEdit()
        self.prompt_text_edit.setMaximumHeight(80)
        self.prompt_text_edit.setPlaceholderText("请输入音频内容的文本描述...")
        voice_layout.addWidget(self.prompt_text_edit, 1, 1, 1, 3)
        
        layout.addWidget(voice_group)
        
        # 批量合成设置组
        batch_group = QGroupBox("批量合成设置")
        batch_layout = QGridLayout(batch_group)
        
        # 输入文本文件夹
        batch_layout.addWidget(QLabel("输入文本文件夹:"), 0, 0)
        self.input_dir_edit = QLineEdit("BatchGenerate/texts")
        batch_layout.addWidget(self.input_dir_edit, 0, 1)
        input_dir_btn = QPushButton("浏览")
        input_dir_btn.clicked.connect(self.select_input_dir)
        batch_layout.addWidget(input_dir_btn, 0, 2)
        
        # 输出文件夹
        batch_layout.addWidget(QLabel("输出文件夹:"), 1, 0)
        self.output_dir_edit = QLineEdit("BatchGenerate/voice_output")
        batch_layout.addWidget(self.output_dir_edit, 1, 1)
        output_dir_btn = QPushButton("浏览")
        output_dir_btn.clicked.connect(self.select_output_dir)
        batch_layout.addWidget(output_dir_btn, 1, 2)
        
        layout.addWidget(batch_group)
        
        # 参数设置组
        params_group = QGroupBox("参数设置")
        params_layout = QGridLayout(params_group)
        
        # 语音速度
        params_layout.addWidget(QLabel("语音速度:"), 0, 0)
        self.speed_spinbox = QDoubleSpinBox()
        self.speed_spinbox.setRange(0.5, 2.0)
        self.speed_spinbox.setSingleStep(0.1)
        self.speed_spinbox.setValue(1.0)
        params_layout.addWidget(self.speed_spinbox, 0, 1)
        
        # 随机种子
        params_layout.addWidget(QLabel("随机种子:"), 1, 0)
        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(-1, 999999999)
        self.seed_spinbox.setValue(-1)
        self.seed_spinbox.setSpecialValueText("随机")
        params_layout.addWidget(self.seed_spinbox, 1, 1)
        
        # 采样率
        params_layout.addWidget(QLabel("采样率:"), 2, 0)
        self.sample_rate_spinbox = QSpinBox()
        self.sample_rate_spinbox.setRange(16000, 48000)
        self.sample_rate_spinbox.setSingleStep(1000)
        self.sample_rate_spinbox.setValue(22050)
        params_layout.addWidget(self.sample_rate_spinbox, 2, 1)
        
        layout.addWidget(params_group)
        
        # 音色注册状态
        self.voice_status_label = QLabel("音色状态: 未注册")
        self.voice_status_label.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.voice_status_label)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        
        self.register_btn = QPushButton("注册音色")
        self.register_btn.clicked.connect(self.register_voice_only)
        btn_layout.addWidget(self.register_btn)
        
        self.start_btn = QPushButton("开始批量合成")
        self.start_btn.clicked.connect(self.start_synthesis)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止合成")
        self.stop_btn.clicked.connect(self.stop_synthesis)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        layout.addLayout(btn_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        return control_widget
    
    def create_log_panel(self):
        """创建日志面板"""
        log_widget = QWidget()
        layout = QVBoxLayout(log_widget)
        
        log_label = QLabel("运行日志")
        log_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)
        
        # 清除日志按钮
        clear_btn = QPushButton("清除日志")
        clear_btn.clicked.connect(self.clear_log)
        layout.addWidget(clear_btn)
        
        return log_widget
    
    def select_model_dir(self):
        """选择模型目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if dir_path:
            self.model_dir_edit.setText(dir_path)
    
    def select_audio_file(self):
        """选择音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "", 
            "音频文件 (*.wav *.mp3 *.flac *.m4a *.ogg);;所有文件 (*)"
        )
        if file_path:
            self.audio_file_edit.setText(file_path)
            self.play_audio_btn.setEnabled(True)
            # 重置音色状态
            self.voice_status_label.setText("音色状态: 未注册")
            self.voice_status_label.setStyleSheet("color: red; font-weight: bold;")
    
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
    
    def play_audio(self):
        """播放音频文件"""
        audio_path = self.audio_file_edit.text()
        if audio_path and os.path.exists(audio_path):
            self.media_player.setSource(QUrl.fromLocalFile(audio_path))
            self.media_player.play()
            self.log_text.append(f"正在播放: {os.path.basename(audio_path)}")
    
    def clear_log(self):
        """清除日志"""
        self.log_text.clear()
    
    def register_voice_only(self):
        """仅注册音色（不进行批量合成）"""
        if not self.validate_voice_inputs():
            return
        
        # 创建临时工作线程仅用于注册音色
        self.register_btn.setEnabled(False)
        self.status_label.setText("正在注册音色...")
        
        # 获取参数
        seed = self.seed_spinbox.value() if self.seed_spinbox.value() != -1 else None
        
        # 创建工作线程（仅用于注册音色）
        self.worker_thread = QThread()
        self.worker = VoiceRegisterWorker(
            model_dir=self.model_dir_edit.text(),
            prompt_audio_path=self.audio_file_edit.text(),
            prompt_text=self.prompt_text_edit.toPlainText(),
            input_dir="",  # 注册时不需要
            output_dir="",  # 注册时不需要
            speed=self.speed_spinbox.value(),
            seed=seed,
            sample_rate=self.sample_rate_spinbox.value()
        )
        
        # 连接信号
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.log_updated.connect(self.log_text.append)
        self.worker.voice_registered.connect(self.voice_registered)
        
        # 移动到线程并启动
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.register_voice)
        self.worker_thread.start()
    
    def voice_registered(self, audio_name):
        """音色注册完成"""
        self.voice_status_label.setText(f"音色状态: 已注册 ({audio_name})")
        self.voice_status_label.setStyleSheet("color: green; font-weight: bold;")
        self.register_btn.setEnabled(True)
        self.status_label.setText("音色注册完成")
        
        # 清理线程
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
    
    def validate_voice_inputs(self):
        """验证音色相关输入"""
        if not os.path.exists(self.model_dir_edit.text()):
            QMessageBox.warning(self, "错误", "模型路径不存在！")
            return False
        
        if not self.audio_file_edit.text():
            QMessageBox.warning(self, "错误", "请选择音频文件！")
            return False
        
        if not os.path.exists(self.audio_file_edit.text()):
            QMessageBox.warning(self, "错误", "音频文件不存在！")
            return False
        
        if not self.prompt_text_edit.toPlainText().strip():
            QMessageBox.warning(self, "错误", "请输入Prompt文本！")
            return False
        
        return True
    
    def start_synthesis(self):
        """开始批量合成"""
        # 验证所有输入
        if not self.validate_voice_inputs():
            return
        
        if not os.path.exists(self.input_dir_edit.text()):
            QMessageBox.warning(self, "错误", "输入文本文件夹不存在！")
            return
        
        # 禁用按钮
        self.start_btn.setEnabled(False)
        self.register_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # 重置进度条
        self.progress_bar.setValue(0)
        
        # 获取参数
        seed = self.seed_spinbox.value() if self.seed_spinbox.value() != -1 else None
        
        # 创建工作线程
        self.worker_thread = QThread()
        self.worker = VoiceRegisterWorker(
            model_dir=self.model_dir_edit.text(),
            prompt_audio_path=self.audio_file_edit.text(),
            prompt_text=self.prompt_text_edit.toPlainText(),
            input_dir=self.input_dir_edit.text(),
            output_dir=self.output_dir_edit.text(),
            speed=self.speed_spinbox.value(),
            seed=seed,
            sample_rate=self.sample_rate_spinbox.value()
        )
        
        # 连接信号
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.log_updated.connect(self.log_text.append)
        self.worker.finished.connect(self.synthesis_finished)
        self.worker.voice_registered.connect(self.voice_registered)
        
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
    
    def synthesis_finished(self, success_count, total_count):
        """合成完成"""
        # 重新启用按钮
        self.start_btn.setEnabled(True)
        self.register_btn.setEnabled(True)
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
    
    window = VoiceRegisterGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 