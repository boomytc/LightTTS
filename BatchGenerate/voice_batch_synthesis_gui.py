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
import json
import shutil
import uuid
from datetime import datetime

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                               QPushButton, QFileDialog, QProgressBar, QTextEdit,
                               QSpinBox, QDoubleSpinBox, QGroupBox, QMessageBox,
                               QSplitter, QFrame, QTabWidget, QListWidget, 
                               QListWidgetItem, QInputDialog, QComboBox)
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

class VoiceSynthesisWorker(QObject):
    """音色批量合成工作线程"""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    log_updated = Signal(str)
    finished = Signal(int, int)  # success_count, total_count
    
    def __init__(self, model_dir, voice_data, input_dir, output_dir, 
                 speed=1.0, seed=None, sample_rate=22050):
        super().__init__()
        self.model_dir = model_dir
        self.voice_data = voice_data
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
    
    def load_voice(self):
        """加载音色"""
        try:
            # 初始化CosyVoice2模型
            self.status_updated.emit("正在加载模型...")
            self.log_updated.emit(f"正在加载CosyVoice2模型: {self.model_dir}")
            self.cosyvoice = CosyVoice2(self.model_dir)
            self.log_updated.emit("模型加载完成")
            
            # 加载音色
            prompt_audio_path = self.voice_data.get('source', '')
            if not os.path.exists(prompt_audio_path):
                self.log_updated.emit(f"错误: 音频文件不存在: {prompt_audio_path}")
                return False
            
            # 检查音频采样率
            audio_info = torchaudio.info(prompt_audio_path)
            if audio_info.sample_rate < 16000:
                self.log_updated.emit(f"警告: 音频 {prompt_audio_path} 采样率过低")
                return False
            
            # 加载并处理prompt音频
            self.status_updated.emit("正在加载音色...")
            self.log_updated.emit(f"正在处理音色: {self.voice_data.get('key', 'Unknown')}")
            self.log_updated.emit(f"音频采样率: {audio_info.sample_rate} Hz")
            self.log_updated.emit(f"音频时长: {audio_info.num_frames / audio_info.sample_rate:.2f} 秒")
            self.log_updated.emit(f"Prompt文本: {self.voice_data.get('target', '')}")
            
            self.prompt_speech_16k = self.postprocess(load_wav(prompt_audio_path, 16000))
            
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
            for i in self.cosyvoice.inference_zero_shot(tts_text, prompt_text, self.prompt_speech_16k, 
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
            
            # 加载音色
            if not self.load_voice():
                self.status_updated.emit("音色加载失败")
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

class VoiceBatchSynthesisGUI(QMainWindow):
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
        self.setWindowTitle("LightTTS 音色选择与批量语音合成系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧主面板
        main_panel = self.create_main_panel()
        splitter.addWidget(main_panel)
        
        # 右侧日志面板
        log_panel = self.create_log_panel()
        splitter.addWidget(log_panel)
        
        # 设置分割器比例
        splitter.setSizes([700, 500])
        
        # 初始化音色下拉框
        QTimer.singleShot(100, self.refresh_voice_combo)
    
    def create_main_panel(self):
        """创建主面板"""
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        
        # 标题
        title_label = QLabel("音色选择与批量语音合成")
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
        
        # 音色选择组
        voice_select_group = QGroupBox("音色选择")
        voice_select_layout = QGridLayout(voice_select_group)
        
        voice_select_layout.addWidget(QLabel("选择音色:"), 0, 0)
        self.voice_combo = QComboBox()
        voice_select_layout.addWidget(self.voice_combo, 0, 1)
        
        refresh_combo_btn = QPushButton("🔄 刷新")
        refresh_combo_btn.clicked.connect(self.refresh_voice_combo)
        voice_select_layout.addWidget(refresh_combo_btn, 0, 2)
        
        # 播放选择的音色
        self.play_voice_btn = QPushButton("🔊 播放音色")
        self.play_voice_btn.clicked.connect(self.play_selected_voice)
        self.play_voice_btn.setEnabled(False)
        voice_select_layout.addWidget(self.play_voice_btn, 0, 3)
        
        # 显示选择的音色信息
        voice_select_layout.addWidget(QLabel("音色信息:"), 1, 0)
        self.combo_voice_info_label = QLabel("请选择音色")
        self.combo_voice_info_label.setWordWrap(True)
        self.combo_voice_info_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        voice_select_layout.addWidget(self.combo_voice_info_label, 1, 1, 1, 3)
        
        layout.addWidget(voice_select_group)
        
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
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("🚀 开始批量合成")
        self.start_btn.clicked.connect(self.start_synthesis)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹️ 停止合成")
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
        return main_widget
    
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
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        
        # 清除日志按钮
        clear_btn = QPushButton("清除日志")
        clear_btn.clicked.connect(self.clear_log)
        btn_layout.addWidget(clear_btn)
        
        # 查看音色数据库按钮
        view_db_btn = QPushButton("查看音色数据库")
        view_db_btn.clicked.connect(self.view_voice_database)
        btn_layout.addWidget(view_db_btn)
        
        # 打开音色管理器按钮
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
    
    def clear_log(self):
        """清除日志"""
        self.log_text.clear()
    
    def view_voice_database(self):
        """查看音色数据库"""
        db_clone_dir = os.path.join(_project_root, "BatchGenerate/DB_clone")
        db_clone_jsonl = os.path.join(db_clone_dir, "db_clone.jsonl")
        
        self.log_text.append("=== 音色数据库内容 ===")
        
        if not os.path.exists(db_clone_jsonl):
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
            import shlex
            
            script_path = os.path.join(_batch_generate_dir, "voice_register_manager_gui.py")
            
            # 检查脚本是否存在
            if not os.path.exists(script_path):
                self.log_text.append(f"错误: 音色管理器脚本不存在: {script_path}")
                return
            
            # 获取当前Python解释器路径
            python_executable = sys.executable
            
            # 检查是否在conda环境中
            in_conda = ('conda' in python_executable.lower() or 
                       'anaconda' in python_executable.lower() or
                       'CONDA_DEFAULT_ENV' in os.environ)
            
            # 如果当前Python解释器不是conda环境中的，尝试使用conda环境的python
            if in_conda and ('Cursor' in python_executable or 'vscode' in python_executable.lower()):
                # 在IDE中但有conda环境，尝试直接使用python命令
                self.log_text.append("检测到在IDE中运行，尝试使用conda环境的python")
                self._try_alternative_launch(script_path)
                    
            else:
                # 使用当前Python解释器
                try:
                    cmd = [python_executable, script_path]
                    self.log_text.append(f"启动命令: {' '.join(cmd)}")
                    
                    # 在后台启动音色管理器
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=_project_root,
                        env=os.environ.copy()
                    )
                    
                    self.log_text.append("已启动音色管理器")
                    self.log_text.append(f"进程ID: {process.pid}")
                    
                except subprocess.SubprocessError as se:
                    self.log_text.append(f"使用sys.executable启动失败: {str(se)}")
                    # 尝试备用方法
                    self._try_alternative_launch(script_path)
                
        except Exception as e:
            self.log_text.append(f"启动音色管理器失败: {str(e)}")
            self.log_text.append("请尝试手动运行音色管理器:")
            self.log_text.append(f"python {script_path}")
            
    def _try_alternative_launch(self, script_path):
        """尝试备用的启动方法"""
        try:
            import subprocess
            
            # 方法1：尝试使用python命令
            try:
                cmd = ["python", script_path]
                process = subprocess.Popen(cmd, cwd=_project_root)
                self.log_text.append("使用备用方法启动音色管理器成功")
                self.log_text.append(f"进程ID: {process.pid}")
                return
            except FileNotFoundError:
                pass
            
            # 方法2：尝试使用python3命令
            try:
                cmd = ["python3", script_path]
                process = subprocess.Popen(cmd, cwd=_project_root)
                self.log_text.append("使用python3命令启动音色管理器成功")
                self.log_text.append(f"进程ID: {process.pid}")
                return
            except FileNotFoundError:
                pass
                
            self.log_text.append("所有启动方法都失败了")
            
        except Exception as e:
            self.log_text.append(f"备用启动方法也失败: {str(e)}")
    
    def start_synthesis(self):
        """开始批量合成"""
        # 验证音色选择
        selected_voice = self.get_selected_voice_for_synthesis()
        if not selected_voice:
            QMessageBox.warning(self, "错误", "请先选择一个音色！")
            return
        
        # 验证模型路径
        if not os.path.exists(self.model_dir_edit.text()):
            QMessageBox.warning(self, "错误", "模型路径不存在！")
            return
        
        # 验证输入输出目录
        if not os.path.exists(self.input_dir_edit.text()):
            QMessageBox.warning(self, "错误", "输入文本文件夹不存在！")
            return
        
        # 禁用按钮
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # 重置进度条
        self.progress_bar.setValue(0)
        
        # 获取参数
        seed = self.seed_spinbox.value() if self.seed_spinbox.value() != -1 else None
        
        # 使用选择的音色创建工作线程
        self.worker_thread = QThread()
        self.worker = VoiceSynthesisWorker(
            model_dir=self.model_dir_edit.text(),
            voice_data=selected_voice,
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
        
        # 移动到线程并启动
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run_synthesis)
        self.worker_thread.start()
    
    def get_selected_voice_for_synthesis(self):
        """获取批量合成选择的音色"""
        current_index = self.voice_combo.currentIndex()
        if current_index > 0:  # 跳过"请选择音色..."
            return self.voice_combo.itemData(current_index)
        return None
    
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
        self.stop_btn.setEnabled(False)
        
        # 清理线程
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        
        if success_count > 0 or total_count > 0:
            QMessageBox.information(self, "完成", 
                                  f"批量合成完成！\n成功: {success_count}/{total_count}")
    
    def refresh_voice_combo(self):
        """刷新音色下拉框"""
        current_text = self.voice_combo.currentText()
        
        # 断开旧的信号连接
        try:
            self.voice_combo.currentTextChanged.disconnect()
        except:
            pass
        
        self.voice_combo.clear()
        
        db_clone_dir = os.path.join(_project_root, "BatchGenerate/DB_clone")
        db_clone_jsonl = os.path.join(db_clone_dir, "db_clone.jsonl")
        
        if not os.path.exists(db_clone_jsonl):
            self.voice_combo.addItem("请选择音色...")
            self.combo_voice_info_label.setText("数据库为空，请先注册音色")
            # 重新连接信号
            self.voice_combo.currentTextChanged.connect(self.on_voice_combo_changed)
            return
        
        try:
            voices = []
            with open(db_clone_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        voices.append(entry)
            
            # 添加选项
            self.voice_combo.addItem("请选择音色...")
            for voice in voices:
                voice_key = voice.get('key', 'Unknown')
                self.voice_combo.addItem(voice_key)
                # 存储完整数据到ComboBox
                index = self.voice_combo.count() - 1
                self.voice_combo.setItemData(index, voice)
            
            # 恢复选择
            if current_text:
                index = self.voice_combo.findText(current_text)
                if index >= 0:
                    self.voice_combo.setCurrentIndex(index)
            
            self.log_text.append(f"刷新音色列表完成，共 {len(voices)} 个音色")
            
        except Exception as e:
            self.log_text.append(f"刷新音色下拉框失败: {str(e)}")
        
        # 重新连接选择变化信号
        self.voice_combo.currentTextChanged.connect(self.on_voice_combo_changed)
    
    def on_voice_combo_changed(self, text):
        """音色下拉框选择改变"""
        if text == "请选择音色..." or not text:
            self.combo_voice_info_label.setText("请选择音色")
            self.play_voice_btn.setEnabled(False)
            return
        
        # 获取选择的音色数据
        current_index = self.voice_combo.currentIndex()
        if current_index > 0:  # 跳过"请选择音色..."
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
            if os.path.exists(source_path):
                self.media_player.setSource(QUrl.fromLocalFile(source_path))
                self.media_player.play()
                self.log_text.append(f"播放: {selected_voice.get('key', '')}")
            else:
                QMessageBox.warning(self, "错误", "音频文件不存在！")

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    window = VoiceBatchSynthesisGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 