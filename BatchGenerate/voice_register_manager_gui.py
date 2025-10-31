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

import torch
import torchaudio
import librosa
import json
import shutil
from datetime import datetime

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                               QPushButton, QFileDialog, QTextEdit,
                               QGroupBox, QMessageBox,
                               QSplitter, QTabWidget, QListWidget, 
                               QListWidgetItem, QInputDialog)
from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer, QUrl
from PySide6.QtGui import QFont
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 禁用警告
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

max_val = 0.8

class VoiceRegisterWorker(QObject):
    """音色注册工作线程"""
    status_updated = Signal(str)
    log_updated = Signal(str)
    voice_registered = Signal(str)  # 音色注册完成信号
    
    def __init__(self, model_dir, prompt_audio_path, prompt_text, voice_key=""):
        super().__init__()
        self.model_dir = model_dir
        self.prompt_audio_path = prompt_audio_path
        self.prompt_text = prompt_text
        self.voice_key = voice_key
        self.cosyvoice = None
        self.prompt_speech_16k = None
        
        # DB_clone 路径设置
        self.db_clone_dir = os.path.join(_project_root, "BatchGenerate/DB_clone")
        self.db_clone_jsonl = os.path.join(self.db_clone_dir, "db_clone.jsonl")
    
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
    
    def save_voice_to_db(self, voice_key, original_audio_path, prompt_text):
        """保存音色到数据库"""
        try:
            # 创建 DB_clone 目录
            os.makedirs(self.db_clone_dir, exist_ok=True)
            
            # 生成新的音频文件名
            original_filename = os.path.basename(original_audio_path)
            name_part, ext = os.path.splitext(original_filename)
            
            # 如果没有提供voice_key，使用文件名加时间戳
            if not voice_key.strip():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                voice_key = f"{name_part}_{timestamp}"
            
            # 生成唯一的文件名避免冲突
            saved_filename = f"{voice_key}{ext}"
            counter = 1
            while os.path.exists(os.path.join(self.db_clone_dir, saved_filename)):
                saved_filename = f"{voice_key}_{counter}{ext}"
                counter += 1
            
            saved_audio_path = os.path.join(self.db_clone_dir, saved_filename)
            
            # 复制音频文件到 DB_clone 目录
            shutil.copy2(original_audio_path, saved_audio_path)
            self.log_updated.emit(f"音频文件已保存到: {saved_audio_path}")
            
            # 获取音频信息
            try:
                audio_info = torchaudio.info(saved_audio_path)
                source_len = int(audio_info.num_frames / audio_info.sample_rate * 1000)  # 转换为毫秒
            except:
                source_len = 0
            
            # 计算目标文本长度
            target_len = len(prompt_text)
            
            # 创建数据库条目
            db_entry = {
                "key": voice_key,
                "source": saved_audio_path,  # 使用完整路径
                "source_len": source_len,
                "target": prompt_text,
                "target_len": target_len,
                "created_time": datetime.now().isoformat(),
                "original_path": original_audio_path
            }
            
            # 写入到 JSONL 文件
            with open(self.db_clone_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(db_entry, ensure_ascii=False) + '\n')
            
            self.log_updated.emit(f"音色数据库已更新: {voice_key}")
            self.log_updated.emit(f"数据库文件: {self.db_clone_jsonl}")
            
            return saved_audio_path, voice_key
            
        except Exception as e:
            self.log_updated.emit(f"保存音色到数据库失败: {str(e)}")
            return None, None
    
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
            
            # 保存音色到数据库
            self.status_updated.emit("正在保存音色到数据库...")
            saved_audio_path, final_voice_key = self.save_voice_to_db(
                self.voice_key, self.prompt_audio_path, self.prompt_text
            )
            
            if saved_audio_path and final_voice_key:
                self.log_updated.emit(f"使用prompt文本: {self.prompt_text}")
                self.log_updated.emit("音色注册并保存到数据库完成！")
                self.voice_registered.emit(final_voice_key)
                return True
            else:
                self.log_updated.emit("音色注册失败")
                return False
            
        except Exception as e:
            self.log_updated.emit(f"音色注册失败: {str(e)}")
            return False

class VoiceRegisterManagerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker_thread = None
        self.worker = None
        self.media_player = None
        self.audio_output = None
        self.selected_voice_data = None  # 当前选择的音色数据
        self.init_ui()
        self.init_audio_player()
        
    def init_audio_player(self):
        """初始化音频播放器"""
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
    def init_ui(self):
        self.setWindowTitle("LightTTS 音色注册与管理系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧标签页面板
        tab_panel = self.create_tab_panel()
        splitter.addWidget(tab_panel)
        
        # 右侧日志面板
        log_panel = self.create_log_panel()
        splitter.addWidget(log_panel)
        
        # 设置分割器比例
        splitter.setSizes([700, 500])
        
        # 初始化音色列表
        QTimer.singleShot(100, self.refresh_voice_list)
    
    def create_tab_panel(self):
        """创建标签页面板"""
        tab_widget = QTabWidget()
        
        # 标签页1: 音色注册
        register_tab = self.create_register_tab()
        tab_widget.addTab(register_tab, "🎤 音色注册")
        
        # 标签页2: 音色管理
        manage_tab = self.create_manage_tab()
        tab_widget.addTab(manage_tab, "📚 音色管理")
        
        return tab_widget
    
    def create_register_tab(self):
        """创建音色注册标签页"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        
        # 标题
        title_label = QLabel("音色注册")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 模型设置组
        model_group = QGroupBox("模型设置")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("模型路径:"), 0, 0)
        self.model_dir_edit = QLineEdit("models/CosyVoice2-0.5B")
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
        
        # 音色键名
        voice_layout.addWidget(QLabel("音色键名:"), 1, 0)
        self.voice_key_edit = QLineEdit()
        self.voice_key_edit.setPlaceholderText("输入音色名称，如：周杰伦voice（可选，留空自动生成）")
        voice_layout.addWidget(self.voice_key_edit, 1, 1, 1, 3)
        
        # Prompt文本
        voice_layout.addWidget(QLabel("Prompt文本:"), 2, 0)
        self.prompt_text_edit = QTextEdit()
        self.prompt_text_edit.setMaximumHeight(80)
        self.prompt_text_edit.setPlaceholderText("请输入音频内容的文本描述...")
        voice_layout.addWidget(self.prompt_text_edit, 2, 1, 1, 3)
        
        layout.addWidget(voice_group)
        
        # 音色注册状态
        self.voice_status_label = QLabel("音色状态: 未注册")
        self.voice_status_label.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.voice_status_label)
        
        # 注册按钮
        self.register_btn = QPushButton("注册音色")
        self.register_btn.clicked.connect(self.register_voice_only)
        layout.addWidget(self.register_btn)
        
        layout.addStretch()
        return tab_widget
    
    def create_manage_tab(self):
        """创建音色管理标签页"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        
        # 标题
        title_label = QLabel("音色管理")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 音色列表组
        list_group = QGroupBox("已注册音色列表")
        list_layout = QVBoxLayout(list_group)
        
        # 刷新按钮
        refresh_btn = QPushButton("🔄 刷新列表")
        refresh_btn.clicked.connect(self.refresh_voice_list)
        list_layout.addWidget(refresh_btn)
        
        # 音色列表
        self.voice_list_widget = QListWidget()
        self.voice_list_widget.itemSelectionChanged.connect(self.on_voice_selection_changed)
        list_layout.addWidget(self.voice_list_widget)
        
        layout.addWidget(list_group)
        
        # 选择的音色信息
        info_group = QGroupBox("选择的音色信息")
        info_layout = QGridLayout(info_group)
        
        info_layout.addWidget(QLabel("键名:"), 0, 0)
        self.selected_key_label = QLabel("未选择")
        info_layout.addWidget(self.selected_key_label, 0, 1)
        
        info_layout.addWidget(QLabel("音频文件:"), 1, 0)
        self.selected_source_label = QLabel("未选择")
        info_layout.addWidget(self.selected_source_label, 1, 1)
        
        info_layout.addWidget(QLabel("文本内容:"), 2, 0)
        self.selected_target_label = QLabel("未选择")
        self.selected_target_label.setWordWrap(True)
        info_layout.addWidget(self.selected_target_label, 2, 1)
        
        info_layout.addWidget(QLabel("时长:"), 3, 0)
        self.selected_length_label = QLabel("未选择")
        info_layout.addWidget(self.selected_length_label, 3, 1)
        
        info_layout.addWidget(QLabel("创建时间:"), 4, 0)
        self.selected_time_label = QLabel("未选择")
        info_layout.addWidget(self.selected_time_label, 4, 1)
        
        layout.addWidget(info_group)
        
        # 管理按钮
        btn_layout = QHBoxLayout()
        
        self.play_selected_btn = QPushButton("🔊 播放音频")
        self.play_selected_btn.clicked.connect(self.play_selected_voice)
        self.play_selected_btn.setEnabled(False)
        btn_layout.addWidget(self.play_selected_btn)
        
        self.rename_btn = QPushButton("✏️ 重命名")
        self.rename_btn.clicked.connect(self.rename_voice)
        self.rename_btn.setEnabled(False)
        btn_layout.addWidget(self.rename_btn)
        
        self.delete_btn = QPushButton("🗑️ 删除")
        self.delete_btn.clicked.connect(self.delete_voice)
        self.delete_btn.setEnabled(False)
        btn_layout.addWidget(self.delete_btn)
        
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        return tab_widget
    
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
        
        layout.addLayout(btn_layout)
        
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
            
            # 自动填充音色键名（使用文件名，用户可以修改）
            filename = os.path.splitext(os.path.basename(file_path))[0]
            if not self.voice_key_edit.text():  # 只在为空时自动填充
                self.voice_key_edit.setText(filename)
            
            # 重置音色状态
            self.voice_status_label.setText("音色状态: 未注册")
            self.voice_status_label.setStyleSheet("color: red; font-weight: bold;")
    
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
    
    def register_voice_only(self):
        """仅注册音色"""
        if not self.validate_voice_inputs():
            return
        
        # 创建临时工作线程仅用于注册音色
        self.register_btn.setEnabled(False)
        self.voice_status_label.setText("正在注册音色...")
        self.voice_status_label.setStyleSheet("color: orange; font-weight: bold;")
        
        # 创建工作线程（仅用于注册音色）
        self.worker_thread = QThread()
        self.worker = VoiceRegisterWorker(
            model_dir=self.model_dir_edit.text(),
            prompt_audio_path=self.audio_file_edit.text(),
            prompt_text=self.prompt_text_edit.toPlainText(),
            voice_key=self.voice_key_edit.text()
        )
        
        # 连接信号
        self.worker.status_updated.connect(self.log_text.append)
        self.worker.log_updated.connect(self.log_text.append)
        self.worker.voice_registered.connect(self.voice_registered)
        
        # 移动到线程并启动
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.register_voice)
        self.worker_thread.finished.connect(self.cleanup_worker)
        self.worker_thread.start()
    
    def voice_registered(self, audio_name):
        """音色注册完成"""
        self.voice_status_label.setText(f"音色状态: 已注册 ({audio_name})")
        self.voice_status_label.setStyleSheet("color: green; font-weight: bold;")
        self.register_btn.setEnabled(True)
        self.log_text.append("音色注册完成")
        
        # 刷新音色列表
        self.refresh_voice_list()
    
    def cleanup_worker(self):
        """清理工作线程"""
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None
        self.register_btn.setEnabled(True)
    
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
    
    # === 音色管理相关方法 ===
    def refresh_voice_list(self):
        """刷新音色列表"""
        self.voice_list_widget.clear()
        
        db_clone_dir = os.path.join(_project_root, "BatchGenerate/DB_clone")
        db_clone_jsonl = os.path.join(db_clone_dir, "db_clone.jsonl")
        
        if not os.path.exists(db_clone_jsonl):
            self.log_text.append("数据库文件不存在")
            return
        
        try:
            voices = []
            with open(db_clone_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        voices.append(entry)
            
            for voice in voices:
                item = QListWidgetItem(voice.get('key', 'Unknown'))
                item.setData(Qt.UserRole, voice)  # 存储完整数据
                self.voice_list_widget.addItem(item)
                
            self.log_text.append(f"刷新音色列表完成，共 {len(voices)} 个音色")
            
        except Exception as e:
            self.log_text.append(f"刷新音色列表失败: {str(e)}")
    
    def on_voice_selection_changed(self):
        """音色选择改变"""
        current_item = self.voice_list_widget.currentItem()
        if current_item:
            voice_data = current_item.data(Qt.UserRole)
            self.selected_voice_data = voice_data
            
            # 更新显示信息
            self.selected_key_label.setText(voice_data.get('key', 'N/A'))
            self.selected_source_label.setText(os.path.basename(voice_data.get('source', 'N/A')))
            self.selected_target_label.setText(voice_data.get('target', 'N/A'))
            self.selected_length_label.setText(f"{voice_data.get('source_len', 0)}ms")
            self.selected_time_label.setText(voice_data.get('created_time', 'N/A'))
            
            # 启用按钮
            self.play_selected_btn.setEnabled(True)
            self.rename_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
        else:
            self.selected_voice_data = None
            self.selected_key_label.setText("未选择")
            self.selected_source_label.setText("未选择")
            self.selected_target_label.setText("未选择")
            self.selected_length_label.setText("未选择")
            self.selected_time_label.setText("未选择")
            
            # 禁用按钮
            self.play_selected_btn.setEnabled(False)
            self.rename_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
    
    def play_selected_voice(self):
        """播放选择的音色"""
        if self.selected_voice_data:
            source_path = self.selected_voice_data.get('source', '')
            if os.path.exists(source_path):
                self.media_player.setSource(QUrl.fromLocalFile(source_path))
                self.media_player.play()
                self.log_text.append(f"播放: {self.selected_voice_data.get('key', '')}")
            else:
                QMessageBox.warning(self, "错误", "音频文件不存在！")
    
    def rename_voice(self):
        """重命名音色"""
        if not self.selected_voice_data:
            return
        
        old_key = self.selected_voice_data.get('key', '')
        new_key, ok = QInputDialog.getText(self, "重命名音色", "新的音色名称:", text=old_key)
        
        if ok and new_key.strip() and new_key.strip() != old_key:
            new_key = new_key.strip()
            
            # 检查新键名是否已存在
            if self.is_voice_key_exists(new_key):
                QMessageBox.warning(self, "错误", f"音色键名 '{new_key}' 已存在！")
                return
            
            # 更新数据库
            if self.update_voice_key_in_db(old_key, new_key):
                self.log_text.append(f"音色重命名成功: '{old_key}' -> '{new_key}'")
                self.refresh_voice_list()
            else:
                QMessageBox.warning(self, "错误", "重命名失败！")
    
    def delete_voice(self):
        """删除音色"""
        if not self.selected_voice_data:
            return
        
        voice_key = self.selected_voice_data.get('key', '')
        reply = QMessageBox.question(
            self, "确认删除", 
            f"确定要删除音色 '{voice_key}' 吗？\n\n这将同时删除音频文件和数据库记录。",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.delete_voice_from_db(voice_key):
                self.log_text.append(f"音色删除成功: '{voice_key}'")
                self.refresh_voice_list()
            else:
                QMessageBox.warning(self, "错误", "删除失败！")
    
    def is_voice_key_exists(self, voice_key):
        """检查音色键名是否已存在"""
        db_clone_dir = os.path.join(_project_root, "BatchGenerate/DB_clone")
        db_clone_jsonl = os.path.join(db_clone_dir, "db_clone.jsonl")
        
        if not os.path.exists(db_clone_jsonl):
            return False
        
        try:
            with open(db_clone_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        if entry.get('key') == voice_key:
                            return True
        except Exception:
            pass
        
        return False
    
    def update_voice_key_in_db(self, old_key, new_key):
        """更新数据库中的音色键名"""
        db_clone_dir = os.path.join(_project_root, "BatchGenerate/DB_clone")
        db_clone_jsonl = os.path.join(db_clone_dir, "db_clone.jsonl")
        
        if not os.path.exists(db_clone_jsonl):
            return False
        
        try:
            # 读取所有记录
            voices = []
            with open(db_clone_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        if entry.get('key') == old_key:
                            entry['key'] = new_key
                        voices.append(entry)
            
            # 重写文件
            with open(db_clone_jsonl, 'w', encoding='utf-8') as f:
                for voice in voices:
                    f.write(json.dumps(voice, ensure_ascii=False) + '\n')
            
            return True
            
        except Exception as e:
            self.log_text.append(f"更新数据库失败: {str(e)}")
            return False
    
    def delete_voice_from_db(self, voice_key):
        """从数据库中删除音色"""
        db_clone_dir = os.path.join(_project_root, "BatchGenerate/DB_clone")
        db_clone_jsonl = os.path.join(db_clone_dir, "db_clone.jsonl")
        
        if not os.path.exists(db_clone_jsonl):
            return False
        
        try:
            # 读取所有记录，找到要删除的记录
            voices = []
            deleted_source_path = None
            
            with open(db_clone_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        if entry.get('key') == voice_key:
                            deleted_source_path = entry.get('source')
                        else:
                            voices.append(entry)
            
            # 重写文件
            with open(db_clone_jsonl, 'w', encoding='utf-8') as f:
                for voice in voices:
                    f.write(json.dumps(voice, ensure_ascii=False) + '\n')
            
            # 删除音频文件
            if deleted_source_path and os.path.exists(deleted_source_path):
                try:
                    os.remove(deleted_source_path)
                    self.log_text.append(f"删除音频文件: {deleted_source_path}")
                except Exception as e:
                    self.log_text.append(f"删除音频文件失败: {str(e)}")
            
            return True
            
        except Exception as e:
            self.log_text.append(f"删除音色失败: {str(e)}")
            return False

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    window = VoiceRegisterManagerGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 