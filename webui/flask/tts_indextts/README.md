# IndexTTS Flask WebUI

基于 Flask 实现的 IndexTTS2 语音合成 Web 界面。

## 功能特性

- 支持 CUDA/CPU 设备选择
- 多种情感控制模式：
  - 无情感控制
  - 情感参考音频
  - 情感向量（8维精细控制）
  - 情感文本引导
- 高级参数调节：
  - 句子间静音时长
  - 每段最大 Token 数
  - 随机性开关
- 实时音频播放和下载

## 运行方式

### 开发模式

```bash
cd /media/fl01/data02/WorkSpace/LightTTS/webui/flask/tts_indextts
/home/fl01/miniconda3/envs/LightTTS/bin/python app.py
```

然后访问: http://127.0.0.1:5001

### 生产模式

使用 Gunicorn:

```bash
gunicorn -w 4 -b 127.0.0.1:5001 app:app
```

## 使用说明

1. **加载模型**
   - 选择运行设备（CUDA 或 CPU）
   - 点击"加载模型"按钮

2. **配置参数**
   - 输入待合成文本
   - 上传说话人参考音频（可选，默认使用预设音频）
   - 选择情感控制模式并配置相应参数

3. **生成语音**
   - 点击"生成语音"按钮
   - 等待生成完成
   - 播放或下载生成的音频

## 技术架构

- 后端：Flask
- 前端：HTML5 + CSS3 + JavaScript
- 音频处理：torchaudio
- 数据传输：Base64 编码

## 目录结构

```
tts_indextts/
├── app.py                    # Flask 应用主文件
├── templates/                # 模板目录
│   └── index.html           # 主页面模板
├── static/                   # 静态资源目录
│   ├── css/
│   │   └── style.css        # 样式文件
│   └── js/
│       └── script.js        # 前端交互脚本
└── README.md                # 本文件
```

## API 接口

### GET /
返回主页面

### POST /api/load_model
加载 IndexTTS2 模型

请求体:
```json
{
    "device": "cuda"  // 或 "cpu"
}
```

### POST /api/generate
生成语音

请求: multipart/form-data
- text: 待合成文本
- emo_mode: 情感模式 (none/audio/vector/text)
- prompt_audio: 说话人参考音频（可选）
- emo_audio: 情感参考音频（audio 模式）
- emo_vector: 情感向量 JSON（vector 模式）
- emo_text: 情感引导文本（text 模式）
- emo_alpha: 情感权重
- interval_silence: 静音时长
- max_tokens: 最大 Token 数
- use_random: 是否启用随机性

### GET /api/default_audio
获取默认参考音频

## 注意事项

1. 确保模型文件位于 `models/IndexTTS-2/` 目录
2. 确保配置文件位于 `models/IndexTTS-2/config.yaml`
3. 默认参考音频位于 `asset/zero_shot_prompt.wav`
4. 上传文件大小限制为 50MB

## 与 Gradio 版本的差异

| 特性 | Gradio 版本 | Flask 版本 |
|------|-------------|-----------|
| 框架 | Gradio | Flask + HTML/CSS/JS |
| UI 定制 | 有限 | 完全自定义 |
| API 接口 | 自动生成 | 手动定义 |
| 部署方式 | 内置服务器 | 可集成到现有系统 |
| 端口 | 7860 | 5001 |
