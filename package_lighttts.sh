#!/bin/bash

# LightTTS 项目打包脚本
# 用法: ./package_lighttts.sh <输出目录>

set -e  # 遇到错误时退出

# 检查参数
if [ $# -ne 1 ]; then
    echo "用法: $0 <输出目录>"
    echo "示例: $0 /path/to/output"
    exit 1
fi

OUTPUT_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEMP_DIR="/tmp/lighttts_package_${TIMESTAMP}"

# 创建临时目录和输出目录
mkdir -p "$TEMP_DIR"
mkdir -p "$OUTPUT_DIR"

echo "🚀 开始打包 LightTTS 项目..."
echo "📁 项目目录: $SCRIPT_DIR"
echo "📦 输出目录: $OUTPUT_DIR"
echo "🔧 临时目录: $TEMP_DIR"

# 步骤1: 打包项目文件 (压缩包A)
echo "\n📦 步骤1: 打包项目文件..."
cd "$SCRIPT_DIR"

# 检查必要的文件夹是否存在
for folder in "cosyvoice" "third_party"; do
    if [ ! -d "$folder" ]; then
        echo "❌ 错误: 找不到文件夹 $folder"
        exit 1
    fi
done

# 打包项目文件
echo "正在压缩 cosyvoice, pretrained_models, third_party, webui_cosyvoice2.py, run_webui.sh..."
PROJECT_ARCHIVE="$TEMP_DIR/lighttts_project.zip"

# 检查必要的文件是否存在
for file in "webui_cosyvoice2.py" "run_webui.sh"; do
    if [ ! -f "$file" ]; then
        echo "❌ 错误: 找不到文件 $file"
        exit 1
    fi
done

# 创建临时文件列表
FILES_TO_ZIP="cosyvoice third_party webui_cosyvoice2.py run_webui.sh"
if [ -d "pretrained_models" ]; then
    FILES_TO_ZIP="$FILES_TO_ZIP pretrained_models"
fi

# 使用zip压缩，排除不需要的文件
zip -r "$PROJECT_ARCHIVE" $FILES_TO_ZIP \
    -x "*.pyc" "*/__pycache__/*" "*/.DS_Store" "*/.*"

echo "✅ 项目文件打包完成: $(basename "$PROJECT_ARCHIVE")"
echo "📊 文件大小: $(du -h "$PROJECT_ARCHIVE" | cut -f1)"

# 步骤2: 使用conda-pack打包环境 (压缩包B)
echo "\n🐍 步骤2: 打包conda环境..."

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 找不到conda命令"
    exit 1
fi

# 初始化conda
eval "$(conda shell.bash hook)"

# 检查LightTTS环境是否存在
if ! conda env list | grep -q "LightTTS"; then
    echo "❌ 错误: 找不到conda环境 'LightTTS'"
    echo "请先运行 ./run_webui.sh 创建环境"
    exit 1
fi

# 激活环境并安装conda-pack
echo "正在激活LightTTS环境..."
conda activate LightTTS

# 检查并安装conda-pack
if ! conda list | grep -q "conda-pack"; then
    echo "正在安装conda-pack..."
    conda install -y conda-pack
fi

# 打包conda环境
echo "正在打包conda环境..."
ENV_ARCHIVE="$TEMP_DIR/lighttts_env.tar.gz"
conda pack -n LightTTS -o "$ENV_ARCHIVE" --ignore-missing-files

# 转换为zip格式以提高跨平台兼容性
ENV_ZIP="$TEMP_DIR/lighttts_env.zip"
echo "正在转换环境包为zip格式..."
(cd "$TEMP_DIR" && tar -xzf lighttts_env.tar.gz && zip -r lighttts_env.zip * && rm -rf bin lib include share ssl etc)
ENV_ARCHIVE="$ENV_ZIP"

echo "✅ conda环境打包完成: $(basename "$ENV_ARCHIVE")"
echo "📊 文件大小: $(du -h "$ENV_ARCHIVE" | cut -f1)"

# 步骤3: 合并打包 (压缩包C)
echo "\n📦 步骤3: 创建最终部署包..."

# 创建部署脚本
DEPLOY_SCRIPT="$TEMP_DIR/deploy.sh"
cat > "$DEPLOY_SCRIPT" << 'EOF'
#!/bin/bash

# LightTTS 部署脚本
# 在目标机器上运行此脚本来部署LightTTS

set -e

echo "🚀 开始部署 LightTTS..."

# 获取脚本所在目录
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📁 部署目录: $DEPLOY_DIR"

# 解压项目文件
echo "\n📦 解压项目文件..."
if [ -f "lighttts_project.zip" ]; then
    unzip -q lighttts_project.zip
    echo "✅ 项目文件解压完成"
else
    echo "❌ 错误: 找不到 lighttts_project.zip"
    exit 1
fi

# 解压conda环境
echo "\n🐍 解压conda环境..."
if [ -f "lighttts_env.zip" ]; then
    mkdir -p lighttts_env
    unzip -q lighttts_env.zip -d lighttts_env
    echo "✅ conda环境解压完成"
else
    echo "❌ 错误: 找不到 lighttts_env.zip"
    exit 1
fi

# 创建启动脚本
echo "\n📝 创建启动脚本..."
cat > start_lighttts.sh << 'STARTEOF'
#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 激活conda环境
echo "🐍 激活LightTTS环境..."
source "$SCRIPT_DIR/lighttts_env/bin/activate"

# 设置PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/third_party/Matcha-TTS:$SCRIPT_DIR:$PYTHONPATH"

# 启动WebUI
echo "🌐 启动LightTTS WebUI..."
cd "$SCRIPT_DIR"
# 检查预训练模型是否存在
if [ ! -d "./pretrained_models/CosyVoice2-0.5B" ]; then
    echo "预训练模型不存在，开始下载..."
    modelscope download --model iic/CosyVoice2-0.5B --local_dir ./pretrained_models/CosyVoice2-0.5B
    echo "模型下载完成"
else
    echo "预训练模型已存在"
fi

python webui_cosyvoice2.py
STARTEOF

chmod +x start_lighttts.sh

echo "\n✅ LightTTS 部署完成!"
echo "\n🚀 使用方法:"
echo "   ./start_lighttts.sh"
echo "\n📁 项目结构:"
echo "   ├── cosyvoice/          # 核心代码"
echo "   ├── pretrained_models/  # 预训练模型 (如果存在)"
echo "   ├── third_party/        # 第三方依赖"
echo "   ├── webui_cosyvoice2.py # WebUI主程序"
echo "   ├── run_webui.sh        # 原始启动脚本"
echo "   ├── lighttts_env/       # Python环境"
echo "   └── start_lighttts.sh   # 启动脚本"
EOF

chmod +x "$DEPLOY_SCRIPT"

# 创建README文件
README_FILE="$TEMP_DIR/README.md"
cat > "$README_FILE" << 'EOF'
# LightTTS 部署包

这是一个完整的 LightTTS 部署包，包含了所有必要的代码和依赖环境。

## 系统要求

- macOS (与打包环境相同的操作系统)
- 无需预装 Python 或 conda

## 部署步骤

1. **解压部署包**
   ```bash
   unzip lighttts_deployment_YYYYMMDD_HHMMSS.zip
   cd lighttts_deployment_YYYYMMDD_HHMMSS
   ```

2. **运行部署脚本**
   ```bash
   ./deploy.sh
   ```

3. **启动应用**
   ```bash
   ./start_lighttts.sh
   ```

## 包含内容

- `lighttts_project.zip` - 项目源代码和模型
- `lighttts_env.zip` - 完整的Python环境
- `deploy.sh` - 自动部署脚本
- `README.md` - 本说明文件

## 注意事项

- 首次启动可能需要下载预训练模型
- 确保有足够的磁盘空间 (建议至少5GB)
- 如遇问题，请检查终端输出的错误信息

## 项目结构

部署后的目录结构：
```
├── cosyvoice/          # 核心代码
├── pretrained_models/  # 预训练模型
├── third_party/        # 第三方依赖
├── webui_cosyvoice2.py # WebUI主程序
├── run_webui.sh        # 原始启动脚本
├── lighttts_env/       # Python环境
└── start_lighttts.sh   # 启动脚本
```
EOF

# 创建最终部署包
FINAL_PACKAGE="$OUTPUT_DIR/lighttts_deployment_${TIMESTAMP}.zip"
echo "正在创建最终部署包..."
cd "$TEMP_DIR"
zip "$FINAL_PACKAGE" \
    lighttts_project.zip \
    lighttts_env.zip \
    deploy.sh \
    README.md

echo "✅ 最终部署包创建完成: $(basename "$FINAL_PACKAGE")"
echo "📊 文件大小: $(du -h "$FINAL_PACKAGE" | cut -f1)"

# 清理临时文件
echo "\n🧹 清理临时文件..."
rm -rf "$TEMP_DIR"

echo "\n🎉 打包完成!"
echo "📦 部署包位置: $FINAL_PACKAGE"
echo "📋 部署包内容:"
echo "   ├── lighttts_project.zip     # 项目文件 (包含cosyvoice, third_party, webui_cosyvoice2.py, run_webui.sh等)"
echo "   ├── lighttts_env.zip         # conda环境"
echo "   ├── deploy.sh                # 部署脚本"
echo "   └── README.md                # 说明文档"
echo "\n🚀 在目标机器上的使用方法:"
echo "   1. unzip $(basename "$FINAL_PACKAGE")"
echo "   2. cd $(basename "$FINAL_PACKAGE" .zip)"
echo "   3. ./deploy.sh"
echo "   4. ./start_lighttts.sh"