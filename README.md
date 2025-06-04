# LightTTS
基于 CosyVoice 精简开发的语音合成系统。


git clone https://github.com/boomytc/LightTTS.git
cd LightTTS

conda create -n LightTTS python=3.10 -y
conda activate LightTTS
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements_mac.txt

mkdir -p pretrained_models
modelscope download --model iic/CosyVoice2-0.5B --local_dir ./pretrained_models/CosyVoice2-0.5B