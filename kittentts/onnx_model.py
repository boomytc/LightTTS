import numpy as np
import phonemizer
import soundfile as sf
import onnxruntime as ort

def basic_english_tokenize(text):
    # 基本的英语分词器，按空白和标点符号分割。
    import re
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens

class TextCleaner:
    def __init__(self, dummy=None):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        
        dicts = {}
        for i in range(len(symbols)):
            dicts[symbols[i]] = i

        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
        return indexes

class KittenTTS_Onnx:
    def __init__(self, model_path="kitten_tts_nano_preview.onnx", voices_path="voices.npz"):
        """使用模型和语音数据初始化KittenTTS。
        
        参数：
            model_path: ONNX模型文件的路径
            voices_path: 语音NPZ文件的路径
        """
        self.model_path = model_path
        self.voices = np.load(voices_path)
        self.session = ort.InferenceSession(model_path)
        
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        self.text_cleaner = TextCleaner()
        
        # Available voices
        self.available_voices = [
            'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f', 
            'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f'
        ]
    
    def _prepare_inputs(self, text: str, voice: str, speed: float = 1.0) -> dict:
        # 准备ONNX模型输入
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not available. Choose from: {self.available_voices}")
        
        # 对输入文本进行音素化
        phonemes_list = self.phonemizer.phonemize([text])
        
        # 处理音素以获取令牌ID
        phonemes = basic_english_tokenize(phonemes_list[0])
        phonemes = ' '.join(phonemes)
        tokens = self.text_cleaner(phonemes)
        
        # 添加开始和结束令牌
        tokens.insert(0, 0)
        tokens.append(0)
        
        input_ids = np.array([tokens], dtype=np.int64)
        ref_s = self.voices[voice]
        
        return {
            "input_ids": input_ids,
            "style": ref_s,
            "speed": np.array([speed], dtype=np.float32),
        }
    
    def generate(self, text: str, voice: str = "expr-voice-5-m", speed: float = 1.0) -> np.ndarray:
        """从文本合成语音。
        
        参数：
            text: 要合成的输入文本
            voice: 用于合成的语音
            speed: 语音速度（1.0 = 正常）
            
        返回：
            音频数据作为numpy数组
        """
        onnx_inputs = self._prepare_inputs(text, voice, speed)
        
        outputs = self.session.run(None, onnx_inputs)
        
        # Trim audio
        audio = outputs[0][5000:-10000]

        return audio
    
    def generate_to_file(self, text: str, output_path: str, voice: str = "expr-voice-5-m", 
                          speed: float = 1.0, sample_rate: int = 24000) -> None:
        """合成语音并保存到文件。
        
        参数：
            text: 要合成的输入文本
            output_path: 保存音频文件的路径
            voice: 用于合成的语音
            speed: 语音速度（1.0 = 正常）
            sample_rate: 音频采样率
        """
        audio = self.generate(text, voice, speed)
        sf.write(output_path, audio, sample_rate)
        print(f"Audio saved to {output_path}")

# Example usage
if __name__ == "__main__":
    tts = KittenTTS_Onnx()
    
    text = """
    It begins with an "Ugh!" Another mysterious stain appears on a favorite shirt. Every trick has been tried, but the stain persists.
    """

    tts.generate_to_file(text, "inference_output25.wav", voice="expr-voice-5-m")