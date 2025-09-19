from typing import List
import torch
from transformers import PreTrainedTokenizer


def mask_multichar_chinese_tokens(tokenizer: PreTrainedTokenizer):
    """创建一个 tokenizer 包装器，将多字符中文 token 转换为单字符。
    
    该函数为提供的 tokenizer 创建包装器，自动将多字符中文 token
    拆分为单个字符。这有助于确保中文文本的 token 化保持一致。
    
    参数:
        tokenizer: 需要包装的基础 tokenizer
        
    返回:
        处理多字符中文 token 的 CharTokenizerWrapper 实例
        
    示例:
        >>> from transformers import LlamaTokenizerFast
        >>> tokenizer = LlamaTokenizerFast.from_pretrained("path/to/tokenizer")
        >>> wrapped_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        >>> tokens = wrapped_tokenizer("你好世界")
    """
    # 预先计算多字符 token（长度 >= 2，纯中文字符）
    multichar_tokens = {
        token for token in tokenizer.vocab.keys() 
        if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }

    class CharTokenizerWrapper:
        """处理多字符中文 token 的 tokenizer 包装类。
    
    该包装类会自动将多字符中文 token 拆分为单字符，同时保持
    原始 tokenizer 的接口不变。
    """
        
        def __init__(self, base_tokenizer: PreTrainedTokenizer) -> None:
            """使用基础 tokenizer 初始化包装器。
            
            参数:
                base_tokenizer: 需要包装的 tokenizer
            """
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def tokenize(self, text: str, **kwargs) -> List[str]:
            """对文本进行 token 化，并将多字符中文 token 拆分为单字符。
            
            参数:
                text: 待 token 化的文本
                **kwargs: 传递给基础 tokenizer 的额外参数
                
            返回:
                已处理的 token 列表，已将多字符中文 token 拆分
                
            示例:
                >>> wrapper = CharTokenizerWrapper(tokenizer)
                >>> tokens = wrapper.tokenize("你好世界")
                >>> # 返回 ["你", "好", "世", "界"] 而不是 ["你好", "世界"]
            """
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")
                
            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed = []
            
            for token in tokens:
                # 移除可能的子词前缀
                clean_token = token.replace("▁", "")

                if clean_token in self.multichar_tokens:
                    # 将多字符 token 拆分为单个字符
                    chars = list(clean_token)
                    processed.extend(chars)
                else:
                    processed.append(token)
                    
            return processed

        def __call__(self, text: str, **kwargs) -> List[int]:
            """调用 tokenizer 并返回 token ID。
            
            该方法保持原始 tokenizer 的接口，但加入了多字符中文 token 的处理。
            
            参数:
                text: 待 token 化的文本
                **kwargs: 传递给基础 tokenizer 的额外参数
                
            返回:
                token ID 列表
                
            异常:
                TypeError: 当输入不是字符串时抛出
                ValueError: 当 token 化失败时抛出
            """
            try:
                tokens = self.tokenize(text, **kwargs)
                result = self.tokenizer.convert_tokens_to_ids(tokens)
                return result
            except Exception as e:
                raise ValueError(f"Tokenization failed: {str(e)}") from e

    return CharTokenizerWrapper(tokenizer)


def get_dtype(dtype: str):
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "bf16":
        return torch.bfloat16
    elif dtype == "float16":
        return torch.float16
    elif dtype == "fp16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    elif dtype == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
