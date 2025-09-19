import re
import regex
import inflect
from wetext import Normalizer

chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')

# 是否包含中文字符
def contains_chinese(text):
    return bool(chinese_char_pattern.search(text))


# 替换特殊符号
def replace_corner_mark(text):
    text = text.replace('²', '平方')
    text = text.replace('³', '立方')
    text = text.replace('√', '根号')
    text = text.replace('≈', '约等于')
    text = text.replace('<', '小于')
    return text


# 移除无意义符号
def remove_bracket(text):
    text = text.replace('（', ' ').replace('）', ' ')
    text = text.replace('【', ' ').replace('】', ' ')
    text = text.replace('`', '').replace('`', '')
    text = text.replace("——", " ")
    return text


# 拼写阿拉伯数字
def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st: i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return ''.join(new_text)


# 分割段落逻辑：
# 1. 每句最大长度 token_max_n，最小长度 token_min_n，如果最后一句长度小于 merge_len 则合并
# 2. 根据语言计算句子长度
# 3. 根据标点分割句子
def split_paragraph(text: str, tokenize, lang="zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False):
    def calc_utt_length(_text: str):
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    if lang == "zh":
        pounc = ['。', '？', '！', '；', '：', '、', '.', '?', '!', ';']
    else:
        pounc = ['.', '?', '!', ';', ':']
    if comma_split:
        pounc.extend(['，', ','])
    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st: i]) > 0:
                utts.append(text[st: i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', '”']:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1
    if len(utts) == 0:
        if lang == "zh":
            utts.append(text + '。')
        else:
            utts.append(text + '.')
    final_utts = []
    cur_utt = ""
    for utt in utts:
        if calc_utt_length(cur_utt + utt) > token_max_n and calc_utt_length(cur_utt) > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts


# 移除中文字符之间的空白
def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if ((text[i + 1].isascii() and text[i + 1] != " ") and
                    (text[i - 1].isascii() and text[i - 1] != " ")):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)

def clean_markdown(md_text: str) -> str:
    # 去除代码块 ``` ```（包括多行）
    md_text = re.sub(r"```.*?```", "", md_text, flags=re.DOTALL)

    # 去除内联代码 `code`
    md_text = re.sub(r"`[^`]*`", "", md_text)

    # 去除图片语法 ![alt](url)
    md_text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", md_text)

    # 去除链接但保留文本 [text](url) -> text
    md_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md_text)
    
    # 替换无序列表符号
    md_text = re.sub(r'^(\s*)-\s+', r'\1', md_text, flags=re.MULTILINE)

    # 去除HTML标签
    md_text = re.sub(r"<[^>]+>", "", md_text)

    # 去除标题符号（#）
    md_text = re.sub(r"^#{1,6}\s*", "", md_text, flags=re.MULTILINE)

    # 去除多余空格和空行
    md_text = re.sub(r"\n\s*\n", "\n", md_text)  # 多余空行
    md_text = md_text.strip()

    return md_text


def clean_text(text):
    # 去除 Markdown 语法
    text = clean_markdown(text)
    # 匹配并移除表情符号
    text = regex.compile(r'\p{Emoji_Presentation}|\p{Emoji}\uFE0F', flags=regex.UNICODE).sub("",text)
    # 去除换行符
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace('"', "\“")
    return text

class TextNormalizer:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.zh_tn_model = Normalizer(lang="zh", operator="tn", remove_erhua=True)
        self.en_tn_model = Normalizer(lang="en", operator="tn")
        self.inflect_parser = inflect.engine()
    
    def normalize(self, text, split=False):
        # 去除 Markdown 语法，去除表情符号，去除换行符
        lang = "zh" if contains_chinese(text) else "en"
        text = clean_text(text)
        if lang == "zh":
            text = text.replace("=", "等于") # 修复 ”550 + 320 等于 870 千卡。“ 被错误正则为 ”五百五十加三百二十等于八七十千卡.“
            if re.search(r'([\d$%^*_+≥≤≠×÷?=])', text): # 避免 英文连字符被错误正则为减
                text = re.sub(r'(?<=[a-zA-Z0-9])-(?=\d)', ' - ', text) # 修复 x-2 被正则为 x负2
            text = self.zh_tn_model.normalize(text)
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = remove_bracket(text)
        else:
            text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)
        if split is False:
            return text