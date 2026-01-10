#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from typing import List, Tuple, Optional
from enum import Enum

_TRANSLATIONS = {
    "测试文本：": {"en": "Test text:"},
    "处理结果：": {"en": "Processed result:"},
    "分离结果：": {"en": "Separated result:"},
    "格式化输出：": {"en": "Formatted output:"},
    "只获取拼音：": {"en": "Pinyin only:"},
    "只获取音标：": {"en": "Phonetic only:"},
    "只获取文本（去除拼音和音标）：": {"en": "Text only (remove pinyin and phonetic):"},
    "其他测试用例：": {"en": "Other test cases:"},
    "输入: {text}": {"en": "Input: {text}"},
    "拼音: {pinyin}": {"en": "Pinyin: {pinyin}"},
    "音标: {phonetic}": {"en": "Phonetic: {phonetic}"},
    "纯文本: '{text}'": {"en": "Plain text: '{text}'"},
}


def _t(text: str, **kwargs) -> str:
    lang = os.getenv("HYDRAVOX_LANG", os.getenv("HYDRAVOX_UI_LANG", "zh")).lower()
    if lang not in ("zh", "en"):
        lang = "zh"
    entry = _TRANSLATIONS.get(text)
    result = entry.get(lang, text) if entry else text
    if kwargs:
        try:
            return result.format(**kwargs)
        except Exception:
            return result
    return result


class ContentType(Enum):
    """内容类型枚举"""
    TEXT = "text"  # 普通文本
    PINYIN = "pinyin"  # 拼音
    PHONETIC = "phonetic"  # 音标
    
    
def is_english(content: str) -> bool:
    """
    判断是否为普通英文
    英文特征：常见英文单词或短语，不包含数字声调
    """
    content = content.strip()
    if not content:
        return False
    
    # 常见英文单词列表（可扩展）
    common_english_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'it', 'its',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about',
        'and', 'or', 'but', 'not', 'no', 'yes', 'i', 'you', 'he', 'she', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'our', 'their',
        'what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose',
        'here', 'there', 'now', 'then', 'today', 'tomorrow', 'yesterday',
        'good', 'bad', 'big', 'small', 'new', 'old', 'first', 'last',
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'
    }
    
    # 检查是否只包含英文字母和空格
    if not re.match(r'^[a-zA-Z\s]+$', content):
        return False
    
    # 分词检查
    words = content.lower().split()
    
    # 如果所有单词都是常见英文单词，则判定为英文
    if all(word in common_english_words for word in words):
        return True
    
    # 如果是单个词且长度大于2，可能是英文单词（非拼音）
    if len(words) == 1 and len(words[0]) > 2:
        # 检查是否符合英文单词模式（不是拼音模式）
        # 拼音通常是2-6个字母，英文单词变化更大
        word = words[0]
        # 检查常见英文单词模式
        # 包含常见英文词根、前缀、后缀
        english_patterns = [
            r'.*ing$', r'.*ed$', r'.*er$', r'.*est$', r'.*ly$', r'.*tion$',
            r'.*sion$', r'.*ness$', r'.*ment$', r'.*ful$', r'.*less$',
            r'^un.*', r'^re.*', r'^pre.*', r'^dis.*', r'^mis.*'
        ]
        for pattern in english_patterns:
            if re.match(pattern, word):
                return True
    
    # 如果是短语形式（包含多个单词），且不是拼音组合
    if len(words) > 1:
        # 英文短语通常包含冠词、介词等
        has_article_or_prep = any(word in ['a', 'an', 'the', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'of'] 
                                  for word in words)
        if has_article_or_prep:
            return True
    
    return False


def is_pinyin(content: str) -> bool:
    """
    判断是否为拼音
    拼音特征：字母开头，可能包含数字（声调1-5）
    """
    # 去除首尾空格
    content = content.strip()
    if not content:
        return False
    
    # 如果是普通英文，不是拼音
    if is_english(content):
        return False
    
    # 拼音模式：字母（可能有空格分隔的多个拼音）和数字1-5
    pinyin_pattern = r'^[a-zA-Z]+[1-5]?(\s+[a-zA-Z]+[1-5]?)*$'
    if not re.match(pinyin_pattern, content):
        return False
    
    # 进一步检查是否符合拼音规则
    words = content.split()
    for word in words:
        # 去掉数字声调
        base = re.sub(r'[1-5]$', '', word)
        # 拼音通常是1-6个字母
        if len(base) < 1 or len(base) > 6:
            return False
        # 常见的拼音音节（可以扩展）
        common_pinyin_syllables = {
            'a', 'o', 'e', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'ang', 'eng', 'er',
            'ba', 'pa', 'ma', 'fa', 'da', 'ta', 'na', 'la', 'ga', 'ka', 'ha',
            'za', 'ca', 'sa', 'zha', 'cha', 'sha', 'ra',
            'ji', 'qi', 'xi', 'zi', 'ci', 'si', 'zhi', 'chi', 'shi', 'ri',
            'bo', 'po', 'mo', 'fo', 'wo', 'yo',
            'bi', 'pi', 'mi', 'di', 'ti', 'ni', 'li',
            'bu', 'pu', 'mu', 'fu', 'du', 'tu', 'nu', 'lu', 'gu', 'ku', 'hu',
            'zu', 'cu', 'su', 'zhu', 'chu', 'shu', 'ru',
            'ge', 'ke', 'he', 'ze', 'ce', 'se', 'zhe', 'che', 'she', 're',
            'ben', 'pen', 'men', 'fen', 'den', 'nen', 'gen', 'ken', 'hen', 'zen', 'cen', 'sen',
            'bang', 'pang', 'mang', 'fang', 'dang', 'tang', 'nang', 'lang', 'gang', 'kang', 'hang',
            'beng', 'peng', 'meng', 'feng', 'deng', 'teng', 'neng', 'leng', 'geng', 'keng', 'heng',
            'bin', 'pin', 'min', 'nin', 'lin', 'jin', 'qin', 'xin',
            'bing', 'ping', 'ming', 'ding', 'ting', 'ning', 'ling', 'jing', 'qing', 'xing',
            'jia', 'qia', 'xia', 'yan', 'qian', 'xian', 'jiang', 'qiang', 'xiang',
            'jie', 'qie', 'xie', 'jian', 'qian', 'xian',
            'ju', 'qu', 'xu', 'juan', 'quan', 'xuan',
            'yue', 'yun', 'yuan', 'yin', 'ying', 'yong', 'you',
            'wu', 'wei', 'wen', 'wang', 'weng',
            'lv', 'lve', 'nv', 'nve'
        }
        
        # 如果包含声调数字，说明更可能是拼音
        if re.search(r'[1-5]$', word):
            return True
        
        # 如果在常见拼音音节中，更可能是拼音
        if base.lower() in common_pinyin_syllables:
            return True
    
    return False


def is_phonetic(content: str) -> bool:
    """
    判断是否为音标
    音标特征：包含特殊音标符号或IPA字符
    """
    # 常见的音标符号
    phonetic_chars = [
        'ɪ', 'ə', 'ʊ', 'ɔ', 'æ', 'ʌ', 'ɑ', 'ɛ', 'ɜ', 'ɒ',  # 元音
        'θ', 'ð', 'ʃ', 'ʒ', 'tʃ', 'dʒ', 'ŋ',  # 辅音
        'ˈ', 'ˌ', 'ː',  # 重音和长音符号
    ]
    
    # 检查是否包含音标特征字符
    for char in phonetic_chars:
        if char in content:
            return True
    
    # 检查是否符合简单的英文音标模式（如 saɪn）
    # 包含英文字母和音标符号的组合
    if re.search(r'[a-z]+[ɪəʊɔæʌɑɛɜɒθðʃʒŋ]+|[ɪəʊɔæʌɑɛɜɒθðʃʒŋ]+[a-z]+', content, re.IGNORECASE):
        return True
    
    return False


def parse_text_with_pinyin(text: str) -> Tuple[List[str], List[str]]:
    """
    解析包含拼音/音标的文本
    
    Args:
        text: 输入文本，可能包含括号内的拼音、音标或普通内容
        
    Returns:
        Tuple[List[str], List[str]]: 
            - 第一个列表：分离的文字片段
            - 第二个列表：对应的类型标记
    """
    segments = []
    types = []
    
    i = 0
    current_text = ""
    
    while i < len(text):
        char = text[i]
        
        # 检测到左括号
        if char == '(' or char == '（':
            # 先保存括号前的文本
            if current_text:
                segments.append(current_text)
                types.append(ContentType.TEXT.value)
                current_text = ""
            
            # 找到对应的右括号，使用括号匹配计数
            bracket_depth = 1
            j = i + 1
            while j < len(text) and bracket_depth > 0:
                if text[j] == '(' or text[j] == '（':
                    bracket_depth += 1
                elif text[j] == ')' or text[j] == '）':
                    bracket_depth -= 1
                j += 1
            
            # 检查是否找到了匹配的右括号
            if bracket_depth == 0:
                # 提取括号内的内容（不包括最外层括号）
                bracket_content = text[i+1:j-1]
                
                if bracket_content:
                    # 检查是否有嵌套括号
                    has_nested = '(' in bracket_content or '（' in bracket_content
                    
                    if has_nested:
                        # 如果有嵌套括号，递归处理内容
                        nested_segments, nested_types = parse_text_with_pinyin(bracket_content)
                        # 将递归处理的结果合并
                        segments.extend(nested_segments)
                        types.extend(nested_types)
                    else:
                        # 没有嵌套括号，判断内容类型
                        # 优先级：音标 > 拼音 > 英文 > 普通文本
                        if is_phonetic(bracket_content):
                            segments.append(bracket_content)
                            types.append(ContentType.PHONETIC.value)
                        elif is_pinyin(bracket_content):
                            segments.append(bracket_content) 
                            types.append(ContentType.PINYIN.value)
                        elif is_english(bracket_content):
                            # 英文内容作为普通文本处理
                            segments.append(bracket_content)
                            types.append(ContentType.TEXT.value)
                        else:
                            # 其他内容作为文本处理
                            segments.append(bracket_content)
                            types.append(ContentType.TEXT.value)
                
                # 移动索引到右括号之后
                i = j
            else:
                # 没有找到匹配的右括号，将左括号作为普通文本
                current_text += char
                i += 1
        else:
            # 普通字符，加入当前文本
            current_text += char
            i += 1
    
    # 处理剩余的文本
    if current_text:
        segments.append(current_text)
        types.append(ContentType.TEXT.value)
    
    return segments, types


# 便捷函数
def process_pinyin_text(text: str) -> Tuple[List[str], List[str]]:
    """
    便捷函数：处理包含拼音的文本
    
    Args:
        text: 输入文本
        
    Returns:
        Tuple[List[str], List[str]]: (片段列表, 类型列表)
    """
    return parse_text_with_pinyin(text)


def get_segments_by_type(segments: List[str], types: List[str], content_type: str) -> List[Tuple[int, str]]:
    """
    获取指定类型的片段及其索引
    
    Args:
        segments: 分离的文字片段列表
        types: 对应的类型标记列表
        content_type: 要筛选的内容类型 ("text", "pinyin", "phonetic")
    
    Returns:
        List[Tuple[int, str]]: 符合条件的片段及其索引
    """
    result = []
    for i, (seg, typ) in enumerate(zip(segments, types)):
        if typ == content_type:
            result.append((i, seg))
    return result


def get_text_only(segments: List[str], types: List[str]) -> str:
    """
    只获取文本部分（不包括拼音和音标）
    
    Args:
        segments: 分离的文字片段列表
        types: 对应的类型标记列表
    
    Returns:
        str: 拼接后的纯文本内容
    """
    return ''.join(seg for seg, typ in zip(segments, types) if typ == ContentType.TEXT.value)


def get_pinyin_only(segments: List[str], types: List[str]) -> List[str]:
    """
    只获取拼音部分
    
    Args:
        segments: 分离的文字片段列表
        types: 对应的类型标记列表
    
    Returns:
        List[str]: 拼音片段列表
    """
    return [seg for seg, typ in zip(segments, types) if typ == ContentType.PINYIN.value]


def get_phonetic_only(segments: List[str], types: List[str]) -> List[str]:
    """
    只获取音标部分
    
    Args:
        segments: 分离的文字片段列表
        types: 对应的类型标记列表
    
    Returns:
        List[str]: 音标片段列表
    """
    return [seg for seg, typ in zip(segments, types) if typ == ContentType.PHONETIC.value]


def format_segments(segments: List[str], types: List[str], separator: str = " | ") -> str:
    """
    格式化输出，便于查看
    
    Args:
        segments: 分离的文字片段列表
        types: 对应的类型标记列表
        separator: 分隔符
    
    Returns:
        str: 格式化的字符串
    """
    output_lines = []
    for seg, typ in zip(segments, types):
        output_lines.append(f"[{typ}]: {seg}")
    return separator.join(output_lines)


if __name__ == "__main__":
    # 测试用例
    test_text = """咱们(ke3)以结合正弦函数图象来算。正弦函数(y) 等于 (saɪn)t在t等于 (2分之派)时取到最大值1，
在t等于 (6分之7派)时，(saɪn) (6分之7派)等于至 (2分之1)  ，在t等于 (6分之派)时，(saɪn) (6分之派)等于 (2分之1) 。
所以当t属于 (6分之派)到 (6分之7派)的左闭右开区间时，(saɪn)t的取值范围是至 (2分之1) 到1的闭区间。
那2(saɪn)(2x 加  (6分之派))也就是f(x)的值域是多少呢？"""
    
    print("=" * 80)
    print(_t("测试文本："))
    print(test_text)
    print("=" * 80)
    
    # 处理文本
    print("\n" + _t("处理结果："))
    segments, types = process_pinyin_text(test_text)
    
    print("\n" + _t("分离结果："))
    for i, (seg, typ) in enumerate(zip(segments, types)):
        print(f"{i}: [{typ}] '{seg}'")
    
    print("\n" + _t("格式化输出："))
    print(format_segments(segments, types, "\n"))
    
    print("\n" + _t("只获取拼音："))
    print(get_pinyin_only(segments, types))
    
    print("\n" + _t("只获取音标："))
    print(get_phonetic_only(segments, types))
    
    print("\n" + _t("只获取文本（去除拼音和音标）："))
    print(get_text_only(segments, types))
    
    # 更多测试用例
    print("\n" + "=" * 80)
    print("\n" + _t("其他测试用例："))
    
    test_cases = [
        "你好(ni3 hao3)世界",
        "这是(this is)英文",
        "音标测试(θɪs)和(ðæt)",
        "嵌套测试((内层))外层",
        "混合(wo3)测试(test)内容(kənˈtent)",
        "简单(abc)测试",
        "数学公式f(x)=sin(x)",
    ]
    
    for test in test_cases:
        print("\n" + _t("输入: {text}", text=test))
        segs, typs = process_pinyin_text(test)
        for seg, typ in zip(segs, typs):
            print(f"  [{typ}]: {seg}")
        
        # 演示各种提取功能
        pinyin_list = get_pinyin_only(segs, typs)
        phonetic_list = get_phonetic_only(segs, typs)
        text_only = get_text_only(segs, typs)
        
        if pinyin_list:
            print("  " + _t("拼音: {pinyin}", pinyin=pinyin_list))
        if phonetic_list:
            print("  " + _t("音标: {phonetic}", phonetic=phonetic_list))
        print("  " + _t("纯文本: '{text}'", text=text_only))
