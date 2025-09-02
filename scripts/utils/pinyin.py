#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import List, Tuple, Optional
from enum import Enum


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
    
    # 用于处理括号嵌套的栈
    bracket_stack = []
    current_text = ""
    in_bracket = False
    bracket_content = ""
    i = 0
    
    while i < len(text):
        char = text[i]
        
        if char == '(' or char == '（':
            if not in_bracket:
                # 进入括号
                if current_text:
                    segments.append(current_text)
                    types.append(ContentType.TEXT.value)
                    current_text = ""
                in_bracket = True
                bracket_stack.append(i)
            else:
                # 嵌套括号
                bracket_content += char
                bracket_stack.append(i)
            i += 1
            
        elif char == ')' or char == '）':
            if bracket_stack:
                bracket_stack.pop()
                if not bracket_stack:
                    # 最外层括号结束
                    if bracket_content:
                        # 判断括号内容类型，优先级：音标 > 英文 > 拼音 > 普通文本
                        if is_phonetic(bracket_content):
                            segments.append(bracket_content)
                            types.append(ContentType.PHONETIC.value)
                        elif is_english(bracket_content):
                            # 英文内容作为普通文本处理
                            segments.append(bracket_content)
                            types.append(ContentType.TEXT.value)
                        elif is_pinyin(bracket_content):
                            segments.append(bracket_content)
                            types.append(ContentType.PINYIN.value)
                        else:
                            # 其他内容，作为文本处理
                            segments.append(bracket_content)
                            types.append(ContentType.TEXT.value)
                        bracket_content = ""
                    in_bracket = False
                else:
                    # 还在嵌套括号中
                    bracket_content += char
            else:
                # 没有匹配的左括号，当作普通文本
                if in_bracket:
                    bracket_content += char
                else:
                    current_text += char
            i += 1
            
        else:
            if in_bracket:
                bracket_content += char
            else:
                current_text += char
            i += 1
    
    # 处理剩余内容
    if current_text:
        segments.append(current_text)
        types.append(ContentType.TEXT.value)
    if bracket_content and in_bracket:
        # 未闭合的括号内容，当作普通文本
        segments.append(bracket_content)
        types.append(ContentType.TEXT.value)
    
    return segments, types


class PinyinProcessor:
    """拼音处理器类，提供更高级的功能"""
    
    def __init__(self):
        self.segments = []
        self.types = []
    
    def parse(self, text: str) -> 'PinyinProcessor':
        """解析文本"""
        self.segments, self.types = parse_text_with_pinyin(text)
        return self
    
    def get_segments(self) -> List[str]:
        """获取所有片段"""
        return self.segments
    
    def get_types(self) -> List[str]:
        """获取所有类型"""
        return self.types
    
    def get_by_type(self, content_type: ContentType) -> List[Tuple[int, str]]:
        """获取指定类型的片段及其索引"""
        result = []
        for i, (seg, typ) in enumerate(zip(self.segments, self.types)):
            if typ == content_type.value:
                result.append((i, seg))
        return result
    
    def get_text_only(self) -> str:
        """只获取文本部分（不包括拼音和音标）"""
        return ''.join(seg for seg, typ in zip(self.segments, self.types) 
                      if typ == ContentType.TEXT.value)
    
    def get_pinyin_only(self) -> List[str]:
        """只获取拼音部分"""
        return [seg for seg, typ in zip(self.segments, self.types) 
                if typ == ContentType.PINYIN.value]
    
    def get_phonetic_only(self) -> List[str]:
        """只获取音标部分"""
        return [seg for seg, typ in zip(self.segments, self.types) 
                if typ == ContentType.PHONETIC.value]
    
    def format_output(self, separator: str = " | ") -> str:
        """格式化输出，便于查看"""
        output_lines = []
        for seg, typ in zip(self.segments, self.types):
            output_lines.append(f"[{typ}]: {seg}")
        return separator.join(output_lines)


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


if __name__ == "__main__":
    # 测试用例
    test_text = """咱们(ke3)以结合正弦函数图象来算。正弦函数(y) 等于 (saɪn)t在t等于 (2分之派)时取到最大值1，
在t等于 (6分之7派)时，(saɪn) (6分之7派)等于至 (2分之1)  ，在t等于 (6分之派)时，(saɪn) (6分之派)等于 (2分之1) 。
所以当t属于 (6分之派)到 (6分之7派)的左闭右开区间时，(saɪn)t的取值范围是至 (2分之1) 到1的闭区间。
那2(saɪn)(2x 加  (6分之派))也就是f(x)的值域是多少呢？"""
    
    print("=" * 80)
    print("测试文本：")
    print(test_text)
    print("=" * 80)
    
    # 使用函数方式
    print("\n使用函数方式处理：")
    segments, types = process_pinyin_text(test_text)
    
    print("\n分离结果：")
    for i, (seg, typ) in enumerate(zip(segments, types)):
        print(f"{i}: [{typ}] '{seg}'")
    
    print("\n" + "=" * 80)
    
    # 使用类方式
    print("\n使用类方式处理：")
    processor = PinyinProcessor()
    processor.parse(test_text)
    
    print("\n格式化输出：")
    print(processor.format_output("\n"))
    
    print("\n只获取拼音：")
    print(processor.get_pinyin_only())
    
    print("\n只获取音标：")
    print(processor.get_phonetic_only())
    
    print("\n只获取文本（去除拼音和音标）：")
    print(processor.get_text_only())
    
    # 更多测试用例
    print("\n" + "=" * 80)
    print("\n其他测试用例：")
    
    test_cases = [
        "你好(ni3 hao3)世界",
        "这是(this is)英文",
        "音标测试(θɪs)和(ðæt)",
        "嵌套测试((内层))外层",
        "混合(wo3)测试(test)内容(kənˈtent)",
    ]
    
    for test in test_cases:
        print(f"\n输入: {test}")
        segs, typs = process_pinyin_text(test)
        for seg, typ in zip(segs, typs):
            print(f"  [{typ}]: {seg}")