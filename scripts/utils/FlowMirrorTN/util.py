import re
import os
from tqdm import tqdm
import json

def detect_english_phrases(text):
    """
    检测文本中的连续英文单词
    
    Args:
        text (str): 输入的文本
    
    Returns:
        tuple: 包含两个列表:
            - positions: 列表，每个元素是元组 (start, end)
            - phrases: 列表，包含对应的英文短语
    
    Examples:
        >>> detect_english_phrases("今天讨论global warming和weather")
        ([(4, 17)], ['global warming'])
    """
    positions = []
    phrases = []
    i = 0
    text_len = len(text)
    
    white_list = [':',"-"]

    def is_english_char(char):
        return char.isascii() and (char.isalpha() or char in white_list)
    
    def is_word_boundary(char):
        return not (char.isspace() or is_english_char(char) or char == "'")
    
    while i < text_len:
        # 找到第一个英文字符
        if not is_english_char(text[i]):
            i += 1
            continue
            
        # 找到单词的起始位置
        start = i
        
        # 找到第一个单词的结束位置
        while i < text_len and is_english_char(text[i]):
            i += 1
            
        # 检查是否有后续单词
        has_next_word = False
        next_pos = i
        
        # 跳过空格
        while next_pos < text_len and (text[next_pos].isspace() or text[next_pos] == "'"):
            next_pos += 1
            
        # 检查空格后是否还有英文字符
        if next_pos < text_len and is_english_char(text[next_pos]):
            has_next_word = True
            i = next_pos  # 继续处理下一个单词
        
        # 如果有多个单词，继续查找直到遇到边界
        if has_next_word:
            while i < text_len:
                # 跳过当前单词
                while i < text_len and is_english_char(text[i]):
                    i += 1
                    
                # 如果遇到边界字符，结束查找
                if i >= text_len or is_word_boundary(text[i]):
                    break
                    
                # 跳过空格
                while i < text_len and text[i].isspace() or text[i]=="'":
                    i += 1
                    
                # 如果空格后不是英文字符，结束查找
                if i >= text_len or not is_english_char(text[i]):
                    break
            
            # 添加找到的短语
            positions.append((start, i))
            phrases.append(text[start:i])
        else:
            # 单个单词，继续下一个位置的查找
            continue
            
    return positions, phrases

def protect_alphabet(text):
    """
    保护前后是汉字的单个字母前面不出现标点，但保留多个字母之间的标点
    
    Args:
        text (str): 输入的文本
    
    Returns:
        str: 处理后的文本
    
    Examples:
        >>> protect_alphabet("看一看,X减一分之X")
        "看一看X减一分之X"
        >>> protect_alphabet("P,M,N")
        "P,M,N"
    """
    def is_chinese(char):
        """判断一个字符是否是汉字"""
        if char:
            return '\u4e00' <= char <= '\u9fff'
        return False
    
    def should_remove_punctuation(prev_char, curr_char, next_char):
        """判断是否应该删除标点"""
        # 如果前后都是汉字，且当前字符是单个字母，则删除标点
        return (is_chinese(prev_char) and 
                len(curr_char) == 1 and 
                curr_char.isalpha() and 
                is_chinese(next_char))
    
    def process_match(match):
        """处理每个匹配到的部分"""
        full_match = match.group(0)
        punctuation = match.group(1)  # 标点符号
        letter = match.group(2)       # 字母
        
        # 获取前后文的字符
        start_pos = match.start()
        end_pos = match.end()
        
        # 获取前一个字符（如果存在）
        prev_char = text[start_pos - 1] if start_pos > 0 else None
        # 获取后一个字符（如果存在）
        next_char = text[end_pos] if end_pos < len(text) else None
        
        # 检查是否应该移除标点
        if should_remove_punctuation(prev_char, letter, next_char):
            return letter
        return full_match

    # 匹配标点+字母的模式
    # 使用更全面的标点符号列表
    punctuation_pattern = r'([,，.。!！?？:：;；、\s])([A-Z])'
    
    # 应用替换
    processed_text = re.sub(punctuation_pattern, process_match, text)
    
    return processed_text

def process_chinese_slash(text):
    """处理汉字之间的斜杠，将其替换为"每"
    例如：把"千米/小时"转换为"千米每小时"
    """
    
    def is_chinese(char):
        """判断一个字符是否是汉字"""
        return '\u4e00' <= char <= '\u9fff'
    
    def replace_slash(match):
        """替换函数：检查斜杠前后是否都是汉字"""
        before = match.group(1)[-1]  # 斜杠前的最后一个字符
        after = match.group(2)[0]    # 斜杠后的第一个字符
        if is_chinese(before) and is_chinese(after):
            return match.group(1) + "每" + match.group(2)
        return match.group(0)  # 如果不满足条件，保持原样
    
    # 使用正则表达式查找斜杠及其前后的文本
    pattern = r'([^/]+)/([^/]+)'
    processed_text = re.sub(pattern, replace_slash, text)
    
    return processed_text

def load_symbol_dict(dict_file_path):
    """从.dic文件加载符号字典"""
    symbol_dict = {}
    try:
        with open(dict_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 确保行不是空的
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        symbol_dict[parts[0]] = parts[1]
        print(f"Successfully loaded {len(symbol_dict)} symbols from dictionary")
    except Exception as e:
        print(f"Error loading dictionary file: {e}")
        return {}
    return symbol_dict

def process_fractions(text):
    """处理文本中的各类分数表达式"""
    
    # 定义特殊词组
    SPECIAL_TERMS = ['根号', '正负', '负正', '求和', '求积', '点','π']
    
    def find_matching_parenthesis(text, pos, direction='right'):
        """找到与pos位置的括号匹配的另一个括号
        direction: 'right' 向右找右括号，'left' 向左找左括号
        """
        if direction == 'right':
            count = 1
            i = pos + 1
            while i < len(text) and count > 0:
                if text[i] == '(':
                    count += 1
                elif text[i] == ')':
                    count -= 1
                i += 1
            return i - 1 if count == 0 else -1
        else:  # left
            count = 1
            i = pos - 1
            while i >= 0 and count > 0:
                if text[i] == ')':
                    count += 1
                elif text[i] == '(':
                    count -= 1
                i -= 1
            return i + 1 if count == 0 else -1

    def is_valid_char(char):
        """判断单个字符是否为合法字符（数字或英文字母）"""
        if char.isdigit() or char == '.':
            return True
        if char.isascii():
            return char.isalpha()
        return False

    def get_special_term(text, pos, direction='right'):
        """检查在pos位置是否是特殊词组的一部分"""
        if direction == 'right':
            for term in SPECIAL_TERMS:
                if text[pos:].startswith(term):
                    return True, len(term), True, pos
                if any(term.startswith(text[pos:pos+i]) for i in range(1, len(term)+1)):
                    return True, 1, False, pos
        else:  # left
            for term in SPECIAL_TERMS:
                # 检查完整词组
                start_pos = pos - len(term) + 1
                if start_pos >= 0 and text[start_pos:pos+1] == term:
                    return True, len(term), True, start_pos
                
                # 关键修改：检查当前位置是否是某个词组的一部分
                # 向后看是否能构成完整词组
                for i in range(1, len(term)):  # 注意这里改成了len(term)
                    if pos + i <= len(text):
                        current_text = text[pos:pos+i]
                        if any(term.startswith(current_text) for term in SPECIAL_TERMS):
                            # 是词组的开始部分
                            return True, 2, True, pos  # 修改这里返回True表示是有效词组
                
                # 检查是否是词组的后半部分
                for i in range(1, len(term)+1):
                    if pos - i + 1 >= 0:
                        current_text = text[pos-i+1:pos+1]
                        if any(term.endswith(current_text) for term in SPECIAL_TERMS):
                            return True, i, False, pos-i+1

        return False, 0, False, -1

    def find_next_term(text, pos, direction='right'):
        """从pos位置开始向指定方向寻找下一个合法词组"""
        # print(f"\nfind_next_term 开始: pos={pos}, direction={direction}, 当前字符='{text[pos] if pos < len(text) else 'EOF'}'")
        
        if direction == 'right':
            if pos >= len(text):
                # print("已到达文本末尾")
                return 0, False, -1
                
            is_special, term_len, complete, start_pos = get_special_term(text, pos, 'right')
            # print(f"special term 检查结果: is_special={is_special}, term_len={term_len}, complete={complete}, start_pos={start_pos}")
            
            if is_special:
                if complete:
                    return term_len, True, start_pos
                return 0, False, -1
                
            # if text[pos] == '(':
            #     return 1, True, pos
                
            if not is_valid_char(text[pos]):
                # print(f"字符 '{text[pos]}' 不是有效字符")
                return 0, False, -1
                
            if text[pos].isdigit() or text[pos] == '.':
                i = pos + 1
                while i < len(text) and (text[i].isdigit() or text[i] == '.'):
                    i += 1
                # print(f"找到数字序列，长度: {i - pos}")
                return i - pos, True, pos
                
            if text[pos].isascii() and text[pos].isalpha():
                # print(f"找到字母: {text[pos]}")
                return 1, True, pos
                
            return 0, False, -1
            
        else:  # left
            if pos < 0:
                return 0, False, -1
                
            is_special, term_len, complete, start_pos = get_special_term(text, pos, 'left')
            # print('is_special', is_special, 'term_len', term_len, 'complete', complete, 'start_pos', start_pos)
            if is_special:
                # print('found special term', text[start_pos:start_pos+term_len])
                if complete:
                    return term_len, True, start_pos
                return 0, False, -1
                    
            if text[pos] == ')':
                return 1, True, pos
                
            if not is_valid_char(text[pos]):
                return 0, False, -1
                
            if text[pos].isdigit() or text[pos] == '.':
                i = pos - 1
                while i >= 0 and (text[i].isdigit() or text[i] == '.'):
                    i -= 1
                return pos - i, True, i + 1
                
            if text[pos].isascii() and text[pos].isalpha():
                return 1, True, pos
                
            return 0, False, -1

    def find_fraction_bounds(text, slash_pos):
        """找到分数的完整边界"""
        # print(f"\n开始处理分数，原始文本: {text}")
        # print(f"斜杠位置: {slash_pos}, 斜杠字符: {text[slash_pos]}")
        
        left = slash_pos - 1
        right = slash_pos + 1
        
        # 跳过空格
        while left >= 0 and text[left].isspace():
            left -= 1
        while right < len(text) and text[right].isspace():
            right += 1
        
        # 检查是否是括号相关的分数
        is_left_bracket_fraction = False
        is_right_bracket_fraction = False
        if left >= 0 and text[left] == ')':  # 情况1：(a+b)/c
            is_left_bracket_fraction = True
        if right < len(text) and text[right] == '(':  # 情况2：a/(b+c)
            is_right_bracket_fraction = True
        
        if right < len(text):
            if is_right_bracket_fraction and text[right] == '(':
                # 只有确实是括号分数时才进行括号匹配
                right_end = find_matching_parenthesis(text, right, 'right')
                # print(f"找到右括号位置: {right_end}")
                if right_end != -1:
                    right = right_end + 1
            else:
                # 否则正常检测边界
                current_pos = right
                while current_pos < len(text):
                    term_len, is_valid, _ = find_next_term(text, current_pos, 'right')
                    # print(f"右边界检查: pos={current_pos}, current_char={text[current_pos]}, len={term_len}, valid={is_valid}")
                    if not is_valid:
                        break
                    current_pos += term_len
                right = current_pos
        
        if left >= 0:
            if is_left_bracket_fraction and text[left] == ')':
                # 只有确实是括号分数时才进行括号匹配
                left_start = find_matching_parenthesis(text, left, 'left')
                # print(f"找到左括号位置: {left_start}")
                if left_start != -1:
                    left = left_start
            else:
                # 否则正常检测边界
                min_left = left
                current_pos = left
                while current_pos >= 0:
                    term_len, is_valid, start_pos = find_next_term(text, current_pos, 'left')
                    # print(f"左边界检查: pos={current_pos}, current_char={text[current_pos]}, len={term_len}, valid={is_valid}, start={start_pos}")
                    if not is_valid:
                        break
                    if start_pos >= 0:
                        min_left = min(min_left, start_pos)
                    current_pos -= 1
                left = min_left
        
        # print(f"最终边界: left={left}, right={right}")
        # print(f"提取的分数文本: {text[left:right]}")
        return left, right
    
    def process_single_fraction(fraction):
        """处理单个分数表达式"""
        num, den = fraction.split('/')
        return f"{den.strip()}分之{num.strip()}"
    
    # 主处理流程
    fractions = []
    i = 0
    while i < len(text):
        slash_pos = text.find('/', i)
        if slash_pos == -1:
            break
            
        left, right = find_fraction_bounds(text, slash_pos)
        if left < slash_pos < right:
            fractions.append((left, right))
        i = slash_pos + 1
    
    result = text
    for start, end in sorted(fractions, reverse=True):
        fraction = result[start:end]
        if '/' in fraction:
            converted = process_single_fraction(fraction)
            result = result[:start] + ',' + converted + ',' + result[end:]
    
    return result

def process_special_symbols(text):
    """处理特殊符号，如百分比等"""
    # 存储所有的替换规则
    replacements = []

    # 处理序号
    def number_marker_handler(match):
        number = match.group(1)
        # 将阿拉伯数字转换为中文数字
        number_map = {
            '1': '一', '2': '二', '3': '三', '4': '四', '5': '五',
            '6': '六', '7': '七', '8': '八', '9': '九', '10': '十'
        }
        # 如果已经是中文数字，直接使用
        if number in number_map.values():
            return f"，{number}，"
        # 如果是阿拉伯数字，转换后使用
        return f"，{number_map.get(number, number)}，"

    # 匹配(数字)或(中文数字)的模式
    text = re.sub(r'[（\(](一|二|三|四|五|六|七|八|九|十|\d+)[）\)]', number_marker_handler, text)

    # 修改百分比处理部分
    def percent_handler(match):
        """处理百分号"""
        number = match.group(1)  # 直接获取完整的数字部分
        return f"百分之{number}"
    
    # 修改正则表达式以匹配包含小数点的数字
    text = re.sub(r'(\d+\.?\d*)%', percent_handler, text)

    def decimal_point_handler(match):
        number1 = match.group(1)
        number2 = ' '.join(list(match.group(2)))
        return f"{number1}点{number2}"
    # 使用零宽负向后瞻和前瞻确保点号前后都是纯数字
    text = re.sub(r'(?<![A-Za-z])(\d+)\.(\d+)(?![A-Za-z])', decimal_point_handler, text)

    # 处理坐标点
    def coordinate_handler(match):
        content = match.group(1)  # 获取括号内的内容
        # 处理中英文逗号，将内容分割
        if ',' in content:
            x, y = content.split(',')
        else:
            x, y = content.split('，')
        x = x.upper()
        y = y.upper()
        return f",{x},{y},"
    # 匹配形如(x,y)的坐标点，允许括号内的字母、数字和正负号，同时支持中英文逗号
    text = re.sub(r'\(([A-Za-z0-9+\-]+[,，][A-Za-z0-9+\-]+)\)', coordinate_handler, text)

    # 处理前面是汉字或标点的正负号
    def sign_handler(match):
        # print(f"匹配到的完整文本: {match.group(0)}")
        # print(f"第一组(前缀): {match.group(1)}")
        # print(f"第二组(符号): {match.group(2)}")
        # print(f"第三组(数字/字母): {match.group(3)}")
        
        before_char = match.group(1) if match.group(1) else ''
        sign = match.group(2)
        content = match.group(3)

        # 如果前面是空格，返回原始匹配
        if before_char and before_char.isspace():
            return match.group(0)
            
        if sign in ['+', '＋']:  # 同时处理全角和半角加号
            return f"{before_char}正{content}"
        elif sign in ['-', '－', '﹣']:  # 处理各种减号
            return f"{before_char}负{content}"
        return match.group(0)
    
    # 先处理正负号
    # 修改后的正则表达式，匹配数字或字母
    pattern = r'(^|[^a-zA-Z0-9\s\)])([＋+\-－﹣])([a-zA-Z0-9]+)'
    # print(f"处理正负号前文本: {text}")
    text = re.sub(pattern, sign_handler, text)
    # print(f"处理正负号后文本: {text}")

    text = text.replace('+', '加')
    
    # 处理幂次
    def power_handler(match):
        base = match.group(1)
        power = match.group(2)
        # 只有当幂次数字长度大于1时才转换
        if len(power) > 1:
            return f"{base}的{power}次方"
        return match.group(0)
    # 匹配字母/数字后面跟着^和多个数字的情况
    text = re.sub(r'([A-Za-z\d])\^(\d+)', power_handler, text)

    # print(f"\n处理分数前文本: {text}")
    # 处理分数
    text = process_fractions(text)
    # print(f"处理分数后文本: {text}")
    
    # 处理区间范围
    def range_handler(match):
        start = match.group(1)
        end = match.group(2)
        return f"从{start}到{end}"
    text = re.sub(r'(\d+\.?\d*)[~～](\d+\.?\d*)', range_handler, text)
    
    # # 处理时间
    # def time_handler(match):
    #     hours = match.group(1)
    #     minutes = match.group(2)
    #     return f"{hours}点{minutes}分"
    # text = re.sub(r'(\d{1,2}):(\d{2})', time_handler, text)
    
    # 处理单位
    def unit_handler(match):
        number = match.group(1)
        unit = match.group(2)
        # 可以添加单位的特殊处理规则
        unit_map = {
            'km/h': '千米每小时',
            'm/s': '米每秒',
            'km²': '平方千米',
            'cm²': '平方厘米'
        }
        unit = unit_map.get(unit, unit)
        return f"{number}{unit}"
    text = re.sub(r'(\d+\.?\d*)\s*(km/h|m/s|km²|m²|cm²)', unit_handler, text)
    


    # 处理绝对值
    def absolute_value_handler(match):
        content = match.group(1)  # 获取绝对值符号内的内容
        return f"{content}的绝对值,"
    # 匹配由|包围的表达式，内容可以包含字母、数字、加减号和空格
    text = re.sub(r'\|([^|]+)\|', absolute_value_handler, text)

    return text

def replace_english_with_markers(text):
    """
    将文本中的连续英文短语替换为标记（EN1, EN2等）
    
    Args:
        text (str): 输入的文本
    
    Returns:
        tuple: 包含两个元素:
            - str: 替换后的文本
            - dict: 标记到原始英文的映射
    
    Examples:
        >>> replace_english_with_markers("今天讨论global warming和weather")
        ("今天讨论EN1和weather", {"EN1": "global warming"})
    """
    positions, phrases = detect_english_phrases(text)
    
    # 如果没有找到英文短语，直接返回原文和空字典
    if not positions:
        return text, {}
    
    # 创建映射字典
    markers = {f"EN{i+1}": phrase for i, phrase in enumerate(phrases)}
    
    # 从后向前替换，避免位置变化
    result = text
    for (start, end), marker in zip(reversed(positions), reversed(markers.keys())):
        result = result[:start] + marker + result[end:]
    
    return result, markers

def restore_english_from_markers(text, markers):
    """
    将标记（EN1/EN一等）还原为原始英文短语
    
    Args:
        text (str): 包含标记的文本
        markers (dict): 标记到原始英文的映射
    
    Returns:
        str: 还原后的文本
    
    Examples:
        >>> restore_english_from_markers("今天讨论EN1和EN10", {"EN1": "global", "EN10": "warming"})
        "今天讨论global和warming"
    """
    result = text
    chinese_numbers = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
                      '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'}
    
    # 先处理中文数字的情况
    for cn_num, num in chinese_numbers.items():
        result = result.replace(f'EN{cn_num}', f'EN{num}')
    
    # 按标记长度降序排序，确保先处理较长的标记（如EN10）再处理较短的标记（如EN1）
    sorted_markers = sorted(markers.items(), key=lambda x: len(x[0]), reverse=True)
    
    # 然后进行正常的替换
    for marker, phrase in sorted_markers:
        result = result.replace(marker, phrase)
    
    return result

def normalize_punctuation(text: str) -> str:
    """处理文本中的连续标点符号
    
    1. 将中文标点转换为英文标点
    2. 处理连续标点的组合（主要处理逗号和句号）
    3. 使用迭代方式处理长度大于2的连续标点
    4. 处理带空格的连续标点
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 处理后的文本，保持英文标点
        
    Examples:
        >>> normalize_punctuation("你好，。")
        "你好."
        >>> normalize_punctuation("你好, .")
        "你好."
        >>> normalize_punctuation("测试，，，。")
        "测试."
    """
    # 1. 标点符号映射（中文转英文）
    punct_map = {
        '，': ',',
        '。': '.',
        '！': '!',
        '？': '?',
        '；': ';',
        '：': ':',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '（': '(',
        '）': ')',
        '【': '[',
        '】': ']',
        '《': '<',
        '》': '>',
        '、': ',',
        '…': '...'
    }
    
    # 转换所有标点为英文
    for cn, en in punct_map.items():
        text = text.replace(cn, en)
    
    # 2. 定义标点组合的处理规则（只处理长度为2的组合）
    punct_rules = {
        ',.': '.',  # 逗号+句号 -> 句号
        '.,': '.',  # 句号+逗号 -> 句号
        ',,': ',',  # 双逗号 -> 单逗号
        '..': '.',  # 双句号 -> 单句号
        '.?': '?',  # 句号+问号 -> 问号
        '?.': '?',  # 问号+句号 -> 问号
        '?,': '?',  # 逗号+问号 -> 逗号
        '?,': '?',  # 问号+逗号 -> 逗号
        "',": ",",
        ",'": ",",
        "'.": ".",
        ".'": ".",
    }
    
    # 先处理带空格的连续标点
    # 使用正则表达式匹配标点+可选空格+标点的模式
    text = re.sub(r'([,.!?;:])\s*([,.!?;:])', lambda m: punct_rules.get(m.group(1) + m.group(2), m.group(2)), text)
    
    # 3. 迭代处理直到没有变化
    prev_text = ''
    while prev_text != text:
        prev_text = text
        
        # 应用长度为2的规则
        for pattern, replacement in punct_rules.items():
            text = text.replace(pattern, replacement)
        
        # 处理超过2个的连续标点（包括带空格的情况）
        text = re.sub(r'[,.!?;:]\s*[,.!?;:]{2,}', '.', text)  # 3个及以上的标点组合替换为单个句号
    
    if text.startswith(','):
        text = text[1:]
    return text

def normalize_spacing_and_punctuation(text: str) -> str:
    """处理中英文混合文本的空格和标点符号
    
    规则：
    1. 中文相关：
       - 如果标点前后都是中文，使用中文标点
       - 如果标点前或后任一个是中文，中文和标点之间不加空格
       - 如果句子末尾是中文+标点，将标点转换为中文标点
       - 如果句子末尾是中文但没有标点，添加中文句号
    2. 英文相关：
       - 标点后面是英文，标点和英文之间要加空格
       - 英文后面是标点，英文和标点之间不加空格
       - 如果句子末尾是英文+标点，保持英文标点
       - 如果句子末尾是英文但没有标点，添加英文句号
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 处理后的文本
    """
    # 定义中文字符范围的正则表达式
    chinese_char = r'[\u4e00-\u9fff]'
    # 定义英文字符的正则表达式（包括数字）
    english_char = r'[a-zA-Z0-9]'
    
    # 标点符号映射
    punct_map = {
        '，': ',',
        '。': '.',
        '！': '!',
        '？': '?',
        '；': ';',
    }
    reverse_punct_map = {v: k for k, v in punct_map.items()}
    
    # 首先统一将中文标点转换为英文标点
    for cn, en in punct_map.items():
        text = text.replace(cn, en)
    
    # 先处理所有多余的空格
    text = re.sub(r'\s+', ' ', text)
    
    # 移除中文字符和标点符号之间的空格
    text = re.sub(rf'({chinese_char})\s+([,.!?;])', r'\1\2', text)  # 中文后面的标点
    text = re.sub(rf'([,.!?;:])\s+({chinese_char})', r'\1\2', text)  # 标点后面的中文
    
    # 移除英文和后面标点之间的空格
    text = re.sub(rf'({english_char})\s+([,.!?;])', r'\1\2', text)
    
    # 确保标点后面的英文有空格
    text = re.sub(rf'([,.!?;])({english_char})', r'\1 \2', text)
    
    def replace_punct(match):
        full_match = match.group(0)
        prev_char = match.group(1) if match.group(1) else ''
        punct = match.group(2)
        next_char = match.group(3) if match.group(3) else ''
        
        # 如果是引号或冒号，保持原样
        if punct in ['"', "'", ':']:
            return full_match
        
        # 判断前后字符的类型
        prev_is_chinese = bool(re.match(chinese_char, prev_char)) if prev_char else False
        next_is_chinese = bool(re.match(chinese_char, next_char)) if next_char else False
        prev_is_english = bool(re.match(english_char, prev_char)) if prev_char else False
        next_is_english = bool(re.match(english_char, next_char)) if next_char else False
        
        # 如果前后都是中文，使用中文标点
        if prev_is_chinese and next_is_chinese:
            return f"{prev_char}{reverse_punct_map.get(punct, punct)}{next_char}"
        
        # 如果是句子末尾且前面是中文，使用中文标点
        if prev_is_chinese and not next_char:
            return f"{prev_char}{reverse_punct_map.get(punct, punct)}"
        
        # 如果是句子末尾且前面是英文，使用英文标点
        if prev_is_english and not next_char:
            return f"{prev_char}{punct}"
        
        # 如果前面是中文
        if prev_is_chinese:
            if next_is_english:
                return f"{prev_char}{punct} {next_char}"
            return f"{prev_char}{punct}{next_char if next_char else ''}"
        
        # 如果前面是英文
        if prev_is_english:
            if next_is_english:
                return f"{prev_char}{punct} {next_char}"
            return f"{prev_char}{punct}{next_char if next_char else ''}"
        
        # 其他情况
        result = ''
        if prev_char:
            result += prev_char
        result += punct
        if next_char:
            if next_is_english:
                result += ' '
            result += next_char
        return result
    
    # 使用正则表达式查找所有标点符号及其前后字符
    pattern = r'(.)([,.!?;:])(.)?'
    text = re.sub(pattern, replace_punct, text)
    
    # 处理句尾标点的情况
    if text:
        if not text[-1] in ',.!?;:。，！？；：':
            # 检查最后一个字符是否为中文
            if re.match(chinese_char, text[-1]):
                text += '。'  # 添加中文句号
            elif re.match(english_char, text[-1]):
                text += '.'   # 添加英文句号
        elif text[-1] in [',', '，']:  # 处理句末逗号
            # 检查逗号前面的字符
            if re.match(chinese_char, text[-2]):
                text = text[:-1] + '。'  # 将逗号改为中文句号
            else:
                text = text[:-1] + '.'   # 将逗号改为英文句号
    
    # 最后清理可能残留的多余空格
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text
        
def add_split_markers(text: str, sentences_per_split: int = 2, en_zh_split: bool = True, subject: str = 'en') -> str:
    """在文本中添加分割标记，只在标点符号后添加分割标记
    
    判断句子的规则：
    1. 遇到句子结束标点（。！？!?;；?？.）
    2. 遇到换行符 \n
    3. 与上一个有效句子结束标点之间的距离：
       - 大于等于5个中文字符或
       - 大于等于10个英文字母
    """
    if not text:
        return text
    
    # 先处理换行符
    text = text.replace('\n', '<|split|>')
    
    if subject == 'en':
        punctuation = set('。！？!??？.')
    else:
        punctuation = set('。！？!??？')
    
    result = []
    sentence_count = 0
    last_split_pos = 0
    last_valid_punct_pos = -1  # 记录上一个有效句子结束标点的位置
    
    def is_valid_sentence_length(start_pos: int, end_pos: int) -> bool:
        """判断两个标点之间的文本是否构成有效句子长度"""
        if start_pos < 0:  # 如果是第一个句子，直接返回True
            return True
            
        segment = text[start_pos+1:end_pos]  # 获取两个标点之间的文本
        chinese_count = sum(1 for c in segment if '\u4e00' <= c <= '\u9fff')
        english_count = sum(1 for c in segment if c.isascii() and c.isalpha())
        
        total_count = chinese_count*2 + english_count

        return total_count >= 10
    
    # 如果启用中英文分割，预处理英文短语位置
    split_points = set()
    if en_zh_split:
        positions, phrases = detect_english_phrases(text)
        
        # 检查每个位置的左右边界，合并连续的英文片段
        english_ranges = []
        if positions:
            current_start, current_end = positions[0]
            # print(f"\n开始处理第一个范围: {current_start}-{current_end}")
            # print(f"文本: '{text[current_start:current_end]}'")
            
            # 检查第一个片段的左边是否有英文
            if current_start > 0:
                # print(f"检查左边字符: '{text[current_start-1]}'")
                if text[current_start-1].isalpha():
                    # print("左边是英文字符，开始向左扩展")
                    while current_start > 0 and text[current_start-1].isalpha():
                        current_start -= 1
                        # print(f"扩展到 {current_start}: '{text[current_start]}'")
            
            for i in range(1, len(positions)):
                next_start, next_end = positions[i]
                # print(f"\n检查相邻范围:")
                # print(f"当前范围: {current_start}-{current_end}: '{text[current_start:current_end]}'")
                # print(f"下一范围: {next_start}-{next_end}: '{text[next_start:next_end]}'")
                
                # 检查两个范围之间的字符
                gap_text = text[current_end:next_start]
                # print(f"间隔文本: '{gap_text}'")
                # print(f"间隔字符列表: {[c for c in gap_text]}")
                
                # 检查间隔中的每个字符
                valid_chars = []
                for c in gap_text:
                    is_valid = c.isspace() or c == '.' or (len(c.strip()) == 1 and c.isalpha())
                    valid_chars.append((c, is_valid))
                # print(f"间隔字符有效性: {valid_chars}")
                
                is_valid_gap = all(c.isspace() or c == '.' or (len(c.strip()) == 1 and c.isalpha()) for c in gap_text)
                # print(f"是否有效间隔? {is_valid_gap}")
                
                if is_valid_gap:
                    # print(f"合并范围: {current_start}-{next_end}")
                    current_end = next_end
                else:
                    # print(f"添加独立范围: {current_start}-{current_end}")
                    english_ranges.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            
            # 检查最后一个片段的右边是否有英文
            if current_end < len(text):
                # print(f"\n检查最后范围的右边字符: '{text[current_end]}'")
                if text[current_end].isalpha():
                    # print("右边是英文字符，开始向右扩展")
                    while current_end < len(text) and text[current_end].isalpha():
                        current_end += 1
                        # print(f"扩展到 {current_end}: '{text[current_end-1]}'")
            
            english_ranges.append((current_start, current_end))
            # print(f"\n添加最后范围: {current_start}-{current_end}")
        
        # print(f"\n最终英文范围: {english_ranges}")
        # print("最终范围内容:")
        # for start, end in english_ranges:
        #     print(f"{start}-{end}: '{text[start:end]}'")
        
        # 对每个合并后的范围，找到其前面的标点符号位置
        for start, end in english_ranges:
            # print(f"\n处理范围 {start}-{end}: '{text[start:end]}'")
            # 向前找最近的标点
            prev_punct = start - 1
            while prev_punct >= 0 and text[prev_punct] not in punctuation:
                prev_punct -= 1
            if prev_punct >= 0:
                # print(f"找到前置标点: '{text[prev_punct]}' 位置 {prev_punct}")
                split_points.add(prev_punct)
    
    # print("\n最终分割点:", split_points)
    
    # 遍历文本添加分割标记
    i = 0
    while i < len(text):
        result.append(text[i])
        
        if text[i] in punctuation:
            # 检查是否是有效的句子长度
            if is_valid_sentence_length(last_valid_punct_pos, i):
                if is_long_sentence(last_valid_punct_pos, i):
                    sentence_count += 1
                sentence_count += 1
                last_valid_punct_pos = i  # 更新最后一个有效句子结束标点的位置
                
                should_split = False

                if sentence_count >= sentences_per_split:
                    should_split = True
                
                if en_zh_split and i in split_points:
                    should_split = True
                
                if should_split and i < len(text) - 1:
                    result.append('<|split|>')
                    sentence_count = 0
                    last_split_pos = i + 1
        
        i += 1
    
    final_text = ''.join(result)
    # print(f"\n清洗前文本: {final_text}")
    
    # 清洗逻辑：
    # 1. 移除<|split|>后面的空格
    # 2. 合并连续的<|split|>标记
    final_text = re.sub(r'<\|split\|>\s+', '<|split|>', final_text)
    final_text = re.sub(r'(<\|split\|>){2,}', '<|split|>', final_text)
    # print(f"清洗后文本: {final_text}")
    
    return final_text

def fix_option_punctuation(text: str) -> str:
    """修复选项标记的标点符号
    
    当检测到单个大写字母+点号时，检查其左侧：
    如果是空格且空格左侧是英文字母，则在空格前添加英文句号
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 修复后的文本
        
    Examples:
        >>> fix_option_punctuation("hello B. world")
        "hello. B. world"
    """
    # print(f"\n修复选项标点前文本: {text}")
    
    # 查找所有单个大写字母+点号的位置
    pattern = r'([A-Z]\.)'
    matches = list(re.finditer(pattern, text))
    
    # 从后向前处理，避免位置变化影响
    result = text
    for match in reversed(matches):
        start = match.start()
        if start > 0:  # 确保不是在文本开头
            # 检查左侧字符
            if text[start-1].isspace():  # 是空格
                # 继续检查空格左侧
                if start > 1 and text[start-2].isalpha():  # 是英文字母
                    # print(f"在位置 {start-1} 处添加句号")
                    result = result[:start-1] + '.' + result[start-1:]
    
    # print(f"修复选项标点后文本: {result}")
    return result

def fix_english_punctuation(text: str) -> str:
    """修复英文标点符号,将英文标点转换为对应的中文标点
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 转换后的文本
        
    Examples:
        >>> fix_english_punctuation("Hello! How are you?")
        "Hello。 How are you？"
        >>> fix_english_punctuation("12:34")  # 时间格式保持不变
        "12:34"
    """
    # 定义英文标点到中文标点的映射
    punct_map = {
        '!': '.',
        '！': '。',
        '(': ',',
        ')': ',',
        '（': ',',
        '）': ',',
        '[': ',',
        ']': ',',
        '【': ',',
        '】': ',',
        '｛': ',',
        '｝': ',',
        '【': ',',
        '】': ',',
        '/': ',',
        '_': ',',
        ' - ': ',',
        '<':',',
        '>':',',
        '《':',',
        '》':',',
    }
    
    # 使用字典进行替换
    for raw, target in punct_map.items():
        text = text.replace(raw, target)
    
    # 特殊处理冒号：只在前后不都是数字的情况下转换为逗号
    def handle_colon(match):
        before = match.group(1)  # 冒号前的字符
        after = match.group(2)   # 冒号后的字符
        # 如果前后都是数字，保持冒号不变
        if before.isdigit() and after.isdigit():
            return f"{before}:{after}"
        return f"{before},{after}"
    
    # 使用正则表达式匹配冒号及其前后的字符
    text = re.sub(r"(.):(.)", handle_colon, text)
    
    # 处理单引号的特殊情况
    def replace_apostrophe(match):
        before = match.group(1)  # 单引号前的字符
        after = match.group(2)   # 单引号后的字符
        # 严格检查后面的字符是空格或中文
        if (after.isspace() or '\u4e00' <= after <= '\u9fff') or (before.isspace() and '\u4e00' <= before <= '\u9fff'):
            return before + ',' + after
        return match.group(0)
    
    # 使用正则表达式匹配单引号及其前后的字符
    text = re.sub(r"(.)\'(.)", replace_apostrophe, text)
        
    return text

def fix_chinese_punctuation(text: str) -> str:
    """修复中文标点符号
    
    将中文标点符号转换为英文标点符号
    """
    punct_map = {
        '!': '.',
        '！': '。',
        '/': ',',
        '_': ',',
        '：':',',
        '；':'。',
        ':':',',
        ';':'.',
    }
    
    # 使用字典进行替换
    for raw, target in punct_map.items():
        text = text.replace(raw, target)
    
    # 特殊处理冒号：只在前后不都是数字的情况下转换为逗号
    def handle_colon(match):
        before = match.group(1)  # 冒号前的字符
        after = match.group(2)   # 冒号后的字符
        # 如果前后都是数字，保持冒号不变
        if before.isdigit() and after.isdigit():
            return f"{before}:{after}"
        return f"{before},{after}"
    
    # 使用正则表达式匹配冒号及其前后的字符
    text = re.sub(r"(.):(.)", handle_colon, text)

    return text

def simplify_punctuation(text: str, length_threshold: int = 3) -> str:
    """简化标点符号
    
    规则：
    1. 删除中英文之间的标点和空格
    2. 合并由标点分隔的短中文片段（长度<=2），但保护数字词（一到十）
    3. 将非句首的英文单词首字母大写转换为小写（不包括单字母）
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 简化后的文本
    """
    def is_chinese(char):
        """判断字符是否为中文"""
        return '\u4e00' <= char <= '\u9fff'
    
    def is_english(char):
        """判断字符是否为英文"""
        return char.isascii() and char.isalpha()
    
    def is_punctuation(char):
        """判断是否为需要处理的标点"""
        return char in {',', '，', '、', ':', '：', ';', '；'}
    
    def is_number_word(text: str) -> bool:
        """判断文本是否包含数字词（一到十）"""
        number_words = {'一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
        return any(word in text for word in number_words)
    
    def should_convert_case(word: str) -> bool:
        """判断是否需要转换大小写
        
        条件：
        1. 长度大于1的英文单词
        2. 首字母大写
        """
        return (len(word) > 1 and 
                word.isascii() and 
                word[0].isupper() and 
                any(c.islower() for c in word[1:]))  # 确保不是全大写的缩写
    
    # 添加一些常见的短语，即使长度小于阈值也要保留标点
    protected_phrases = ["没关系", "对不起", "谢谢你", "不客气", "请问", "抱歉"]

    # 处理中英文之间的标点和空格
    i = 0
    result = []
    while i < len(text):
        curr_char = text[i]
        
        # 如果当前字符是空格或标点，检查前后文字
        if curr_char.isspace() or is_punctuation(curr_char):
            prev_char = result[-1] if result else None
            
            # 找到下一个非空格非标点的字符
            next_idx = i + 1
            while next_idx < len(text) and (text[next_idx].isspace() or is_punctuation(text[next_idx])):
                next_idx += 1
            next_char = text[next_idx] if next_idx < len(text) else None
            
            # 如果前后分别是中文和英文（或英文和中文），跳过当前字符
            if prev_char and next_char:
                if ((is_chinese(prev_char) and is_english(next_char)) or
                    (is_english(prev_char) and is_chinese(next_char))):
                    i = next_idx
                    continue
        
        result.append(curr_char)
        i += 1
    
    # 处理短中文片段和英文大小写
    text = ''.join(result)
    i = 0
    result = []
    current_word = []
    
    while i < len(text):
        curr_char = text[i]
        
        # 收集英文单词
        if is_english(curr_char):
            current_word.append(curr_char)
        else:
            # 处理已收集的英文单词
            if current_word:
                word = ''.join(current_word)
                # 检查是否需要转换大小写
                if should_convert_case(word) and len(result) > 0:  # 确保不是句首
                    word = word[0].lower() + word[1:]
                result.extend(word)
                current_word = []
            
            # 处理标点符号
            if is_punctuation(curr_char):
                # 检查前面是否是中文
                prev_is_chinese = result and is_chinese(result[-1])
                
                # 查找下一个标点或文本结束
                j = i + 1
                while j < len(text) and not (is_punctuation(text[j]) or text[j] in {'。', '.', '！', '!', '？', '?'}):
                    j += 1
                
                # 计算后半部分长度（不包括空格）
                segment = text[i+1:j]
                segment_length = sum(1 for char in segment if is_chinese(char))
                
                # 如果前面是中文且后半部分长度<length_threshold（而不是<=），删除标点
                if prev_is_chinese and segment_length < length_threshold:
                    # 检查段落是否包含数字词或是受保护的短语，如果是则保留标点
                    if not is_number_word(segment) and segment not in protected_phrases:
                        i += 1
                        continue
            
            result.append(curr_char)
        
        i += 1
    
    # 处理最后一个英文单词
    if current_word:
        word = ''.join(current_word)
        if should_convert_case(word) and len(result) > 0:  # 确保不是句首
            word = word[0].lower() + word[1:]
        result.extend(word)
    
    return ''.join(result)

def replace_geographical_names(text: str) -> str:
    """替换文本中的英文地理名称为对应的中文名称
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 替换后的文本
        
    Examples:
        >>> replace_geographical_names("I live in Beijing")
        "I live in 北京"
    """
    # 中国城市
    cities = {
        'beijing': '北京',
        'shanghai': '上海',
        'guangzhou': '广州',
        'shenzhen': '深圳',
        'hangzhou': '杭州',
        'nanjing': '南京',
        'chengdu': '成都',
        'wuhan': '武汉',
        'xian': "西安",
        "xi'an": "西安",
        'tianjin': '天津',
        'chongqing': '重庆',
        'suzhou': '苏州',
        'qingdao': '青岛',
        'dalian': '大连',
        'xiamen': '厦门',
        'ningbo': '宁波',
        'kunming': '昆明',
        'changsha': '长沙',
        'fuzhou': '福州',
        'shenyang': '沈阳',
        'jinan': '济南',
        'harbin': '哈尔滨',
        'zhengzhou': '郑州',
    }
    
    # 中国省份/地区
    provinces = {
        'guangdong': '广东',
        'shandong': '山东',
        'jiangsu': '江苏',
        'zhejiang': '浙江',
        'henan': '河南',
        'sichuan': '四川',
        'hubei': '湖北',
        'hunan': '湖南',
        'hebei': '河北',
        'fujian': '福建',
        'shanxi': '山西',
        'shaanxi': '陕西',
        'liaoning': '辽宁',
        'jilin': '吉林',
        'heilongjiang': '黑龙江',
        'anhui': '安徽',
        'jiangxi': '江西',
        'yunnan': '云南',
        'guizhou': '贵州',
        'gansu': '甘肃',
        'hainan': '海南',
        'qinghai': '青海',
        'xinjiang': '新疆',
        'xizang': '西藏',
        'tibet': '西藏',
        'guangxi': '广西',
        'inner mongolia': '内蒙古',
        'ningxia': '宁夏',
        'hong kong': '香港',
        'hongkong': '香港',
        'macau': '澳门',
        'macao': '澳门',
        'taiwan': '台湾',
    }
    
    # 合并所有字典
    all_names = {}
    all_names.update(cities)
    all_names.update(provinces)
    # 创建正则表达式模式
    # 注意：这里的模式会匹配完整的词（词边界），不区分大小写
    patterns = [
        (re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE), v)
        for k, v in all_names.items()
    ]
    
    # 替换文本
    result = text
    for pattern, replacement in patterns:
        result = pattern.sub(replacement, result)
    
    return result

def is_long_sentence(start_pos: int, end_pos: int) -> bool:
    """判断句子是否超过40个字符
    
    Args:
        start_pos (int): 句子开始位置
        end_pos (int): 句子结束位置
    
    Returns:
        bool: 如果句子长度超过40个字符返回True，否则返回False
    """
    return (end_pos - start_pos) > 30

def fix_math_punctuation(text: str) -> str:
    """处理数学文本中的标点符号
    
    规则：
    1. 当冒号前后都是数字时，保留冒号（表示比例关系）
    2. 当冒号前后不都是数字时，将冒号转换为逗号
    3. 保留其他数学相关的标点符号
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 处理后的文本
        
    Examples:
        >>> fix_math_punctuation("10:20")  # 前后都是数字，保留冒号
        "10:20"
        >>> fix_math_punctuation("提升:60的一半")  # 前面不是数字，转换为逗号
        "提升,60的一半"
    """
    # 特殊处理冒号：只在前后不都是数字的情况下转换为逗号
    def handle_colon(match):
        before = match.group(1)  # 冒号前的字符
        after = match.group(2)   # 冒号后的字符
        # 如果前后都是数字，保持冒号不变（表示比例关系）
        if before.isdigit() and after.isdigit():
            return f"{before}:{after}"
        return f"{before},{after}"
    
    # 使用正则表达式匹配冒号及其前后的字符
    text = re.sub(r"(.):(.)", handle_colon, text)
    
    # 处理中文冒号
    text = re.sub(r"(.)：(.)", handle_colon, text)
    
    return text

def merge_short_splits(text: str, min_length: int = 50) -> str:
    """合并文本中长度小于指定阈值的相邻<|split|>片段
    
    算法逻辑：
    1. 在文本首尾添加<|split|>标记，便于统一处理
    2. 将文本按<|split|>分割成片段列表
    3. 从头开始遍历片段，检查相邻片段合并后的长度是否小于阈值
    4. 如果小于阈值，则合并片段，并从头重新开始检测
    5. 最后移除首尾的<|split|>标记
    
    Args:
        text (str): 包含<|split|>标记的文本
        min_length (int, optional): 最小片段长度阈值，默认为50
        
    Returns:
        str: 合并短片段后的文本
    """
    if '<|split|>' not in text:
        return text
    
    # 在文本首尾添加<|split|>标记，便于统一处理
    processed_text = '<|split|>' + text + '<|split|>'
    
    # 将文本按<|split|>分割成片段列表
    segments = processed_text.split('<|split|>')
    
    # 由于首尾添加了<|split|>，分割后的第一个和最后一个元素为空字符串
    if segments[0] == '':
        segments.pop(0)
    if segments[-1] == '':
        segments.pop()
    
    # 合并短片段的主循环
    i = 0
    while i < len(segments) - 1:
        # 检查当前片段与下一个片段合并后的长度
        current_length = len(segments[i])
        next_length = len(segments[i+1])
        
        # 如果当前片段和下一个片段的总长度小于阈值
        if current_length + next_length < min_length:
            # 合并片段
            segments[i] = segments[i] + segments[i+1]
            # 删除下一个片段
            segments.pop(i+1)
            # 重新从头开始检测
            i = 0
        else:
            # 如果不需要合并，继续检查下一对片段
            i += 1
    
    # 重新组合文本，不包含首尾的<|split|>
    result = '<|split|>'.join(segments)
    
    return result

if __name__ == '__main__':
    """测试函数"""
    test_cases = [
        "你的答案是四百六十二，但正确顺序应该是四，再次汇入大河六，变回小溪二，到达大海。我们再来看看第十三题，纹丝不动，的意思是什么呢。",
        "好的，我们先来看第十二题。题目要求根据故事内容，按顺序补充下图。我们一起来回顾一下故事的顺序首先，溪溪汇入了小河。然后经过一系列事件后，他再次汇入大河。最后到达大海。你觉得一，应该填哪个序号呢。"
    ]
    
    for text in test_cases:
        print(f"\n测试文本: {text}")
        result = add_split_markers(text,subject='cn')
        print(f"处理后: {result}")
        # 替换
        # replaced_text, markers = replace_english_with_markers(text)
        # print(f"替换后: {replaced_text}")
        # print(f"映射关系: {markers}")
        
        # # 还原
        # restored_text = restore_english_from_markers(replaced_text, markers)
        # print(f"还原后: {restored_text}")