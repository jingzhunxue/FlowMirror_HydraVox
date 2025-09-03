import re

def process_text(text):
    # 规则1: 处理大写字母之间的空格
    result = ''
    for i in range(len(text)):
        if (i > 0 and text[i].isupper() and 
            text[i-1].isalpha() and 
            ord(text[i-1]) < 128):  # 确保前一个字符是英文字母
            result += ' ' + text[i]
        else:
            result += text[i]
    
    # 规则2和3: 处理句末标点
    # 检查是否包含英文单词（由两个或更多字母组成）
    has_english_word = bool(re.search(r'\b[a-zA-Z]{2,}\b', result))
    
    # 删除句末标点
    result = re.sub(r'[.。！!？?]+$', '', result)
    
    # 根据是否包含英文单词添加适当的句末标点
    if has_english_word:
        result += '.'
    
    return result

if __name__ == "__main__":
    text1 = "Hello World你好"  # 应该变成 "Hello World你好"
    text2 = "这是一个test案例。"  # 应该变成 "这是一个test案例."
    text3 = "你好世界。"  # 应该变成 "你好世界"
    text4 = "我ABCD啊"
    print(process_text(text1))  # Hello World你好
    print(process_text(text2))  # 这是一个test案例.
    print(process_text(text3)) 
    print(process_text(text4)) 