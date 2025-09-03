import os
import sys
import re
import json
from tqdm import tqdm
from util import load_symbol_dict
from flowMirrorTN import FlowMirrorTN 

current_dir = os.path.dirname(os.path.abspath(__file__))
wetext_processing_dir = os.path.join(current_dir, 'WeTextProcessing')
sys.path.append(wetext_processing_dir)

from WeTextProcessing.tn.chinese.normalizer import Normalizer

if __name__ == '__main__':
    # 设置字典文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dict_file_path = os.path.join(current_dir, './symbols.dic')
    unit_dict_path = os.path.join(current_dir, './unit.dic')
    normalizer_cache_dir = os.path.join(current_dir, './WeTextProcessing/cache')

    # 初始化转换器
    normalizer = Normalizer(cache_dir=normalizer_cache_dir)
    
    tn = FlowMirrorTN(
        normalizer=normalizer,
        dict_file_path=os.path.join(current_dir, "symbols.dic"),
        unit_dict_path=os.path.join(current_dir, "unit.dic"),
        char_replace_path=os.path.join(current_dir, "math_char_replace.txt"),
        mathcase=False,
        pinyin_to_hanzi=False
    )
    
    # process_text = tn.process_text(input_text="这是一个要求ABc", log=True, pinyin_to_hanzi=True, mathcase=False)
    # print(process_text)

    test_cases = [
    # '今天我们要讨论global warming和climate change，请大家先看课本Unit five的阅读材料.',
    # '让我们学习如何运用sine函数和cosine函数来进行角度计算',
    # "提升：60的一半。",
    # "即180°。我们设∠1为20°，那么∠2就是20°乘以2等于40°，∠3就是20°乘以6等于160°。",
    # "2个正方形加上1个圆形等于8个三角形。因为1个正方形等于3个三角形，所以2个正方形就是6个三角形，再加上1个圆形等于2个三角形，总共就是8个三角形。明白了吗？",
    # "2小时是总时间的2/3，所以我们要找的是2小时对应的整个时间。你刚才的想法接近了，但我们需要用2小时去除以2/3来找到总时间。你知道为什么吗？",
    # "12x加35x等于19加28。",
    # "我们一起算一下，32乘以11等于352，然后352除以526，简化分数后得到的结果是176除以263。",
    # "然后352除以526"
    # "看看总长AD是多少？"
    # "当 x＜2 时，",
    # "看起来你可能有点分心了没关系，我们继续来看这个问题。《三月桃花水》的第三到第六自然段里，把三月桃花水比作了什么。"
    # "我们来一步步分析，首先根据三角形外角定理，∠BCF等于∠A加上∠ABC。因为∠BCF是∠BCE的两倍，∠ABC是∠DBC的两倍，所以2∠BCE - 2∠DBC = 28°。简化后得到∠BCE - ∠DBC = 14°。而∠BDC正好就是这个差值，所以∠BDC是多少度呢？",
    "好的，我们一起来看，把 y = 3 代入 y + 3m = 24 后，方程变成了 3 + 3m = 24。接下来，我们需要解这个方程找出 m 的值。你再试一次，把m=-4代入-m²-4m+4中，计算结果。",
    "今天我们要讨论global warming和climate change，请大家先看课本Unit five的阅读材料.",
    "让我们学习如何运用sine函数和cosine函数来进行角度计算, sine alpha等于什么？",
    ]
    
    for test in test_cases:
        result = tn.process_text(test,subject='math')
        print(f"输入: {test}")
        print(f"输出: {result}\n")


    # root_dir = os.path.dirname(current_dir)
    # input_file = os.path.join(root_dir, "/home/ecs-user/code/zeying/sllm/sllm_infer/label_studio_dataset/2025_test/cn_2025_3/cn_test.jsonl")
    # output_folder = os.path.dirname(input_file)
    # output_file = os.path.join(output_folder, "cn_test_processed.jsonl")
    # input_file = os.path.join(root_dir, "TTS_test/test.jsonl")
    # output_file = os.path.join(root_dir, "TTS_test/processed/test_processed.jsonl")
    # tn.process_jsonl(input_file, output_file, subject='cn')
    # input_file = '/home/ecs-user/code/zeying/TTS_test/data/en/题库/en_data_entries_300.csv'
    # output_file = input_file.replace('.csv', '_processed.csv')
    # tn.process_csv(input_file, output_file, subject='en',need_split=False)