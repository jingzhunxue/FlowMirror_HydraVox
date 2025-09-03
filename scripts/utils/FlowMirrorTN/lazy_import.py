import os
import sys
import re
import json
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from util import load_symbol_dict
from flowMirrorTN import FlowMirrorTN 

wetext_processing_dir = os.path.join(current_dir, 'WeTextProcessing')
sys.path.append(wetext_processing_dir)

from WeTextProcessing.tn.chinese.normalizer import Normalizer

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

class LazyImport:
    def __init__(self):
        self.tn = tn
        self.normalizer = normalizer

    def process_text(self, text):
        return self.tn.process_text(text)