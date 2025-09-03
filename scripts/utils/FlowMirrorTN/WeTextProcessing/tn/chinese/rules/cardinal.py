# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tn.processor import Processor
from tn.utils import get_abs_path

from pynini import accep, cross, string_file
from pynini.lib.pynutil import add_weight, delete, insert


class Cardinal(Processor):

    def __init__(self):
        super().__init__('cardinal')
        self.number = None
        self.digits = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        zero = string_file(get_abs_path('chinese/data/number/zero.tsv'))
        digit = string_file(get_abs_path('chinese/data/number/digit.tsv'))
        teen = string_file(get_abs_path('chinese/data/number/teen.tsv'))
        sign = string_file(get_abs_path('chinese/data/number/sign.tsv'))
        dot = string_file(get_abs_path('chinese/data/number/dot.tsv'))
        math = string_file(get_abs_path('chinese/data/number/math.tsv'))

        rmzero = delete('0') | delete('０')
        rmpunct = delete(',').ques
        digits = zero | digit
        self.digits = digits

        # 11 => 十一
        ten = teen + insert('十') + (digit | rmzero)
        # 11 => 一十一
        tens = digit + insert('十') + (digit | rmzero)
        # 111, 101, 100
        hundred = (digit + insert('百') + (tens | (zero + digit) | rmzero**2))
        # 1111, 1011, 1001, 1000
        thousand = (digit + insert('千') + rmpunct + (hundred
                                                     | (zero + tens)
                                                     | (rmzero + zero + digit)
                                                     | rmzero**3))
        # 10001111, 1001111, 101111, 11111, 10111, 10011, 10001, 10000
        ten_thousand = ((thousand | hundred | ten | digit) + insert('万') +
                        (thousand
                         | (zero + rmpunct + hundred)
                         | (rmzero + rmpunct + zero + tens)
                         | (rmzero + rmpunct + rmzero + zero + digit)
                         | rmzero**4))

        # 1.11, 1.01
        number = digits | ten | hundred | thousand | ten_thousand
        number = sign.ques + number + (dot + digits.plus).ques

        number @= self.build_rule(
            cross('二百', '两百')
            | cross('二千', '两千')
            | cross('二分钟', '两分钟')
            | cross('二个', '两个')
            | cross('二万', '两万')).optimize()
        percent = insert('百分之') + number + delete('%')
        self.number = (accep('约').ques + 
                      accep('人均').ques + 
                    #   accep('加').ques + 
                      # accep('减').ques + 
                      # accep('乘').ques + 
                      # accep('除').ques + 
                      # accep('加上').ques + 
                      # accep('减去').ques + 
                      # accep('乘上').ques + 
                    #   accep('除以').ques + 
                      # accep('负').ques + 
                      # accep('正').ques + 
                      accep('数学加').ques + 
                    #   accep('等于').ques + 
                      accep('是').ques + 
                      accep('比').ques + 
                      accep('Math_Tag').ques +
                    #   accep('想一想').ques +
                      (number | percent) +
                      accep('度').ques +
                      accep('加').ques + 
                      accep('减').ques + 
                      accep('乘').ques + 
                      accep('除').ques + 
                      accep('加上').ques + 
                      accep('减去').ques + 
                      accep('乘上').ques + 
                      accep('除以').ques + 
                      accep('等于').ques)

        # 定义数字+操作符+数字的模式，考虑更多操作符和表达方式
        number_operation_number = (
            number + 
            (
                # 基本运算符
                accep('加') | accep('减') | accep('乘') | accep('除') | 
                accep('加上') | accep('减去') | accep('乘以') | accep('除以') |
                
                # 扩展运算符
                accep('再加') | accep('再减') | accep('再乘') | accep('再除') |
                accep('增加') | accep('减少') | accep('乘上') | accep('除掉') |
                
                # 比较运算符
                accep('大于') | accep('小于') | accep('等于') | accep('不等于') |
                accep('大于等于') | accep('小于等于') | accep('比') |
                
                
                # 分数相关
                accep('分之') | accep('除以') | accep('比例是') 
            ) + 
            number
        )

        # 将这个模式添加到cardinal定义中，并给予更高优先级
        cardinal = add_weight(number_operation_number, -2.0) | number
        cardinal |= percent
        # cardinal string like 127.0.0.1, used in ID, IP, etc.
        cardinal |= digits.plus + (dot + digits.plus)**3
        # xxxx-xxx-xxx
        cardinal |= digits.plus + (delete('-') + digits.plus)**2
        # xxx-xxxxxxxx
        cardinal |= digits**3 + delete('-') + digits**8

        cardinal |= accep("点") + digits
        # three or five or eleven phone numbers
        phone_digits = digits @ self.build_rule(cross('一', '幺'))
        phone = phone_digits**3 | phone_digits**5 | phone_digits**7 | phone_digits**9 | phone_digits**11
        phone |= accep("尾号") + (accep("是") | accep("为")).ques + phone_digits**4
        cardinal |= add_weight(phone, -1.0)

        tagger = insert('value: "') + cardinal + insert('"')
        self.tagger = self.add_tokens(tagger)
