from typing import Union, List

# from fastapi import FastAPI
import latex2mathml.converter

from Access8Math.A8M_PM import MathContent, initialize
from Access8Math.speech import flatten
from Access8Math.reshape.percent import reshape_percent


# from A8M_PM import MathContent, initialize
# from speech import flatten
# from reshape.percent import reshape_percent

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# app = FastAPI()

Access8MathConfig = {
	"settings": {
		"analyze_math_meaning": True,
		"language": "zh_CN",
		"LaTeX_delimiter": 'bracket',
		"Nemeth_delimiter": "at",
		"auto_generate": False,
		"dictionary_generate": True,
		"no_move_beep": True,
		"command_mode": False,
		"navigate_mode": False,
		"shortcut_mode": False,
		"writeNavAudioIndication": True,
		"writeNavAcrossLine": True,
		"LaTeX_delimiter": "bracket",
		"Nemeth_delimiter": "at",
		"speech_source": "Access8Math",
		"braille_source": "Access8Math",
		"interact_source": "Access8Math"
	},
	"rules": {
		"config_file_path": os.path.join(current_dir, "config_file.json"),
		# "NodeType": False,
		# "TerminalNodeType": False,
		# "NonTerminalNodeType": False,
		# "SiblingNodeType": False,
		# "CompoundNodeType": False,
		# "FractionType": False,
		# "MiOperandType": False,
		# "MnOperandType": False,
		# "OperandType": False,
		# "OperatorType": False,
		# "FromToOperatorType": False,
		# "LogOperatorType": False,
		# "MiType": False,
		# "MnType": False,
		# "MoType": False,
		# "MtableType": False,
		# "TwoMnType": False,
		# "ThreeMnType": False,
		# "TwoMiOperandItemType": False,
		# "MoLineType": False,
		# "LineType": False,
		# "MoLineSegmentType": False,
		# "LineSegmentType": False,
		# "MoVectorType": False,
		# "VectorSingleType": False,
		# "VectorDoubleType": False,
		# "MoRayType": False,
		# "RayType": False,
		# "ArrowOverSingleSymbolType": False,
		# "MoFrownType": False,
		# "FrownType": False,
		# "MoDegreeType": False,
		# "DegreeType": False,
		# "SingleType": False,
		# "SingleMsubsupType": False,
		# "SingleMsubType": False,
		# "SingleMsupType": False,
		# "SingleMunderoverType": False,
		# "SingleMunderType": False,
		# "SingleMoverType": False,
		# "SingleFractionType": False,
		# "SingleSqrtType": False,
		# "PowerType": False,
		# "SquarePowerType": False,
		# "CubePowerType": False,
		# "MsubsupFromToType": False,
		# "MunderoverFromToType": False,
		# "MsubFromType": False,
		# "MunderFromType": False,
		# "MsupToType": False,
		# "MoverToType": False,
		# "MsubLogType": False,
		# "VerticalBarType": False,
		# "AbsoluteType": False,
		# "OpenMatrixType": False,
		# "CloseMatrixType": False,
		# "MatrixType": False,
		# "OpenSimultaneousEquationsType": False,
		# "SimultaneousEquationsType": False,
		# "DeterminantType": False,
		# "BinomialType": False,
		# "SingleNumberFractionType": False,
		# "AddIntegerFractionType": False,
		# "SignPreviousMoType": False,
		# "MinusType": False,
		# "NegativeSignType": False,
		# "FirstNegativeSignType": False,
		# "PlusType": False,
		# "PositiveSignType": False,
		# "FirstPositiveSignType": False
	}
}


initialize(Access8MathConfig)


# 提取核心逻辑到一个函数
def convert_latex_to_spoken(latex_input: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    将LaTeX转换为语音文本
    Args:
        latex_input: 可以是单个LaTeX字符串或LaTeX字符串列表
    Returns:
        如果输入是字符串，返回单个处理结果
        如果输入是列表，返回处理结果列表
    """
    # 判断输入类型，将字符串转换为单元素列表处理
    is_single_string = isinstance(latex_input, str)
    latexs = [latex_input] if is_single_string else latex_input
    
    ssmls = []
    for latex in latexs:
        mathMl = latex2mathml.converter.convert(latex)
        mathMl = mathMl.replace("<<", "&lt;<").replace(">>", ">&gt;")
        mathMl = reshape_percent(mathMl)
        mathcontent = MathContent("zh_CN", mathMl)
        serialized_tree = mathcontent.pointer.serialized()
        
        item = flatten(serialized_tree)
        ssml = " ".join(item)
        ssmls.append(ssml)
    
    # 如果输入是单个字符串，返回单个结果而不是列表
    return ssmls[0] if is_single_string else ssmls

if __name__ == "__main__":
	# 现在你可以在本地调用这个函数
	latex_expressions = ["我们现在已经知道了AB=\\frac{8}{7}BC,和AB+BC=37.5,接下来是不是可以把AB进行替换呀?", "|x-1|+|x-3|+|x-4|可以表示x到三个数的距离和,|x-1|表示x到哪个数的距离呀?","85%"]
	ssml_results = convert_latex_to_spoken(latex_expressions)

	# 输出结果
	for result in ssml_results:
		print(result)