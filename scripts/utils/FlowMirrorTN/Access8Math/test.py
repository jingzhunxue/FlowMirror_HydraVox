import latex2mathml.converter
from A8M_PM import MathContent, initialize
from speech import flatten
from reshape.percent import reshape_percent 


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

def get_output(latex):
	print('*' * 80)
	mathMl = latex2mathml.converter.convert(latex)
	mathMl = mathMl.replace("<<", "&lt;<").replace(">>", ">&gt;")

	print(mathMl)

	mathMl = reshape_percent(mathMl)

	print(mathMl)

	mathcontent = MathContent("zh_CN", mathMl)
	# print(translate_SpeechCommand_CapNotification(mathcontent.pointer.serialized()))
	# print(mathcontent.pointer.serialized())
	serialized_tree = mathcontent.pointer.serialized()

	# print(serialized_tree)

	item = flatten(serialized_tree)
	# ssml = "<speak>" + "".join(item) + "</speak>"
	ssml = " ".join(item)

	print(ssml)

if __name__ == "__main__":
	print('*' * 80)

	latex_list = []

	# latex_list.append(r"\angle MAD = \theta, \angle MBA = \theta_2，\angle MCB = \theta_3，\angle MDC = \theta_4")
	# latex_list.append(r"\angle AMB = 110^\circ，\angle CMD = 90^\circ，\angle BCD = 60^\circ")
	# latex_list.append(r"\theta_2 + \theta_4 - \theta_1 - \theta_3 =")
	# latex_list.append(r"ABCDE  内接于 \odot O")
	# latex_list.append(r"PC ，PD ，DG \perp PC")
	# latex_list.append(r"C(-\sqrt{2}, \sqrt{7})")
	# latex_list.append(r"A(-1, 0) B(1, 0)")
	# latex_list.append(r"PA^2 + PB^2")
	# latex_list.append(r"AM\perp x")
	# latex_list.append(r"\sin \angle AOB = \frac{AM}{OA} = \frac{4}{5} ， AM = \frac{4}{5}a")
	# latex_list.append(r"OM = \frac{3}{5}a")
	# latex_list.append(r"\left( \frac{3a}{5} , \frac{4a}{5} \right)")
	# latex_list.append(r"a = 5 ， OB = OA = 5 ， S_{\triangle AOF} = \frac{1}{2} \times OB \times AM")
	# latex_list.append(r"f(x) = x^2 + ax + b")
	# latex_list.append(r"y = ax^2 + bx + c （其中  a \neq 0 ）")
	# latex_list.append(r"ax^2 + bx + c = n - 1")
	# latex_list.append(r"\Delta h_b = 6V_a \cdot t")
	# latex_list.append(r"\therefore \angle ACD = 45^\circ ，  \angle DCB = 35^\circ")
	# latex_list.append(r"\therefore  DB = BC \cdot \sin 35^\circ = 100\cdot \sin 35^\circ")
	# latex_list.append(r"\text{Rt}\triangle ACD  中， AD = CD ，\therefore  AB = AD + DB = 100(\sin 35^\circ + \cos 35^\circ)")
	# latex_list.append(r"\angle A = 30^\circ ， \tan B = \sqrt{32} ， AC = 2\sqrt{3}")
	# latex_list.append(r"y = kx + b （ k \neq 0 ）")
	# latex_list.append(r"kx + b > mx")
	# latex_list.append(r"x < -2  \text{或}  0 < x < 6")

	# latex_list.append(r"a = 5 ， OB = OA = 5 ， S_{\triangle AOF} = \frac{1}{2} \times OB \times AM")
	# latex_list.append(r"I = \{(x,y) \in A \times B : x < y \text{ and } x \in C\}")
	# latex_list.append(r"H = \{x \in A \cup B : x \notin C \text{ and } x^2 \in D\}")
	# latex_list.append(r"这是一元二次方程：f(x) = x^2 + ax + b")
	# latex_list.append(r"A = \left\{ x \in \mathbb{R} : x^2 < 4 \right\}, \quad B = \left\{ x \in \mathbb{R} : \sin(x) > 0 \right\}")

	# latex_list.append( r"S = \sum_{n=1}^{100} \frac{n^2}{1 + e^{n}}" )
	# latex_list.append( r"P = \prod_{i=1}^{n} \left(1 - \frac{x_i}{1 + x_i^2}\right)" )
	# latex_list.append( r"(a + b)^2 = a^2 + 2ab + b^2" )
	# latex_list.append( r"因为f′x＞fx，所以F′x＞0，故函数Fx是定义在R上的增函数" )
	# latex_list.append( r"\begin{cases}ax - 3y = 9 \\2x - y = 1\end{cases}" )
	# latex_list.append( r"720÷（9-1）=90（千克）" )
	# latex_list.append( r"两人的速度差是40-30=10(米/分钟）" )
	# latex_list.append( r"让x=0求得直线与y轴的交点的纵坐标，让y=0，求得直线与x轴的交点的横坐标，△AOB的面积=$\\frac{1}{2}$×直线与x轴交点横坐标的绝对值×直线与y轴交点纵坐标的绝对值．" )
	# latex_list.append( r"让x=0求得直线与y轴的交点的纵坐标，让y=0，求得直线与x轴的交点的横坐标，△AOB的面积=\frac{1}{2}×直线与x轴交点横坐标的绝对值×直线与y轴交点纵坐标的绝对值．" )

	# latex_list.append( r"x = \frac{{-b \pm \sqrt{{b^2 - 4ac}}}}{{2a}}")
	# latex_list.append( r"解：∵直线l与直线y=$\\frac{1}{2}$x-1关于x轴对称，<br />∴直线l的解析式为-y=$\\frac{1}{2}$x-1<br />即y=-$\\frac{1}{2}$x+1．<br />故选：C．")
	# latex_list.append( r"\frac{1}{2}\frac{1}{2}\frac{1}{2}")
	# latex_list.append( r"1-3(8-x)=-2(15-2x)")
	# latex_list.append( "5\\%")
	latex_list.append( "已知∠A、∠B互余，∠A比∠B大30°，设∠A、∠B的度数分别为x°、y°，下列方程组中符合题意的是（　　）")
	# latex_list.append( "对于一次函数y=-2x+4，当-2≤x≤4时，函数y的取值范围是（　　）")
	# latex_list.append( "56乘以百分之95乘以1点5%等于53点2乘以1点5%等于0点798万元0点798")
	# latex_list.append( "4\\times120\\times70\\%-4\\times80=16(元)")

	# latex_list.append(r"\frac{a}{\sin A} = \frac{b}{\sin B} = \frac{c}{\sin C}")

	# latex_list.append(r"f^{-1}(x) = \frac{1}{f'(f^{-1}(x))}")

	# latex_list.append(r"c^2 = a^2 + b^2 - 2ab \\cdot \\cos(\\gamma)")

	# latex_list.append(r"\frac{a}{x+y}")

	# latex_list.append(r"\angle ACE=180^{\circ}")

	for latex in latex_list:
		get_output(latex)
