from lxml import etree


def find_left_bracket(target):
	previous_siblings = []
	current = target.getprevious()

	while current is not None:
		previous_siblings.insert(0, current)
		if current.tag == "mo" and current.text == "(":
			break
		current = current.getprevious()

	return previous_siblings


# MathML 数据
xml_data = """
<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline">
    <mrow>
        <mn>5</mn>
        <mi>%</mi>
    </mrow>
</math>
"""

xml_data = xml_data.replace('<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline">', '<math>')

# 解析 XML
parser = etree.XMLParser(ns_clean=True)
root = etree.fromstring(xml_data, parser=parser)

# 我们需要找到包含 % 的结构
xpath_str = """//mrow
    /following-sibling::mi[1][text()='%']
"""

xpath_str = "//mn/following-sibling::mi[text()='%']"

target_expression = root.xpath(xpath_str)
target_element = target_expression[0]
previous_sibling = target_element.getprevious()
parent = target_element.getparent()
index = parent.index(target_element)
parent.remove(target_element)
parent.remove(previous_sibling)

percent = etree.Element('percent')
percent.append(previous_sibling)
percent.append(target_element)

parent.insert(index, percent)

print(etree.tostring(root, pretty_print=True).decode('utf-8'))

# if target_expression:
#     for ex in target_expression:
#         print(etree.tostring(ex, pretty_print=True).decode('utf-8'))
#         # print(find_left_bracket(ex))
# else:
#     print("No matching element found.")
