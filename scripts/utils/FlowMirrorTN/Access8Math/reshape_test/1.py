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
<math>
  <mrow>
    <mo stretchy="false">(</mo>
    <mi>a</mi>
    <mo>+</mo>
    <mo>+</mo>
    <mi>b</mi>
    <msup>
      <mo stretchy="false">)</mo>
      <mn>2</mn>
    </msup>
    <mo>=</mo>
    <mo stretchy="false">(</mo>
    <mi>a</mi>
    <mo>*</mo>
    <mi>b</mi>
    <msup>
      <mo stretchy="false">)</mo>
      <mn>2</mn>
    </msup>
    <mo>=</mo>
    <mo stretchy="false">(</mo>
    <mi>f</mi>
    <mo>+</mo>
    <mi>g</mi>
    <msup>
      <mo stretchy="false">)</mo>
      <mn>2</mn>
    </msup>
    <mo>=</mo>
    <msup>
      <mi>a</mi>
      <mn>2</mn>
    </msup>
    <mo>+</mo>
    <mn>2</mn>
    <mi>a</mi>
    <mi>b</mi>
    <mo>+</mo>
    <msup>
      <mi>b</mi>
      <mn>2</mn>
    </msup>
  </mrow>
</math>
"""

# 解析 XML
parser = etree.XMLParser(ns_clean=True)
root = etree.fromstring(xml_data, parser=parser)

# 使用 XPath 查找特定的子结构
ns = {'mml': 'http://www.w3.org/1998/Math/MathML'}

# 我们需要找到包含 (a+b)^2 的结构
xpath_str = """//mo[text()='(']
    /following-sibling::mo[1][text()='+']
    /following-sibling::msup[1] [mo[text()=')'] and mn[text()='2']]
"""

# xpath_str = "//mn[text()='2']"
# xpath_str = "//mi[text()='a']"

# target_expression = root.xpath(xpath_str, namespaces=ns)
target_expression = root.xpath(xpath_str)

print(target_expression)

if target_expression:
    for ex in target_expression:
        print(etree.tostring(ex, pretty_print=True).decode('utf-8'))
        print(find_left_bracket(ex))
else:
    print("No matching element found.")
