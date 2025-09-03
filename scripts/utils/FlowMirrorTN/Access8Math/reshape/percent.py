from lxml import etree


def reshape_percent(xml_data):

	xml_data = xml_data.replace('<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline">', '<math>')

	# 解析 XML
	parser = etree.XMLParser(ns_clean=True)
	root = etree.fromstring(xml_data, parser=parser)

	# 包含 % 的结构
	xpath_str = "//mn/following-sibling::mi[text()='%']"

	target_expression = root.xpath(xpath_str)

	if len(target_expression) == 0:
		return xml_data.replace('<math>', '<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline">')

	for target_element in target_expression:
		previous_sibling = target_element.getprevious()
		parent = target_element.getparent()
		index = parent.index(previous_sibling)
		parent.remove(target_element)
		parent.remove(previous_sibling)

		percent = etree.Element('percent')
		percent.append(previous_sibling)
		percent.append(target_element)

		parent.insert(index, percent)

		# print(etree.tostring(root, pretty_print=True).decode('utf-8'))

	reshaped_xml_data = etree.tostring(root, pretty_print=True).decode('utf-8')
	reshaped_xml_data = reshaped_xml_data.replace('<math>', '<math xmlns="http://www.w3.org/1998/Math/MathML" display="inline">')

	return reshaped_xml_data
