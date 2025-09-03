from lxml import etree

# 原始的 MathML 数据
data = '''
<math>
    <mo>(</mo>
    <mi>a</mi>
    <mo>+</mo>
    <mi>b</mi>
    <msup>
        <mo>)</mo>
        <mn>2</mn>
    </msup>
</math>
'''

def fix_mathml(data):
    # 解析 MathML XML
    root = etree.XML(data)
    
    # 创建一个新的 mrow 元素
    mrow = etree.Element("mrow")
    
    # 遍历根元素的子元素，并根据需要移动它们
    for elem in list(root):
        if elem.tag != 'msup':  # 将除 'msup' 外的所有元素移动到 'mrow'
            mrow.append(elem)
    
    # 查找 'msup' 元素
    msup = root.find('.//msup')
    
    # 如果存在，从 'msup' 中移除并添加到 'mrow' 的最后
    if msup is not None:
        # 移动 'msup' 中的 '(' 到 'mrow'
        mo_in_msup = msup.find('.//mo')
        if mo_in_msup is not None:
            mrow.append(mo_in_msup)
        
        # 重新创建 'msup' 元素，并只包含指数部分
        new_msup = etree.Element("msup")
        new_msup.append(mrow)  # 将 'mrow' 作为基数
        new_msup.append(msup.find('.//mn'))  # 将指数部分添加到 'msup'
        
        # 清除原根元素的所有子元素并添加新结构
        root.clear()
        root.append(new_msup)
    
    # 返回修改后的 XML 字符串
    return etree.tostring(root, pretty_print=True, encoding='unicode')

# 调用函数并打印结果
modified_xml = fix_mathml(data)
print(modified_xml)
