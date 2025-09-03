# coding=utf-8
# Copyright (C) 2017-2024 Tseng Woody <tsengwoody.tw@gmail.com>

from xml.etree import ElementTree as ET

import collections
import csv
from csv import DictReader
from dataclasses import dataclass
import html
import inspect
import os
import re
import shutil
import weakref

from Access8Math.lib.dataProcess import joinObjectArray
# from lib.dataProcess import joinObjectArray

AUTO_GENERATE = 0
DIC_GENERATE = 1


def includes_unicode_range(string, lower, high):
	for char in string:
		if ord(char) < high and ord(char) >= lower:
			return True
	return False


def NVDASymbolsFetch(language):
	from characterProcessing import _getSpeechSymbolsForLocale
	builtin_dict = {}
	user_dict = {}

	builtin, user = _getSpeechSymbolsForLocale(language)
	for value in builtin.symbols.values():
		if not value.identifier.isspace() and " " not in value.identifier:
			builtin_dict[value.identifier] = value.replacement
	for value in user.symbols.values():
		if not value.identifier.isspace() and " " not in value.identifier:
			user_dict[value.identifier] = value.replacement

	return builtin_dict, user_dict


def mathml2etree(mathml):
	"""
	Convert mathml to XML etree object
	@param mathml: source mathml
	@type mathml: str
	@rtype: XML etree
	"""

	gtlt_pattern = re.compile(r"([\>])(.*?)([\<])")
	mathml = gtlt_pattern.sub(
		lambda m: m.group(1) + html.escape(html.unescape(m.group(2))) + m.group(3),
		mathml
	)

	quote_pattern = re.compile(r"=([\"\'])(.*?)\1")
	mathml = quote_pattern.sub(lambda m: '=' + m.group(1) + html.escape(m.group(2)) + m.group(1), mathml)

	parser = ET.XMLParser()

	try:
		tree = ET.fromstring(mathml.encode('utf-8'), parser=parser)
	except BaseException as error:
		raise SystemError(error)
	return tree


def create_node(et):
	p_tag = re.compile(r"[\{].*[\}](?P<mp_type>.+)")
	# print(et.tag)
	# print('et.attrib')
	# print(et.attrib)
	r = p_tag.search(et.tag)
	try:
		mp_tag = r.group('mp_type')
	except BaseException:
		mp_tag = et.tag

	# print(nodes)
	node_class = nodes[mp_tag.capitalize()] if mp_tag.capitalize() in nodes.keys() else object
	# print(node_class)
	if mp_tag == 'none':
		node_class = Nones

	if issubclass(node_class, NonTerminalNode) or issubclass(node_class, BlockNode):
		child = []
		for c in et:
			node = create_node(c)
			child.append(node)
		node = node_class(child, et.attrib)
	elif issubclass(node_class, TerminalNode):
		node = node_class([], et.attrib, data=et.text)
	elif mp_tag == 'none':
		node = Nones()
	else:
		child = []
		for c in et:
			node = create_node(c)
			child.append(node)
		node = Mrow(child, et.attrib)
		# raise RuntimeError('unknown tag : {}'.format(mp_tag))

	# print('!' * 100)
	# print(node)
	# if issubclass(node.__class__, TerminalNode):
	# 	print(node.name)
	# 	print(node.data)
	# 	print(node.rule)
	# if issubclass(node.__class__, NonTerminalNode):
	# 	print(node.rule)
	# 	print(node.child)
	# print('!' * 100)
	return node


def clean_allnode(node):

	for child in node.child:
		clean_allnode(child)

	if isinstance(node, Mphantom):
		# remove node
		parent_new_child = node.parent.child[0:node.index_in_parent()]
		if node.index_in_parent() + 1 < len(node.parent.child):
			parent_new_child = parent_new_child + node.parent.child[node.index_in_parent() + 1:]
		node.parent.child = parent_new_child
		return node

	if node.parent and isinstance(node, BlockNode):
		if len(node.child) == 1 or (isinstance(node.parent, AlterNode) and len(node.child) > 0):
			# remove node
			parent_new_child = node.parent.child[0:node.index_in_parent()] + node.child
			if node.index_in_parent() + 1 < len(node.parent.child):
				parent_new_child = parent_new_child + node.parent.child[node.index_in_parent() + 1:]
			node.parent.child = parent_new_child
			for child in node.child:
				child.parent = node.parent

		elif isinstance(node.parent, AlterNode) and len(node.child) == 0:
			index = node.index_in_parent()
			node.parent.child[index].child = []
		return node

	return node


def set_mathcontent_allnode(node, mathcontent):
	for child in node.child:
		set_mathcontent_allnode(child, mathcontent)
	node.mathcontent = mathcontent
	if node.type:
		node.type.mathcontent = mathcontent
	return node


def set_mathrule_allnode(node, mathrule):
	for child in node.child:
		set_mathrule_allnode(child, mathrule)
	node.set_mathrule(mathrule)
	return node


def set_braillemathrule_allnode(node, braillemathrule):
	for child in node.child:
		set_braillemathrule_allnode(child, braillemathrule)
	node.set_braillemathrule(braillemathrule)
	return node


def clear_type_allnode(node):
	for child in node.child:
		clear_type_allnode(child)
	node.type = None
	return node


def check_type_allnode(node):
	for child in node.child:
		check_type_allnode(child)
	node.check_type()
	return node


def check_in_allnode(node, check_node):
	if id(node) == id(check_node):
		return True

	for child in node.child:
		if check_in_allnode(child, check_node):
			return True

	return False


class MathContent(object):
	def __init__(self, language, mathMl):
		# print(mathMl)
		self.raw_mathMl = mathMl
		et = mathml2etree(mathMl)
		# print(et)
		self.root = self._pointer = self._history = create_node(et)
		# print('-' * 80)
		# print(self._pointer)

		clean_allnode(self.root)

		set_mathcontent_allnode(self.root, self)

		symbol = load_unicode_dic(language=language, category="speech", NVDASymbol=True)
		# save_unicode_dic_codic(symbol, path=r'D:\Access8Math\addon\globalPlugins\Access8Math\locale\zh_CN\speech\unicode_user.codic')
		# print(symbol)
		self.set_symbol(symbol)

		mathrule = load_math_rule(language=language)
		# print('#' * 100)
		# print(mathrule)
		self.set_mathrule(mathrule)

		def bianli(node: Node):
			if issubclass(node.__class__, TerminalNode):
				# print(node)
				# print(node.name)
				# print(node.data)
				# print('node.rule start')
				# print(node.rule)
				# print('node.rule end')
				# print(node.role)
				# print('*')
				# print(node.type)
				# print(node.type.tag if node.type else '**')
				# print('\n')
				return
			elif issubclass(node.__class__, NonTerminalNode):
				# print(node)
				# print(node.name)
				# print(node.data)
				# print('node.rule start --------')
				# print(node.rule)
				# print('node.rule end ----------')
				# print(node.role)
				# print('*')
				# print(node.type)
				# print(node.type.tag if node.type else '**')
				for clild in node.child:
					bianli(clild)
			else:
				print('类型错误')
		# print('#' * 100)
		# bianli(self.root)

		# braillesymbol = load_unicode_dic(language=language, category="braille")
		# self.set_braillesymbol(braillesymbol)

		# braillemathrule = load_math_rule(language=language, category="braille")
		# self.set_braillemathrule(braillemathrule)

	@property
	def mathML(self):
		mathmlNamespace = 'xmlns="http://www.w3.org/1998/Math/MathML"'
		result = self.pointer.get_mathml()
		if not result.startswith("<math"):
			result = f"<math {mathmlNamespace}>{result}</math>"
		return result

	def set_mathrule(self, mathrule):
		self.mathrule = mathrule
		set_mathrule_allnode(self.root, mathrule)
		clear_type_allnode(self.root)
		check_type_allnode(self.root)
		set_mathrule_allnode(self.root, mathrule)
		set_mathcontent_allnode(self.root, self)

	def set_braillemathrule(self, mathrule):
		self.braillemathrule = mathrule
		set_braillemathrule_allnode(self.root, mathrule)
		clear_type_allnode(self.root)
		check_type_allnode(self.root)
		set_braillemathrule_allnode(self.root, mathrule)
		set_mathcontent_allnode(self.root, self)

	def set_symbol(self, symbol):
		self.symbol = symbol
		symbol_list = sorted(list(symbol.keys()), key=lambda i: -len(i))
		symbol_list = [item.replace("|", "\\|") for item in symbol_list if not includes_unicode_range(item, 124, 125)]
		restring = "|".join(symbol_list).translate({
			ord("$"): "\\$",
			ord("("): "\\(",
			ord(")"): "\\)",
			ord("*"): "\\*",
			ord("+"): "\\+",
			ord("-"): "\\-",
			ord("."): "\\.",
			ord("?"): "\\?",
			ord("["): "\\[",
			ord("\\"): "\\\\",
			ord("]"): "\\]",
			ord("^"): "\\^",
			ord("{"): "\\{",
			ord("}"): "\\}",
		})
		restring += "|\\|"
		self.symbol_repattern = re.compile(restring)

	def set_braillesymbol(self, symbol):
		self.braillesymbol = symbol
		symbol_list = sorted(list(symbol.keys()), key=lambda i: -len(i))
		symbol_list = [item.replace("|", "\\|") for item in symbol_list if not includes_unicode_range(item, 124, 125)]
		restring = "|".join(symbol_list).translate({
			ord("$"): "\\$",
			ord("("): "\\(",
			ord(")"): "\\)",
			ord("*"): "\\*",
			ord("+"): "\\+",
			ord("-"): "\\-",
			ord("."): "\\.",
			ord("?"): "\\?",
			ord("["): "\\[",
			ord("\\"): "\\\\",
			ord("]"): "\\]",
			ord("^"): "\\^",
			ord("{"): "\\{",
			ord("}"): "\\}",
		})
		restring += "|\\|"
		self.braillesymbol_repattern = re.compile(restring)

	def navigate(self, action):
		pointer = None
		if action == "downArrow":
			pointer = self.pointer.down
		elif action == "upArrow":
			pointer = self.pointer.up
		elif action == "leftArrow":
			pointer = self.pointer.previous_sibling
		elif action == "rightArrow":
			pointer = self.pointer.next_sibling
		elif action == "home":
			pointer = self.root

		if pointer is not None:
			self.pointer = pointer
			return True

		return False

	def table_navigate(self, action):
		if action == "downArrow":
			pointer = self.pointer.vertical_down
		elif action == "upArrow":
			pointer = self.pointer.vertical_up
		elif action == "leftArrow":
			pointer = self.pointer.previous_sibling
		elif action == "rightArrow":
			pointer = self.pointer.next_sibling

		if pointer is not None:
			self.pointer = pointer
			return True

		return False

	def symbol_translate(self, string):
		try:
			string = self.symbol_repattern.sub(lambda m: self.symbol[m.group(0)], string)
		except BaseException:
			pass
		
		return string

	def braillesymbol_translate(self, string):
		try:
			string = self.braillesymbol_repattern.sub(lambda m: self.braillesymbol[m.group(0)], string)
		except BaseException:
			pass

		return string

	@property
	def pointer(self):
		return self._pointer

	@pointer.setter
	def pointer(self, node):
		self._history = self._pointer
		self._pointer = node

	@property
	def hint(self):
		if self._history.tag == "mtd" and self._pointer.tag == "mtd" \
		and self._history.parent.parent == self._pointer.parent.parent \
		and self._history.parent.index_in_parent() != self._pointer.parent.index_in_parent():
			return [self._pointer.parent.des]
		return []


class Node(object):
	def __init__(self, child=None, attrib=None, data=None):
		self._mathcontent = None
		self._parent = None

		self.child = list(child) if child else []
		for c in self.child:
			if isinstance(c, Node):
				c.parent = self

		self.attrib = attrib if attrib else {}
		self.data = str(data.strip()) if data else ''

		self.mathrule = {}
		self.rule = []
		self.role = []

		self.braillemathrule = {}
		self.braillerule = []
		self.braillerole = []

		self.type = None
		self.role_level = None

	def check_type(self):
		for nodetype in nodetypes_check:
			if nodetype.check(self) and nodetype.name in self.mathrule:
				# print('$$')
				# print('1')
				# print(self)
				# print('2')
				# print(self.name)
				# print('3')
				# print(nodetype)
				# print('4')
				# print(nodetype.name)
				# print('5')
				# print(nodetype.check(self))
				if not self.type:
					self.type = nodetype()
					self.type.mathcontent = self.mathcontent
					self.type.set_mathrule(self.mathrule)
				elif self.type and issubclass(nodetype, self.type.__class__):
					# print('%' * 100)
					# print(self)
					self.type = nodetype()
					self.type.mathcontent = self.mathcontent
					self.type.set_mathrule(self.mathrule)

	def set_mathrule(self, mathrule):
		self.mathrule = mathrule
		self.set_role()
		self.set_rule()
		if self.type:
			self.type.set_mathrule(self.mathrule)

	def set_role(self):
		self.role = self.mathrule[self.name].role if self.name in self.mathrule else [self.mathcontent.symbol_translate('item')]
		d = len(self.child) - len(self.role)
		if d > 0:
			append = self.role[-1]
			try:
				before, after = append.split(".")
			except ValueError:
				before, after = f"{append}.".split(".")
			self.role = self.role[:-1]
			for i in range(d + 1):
				self.role.append(f"{before}{i+1}{after}")
			self.role_level = AUTO_GENERATE
		else:
			self.role_level = DIC_GENERATE
		# print(self.role)

	def set_rule(self):
		# print('w' * 20)
		# print(self)
		# print(self.type)
		if self.type and self.type.rule:
			rule = self.type.rule
			# print('e' * 20)
			# print(self.type)
			# print(rule)
		else:
			rule = self.mathrule[self.name].serialized_order
			# print('f' * 20)
			# print(self.name)
			# print(self.mathrule[self.name])
			# print(rule)
		# print('r' * 100)
		# print(rule)
		if len(rule) >= 2 and isinstance(rule[1], tuple):
			result = []
			for i in range(len(self.child)):
				before_empty = (rule[1][0].isspace() or rule[1][0] == '')
				after_empty = (rule[1][1].isspace() or rule[1][1] == '')
				if not (before_empty and after_empty):
					result.append('{0}{1}{2}'.format(rule[1][0], i + 1, rule[1][1]))
				result.append(i)

			rule = rule[0:1] + result + rule[-1:]

		self.rule = rule
		# print('t' * 100)
		# print(self.rule)

	def set_braillemathrule(self, braillemathrule):
		self.braillemathrule = braillemathrule
		self.set_braillerole()
		self.set_braillerule()
		if self.type:
			self.type.set_braillemathrule(self.braillemathrule)

	def set_braillerole(self):
		self.braillerole = self.braillemathrule[self.name].role if self.name in self.braillemathrule else [self.mathcontent.symbol_translate('item')]
		d = len(self.child) - len(self.braillerole)
		if d > 0:
			append = self.braillerole[-1]
			self.braillerole = self.braillerole[:-1]
			for i in range(d + 1):
				self.braillerole.append('{0}{1}'.format(append, i + 1))
			self.braillerole_level = AUTO_GENERATE
		else:
			self.braillerole_level = DIC_GENERATE

	def set_braillerule(self):
		if self.type and self.type.braillerule:
			braillerule = self.type.braillerule
		else:
			braillerule = self.braillemathrule[self.name].serialized_order

		if len(braillerule) >= 2 and isinstance(braillerule[1], tuple):
			result = []
			for i in range(len(self.child)):
				before_empty = (braillerule[1][0].isspace() or braillerule[1][0] == '')
				after_empty = (braillerule[1][1].isspace() or braillerule[1][1] == '')
				if not (before_empty and after_empty):
					result.append('{0}{1}{2}'.format(braillerule[1][0], i + 1, braillerule[1][1]))
				result.append(i)

			braillerule = braillerule[0:1] + result + braillerule[-1:]

		self.braillerule = braillerule

	def serialized(self):
		serialized = []
		for r in self.rule:
			if isinstance(r, int):
				serialized.append(self.child[r].serialized())
			elif r == '*':
				for c in self.child:
					if c:
						serialized.append(c.serialized())
						serialized.append(['<break time="10ms" />'])
			elif isinstance(r, str):
				serialized.append([r])
			else:
				raise TypeError('rule element type error : expect int or str (get {0})'.format(type(r)))
		return serialized

	def brailleserialized(self):
		serialized = []
		for r in self.braillerule:
			if isinstance(r, int):
				serialized.append(self.child[r].brailleserialized())
			elif r == '*':
				for c in self.child:
					if c:
						serialized.append(c.brailleserialized())
			elif isinstance(r, str):
				serialized.append([r])
			else:
				raise TypeError('rule element type error : expect int or str (get {0})'.format(type(r)))
		return serialized

	def get_mathml(self):
		mathml = ''
		if len(self.child) > 0:
			for c in self.child:
				mathml = mathml + c.get_mathml()
		else:
			mathml = mathml + " "

		attrib = ''
		for k, v in self.attrib.items():
			attrib = attrib + ' {0}="{1}"'.format(k, v)

		if len(attrib) > 0:
			result = f'<{self.tag} {attrib}>{mathml}</{self.tag}>'
		else:
			result = f'<{self.tag}>{mathml}</{self.tag}>'

		return result

	@property
	def mathcontent(self):
		return None if self._mathcontent is None else self._mathcontent()

	@mathcontent.setter
	def mathcontent(self, mathcontent):
		# self._mathcontent = mathcontent
		self._mathcontent = weakref.ref(mathcontent)

	@property
	def des(self):
		return self.parent.role[self.index_in_parent()] if self.parent else self.mathcontent.symbol_translate('math')

	@property
	def name(self):
		return self.type.name if self.type else self.tag

	@property
	def tag(self):
		return self.__class__.__name__.lower()

	@property
	def parent(self):
		return None if self._parent is None else self._parent()

	@parent.setter
	def parent(self, node):
		self._parent = weakref.ref(node)

	def index_in_parent(self):
		try:
			return self.parent.child.index(self)
		except BaseException:
			return None

	@property
	def next_sibling(self):
		try:
			index = self.index_in_parent() + 1
			if index < 0:
				raise IndexError('index out of range')
		except BaseException:
			index = None

		try:
			return self.parent.child[index]
		except BaseException:
			return None

	@property
	def previous_sibling(self):
		try:
			index = self.index_in_parent() - 1
			if index < 0:
				raise IndexError('index out of range')
		except BaseException:
			index = None

		try:
			return self.parent.child[index]
		except BaseException:
			return None

	@property
	def next(self):
		current = self
		_next = current.next_sibling
		while not _next:
			current = current.parent
			if not current:
				return None
			_next = current.next_sibling
		return _next

	@property
	def previous(self):
		current = self
		previous = current.previous_sibling
		while not previous:
			current = current.parent
			if not current:
				return None
			previous = current.previous_sibling
		return previous

	@property
	def down(self):
		try:
			return self.child[0]
		except BaseException:
			return None

	@property
	def up(self):
		try:
			return self.parent
		except BaseException:
			return None

	@property
	def vertical_down(self):
		if self.tag == "mtd":
			index = self.index_in_parent()
			if index >= 0:
				parent_next_sibling = self.parent.next_sibling
				if parent_next_sibling:
					try:
						return parent_next_sibling.child[index]
					except IndexError:
						pass

		return None

	@property
	def vertical_up(self):
		if self.tag == "mtd":
			index = self.index_in_parent()
			if index >= 0:
				parent_previous_sibling = self.parent.previous_sibling
				if parent_previous_sibling:
					try:
						return parent_previous_sibling.child[index]
					except IndexError:
						pass

		return None


class NonTerminalNode(Node):
	def set_rule(self):
		try:
			super().set_rule()
		except BaseException:
			self.rule = range(len(self.child))

	def set_role(self):
		try:
			super().set_role()
		except BaseException:
			self.rule = range(len(self.child))

	def set_braillerule(self):
		try:
			super().set_braillerule()
		except BaseException:
			self.braillerule = range(len(self.child))

	def set_braillerole(self):
		try:
			super().set_braillerole()
		except BaseException:
			self.braillerule = range(len(self.child))


class TerminalNode(Node):
	def set_rule(self):
		# print('&' * 100)
		# print(self.mathrule)
		try:
			super().set_rule()
		except BaseException:
			# self.rule = ['<break time="10ms" />', str(self.mathcontent.symbol_translate(self.data)), '<break time="10ms" />']
			# self.rule = ['<break>', str(self.mathcontent.symbol_translate(self.data)), '</break>']
			self.rule = [str(self.mathcontent.symbol_translate(self.data))]

	def set_braillerule(self):
		try:
			super().set_braillerule()
		except BaseException:
			self.braillerule = [str(self.mathcontent.braillesymbol_translate(self.data))]

	def get_mathml(self):
		mathml = ''
		mathml = mathml + self.data if self.data else mathml

		attrib = ''
		for k, v in self.attrib.items():
			attrib = attrib + ' {0}="{1}"'.format(k, v)

		if len(attrib) > 0:
			result = f'<{self.tag} {attrib}>{mathml}</{self.tag}>'
		else:
			result = f'<{self.tag}>{mathml}</{self.tag}>'

		return result


class AlterNode(NonTerminalNode):
	pass


class FixNode(NonTerminalNode):
	pass


class BlockNode(AlterNode):
	pass


class Mrow(BlockNode):
	pass


class Percent(FixNode):
	pass


class Mfrac(FixNode):
	pass


class Msqrt(AlterNode):
	pass


class Mroot(FixNode):
	pass


class Mstyle(BlockNode):
	pass


class Merror(AlterNode):
	pass


class Mpadded(AlterNode):
	pass


class Mphantom(AlterNode):
	def set_rule(self):
		return []

	def set_braillerule(self):
		return []


class Mfenced(AlterNode):
	def set_rule(self):
		super().set_rule()
		rule = self.rule
		if not self.type:
			if 'open' in self.attrib:
				rule = [str(self.attrib['open'])] + rule
			if 'close' in self.attrib:
				rule = rule[0:-1] + [str(self.attrib['close'])] + rule[-1:]
			if ('open' not in self.attrib) and ('close' not in self.attrib):
				rule = ['('] + rule[0:-1] + [')'] + rule[-1:]

		self.rule = rule

	def set_braillerule(self):
		pass


class Menclose(AlterNode):
	def set_rule(self):
		super().set_rule()
		rule = self.rule

		value2description = {
			"longdiv": "long division symbol",
			"actuarial": "actuarial symbol",
			"radical": "radical",
			"box": "box",
			"roundedbox": "round box",
			"circle": "circle",
			"left": "line on left",
			"right": "line on right",
			"top": "line on top",
			"bottom": "line on bottom",
			"updiagonalstrike": "up diagonal cross out",
			"downdiagonalstrike": "down diagonal cross out",
			"verticalstrike": "vertical cross out",
			"horizontalstrike": "horizontal cross out",
			"madruwb": "Arabic factorial symbol",
			"updiagonalarrow": "diagonal arrow",
			"phasorangle": "phasor angle",
		}
		try:
			notation = self.attrib["notation"]
		except BaseException:
			notation = "longdiv"

		try:
			description = self.mathcontent.symbol_translate("notation:{}".format(value2description[notation]))
		except BaseException:
			description = ""

		head = rule[0].replace("{start}", description)
		cell = rule[1:-1]
		tail = rule[-1].replace("{end}", description)
		self.rule = [head] + cell + [tail]

	def set_braillerule(self):
		pass


class Msub(FixNode):
	pass


class Msup(FixNode):
	pass


class Msubsup(FixNode):
	pass


class Munder(FixNode):
	pass


class Mover(FixNode):
	pass


class Munderover(FixNode):
	pass


class Mtable(AlterNode):
	def set_rule(self):
		super().set_rule()

		rule = self.rule

		if not self.type:
			row_count = len(self.child)
			column_count_list = [len(i.child) for i in self.child]
			column_count = max(column_count_list)
			table_head = [rule[0] + '{0}{1}{2}{3}{4}'.format(
				self.mathcontent.symbol_translate('has'),
				row_count, self.mathcontent.symbol_translate('row'),
				column_count, self.mathcontent.symbol_translate('column')
			)]
		else:
			table_head = [rule[0]]
		cell = rule[1:-1]
		table_tail = rule[-1:]
		self.rule = table_head + cell + table_tail

	def set_braillerule(self):
		super().set_braillerule()
		braillerule = self.braillerule

		if not self.type:
			row_count = len(self.child)
			column_count_list = [len(i.child) for i in self.child]
			column_count = max(column_count_list)
			table_head = [braillerule[0] + '{0}{1}{2}{3}{4}⠀'.format(
				self.mathcontent.braillesymbol_translate(''),
				row_count, self.mathcontent.braillesymbol_translate('r'),
				column_count, self.mathcontent.braillesymbol_translate('c')
			)]
		else:
			table_head = [braillerule[0]]
		cell = braillerule[1:-1]
		table_tail = braillerule[-1:]
		self.braillerule = table_head + cell + table_tail


class Mlabeledtr(AlterNode):
	pass


class Mtr(AlterNode):
	def set_rule(self):
		super().set_rule()
		rule = self.rule
		cell = rule[1:-1]
		self.rule = rule[:1] + cell + rule[-1:]

	def set_braillerule(self):
		super().set_braillerule()
		braillerule = self.braillerule
		cell = braillerule[1:-1]
		self.braillerule = braillerule[:1] + cell + braillerule[-1:]


class Mtd(AlterNode):
	pass


class Mstack(AlterNode):
	pass


class Mlongdiv(AlterNode):
	pass


class Msgroup(AlterNode):
	pass


class Msrow(AlterNode):
	pass


class Mscarries(AlterNode):
	pass


class Mscarry(AlterNode):
	pass


class Maction(AlterNode):
	pass


class Math(AlterNode):
	def get_mathml(self):
		mathml = ''
		# the xml namespace for mathml. Needed to ensure newer versions of word recognise and render the mathml
		mathmlNamespace = 'xmlns="http://www.w3.org/1998/Math/MathML"'
		for c in self.child:
			mathml = mathml + c.get_mathml()

		attrib = ''
		for k, v in self.attrib.items():
			k = re.sub(r"\{.*\}", "", k)
			attrib = attrib + ' {0}="{1}"'.format(k, v)

		if len(attrib) > 0:
			result = f'<{self.tag} {mathmlNamespace} {attrib}>{mathml}</{self.tag}>'
		else:
			result = f'<{self.tag} {mathmlNamespace}>{mathml}</{self.tag}>'

		return result


class Mi(TerminalNode):
	pass


class Mn(TerminalNode):
	pass


class Mo(TerminalNode):
	pass


class Mtext(TerminalNode):
	pass


class Mspace(TerminalNode):
	def set_rule(self):
		self.rule = []


class Ms(TerminalNode):
	pass


class Mmultiscripts(AlterNode):
	def set_rule(self):
		super().set_rule()
		self.role_level = DIC_GENERATE
		rule = self.rule

		mmrule = []

		index = range(self.mprescripts_index_in_child() + 1, len(self.child))
		index_odd = index[0::2]
		index_even = index[1::2]
		index_mix = zip(index_odd, index_even)
		for count, item in enumerate(index_mix):
			temp = [
				'{0}{1}{2}'.format(self.mathcontent.symbol_translate('mmultiscripts:order'), count + 1, self.mathcontent.symbol_translate('mmultiscripts:pre-subscript')),
				item[0],
				'{0}{1}{2}'.format(self.mathcontent.symbol_translate('mmultiscripts:order'), count + 1, self.mathcontent.symbol_translate('mmultiscripts:pre-superscript')),
				item[1],
			]
			mmrule.extend(temp)

		index = range(1, self.mprescripts_index_in_child())
		index_odd = index[0::2]
		index_even = index[1::2]
		index_mix = zip(index_odd, index_even)
		for count, item in enumerate(index_mix):
			temp = [
				'{0}{1}{2}'.format(self.mathcontent.symbol_translate('mmultiscripts:order'), count + 1, self.mathcontent.symbol_translate('mmultiscripts:post-subscript')),
				item[0],
				'{0}{1}{2}'.format(self.mathcontent.symbol_translate('mmultiscripts:order'), count + 1, self.mathcontent.symbol_translate('mmultiscripts:post-superscript')),
				item[1],
			]
			mmrule.extend(temp)

		mmrule.insert(0, 0)

		rule = rule[:1] + mmrule + rule[-1:]
		self.rule = rule

	def set_braillerule(self):
		pass

	def set_role(self):
		super().set_role()
		mmrole = [self.mathcontent.symbol_translate('main')]

		index = range(1, self.mprescripts_index_in_child())
		index_odd = index[0::2]
		index_even = index[1::2]
		index_mix = zip(index_odd, index_even)
		for count, item in enumerate(index_mix):
			temp = [
				'{0}{1}{2}'.format(self.mathcontent.symbol_translate('mmultiscripts:order'), count + 1, self.mathcontent.symbol_translate('mmultiscripts:post-subscript')),
				'{0}{1}{2}'.format(self.mathcontent.symbol_translate('mmultiscripts:order'), count + 1, self.mathcontent.symbol_translate('mmultiscripts:post-superscript')),
			]
			mmrole.extend(temp)

		mmrole.append(self.mathcontent.symbol_translate('mmultiscripts:mprescripts'))

		index = range(self.mprescripts_index_in_child() + 1, len(self.child))
		index_odd = index[0::2]
		index_even = index[1::2]
		index_mix = zip(index_odd, index_even)
		for count, item in enumerate(index_mix):
			temp = [
				'{0}{1}{2}'.format(self.mathcontent.symbol_translate('mmultiscripts:order'), count + 1, self.mathcontent.symbol_translate('mmultiscripts:pre-subscript')),
				'{0}{1}{2}'.format(self.mathcontent.symbol_translate('mmultiscripts:order'), count + 1, self.mathcontent.symbol_translate('mmultiscripts:pre-superscript')),
			]
			mmrole.extend(temp)

		self.role = mmrole

	def mprescripts_index_in_child(self):
		for c in self.child:
			if isinstance(c, Mprescripts):
				return self.child.index(c)
		return None


class Mprescripts(TerminalNode):
	pass


class Nones(TerminalNode):
	def set_rule(self):
		self.rule = [self.mathcontent.symbol_translate('none')]


class NodeType(object):
	tag = object
	child = ['object', '*']
	attrib = {}
	data = re.compile(r".*")
	name = 'nodetype'
	priority = 0

	def __init__(self):
		self.mathrule = {}
		self.rule = []
		self.role = []
		self.braillerule = []
		self.braillerole = []

	@classmethod
	def check(cls, obj):
		if not issubclass(obj.__class__, cls.tag):
			return False

		return True

	def set_mathrule(self, mathrule):
		self.mathrule = mathrule
		self.set_rule()

	def set_rule(self):
		# print('=' * 80)
		# print(self.mathrule)
		try:
			self.rule = self.mathrule[self.name].serialized_order
		except BaseException:
			self.rule = None

	# def set_braillemathrule(self, braillemathrule):
	# 	self.braillemathrule = braillemathrule
	# 	self.set_braillerule()

	# def set_braillerule(self):
	# 	try:
	# 		self.braillerule = self.braillemathrule[self.name].serialized_order
	# 	except BaseException:
	# 		self.braillerule = None


class TerminalNodeType(NodeType):
	@classmethod
	def check(cls, obj):
		if not issubclass(obj.__class__, cls.tag):
			return False

		# check attrib
		for key, value in cls.attrib.items():
			if key not in obj.attrib:
				return False
			elif not value.search(obj.attrib[key]) is not None:
				return False

		# check data
		if not obj.data == '':
			try:
				if not cls.data.search(obj.data) is not None:
					return False
			except BaseException:
				return False
		return True


class NonTerminalNodeType(NodeType):
	@classmethod
	def check(cls, obj):
		if not issubclass(obj.__class__, cls.tag):
			return False

		# check attrib
		for key, value in cls.attrib.items():
			if key not in obj.attrib:
				return False
			elif not value.search(obj.attrib[key]) is not None:
				return False

		# check child
		if cls.child[-1] == '*' and len(cls.child) > 1:
			d = len(obj.child) - (len(cls.child) - 1)
			type_list = cls.child[:-1] + [cls.child[-2]] * d
		else:
			type_list = cls.child
		if not len(type_list) == len(obj.child):
			return False

		# change type
		type_list_str = [t if isinstance(t, str) else t.__name__ for t in type_list]
		type_list = [all_nodetypes_dict[t] for t in type_list_str]

		# check child type
		for mt, o in zip(type_list, obj.child):
			if not mt == object and not mt.check(o):
				return False

		return True


class SiblingNodeType(NodeType):
	previous_siblings = []
	next_siblings = []
	self_ = NodeType

	@classmethod
	def check(cls, obj):
		self_index = obj.index_in_parent()
		if self_index is None:
			return False

		cls_previous_siblings = cls.previous_siblings
		cls_next_siblings = cls.next_siblings

		cpsl = len(cls_previous_siblings)
		cnsl = len(cls_next_siblings)

		start_index = self_index - cpsl
		end_index = self_index + cnsl

		if cpsl > 0 and cls_previous_siblings[0] is None:
			cls_previous_siblings = cls_previous_siblings[1:]
			cpsl = len(cls_previous_siblings)
			start_index = self_index - cpsl
			if not start_index == 0:
				return False
		elif start_index < 0:
			return False

		if cnsl > 0 and cls_next_siblings[-1] is None:
			cls_next_siblings = cls_next_siblings[:-1]
			cnsl = len(cls_next_siblings)
			end_index = self_index + cnsl
			if not end_index == len(obj.parent.child) - 1:
				return False
		elif end_index >= len(obj.parent.child):
			return False

		# change type
		type_list = cls_previous_siblings + [cls.self_] + cls_next_siblings
		type_list_str = [t if isinstance(t, str) else t.__name__ for t in type_list]
		type_list = [all_nodetypes_dict[t] for t in type_list_str]
		objs = obj.parent.child[start_index:end_index + 1]
		for mt, o in zip(type_list, objs):
			if not mt == object and not mt.check(o):
				return False

		return True


class CompoundNodeType(NodeType):
	compound = []

	@classmethod
	def check(cls, obj):
		for mt in cls.compound:
			if mt.check(obj):
				return True
		return False


class FractionType(NonTerminalNodeType):
	tag = Mfrac


class MiOperandType(TerminalNodeType):
	tag = Mi
	data = re.compile(r"^[\d\w]+$")


class MnOperandType(TerminalNodeType):
	tag = Mn
	data = re.compile(r"^[\d\w]+$")


class OperandType(CompoundNodeType):
	compound = [MiOperandType, MnOperandType, FractionType, ]


class OperatorType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^[u'\u2200'-u'\u22FF']$")


class FromToOperatorType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^[∑∫]$")


class LogOperatorType(TerminalNodeType):
	tag = Mi
	data = re.compile(r"^log$")


class MiType(TerminalNodeType):
	tag = Mi


class MnType(TerminalNodeType):
	tag = Mn


class MoType(TerminalNodeType):
	tag = Mo


class MtableType(NonTerminalNodeType):
	tag = Mtable


class TwoMnType(TerminalNodeType):
	tag = Mn
	data = re.compile(r"^[2]$")


class ThreeMnType(TerminalNodeType):
	tag = Mn
	data = re.compile(r"^[3]$")


class TwoMiOperandItemType(NonTerminalNodeType):
	tag = Mrow
	child = [MiOperandType, MiOperandType]


class MoLineType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^[↔]$")


class LineType(NonTerminalNodeType):
	tag = Mover
	child = [TwoMiOperandItemType, MoLineType]
	name = 'LineType'


class MoLineSegmentType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^[¯―]$")


class LineSegmentType(NonTerminalNodeType):
	tag = Mover
	child = [TwoMiOperandItemType, MoLineSegmentType]
	name = 'LineSegmentType'


class MoVectorType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^[→]$")
	attrib = {
		'stretchy': re.compile(r"^false$"),
	}


class VectorSingleType(NonTerminalNodeType):
	tag = Mover
	child = [MiOperandType, MoVectorType]
	name = 'VectorSingleType'


class VectorDoubleType(NonTerminalNodeType):
	tag = Mover
	child = [TwoMiOperandItemType, MoVectorType]
	name = 'VectorDoubleType'


# Arrow above 2 symbols denotes Ray in English notation and vector in French notation
# (equivalent of VectorDoubleType).
# Arrow above 1 symbol denotes also vector in French notation.
class MoRayType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^[→]$")


# Arrow above 2 symbols denotes Ray in English notation and vector in French notation
# (equivalent of VectorDoubleType).
class RayType(NonTerminalNodeType):
	tag = Mover
	child = [TwoMiOperandItemType, MoRayType]
	name = 'RayType'


# Arrow above single symbol denotes vector in French notation.
class ArrowOverSingleSymbolType(NonTerminalNodeType):
	tag = Mover
	child = [MiOperandType, MoRayType]
	name = 'ArrowOverSingleSymbolType'


class MoFrownType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^[⌢]$")


class FrownType(NonTerminalNodeType):
	tag = Mover
	child = [TwoMiOperandItemType, MoFrownType]
	name = 'FrownType'


class MoDegreeType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^[°∘]$")


class DegreeType(NonTerminalNodeType):
	tag = Msup
	child = [MnOperandType, MoDegreeType, ]
	name = 'DegreeType'


class SingleType(CompoundNodeType):
	compound = [MiType, MnType, MoType]


class SingleMsubsupType(NonTerminalNodeType):
	tag = Msubsup
	child = [SingleType, SingleType, SingleType]
	name = 'SingleMsubsup'


class SingleMsubType(NonTerminalNodeType):
	tag = Msub
	child = [SingleType, SingleType]
	name = 'SingleMsub'


class ConcatenationMsubType(SingleMsubType):
	tag = Msub
	child = [MiType, SingleType]
	name = 'ConcatenationMsub'


class PercentType(NonTerminalNodeType):
	tag = Percent
	name = 'Percent'


class SType(TerminalNodeType):
	tag = Mi
	data = re.compile(r"\bS\b")


class TriangleAreaType(SingleMsubType):
	tag = Msub
	child = [SType, NodeType]
	name = 'TriangleArea'


class SingleMsupType(NonTerminalNodeType):
	tag = Msup
	child = [SingleType, SingleType]
	name = 'SingleMsup'


class SingleMunderoverType(NonTerminalNodeType):
	tag = Munderover
	child = [SingleType, SingleType, SingleType]
	name = 'SingleMunderover'


class SingleMunderType(NonTerminalNodeType):
	tag = Munder
	child = [SingleType, SingleType]
	name = 'SingleMunder'


class SingleMoverType(NonTerminalNodeType):
	tag = Mover
	child = [SingleType, SingleType]
	name = 'SingleMover'


class SingleFractionType(FractionType):
	tag = Mfrac
	child = [OperandType, OperandType, ]
	name = 'single_fraction'


class SingleSqrtType(NonTerminalNodeType):
	tag = Msqrt
	child = [OperandType]
	name = 'single_square_root'


class PowerType(SingleMsupType):
	tag = Msup
	child = [OperandType, OperandType, ]
	name = 'power'


class SquarePowerType(PowerType):
	tag = Msup
	child = [OperandType, TwoMnType, ]
	name = 'SquarePowerType'


class CubePowerType(PowerType):
	tag = Msup
	child = [OperandType, ThreeMnType]
	name = 'CubePowerType'


class MsubsupFromToType(SingleMsubsupType):
	tag = Msubsup
	child = [FromToOperatorType, NodeType, NodeType]
	name = 'from_to'


class MunderoverFromToType(SingleMunderoverType):
	tag = Munderover
	child = [FromToOperatorType, NodeType, NodeType]
	name = 'from_to'


class MsubFromType(SingleMsubType):
	tag = Msub
	child = [FromToOperatorType, NodeType]
	name = 'from'


class MunderFromType(SingleMunderType):
	tag = Munder
	child = [FromToOperatorType, NodeType]
	name = 'from'


class MsupToType(SingleMsupType):
	tag = Msup
	child = [FromToOperatorType, NodeType]
	name = 'to'


class MoverToType(SingleMoverType):
	tag = Mover
	child = [FromToOperatorType, NodeType]
	name = 'to'


class MsubLogType(SingleMsubType):
	tag = Msub
	child = [LogOperatorType, OperandType]
	name = 'LogType'


class VerticalBarType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^\|$")


class AbsoluteType(SiblingNodeType):
	previous_siblings = [VerticalBarType]
	self_ = MnOperandType
	next_siblings = [VerticalBarType]
	name = 'absolute'


class OpenMatrixType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^\[$")


class CloseMatrixType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^\]$")


class MatrixType(SiblingNodeType):
	previous_siblings = [OpenMatrixType]
	self_ = MtableType
	next_siblings = [CloseMatrixType]
	name = 'matrix'


class OpenSimultaneousEquationsType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^\{$")


class SimultaneousEquationsType(SiblingNodeType):
	previous_siblings = [OpenSimultaneousEquationsType]
	self_ = MtableType
	name = 'SimultaneousEquations'
	priority = 0

	def set_rule(self):
		super().set_rule()
		rule = self.rule

		row_count = len(self.child)

		table_head = [rule[0] + f'{self.mathcontent.symbol_translate("has")} {row_count} {self.mathcontent.symbol_translate("row")}']
		cell = rule[1:-1]
		table_tail = rule[-1:]
		self.rule = table_head + cell + table_tail


class DeterminantType(SiblingNodeType):
	tag = Mtable
	previous_siblings = [VerticalBarType]
	self_ = MtableType
	next_siblings = [VerticalBarType]
	name = 'determinant'
	priority = 1


class BinomialType(FractionType):
	tag = Mfrac
	attrib = {
		'linethickness': re.compile(r"^[0]$"),
	}
	name = 'BinomialType'
	priority = 1


# SiblingNodeType
class SingleNumberFractionType(SingleFractionType):
	child = [MnOperandType, MnOperandType]
	name = ''


class AddIntegerFractionType(SiblingNodeType):
	previous_siblings = [MnOperandType]
	self_ = SingleNumberFractionType
	name = 'AddIntegerFractionType'
	priority = 1


class SignPreviousMoType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^[(*+-\./:<=>±·×÷−∔]$")


class MinusType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^[-−]$")


class NegativeSignType(SiblingNodeType):
	previous_siblings = [SignPreviousMoType]
	self_ = MinusType
	name = 'NegativeSignType'


class FirstNegativeSignType(SiblingNodeType):
	previous_siblings = [None]
	self_ = MinusType
	name = 'NegativeSignType'


class PlusType(TerminalNodeType):
	tag = Mo
	data = re.compile(r"^[+∔]$")


class PositiveSignType(SiblingNodeType):
	previous_siblings = [SignPreviousMoType]
	self_ = PlusType
	name = 'PositiveSignType'


class FirstPositiveSignType(SiblingNodeType):
	previous_siblings = [None]
	self_ = PlusType
	name = 'PositiveSignType'


@dataclass
class MathRule(object):
	name: str
	description: str
	category: str
	serialized_order: list
	role: list
	example: str


LOCALE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'locale')


def exist_language(language):
	return os.path.exists(os.path.join(LOCALE_DIR, language))


def add_language(language):
	try:
		src_language = language.split("_")[0]
	except IndexError:
		src_language = "en"

	src = os.path.join(LOCALE_DIR, src_language)
	if not os.path.exists(src):
		src = os.path.join(LOCALE_DIR, 'en')
	dst = os.path.join(LOCALE_DIR, language)
	try:
		shutil.copytree(src, dst, ignore=shutil.ignore_patterns('*_user.*'))
	except FileExistsError:
		pass


def remove_language(language):
	try:
		dst = os.path.join(LOCALE_DIR, language)
		shutil.rmtree(dst)
	except BaseException:
		return False
	return True


def export_language(language, dst):
	def ignore_patterns(patterns, filenames):
		return [filename for filename in filenames for pattern in patterns if pattern in filename]

	patterns = ["math.rule", "unicode.dic"]
	src = os.path.join(LOCALE_DIR, language)
	temp = os.path.join(LOCALE_DIR, "export")

	try:
		shutil.rmtree(temp)
	except BaseException:
		pass

	shutil.copytree(src, temp, ignore=lambda directory, contents: ignore_patterns(patterns, contents))
	for dirPath, dirNames, fileNames in os.walk(temp):
		for item in fileNames:
			item = os.path.join(dirPath, item)
			if "_user." in item:
				os.rename(item, item.replace("_user.", "."))

	try:
		shutil.make_archive(dst, 'zip', temp)
	except BaseException:
		return False
	finally:
		shutil.rmtree(temp)

	return True


def available_languages():
	path = os.path.join(LOCALE_DIR)
	languages = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
	return languages


def clean_user_data():
	for dirPath, dirNames, fileNames in os.walk(LOCALE_DIR):
		for item in fileNames:
			item = os.path.join(dirPath, item)
			try:
				if os.path.isfile(item) and ('_user.dic' in item or '_user.rule' in item):
					os.remove(item)
			except BaseException:
				pass


def load_unicode_dic(path=None, language='', category='speech', NVDASymbol=False):
	symbol = {}

	if not path and language:
		path = os.path.join(LOCALE_DIR, language, category)
		if not os.path.exists(path):
			raise OSError(f"{language} i18n file is not found")

		if category == 'speech' and NVDASymbol:
			try:
				builtin, user = NVDASymbolsFetch(language)
			except BaseException:
				builtin, user = {}, {}
			symbol = {**builtin, **user}

		frp = os.path.join(path, 'unicode.dic')
		frp_user = os.path.join(path, 'unicode_user.dic')
		if not os.path.exists(frp_user):
			shutil.copyfile(frp, frp_user)

		path = frp_user

	with open(path, 'r', encoding='utf-8') as fr:
		reader = csv.reader(fr, delimiter='\t')
		for row in reader:
			if len(row) >= 2:
				try:
					symbol[row[0]] = row[1].split(',')[0].strip()
				except BaseException:
					pass

	return symbol


def load_math_rule(path=None, language='', category='speech'):
	meta_path = os.path.join(LOCALE_DIR, 'rule.csv')

	if not path and language:
		path = os.path.join(LOCALE_DIR, language, category)
		if not os.path.exists(path):
			raise OSError(f"{language} i18n file is not found")

		frp = os.path.join(path, 'math.rule')
		frp_user = os.path.join(path, 'math_user.rule')
		if not os.path.exists(frp_user):
			shutil.copyfile(frp, frp_user)
		path = frp_user

	mathrule = collections.OrderedDict({})

	with open(path, 'r', encoding='utf-8') as rule_file, open(meta_path, 'r', encoding='utf-8') as meta_file:
		meta_dict_csv = DictReader(meta_file, delimiter='\t')
		meta = []
		for item in meta_dict_csv:
			meta.append(item)
		# print('=' * 80)
		# print(meta)

		rule_dict_csv = DictReader(rule_file, delimiter='\t')
		rule = []
		for item in rule_dict_csv:
			rule.append(item)
		# print('=' * 80)
		# print(rule)

		rows = joinObjectArray(meta, rule, "name")
		# print('=' * 80)
		# print(rows)
		rows = [{**item, **{"order": int(item["order"])}} for item in rows]
		rows = sorted(rows, key=lambda row: row["order"])
		# print('=' * 80)
		# print(rows)

		for line in rows:
			try:
				if len(line) == 7:
					rule = []
					for i in line["rule"].split(','):
						i = i.strip()
						tuple_pattern = re.compile(r"\((.*)\.(.*)\)")
						s = tuple_pattern.search(i)
						if s:
							i = (s.group(1), s.group(2))

						try:
							rule.append(int(i))
						except BaseException:
							rule.append(i)

					role = []
					for i in line["role"].split(','):
						i = i.strip()
						role.append(i)

					mathrule[line["name"]] = MathRule(line["name"], line["description"].strip(), '', rule, role, line["mathml"])
			except BaseException:
				pass

	# print(len(mathrule))
	# print('@' * 80)
	# print(mathrule)
	mathrule = collections.OrderedDict({k: v for k, v in mathrule.items() if v})
	# print('#' * 80)
	# print(mathrule)
	return mathrule


def save_unicode_dic(symbol, path=None, language='', category='speech'):
	if not path and language:
		path = os.path.join(LOCALE_DIR, language, category)
		path = os.path.join(path, 'unicode_user.dic')

	with open(path, 'w', encoding='utf-8', newline="") as file:
		writer = csv.writer(file, delimiter='\t')
		key = list(symbol.keys())
		key.sort()
		for k in key:
			writer.writerow([k, symbol[k]])

	return True


def save_unicode_dic_codic(symbol, path=None, language='', category='speech'):
	if not path and language:
		path = os.path.join(LOCALE_DIR, language, category)
		path = os.path.join(path, 'unicode_user.codic')

	with open(path, 'w', encoding='utf-8', newline="") as file:
		writer = csv.writer(file, delimiter='\t')
		key = list(symbol.keys())
		# key.sort()
		for k in key:
			if len(k) > 1:
				writer.writerow([k, symbol[k]])
			else:
				writer.writerow([k, symbol[k], ord(k)])

	return True


def save_math_rule(mathrule, path=None, language='', category='speech'):
	if not path and language:
		path = os.path.join(LOCALE_DIR, language, category)
		path = os.path.join(path, 'math_user.rule')

	mathrule_unicode = {}
	for k, v in mathrule.items():
		try:
			so_line = [str(i) if not isinstance(i, tuple) else '(' + '.'.join(i) + ')' for i in v.serialized_order]
			so_line = ', '.join(so_line)
			role_line = ', '.join(v.role)
			mathrule_unicode[k] = [so_line, role_line, v.description]
		except BaseException:
			pass

	with open(path, 'w', encoding='utf-8', newline="") as file:
		writer = csv.writer(file, delimiter='\t')
		key = list(mathrule.keys())
		key.sort()
		for k in key:
			try:
				writer.writerow([k, *mathrule_unicode[k]])
			except BaseException:
				pass

	return True


def initialize(Access8MathConfig):
	if Access8MathConfig:
		language = Access8MathConfig["settings"]["language"]
		if not exist_language(language):
			add_language(language)

	global nodetypes_check
	nodetypes_check = []
	if Access8MathConfig and Access8MathConfig["settings"]["analyze_math_meaning"]:
		for i in nodetypes:
			if i.__name__ in Access8MathConfig["rules"]:
				if Access8MathConfig["rules"][i.__name__]:
					nodetypes_check.append(globals()[i.__name__])
			else:
				nodetypes_check.append(globals()[i.__name__])
	# print('^' * 100)
	# print(nodetypes_check)
	# for nc in nodetypes_check:
	# 	print(nc.name)


def ComplementMethod(method):
	def decorator(cls, obj):
		return not method(obj)
	return decorator


# get class which is Node subclass
nodes = {i.__name__: i for i in locals().values() if inspect.isclass(i) and issubclass(i, Node)}

# get class which is NodeType subclass
nodetypes = [i for i in locals().values() if inspect.isclass(i) and issubclass(i, NodeType)]
nodetypes = sorted(nodetypes, key=lambda c: -c.priority)
nodetypes_dict = {k: v for k, v in locals().items() if inspect.isclass(v) and issubclass(v, NodeType)}
SNT = [i for i in locals().values() if inspect.isclass(i) and issubclass(i, SiblingNodeType)]

notnodetypes = []
notnodetypes_dict = {}
for nodetype in nodetypes:
	dic = dict(nodetype.__dict__)
	dic.update({
		'check': classmethod(ComplementMethod(nodetype.check)),
		'name': '',
	})
	notnodetype = type('Not' + nodetype.__name__, nodetype.__bases__, dic)
	notnodetypes.append(notnodetype)
	notnodetypes_dict.update({
		'Not' + nodetype.__name__: notnodetype,
	})
	locals()['Not' + nodetype.__name__] = notnodetype

nodetypes_dict.update({'object': object})
all_nodetypes = [i for i in locals().values() if inspect.isclass(i) and issubclass(i, NodeType)]
all_nodetypes_dict = {k: v for k, v in locals().items() if inspect.isclass(v) and issubclass(v, NodeType)}
all_nodetypes_dict.update({'object': object})

mathrule_info = {
	"generics": {
		"node": [3, 1, "*"],
		"math": [3, 1, "*", ],
	},
	"fraction": {
		"mfrac": [5, 2, ".", ],
		"single_fraction": [5, 2, ".", ],
		"AddIntegerFractionType": [5, 2, ".", ],
		"BinomialType": [5, 2, ".", ],
	},
	"root": {
		"msqrt": [3, 1, "*", ],
		"mroot": [5, 2, ".", ],
		"single_square_root": [3, 1, ".", ],
	},
	"position": {
		"msubsup": [7, 3, ".", ],
		"msup": [5, 2, ".", ],
		"msub": [5, 2, ".", ],
		"munderover": [7, 3, ".", ],
		"munder": [5, 2, ".", ],
		"mover": [5, 2, ".", ],
		"SingleMsubsup": [7, 3, ".", ],
		"SingleMsub": [5, 2, ".", ],
		"SingleMsup": [5, 2, ".", ],
		"SingleMunderover": [7, 3, ".", ],
		"SingleMunder": [5, 2, ".", ],
		"SingleMover": [5, 2, ".", ],
	},
	"power": {
		"power": [5, 2, ".", ],
		"SquarePowerType": [3, 2, ".", ],
		"CubePowerType": [3, 2, ".", ],
	},
	"from to": {
		"from_to": [7, 3, ".", ],
		"from": [5, 2, ".", ],
		"to": [5, 2, ".", ],
		"LogType": [3, 2, ".", ],
	},
	"table": {
		"mtable": [3, 1, "*", ],
		"mtr": [3, 1, "*", ],
		"mtd": [3, 1, "*", ],
		"determinant": [3, 1, "*", ],
		"matrix": [3, 1, "*", ],
		"SimultaneousEquations": [3, 1, "*", ],
	},
	"line": {
		"LineType": [3, 2, ".", ],
		"RayType": [3, 2, ".", ],
		"LineSegmentType": [3, 2, ".", ],
		"VectorSingleType": [3, 2, ".", ],
		"VectorDoubleType": [3, 2, ".", ],
		"ArrowOverSingleSymbolType": [3, 2, ".", ],
		"FrownType": [3, 2, ".", ],
		"DegreeType": [3, 2, ".", ],
	},
	"other": {
		"NegativeSignType": [1, 1, ".", ],
		"PositiveSignType": [1, 1, ".", ],
		"mmultiscripts": [0, 0, ".", ],
		"menclose": [3, 1, "*", ],
		"mfenced": [3, 1, "*", ],
	},
}


def mathrule_validate(mathrule, validator):
	result = True
	if not len(mathrule.serialized_order) == validator[0]:
		result = False
	if not len(mathrule.role) == validator[1]:
		result = False
	if validator[2] == ".":
		for index in range(len(mathrule.serialized_order)):
			if index % 2 == 0:
				if not isinstance(mathrule.serialized_order[index], str):
					result = False
			else:
				if not isinstance(mathrule.serialized_order[index], int):
					result = False
	elif validator[2] == "*":
		if not isinstance(mathrule.serialized_order[0], str):
			result = False
		elif not isinstance(mathrule.serialized_order[1], tuple):
			result = False
		elif not isinstance(mathrule.serialized_order[2], str):
			result = False
	else:
		pass
	return result


initialize(None)
