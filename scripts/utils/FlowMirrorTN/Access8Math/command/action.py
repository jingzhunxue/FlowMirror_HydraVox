import config

import re

from delimiter import LaTeX as LaTeX_delimiter, AsciiMath as AsciiMath_delimiter, Nemeth as Nemeth_delimiter
from lib.mathProcess import latex2asciimath, asciimath2latex, nemeth2latex
from regularExpression import delimiterRegularExpression

delimiter_dict = {**AsciiMath_delimiter, **LaTeX_delimiter, **Nemeth_delimiter}


def mark(type_):
	if type_ == "latex":
		delimiter = delimiter_dict[config.conf["Access8Math"]["settings"]["LaTeX_delimiter"]]
	elif type_ == "asciimath":
		delimiter = delimiter_dict["graveaccent"]
	elif type_ == "nemeth":
		delimiter = delimiter_dict[config.conf["Access8Math"]["settings"]["Nemeth_delimiter"]]
	delimiter_start = delimiter["start"]
	delimiter_end = delimiter["end"]

	def _mark(text):
		return {
			"text": r'{delimiter_start}{text}{delimiter_end}'.format(
				text=text,
				delimiter_start=delimiter_start,
				delimiter_end=delimiter_end,
			),
			"start_offset": len(delimiter_start),
			"end_offset": len(delimiter_end),
		}
	return _mark


def batch(mode):
	def l2a(m):
		delimiter = AsciiMath_delimiter["graveaccent"]
		data = m.group('latex') or m.group('latex_start')
		try:
			data = delimiter["start"] + latex2asciimath(data) + delimiter["end"]
		except BaseException:
			data = m.group(0)
		return data

	def a2l(m):
		delimiter = LaTeX_delimiter[config.conf["Access8Math"]["settings"]["LaTeX_delimiter"]]
		data = m.group('asciimath') or m.group('asciimath_start')
		try:
			data = delimiter["start"] + asciimath2latex(data)[1:-1] + delimiter["end"]
		except BaseException as e:
			print(e)
			data = m.group(0)
		return data

	def n2l(m):
		delimiter = LaTeX_delimiter[config.conf["Access8Math"]["settings"]["LaTeX_delimiter"]]
		data = m.group('nemeth') or m.group('nemeth_start')
		try:
			data = delimiter["start"] + nemeth2latex(data) + delimiter["end"]
		except BaseException as e:
			print(e)
			data = m.group(0)
		return data

	def reverse(m):
		data = m.group("latex") or m.group("latex_start") or m.group("asciimath") or m.group("asciimath_start")
		print(data)
		if not data:
			data = ''
		if m.group("ld_start") or m.group("lsd_start"):
			type_ = "latex"
		elif m.group("ad_start") or m.group("asd_start"):
			type_ = "asciimath"
		else:
			raise TypeError(m.group(0))
		print(type_)
		if type_ == "latex":
			result = l2a(m)
		elif type_ == "asciimath":
			result = a2l(m)
		else:
			result = m.group(0)
		print(result)
		return result

	def b2d(m):
		data = m.group('latex') or m.group('latex_start')
		return r"${data}$".format(data=data)

	def d2b(m):
		data = m.group('latex') or m.group('latex_start')
		return r"\({data}\)".format(data=data)

	delimiter_regular_expression = delimiterRegularExpression(
		delimiter={
			"latex": config.conf["Access8Math"]["settings"]["LaTeX_delimiter"],
			"asciimath": "graveaccent",
			"nemeth": config.conf["Access8Math"]["settings"]["Nemeth_delimiter"],
		}
	)
	delimiter_regular_expression_latex_bracket = delimiterRegularExpression(
		delimiter={
			"latex": "bracket",
			"asciimath": "graveaccent",
			"nemeth": config.conf["Access8Math"]["settings"]["Nemeth_delimiter"],
		}
	)
	delimiter_regular_expression_latex_dollar = delimiterRegularExpression(
		delimiter={
			"latex": "dollar",
			"asciimath": "graveaccent",
			"nemeth": config.conf["Access8Math"]["settings"]["Nemeth_delimiter"],
		}
	)

	def _batch(text):
		if mode == 'latex2asciimath':
			restring = "|".join([delimiter_regular_expression["latex"], delimiter_regular_expression["latex_start"]])
			pattern = re.compile(restring)
			text = pattern.sub(l2a, text)
		elif mode == 'asciimath2latex':
			restring = "|".join([delimiter_regular_expression["asciimath"], delimiter_regular_expression["asciimath_start"]])
			pattern = re.compile(restring)
			text = pattern.sub(a2l, text)
		elif mode == 'nemeth2latex':
			restring = "|".join([delimiter_regular_expression["nemeth"], delimiter_regular_expression["nemeth_start"]])
			pattern = re.compile(restring)
			text = pattern.sub(n2l, text)
		elif mode == 'reverse':
			restring = "|".join([delimiter_regular_expression["latex"], delimiter_regular_expression["latex_start"], delimiter_regular_expression["asciimath"], delimiter_regular_expression["asciimath_start"]])
			pattern = re.compile(restring)
			text = pattern.sub(reverse, text)
		elif mode == 'bracket2dollar':
			restring = "|".join([delimiter_regular_expression_latex_bracket["latex"], delimiter_regular_expression_latex_bracket["latex_start"]])
			pattern = re.compile(restring)
			text = pattern.sub(b2d, text)
		elif mode == 'dollar2bracket':
			restring = "|".join([delimiter_regular_expression_latex_dollar["latex"], delimiter_regular_expression_latex_bracket["latex_start"]])
			pattern = re.compile(restring)
			text = pattern.sub(d2b, text)
		else:
			text = ''
		return text

	return _batch
