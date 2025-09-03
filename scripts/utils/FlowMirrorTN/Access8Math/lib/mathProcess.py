import html
import re

from regularExpression import delimiterRegularExpression

from latex2mathml import converter
from py_asciimath.translator.translator import (
	ASCIIMath2MathML,
	ASCIIMath2Tex,
	Tex2ASCIIMath,
	MathML2Tex
)

import os

import brailleTables
import config
import louisHelper

from .nemeth.translator import Nemeth2LaTeXTranslator

mathml2latexObj = None
asciimath2mathmlObj = None
latex2asciimathObj = None
asciimath2latexObj = None
nemeth2latexObj = None

translate_dict = {
	ord("$"): r"\$",
	ord("("): r"\(",
	ord(")"): r"\)",
	ord("\\"): r"\\",
}


def mathml2latex(data):
	global mathml2latexObj
	if not mathml2latexObj:
		mathml2latexObj = MathML2Tex()
	return mathml2latexObj.translate(
		data,
		dtd="mathml3",
		dtd_validation=False,
	)[1:-1].strip()


def latex2mathml(data):
	data = data.replace(r'\vec{', r'\overset{⇀}{')
	mathml = converter.convert(data)
	mathml = html.unescape(mathml)
	mathml = mathml.replace('<mi>⇀</mi>', '<mo>⇀</mo>')
	return mathml


def asciimath2mathml(data):
	global asciimath2mathmlObj
	if not asciimath2mathmlObj:
		asciimath2mathmlObj = ASCIIMath2MathML()
	return asciimath2mathmlObj.translate(
		data,
		dtd="mathml3",
		xml_pprint=False,
	)


def latex2asciimath(data):
	global latex2asciimathObj
	if not latex2asciimathObj:
		latex2asciimathObj = Tex2ASCIIMath()
	return latex2asciimathObj.translate(data)


def asciimath2latex(data):
	global asciimath2latexObj
	if not asciimath2latexObj:
		asciimath2latexObj = ASCIIMath2Tex()
	return asciimath2latexObj.translate(data)


def nemeth2latex(data):
	BRAILLE_UNICODE_PATTERNS_START = 0x2800
	temp = ""
	for string in data:
		brailleCells, brailleToRawPos, rawToBraillePos, brailleCursorPos = louisHelper.translate([os.path.join(brailleTables.TABLES_DIR, config.conf["braille"]["translationTable"]), "braille-patterns.cti"], string, mode=4)
		temp += "".join([chr(BRAILLE_UNICODE_PATTERNS_START + cell) for cell in brailleCells])
	data = "".join(temp)

	global nemeth2latexObj
	if not nemeth2latexObj:
		nemeth2latexObj = Nemeth2LaTeXTranslator()
	return nemeth2latexObj.translate(data)


def textmath2laObjFactory(delimiter):
	def wrapper(input):
		delimiter_regular_expression = delimiterRegularExpression(delimiter)
		restring = "|".join(list(delimiter_regular_expression.values()))
		reTexMath = re.compile(restring)

		point = []
		maths = reTexMath.finditer(input)
		previous = None
		index = 0
		count = {
			"all": 0,
			"text": 0,
			"latex": 0,
			"asciimath": 0,
			"nemeth": 0,
			"line": 0,
		}
		for index, item in enumerate(maths):
			if not previous:
				start = 0
				end = item.start(0)
			else:
				start = previous.end(0)
				end = item.start(0)

			if start != end:
				raw = data = input[start:end]
				offset = 0
				first = True
				lines = raw.split("\n")
				for line in lines:
					if not first:
						count['line'] += 1
					first = False
					start_index = start + offset
					end_index = start + offset + len(line) + 1
					point.append({
						"start": start_index,
						"end": end_index,
						"type": "text",
						"raw": raw[offset:offset + len(line) + 1],
						"data": data[offset:offset + len(line)],
						"index": count['all'],
						"line": count['line'],
					})
					offset = offset + len(line) + 1
					count['all'] += 1
				point[-1]["end"] -= 1

			start = item.start(0)
			end = item.end(0)
			raw = item.group(0)
			data = item.group("latex") or item.group("latex_start") or item.group("asciimath") or item.group("asciimath_start") or item.group("nemeth") or item.group("nemeth_start") or item.group("mathml")
			if not data:
				data = ''

			if item.group("ld_start") or item.group("lsd_start"):
				type_ = "latex"
			elif item.group("ad_start") or item.group("asd_start"):
				type_ = "asciimath"
			elif item.group("nd_start") or item.group("nsd_start"):
				type_ = "nemeth"
			elif item.group("mathml"):
				type_ = "mathml"
			else:
				raise TypeError(item.group(0))

			point.append({
				"start": start,
				"end": end,
				"type": type_,
				"raw": raw,
				"data": data,
				"index": count['all'],
				"line": count['line'],
			})
			count['all'] += 1
			previous = item

		start = previous.end(0) if previous else 0
		end = len(input)
		raw = data = input[start:end]
		offset = 0

		first = True
		lines = raw.split("\n")
		for line in lines:
			if not first:
				count['line'] += 1
			first = False
			start_index = start + offset
			end_index = start + offset + len(line) + 1
			point.append({
				"start": start_index,
				"end": end_index,
				"type": "text",
				"raw": raw[offset:offset + len(line)],
				"data": data[offset:offset + len(line)],
				"index": count['all'],
				"line": count['line'],
			})
			offset = offset + len(line) + 1
			count['all'] += 1
		point[-1]["end"] -= 1

		return point
	return wrapper
