from collections.abc import Iterable
from xml.parsers import expat
from typing import Union

class SpeechCommand(object):
	"""The base class for objects that can be inserted between strings of text to perform actions,
	change voice parameters, etc.

	Note: Some of these commands are processed by NVDA and are not directly passed to synth drivers.
	synth drivers will only receive commands derived from L{SynthCommand}.
	"""

SequenceItemT = Union[SpeechCommand, str]
SpeechSequence = list[SequenceItemT]

def flatten(lines):
	"""
	convert tree to linear using generator
	@param lines:
	@type list
	@rtype
	"""
	for line in lines:
		if isinstance(line, Iterable) and not isinstance(line, str):
			for sub in flatten(line):
				yield sub
		else:
			yield line

def convertFromXml(self, xml: str) -> SpeechSequence:
	"""Convert XML to a speech sequence."""
	self._speechSequence = SpeechSequence()
	parser = expat.ParserCreate("utf-8")
	parser.StartElementHandler = parser.EndElementHandler = self._elementHandler
	parser.CharacterDataHandler = self._speechSequence.append
	try:
		parser.Parse(xml)
	except Exception as e:
		raise ValueError(f"XML: {xml}") from e
	return self._speechSequence
