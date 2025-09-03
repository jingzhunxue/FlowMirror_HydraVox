# coding=utf-8

import os
import re
import shutil

import addonHandler
import api
import braille
import config
from keyboardHandler import KeyboardInputGesture
from NVDAObjects import NVDAObject
import mathPres
import nvwave
from scriptHandler import script
import speech
import textInfos
import tones
import ui

from command.latex import A8MLaTeXCommandView
from command.mark import A8MMarkCommandView
from command.review import A8MHTMLCommandView
from command.translate import A8MTranslateCommandView
from command.batch import A8MBatchCommandView
from command.autocomplete import A8MAutocompleteCommandView
from delimiter import LaTeX as LaTeX_delimiter, AsciiMath as AsciiMath_delimiter, Nemeth as Nemeth_delimiter
from lib.braille import display_braille
from lib.mathProcess import textmath2laObjFactory, latex2mathml, asciimath2mathml, nemeth2latex
from lib.viewHTML import Access8MathDocument

addonHandler.initTranslation()
ADDON_SUMMARY = addonHandler.getCodeAddon().manifest["summary"]

PATH = os.path.dirname(__file__)

command_mode = config.conf["Access8Math"]["settings"]["command_mode"]
navigate_mode = config.conf["Access8Math"]["settings"]["navigate_mode"]
shortcut_mode = config.conf["Access8Math"]["settings"]["shortcut_mode"]
greekAlphabet_mode = False
writeNav_mode = False


class SectionManager:
	def __init__(self):
		self.reset()

	def reset(self):
		self.section_index = -1
		self.points = None

	def __enter__(self):
		self.obj = api.getFocusObject()
		self.caret = self.obj.makeTextInfo(textInfos.POSITION_CARET)
		self.reset()
		points = textmath2laObjFactory(
			delimiter={
				"latex": config.conf["Access8Math"]["settings"]["LaTeX_delimiter"],
				"asciimath": "graveaccent",
				"nemeth": config.conf["Access8Math"]["settings"]["Nemeth_delimiter"],
			}
		)(self.obj.makeTextInfo(textInfos.POSITION_ALL).text)
		self.points = list(filter(lambda point: point['start'] < point['end'], points))
		temp = []
		for point in self.points:
			del point['index']
			temp.append(point)
		self.points = temp
		for index, point in enumerate(self.points):
			if self.caret._startOffset >= point['start'] and self.caret._startOffset < point['end']:
				self.section_index = index

		# print(self.obj.makeTextInfo(textInfos.POSITION_ALL).text[self.caret._startOffset])
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		pass

	@property
	def pointer(self):
		try:
			pointer = self.points[self.section_index]
		except BaseException:
			pointer = None

		return pointer

	@property
	def delimiter(self):
		if self.pointer['type'] == 'latex':
			result = LaTeX_delimiter[config.conf["Access8Math"]["settings"]["LaTeX_delimiter"]]
		elif self.pointer['type'] == 'asciimath':
			result = AsciiMath_delimiter["graveaccent"]
		elif self.pointer['type'] == 'nemeth':
			result = Nemeth_delimiter[config.conf["Access8Math"]["settings"]["Nemeth_delimiter"]]
		else:
			result = {
				"start": "",
				"end": "",
				"type": self.pointer['type'],
			}
		return result

	@property
	def inSection(self):
		if self.pointer is None:
			return True

		delimiter_start_length = len(self.delimiter["start"])
		delimiter_end_length = len(self.delimiter["end"])

		if self.caret._startOffset >= self.pointer['start'] + delimiter_start_length and self.caret._endOffset <= self.pointer['end'] - delimiter_end_length:
			return True
		else:
			return False

	@property
	def inMath(self):
		if self.pointer is None:
			return False

		if self.inSection and (self.pointer['type'] == 'latex' or self.pointer['type'] == 'asciimath' or self.pointer['type'] == 'nemeth' or self.pointer['type'] == 'mathml'):
			return True
		else:
			return False

	@property
	def inText(self):
		if self.pointer is None:
			return True

		if self.inSection and self.pointer['type'] == 'text':
			return True
		else:
			return False

	@property
	def inLaTeX(self):
		if self.pointer is None:
			return False

		if self.inSection and self.pointer['type'] == 'latex':
			return True
		else:
			return False

	@property
	def inAsciiMath(self):
		if self.pointer is None:
			return False

		if self.inSection and self.pointer['type'] == 'asciimath':
			return True
		else:
			return False

	@property
	def inNemeth(self):
		if self.pointer is None:
			return False

		if self.inSection and self.pointer['type'] == 'nemeth':
			return True
		else:
			return False

	@property
	def inMathML(self):
		if self.pointer is None:
			return False

		if self.inSection and self.pointer['type'] == 'mathml':
			return True
		else:
			return False

	@property
	def command(self):
		command = {
			"all": None,
			"front": None,
			"back": None,
		}
		if not self.inLaTeX:
			return command

		delimiter_start_length = len(self.delimiter["start"])

		data = self.pointer["data"]

		start = self.pointer['start'] + delimiter_start_length
		current_local = self.caret._startOffset - start

		front_str = data[:current_local]
		back_str = data[current_local:]

		front_result = re.match(r"[A-Za-z]*\\", front_str[::-1])
		if front_result:
			back_result = re.match(r"[A-Za-z]*", back_str)
			front_command = front_result.group(0)[::-1]
			back_command = back_result.group(0)
			command = {
				"all": f"{front_command}{back_command}",
				"front": front_command,
				"back": back_command,
			}

		return command

	def commandSelection(self):
		command = self.command
		if command["all"]:
			current = self.caret._startOffset
			self.caret._startOffset = current - len(command["front"])
			self.caret._endOffset = current + len(command["back"])
			self.caret.updateSelection()

	def move(self, step=0, type_="any", all_index=None):
		MATH_TYPE = ["latex", "asciimath", "nemeth", "mathml"]
		if all_index is not None:
			pointer = self.points[all_index]
		else:
			if step >= 0:
				filte_points = self.points[self.section_index:]
				if self.pointer["type"] != type_ and type_ in MATH_TYPE + ["text"]:
					step -= 1
				if self.pointer["type"] not in MATH_TYPE and type_ == "interactivable":
					step -= 1
			elif step < 0:
				filte_points = self.points[:self.section_index]
			if type_ in MATH_TYPE + ["text"]:
				filte_points = list(filter(lambda i: i['type'] == type_, filte_points))
			elif type_ == 'interactivable':
				filte_points = list(filter(lambda i: i['type'] in MATH_TYPE, filte_points))
			try:
				pointer = filte_points[step]
			except BaseException:
				pointer = None

		if not pointer:
			return None

		if type_ == 'notacrossline' and all_index is not None:
			if self.pointer['line'] != pointer['line']:
				return None

		self.section_index = self.points.index(pointer)
		delimiter_start_length = len(self.delimiter["start"])
		self.caret._startOffset = self.caret._endOffset = pointer['start'] + delimiter_start_length
		self.caret.updateCaret()
		self.caret._startOffset = pointer['start']
		self.caret._endOffset = pointer['end']

		return pointer

	def moveLine(self, step, type=None):
		line = -1
		if self.pointer and self.pointer["line"] + step >= 0 and self.pointer["line"] + step <= self.points[-1]["line"]:
			line = self.pointer["line"] + step
		pointers = []
		for all_index, point in enumerate(self.points):
			if point['line'] == line:
				self.section_index = all_index
				pointers.append(point)

		if len(pointers) > 0:
			if type:
				if type == "home":
					pointers = [pointers[0]]
				elif type == "end":
					pointers = [pointers[-1]]
				self.caret._startOffset = self.caret._endOffset = pointers[0]['start']
				self.caret.updateCaret()
				self.caret._startOffset = pointers[0]['start']
				self.caret._endOffset = pointers[0]['end']
			else:
				self.caret._startOffset = self.caret._endOffset = pointers[0]['start']
				self.caret.updateCaret()
				self.caret._startOffset = pointers[0]['start']
				self.caret._endOffset = pointers[-1]['end']

		return pointers

	def start(self):
		delimiter_start_length = len(self.delimiter["start"])
		if self.pointer:
			self.caret._startOffset = self.caret._endOffset = self.pointer['start'] + delimiter_start_length
			self.caret.updateCaret()
			return self.pointer
		return None

	def end(self):
		delimiter_end_length = len(self.delimiter["end"])
		if self.pointer:
			self.caret._startOffset = self.caret._endOffset = self.pointer['end'] - delimiter_end_length - 1
			self.caret.updateCaret()
			return self.pointer
		return None

	def startMargin(self):
		if self.pointer:
			self.caret._startOffset = self.caret._endOffset = self.pointer['start']
			self.caret.updateCaret()
			return self.pointer
		return None

	def endMargin(self):
		if self.pointer:
			self.caret._startOffset = self.caret._endOffset = self.pointer['end']
			self.caret.updateCaret()
			return self.pointer
		return None


class TextMathEditField(NVDAObject):
	initial = True
	writeNav_mode = False
	_originGestureMap = {}

	def initOverlayClass(self):
		super().initOverlayClass()

	def event_gainFocus(self):
		self.section_manager = SectionManager()
		if self.initial:
			self._originGestureMap = self._gestureMap.copy()
			self.initial = False

		global command_mode, navigate_mode, shortcut_mode, greekAlphabet_mode
		try:
			if command_mode:
				self.bindCommandGestures()
			if navigate_mode:
				self.bindNavigateGestures()
			if shortcut_mode:
				self.bindShortcutGestures()
			if greekAlphabet_mode:
				self.bindGreekAlphabetGestures()
			if self.writeNav_mode:
				self.bindWriteNavGestures()
		except WindowsError:
			pass

		super().event_gainFocus()

	def event_loseFocus(self):
		super().event_loseFocus()

	def event_caret(self):
		super().event_caret()
		# tones.beep(100, 100)

	def bindCommandGestures(self):
		self.bindGesture("kb:alt+b", "batch")
		self.bindGesture("kb:alt+h", "view_math")
		self.bindGesture("kb:alt+i", "interact")
		self.bindGesture("kb:alt+upArrow", "autocomplete")
		self.bindGesture("kb:alt+l", "latex_command")
		self.bindGesture("kb:alt+m", "mark")
		self.bindGesture("kb:alt+t", "translate")

	def unbindCommandGestures(self):
		import inputCore
		for key in ["alt+b", "alt+h", "alt+i", "alt+l", "alt+m", "alt+t", "alt+upArrow"]:
			key = "kb:{}".format(key)
			try:
				self.removeGestureBinding(key)
			except BaseException:
				pass

			key = inputCore.normalizeGestureIdentifier(key)
			if key in self._originGestureMap:
				self._gestureMap[key] = self._originGestureMap[key]

	def bindNavigateGestures(self):
		gestures = [
			"kb:alt+downArrow",
			"kb:alt+leftArrow",
			"kb:alt+rightArrow",
			"kb:alt+shift+downArrow",
			"kb:alt+shift+leftArrow",
			"kb:alt+shift+rightArrow",
			"kb:alt+home",
			"kb:alt+end",
		]
		for key in gestures:
			self.bindGesture(key, "navigate")

	def unbindNavigateGestures(self):
		import inputCore
		gestures = [
			"kb:alt+downArrow",
			"kb:alt+leftArrow",
			"kb:alt+rightArrow",
			"kb:alt+shift+downArrow",
			"kb:alt+shift+leftArrow",
			"kb:alt+shift+rightArrow",
			"kb:alt+home",
			"kb:alt+end",
		]
		for key in gestures:
			try:
				self.removeGestureBinding(key)
			except BaseException:
				pass

			key = inputCore.normalizeGestureIdentifier(key)
			if key in self._originGestureMap:
				self._gestureMap[key] = self._originGestureMap[key]

	def bindShortcutGestures(self):
		for i in range(1, 13):
			key = "kb:f{}".format(i)
			self.bindGesture(key, "shortcut")
			key = "kb:shift+f{}".format(i)
			self.bindGesture(key, "shortcut_help")

		for i in [c for c in "abcdefghijklmnopqrstuvwxyz"]:
			key = "kb:{}".format(i)
			self.bindGesture(key, "shortcut")
			key = "kb:shift+{}".format(i)
			self.bindGesture(key, "shortcut_help")

	def unbindShortcutGestures(self):
		import inputCore
		for i in range(1, 13):
			key = "kb:f{}".format(i)
			try:
				self.removeGestureBinding(key)
			except KeyError:
				pass
			key = inputCore.normalizeGestureIdentifier(key)
			if key in self._originGestureMap:
				self._gestureMap[key] = self._originGestureMap[key]

			key = "kb:shift+f{}".format(i)
			try:
				self.removeGestureBinding(key)
			except KeyError:
				pass
			key = inputCore.normalizeGestureIdentifier(key)
			if key in self._originGestureMap:
				self._gestureMap[key] = self._originGestureMap[key]

		for i in [c for c in "abcdefghijklmnopqrstuvwxyz"]:
			key = "kb:{}".format(i)
			try:
				self.removeGestureBinding(key)
			except KeyError:
				pass
			key = inputCore.normalizeGestureIdentifier(key)
			if key in self._originGestureMap:
				self._gestureMap[key] = self._originGestureMap[key]

			key = "kb:shift+{}".format(i)
			try:
				self.removeGestureBinding(key)
			except KeyError:
				pass
			key = inputCore.normalizeGestureIdentifier(key)
			if key in self._originGestureMap:
				self._gestureMap[key] = self._originGestureMap[key]

	def bindGreekAlphabetGestures(self):
		for i in [c for c in "abcdefghijklmnopqrstuvwxyz"]:
			key = "kb:{}".format(i)
			self.bindGesture(key, "GreekAlphabet")

	def unbindGreekAlphabetGestures(self):
		import inputCore
		for i in [c for c in "abcdefghijklmnopqrstuvwxyz"]:
			key = "kb:{}".format(i)
			try:
				self.removeGestureBinding(key)
			except KeyError:
				pass
			key = inputCore.normalizeGestureIdentifier(key)
			if key in self._originGestureMap:
				self._gestureMap[key] = self._originGestureMap[key]

	def bindWriteNavGestures(self):
		for c in [
			c for c in "abcdefghijklmnopqrstuvwxyz1234567890`-=[];',./\\"
		] \
		+ [
			"upArrow", "downArrow", "leftarrow", "rightarrow", "home", "end", "pageUp", "pageDown", "space", "enter", "delete", "backspace", "tab"
		] \
		+ [
			"control+c", "control+v", "control+x", "control+t",
		] \
		+ [
			"numLockNumpad{number}".format(number=i) for i in range(10)
		] \
		+ [
			"numpadPlus", "numpadMinus", "numpadMultiply", "numpadDivide", "numpadEnter", "numpadDecimal"
		] \
		+ [
			"numLockNumpadPlus", "numLockNumpadMinus", "numLockNumpadMultiply", "numLockNumpadDivide", "numpadDelete"
		]:
			key = "kb:{}".format(c)
			self.bindGesture(key, "writeNav")
			key = "kb:shift+{}".format(c)
			self.bindGesture(key, "writeNav")

	def unbindWriteNavGestures(self):
		import inputCore
		for c in [
			c for c in "abcdefghijklmnopqrstuvwxyz1234567890`-=[];',./\\"
		] \
		+ [
			"upArrow", "downArrow", "leftarrow", "rightarrow", "home", "end", "pageUp", "pageDown", "space", "enter", "control+c", "control+v", "control+x", "delete", "backspace", "tab"
		] \
		+ [
			"control+c", "control+v", "control+x", "control+t",
		] \
		+ [
			"numLockNumpad{number}".format(number=i) for i in range(10)
		] \
		+ [
			"numpadPlus", "numpadMinus", "numpadMultiply", "numpadDivide", "numpadEnter", "numpadDecimal"
		] \
		+ [
			"numLockNumpadPlus", "numLockNumpadMinus", "numLockNumpadMultiply", "numLockNumpadDivide", "numpadDelete"
		]:
			key = "kb:{}".format(c)
			try:
				self.removeGestureBinding(key)
			except KeyError:
				pass
			key = inputCore.normalizeGestureIdentifier(key)
			if key in self._originGestureMap:
				self._gestureMap[key] = self._originGestureMap[key]

			key = "kb:shift+{}".format(c)
			try:
				self.removeGestureBinding(key)
			except KeyError:
				pass
			key = inputCore.normalizeGestureIdentifier(key)
			if key in self._originGestureMap:
				self._gestureMap[key] = self._originGestureMap[key]

	def gesture_not_concurrent(self):
		global shortcut_mode
		if shortcut_mode:
			shortcut_mode = not shortcut_mode
			self.unbindShortcutGestures()
		global greekAlphabet_mode
		if greekAlphabet_mode:
			greekAlphabet_mode = not greekAlphabet_mode
			self.unbindGreekAlphabetGestures()

	@script(
		gestures=["kb:nvda+alt+c"],
		# Translators: The description of a command in input help mode and in the Input gesture dialog
		description=_("Toggle command gestures"),
		category=ADDON_SUMMARY,
	)
	def script_command_toggle(self, gesture):
		global command_mode
		command_mode = not command_mode
		if command_mode:
			self.bindCommandGestures()
			ui.message(_("Command gestures activated"))
		else:
			self.unbindCommandGestures()
			ui.message(_("Command gestures deactivated"))

	@script(
		gestures=["kb:nvda+alt+n"],
		# Translators: The description of a command in input help mode and in the Input gesture dialog
		description=_("Toggle block navigation gestures"),
		category=ADDON_SUMMARY,
	)
	def script_navigate_toggle(self, gesture):
		global navigate_mode
		navigate_mode = not navigate_mode
		if navigate_mode:
			self.bindNavigateGestures()
			ui.message(_("Block navigation gestures activated"))
		else:
			self.unbindNavigateGestures()
			ui.message(_("Block navigation gestures deactivated"))

	@script(
		gestures=["kb:nvda+alt+s"],
		# Translators: The description of a command in input help mode and in the Input gesture dialog
		description=_("Toggle shortcut gestures"),
		category=ADDON_SUMMARY,
	)
	def script_shortcut_toggle(self, gesture):
		if self.writeNav_mode:
			ui.message(_("Cannot activate shortcut gesture in browse navigation mode"))
			return
		global shortcut_mode
		if not shortcut_mode:
			self.gesture_not_concurrent()
			shortcut_mode = not shortcut_mode
			self.bindShortcutGestures()
			ui.message(_("Shortcut gestures activated"))
		else:
			shortcut_mode = not shortcut_mode
			self.unbindShortcutGestures()
			ui.message(_("Shortcut gestures deactivated"))

	@script(
		gestures=["kb:nvda+alt+g"],
		# Translators: The description of a command in input help mode and in the Input gesture dialog
		description=_("Toggle Greek alphabet gesture"),
		category=ADDON_SUMMARY,
	)
	def script_greekAlphabet_toggle(self, gesture):
		if self.writeNav_mode:
			ui.message(_("Cannot activate Greek alphabet gestures in browse navigation mode"))
			return
		global greekAlphabet_mode
		if not greekAlphabet_mode:
			self.gesture_not_concurrent()
			greekAlphabet_mode = not greekAlphabet_mode
			self.bindGreekAlphabetGestures()
			ui.message(_("Greek alphabet gestures activated"))
		else:
			greekAlphabet_mode = not greekAlphabet_mode
			self.unbindGreekAlphabetGestures()
			ui.message(_("Greek alphabet gestures deactivated"))

	@script(
		gestures=["kb:NVDA+space"],
		# Translators: The description of a command in input help mode and in the Input gesture dialog
		description=_("Toggle browse navigation mode"),
		category=ADDON_SUMMARY,
	)
	def script_writeNav_toggle(self, gesture):
		self.writeNav_mode = not self.writeNav_mode
		if self.writeNav_mode:
			self.gesture_not_concurrent()
			self.bindWriteNavGestures()
			if config.conf["Access8Math"]["settings"]["writeNavAudioIndication"]:
				sound = "browseMode.wav"
				nvwave.playWaveFile(os.path.join("waves", sound))
			else:
				ui.message(_("Browse navigation mode on"))
			with self.section_manager as manager:
				result = manager.move(type_='any', step=0)
				self.displayBlocks([result], "view")
		else:
			self.unbindWriteNavGestures()
			if config.conf["Access8Math"]["settings"]["writeNavAudioIndication"]:
				sound = "focusMode.wav"
				nvwave.playWaveFile(os.path.join("waves", sound))
			else:
				ui.message(_("Browse navigation mode off"))
			api.getFocusObject().event_gainFocus()

	def script_writeNav_exit(self, gesture):
		if not self.writeNav_mode:
			self.writeNav_mode = not self.writeNav_mode
			self.bindWriteNavGestures()
			if config.conf["Access8Math"]["settings"]["writeNavAudioIndication"]:
				sound = "browseMode.wav"
				nvwave.playWaveFile(os.path.join("waves", sound))
			else:
				ui.message(_("Browse navigation mode on"))
		else:
			gesture.send()

	def script_shortcut(self, gesture):
		with self.section_manager as manager:
			if manager.inMath:
				view = A8MLaTeXCommandView(inSection=manager.inMath)
				slot = gesture.mainKeyName[1:] if len(gesture.mainKeyName) > 1 else gesture.mainKeyName
				if slot in view.data.shortcut:
					view.command(view.data.shortcut[slot]["id"])
				else:
					gesture.send()
			else:
				gesture.send()

	def script_shortcut_help(self, gesture):
		with self.section_manager as manager:
			if manager.inMath:
				view = A8MLaTeXCommandView(inSection=manager.inMath)
				slot = gesture.mainKeyName[1:] if len(gesture.mainKeyName) > 1 else gesture.mainKeyName
				if slot in view.data.shortcut:
					ui.message(view.data.shortcut[slot]["name"])
				else:
					if True:
						gesture.send()
					else:
						ui.message(_("Shortcut {shortcut} is not set").format(
							shortcut=slot,
						))
			else:
				gesture.send()

	def script_GreekAlphabet(self, gesture):
		with self.section_manager as manager:
			if manager.inMath:
				view = A8MLaTeXCommandView(inSection=manager.inMath)
				slot = gesture.mainKeyName
				if slot in view.data.greekAlphabet:
					view.greekAlphabetCommand(view.data.greekAlphabet[slot]["id"])
				else:
					ui.message(_("Greek alphabet {GreekAlphabet} is not set").format(
						GreekAlphabet=slot,
					))
			else:
				gesture.send()

	def script_writeNav(self, gesture):
		if gesture.mainKeyName in ["space", "enter", "numpadEnter"]:
			self.script_interact(gesture)
		elif gesture.mainKeyName in ["downArrow", "upArrow", "home", "end", "pageUp", "pageDown"]:
			self.script_navigateLine(gesture)
		elif gesture.mainKeyName == "c" and "control" in gesture.modifierNames:
			self.script_navigateCopy(gesture)
		elif gesture.mainKeyName == "v" and "control" in gesture.modifierNames:
			self.script_navigatePaste(gesture)
		elif gesture.mainKeyName == "x" and "control" in gesture.modifierNames:
			self.script_navigateCut(gesture)
		elif gesture.mainKeyName == "t" and "control" in gesture.modifierNames:
			self.script_translate(gesture)
		elif gesture.mainKeyName in ["delete", "backspace"]:
			self.script_navigateDelete(gesture)
		else:
			self.script_navigate(gesture)

	def script_view_math(self, gesture):
		document = self.makeTextInfo(textInfos.POSITION_ALL)

		obj = api.getForegroundObject()
		title = obj.name
		if not isinstance(title, str) or not title or title.isspace():
			title = obj.appModule.appName if obj.appModule else None
			if not isinstance(title, str) or not title or title.isspace():
				title = "index.txt"

		name = title.split("-")[0].strip('* ')
		ext = 'txt'

		data_folder = os.path.join(PATH, 'web', 'workspace', 'default')
		entry_file = f'{name}.{ext}'

		try:
			shutil.rmtree(data_folder)
		except BaseException:
			pass
		if not os.path.exists(data_folder):
			os.makedirs(data_folder)
		with open(os.path.join(data_folder, entry_file), "w", encoding="utf8", newline="") as f:
			f.write(document.text)

		ad = Access8MathDocument(os.path.join(data_folder, entry_file))
		ad.raw2review()
		A8MHTMLCommandView(
			ad=ad
		).setFocus()

	def script_mark(self, gesture):
		with self.section_manager as manager:
			if (manager.pointer and manager.pointer['type'] == 'text') or len(manager.points) == 0:
				view = A8MMarkCommandView(section=manager)
				view.setFocus()
			else:
				ui.message(_("In math section. Please leave math section first and try again."))

	def script_latex_command(self, gesture):
		with self.section_manager as manager:
			if manager.inLaTeX or manager.inText:
				A8MLaTeXCommandView(inSection=manager.inLaTeX).setFocus()
			else:
				ui.message(_("Not in LaTeX block or text block. cannot use LaTeX command"))

	def script_interact(self, gesture):
		with self.section_manager as manager:
			if manager.inMath:
				if manager.pointer['type'] == 'latex':
					mathMl = latex2mathml(manager.pointer['data'])
				elif manager.pointer['type'] == 'asciimath':
					mathMl = asciimath2mathml(manager.pointer['data'])
				elif manager.pointer['type'] == 'nemeth':
					mathMl = latex2mathml(nemeth2latex(manager.pointer['data']))
				elif manager.pointer['type'] == 'mathml':
					mathMl = manager.pointer['data']
				mathMl = mathMl.replace("<<", "&lt;<").replace(">>", ">&gt;")
				mathPres.interactionProvider.interactWithMathMl(mathMl)
			else:
				ui.message(_("This block cannot be interacted"))

	def script_autocomplete(self, gesture):
		with self.section_manager as manager:
			command = manager.command["all"]
			if command:
				view = A8MAutocompleteCommandView(section=manager, command=command)
				if len(view.data.data) > 0:
					view.setFocus()
				else:
					ui.message(_("No autocomplete found"))
			else:
				ui.message(_("No autocomplete found"))

	def script_translate(self, gesture):
		with self.section_manager as manager:
			if manager.inLaTeX or manager.inAsciiMath or manager.inNemeth:
				A8MTranslateCommandView(section=manager).setFocus()
			else:
				ui.message(_("This block cannot be translated"))

	def script_batch(self, gesture):
		with self.section_manager as manager:
			A8MBatchCommandView(section=manager).setFocus()

	def displayBlocks(self, results, mode):
		text = []
		brailleRegion = []
		for result in results:
			if result['data'] == "":
				continue
			if mode == "view" and result['type'] == "latex":
				try:
					mathMl = latex2mathml(result['data'])
					mathMl = mathMl.replace("<<", "&lt;<").replace(">>", ">&gt;")
					text += mathPres.speechProvider.getSpeechForMathMl(mathMl)
					brailleRegion += ["⠀", "".join(mathPres.speechProvider.getBrailleForMathMl(mathMl)), "⠀"]
				except BaseException:
					text += result['data']
					brailleRegion += [result['data']]
			elif mode == "view" and result['type'] == "asciimath":
				try:
					mathMl = asciimath2mathml(result['data'])
					mathMl = mathMl.replace("<<", "&lt;<").replace(">>", ">&gt;")
					text += mathPres.speechProvider.getSpeechForMathMl(mathMl)
					brailleRegion += ["⠀", "".join(mathPres.speechProvider.getBrailleForMathMl(mathMl)), "⠀"]
				except BaseException:
					text += result['data']
					brailleRegion += [result['data']]
			elif mode == "view" and result['type'] == "nemeth":
				try:
					mathMl = latex2mathml(nemeth2latex(result['data']))
					mathMl = mathMl.replace("<<", "&lt;<").replace(">>", ">&gt;")
					text += mathPres.speechProvider.getSpeechForMathMl(mathMl)
					brailleRegion += ["⠀", "".join(mathPres.speechProvider.getBrailleForMathMl(mathMl)), "⠀"]
				except BaseException:
					text += result['data']
					brailleRegion += [result['data']]
			elif mode == "view" and result['type'] == "mathml":
				try:
					mathMl = result['data']
					text += mathPres.speechProvider.getSpeechForMathMl(mathMl)
					brailleRegion += ["⠀", "".join(mathPres.speechProvider.getBrailleForMathMl(mathMl)), "⠀"]
				except BaseException:
					text += result['data']
					brailleRegion += [result['data']]
			else:
				text += [result['data']]
				brailleRegion += [result['data']]
		brailleRegion = [braille.TextRegion("".join(brailleRegion))]
		# brailleRegion = [braille.TextRegion(br) for br in brailleRegion]

		try:
			speech.speak(text)
			display_braille(brailleRegion)
		except BaseException:
			pass

	def script_navigate(self, gesture):
		with self.section_manager as manager:
			selected = False
			if gesture.mainKeyName == "downArrow":
				result = manager.move(type_='any', step=0)
				if "shift" in gesture.modifierNames:
					selected = True
			elif gesture.mainKeyName == "leftArrow":
				if "alt" in gesture.modifierNames or config.conf["Access8Math"]["settings"]["writeNavAcrossLine"]:
					result = manager.move(type_='any', step=-1)
				else:
					result = manager.move(type_='notacrossline', step=-1)
				if "shift" in gesture.modifierNames:
					selected = True
			elif gesture.mainKeyName == "rightArrow":
				if "alt" in gesture.modifierNames or config.conf["Access8Math"]["settings"]["writeNavAcrossLine"]:
					result = manager.move(type_='any', step=1)
				else:
					result = manager.move(type_='notacrossline', step=1)
				if "shift" in gesture.modifierNames:
					selected = True
			elif gesture.mainKeyName == "tab":
				if "shift" in gesture.modifierNames:
					result = manager.move(type_='interactivable', step=-1)
				else:
					result = manager.move(type_='interactivable', step=1)
			elif gesture.mainKeyName == "t":
				if "shift" in gesture.modifierNames:
					result = manager.move(type_='text', step=-1)
				else:
					result = manager.move(type_='text', step=1)
			elif gesture.mainKeyName == "l":
				if "shift" in gesture.modifierNames:
					result = manager.move(type_='latex', step=-1)
				else:
					result = manager.move(type_='latex', step=1)
			elif gesture.mainKeyName == "a":
				if "shift" in gesture.modifierNames:
					result = manager.move(type_='asciimath', step=-1)
				else:
					result = manager.move(type_='asciimath', step=1)
			elif gesture.mainKeyName == "n":
				if "shift" in gesture.modifierNames:
					result = manager.move(type_='nemeth', step=-1)
				else:
					result = manager.move(type_='nemeth', step=1)
			elif gesture.mainKeyName == "m":
				if "shift" in gesture.modifierNames:
					result = manager.move(type_='mathml', step=-1)
				else:
					result = manager.move(type_='mathml', step=1)
			elif gesture.mainKeyName == "home":
				result = manager.start()
			elif gesture.mainKeyName == "end":
				result = manager.end()
			else:
				result = None

			if not result:
				result = manager.move(type_='any', step=0)
				tones.beep(100, 50)

			mode = "raw" if "alt" in gesture.modifierNames else "view"
			if selected:
				manager.caret.updateSelection()
			else:
				self.displayBlocks([result], mode)

	def script_navigateLine(self, gesture):
		with self.section_manager as manager:
			selected = False
			if gesture.mainKeyName == "upArrow":
				results = manager.moveLine(step=-1)
				if "shift" in gesture.modifierNames:
					selected = True
			elif gesture.mainKeyName == "downArrow":
				results = manager.moveLine(step=1)
				if "shift" in gesture.modifierNames:
					selected = True
			elif gesture.mainKeyName == "home":
				results = manager.moveLine(step=0, type=gesture.mainKeyName)
				if "shift" in gesture.modifierNames:
					selected = True
			elif gesture.mainKeyName == "end":
				results = manager.moveLine(step=0, type=gesture.mainKeyName)
				if "shift" in gesture.modifierNames:
					selected = True
			elif gesture.mainKeyName == "pageUp":
				results = manager.moveLine(step=-10)
				if "shift" in gesture.modifierNames:
					selected = True
			elif gesture.mainKeyName == "pageDown":
				results = manager.moveLine(step=10)
				if "shift" in gesture.modifierNames:
					selected = True
			else:
				results = []
				if "shift" in gesture.modifierNames:
					selected = True

			if len(results) == 0:
				results = manager.moveLine(step=0)
				tones.beep(100, 50)

			mode = "view"
			if selected:
				manager.caret.updateSelection()
			else:
				self.displayBlocks(results, mode)

	def script_navigateCopy(self, gesture):
		with self.section_manager as manager:
			result = manager.move(type_='any', step=0)
			api.copyToClip(result["raw"])
			ui.message(_("{data} copied to clipboard").format(data=result["raw"]))

	def script_navigatePaste(self, gesture):
		with self.section_manager as manager:
			manager.endMargin()
			KeyboardInputGesture.fromName("control+v").send()
			ui.message(_("{data} inserted to document").format(data=api.getClipData()))

	def script_navigateCut(self, gesture):
		with self.section_manager as manager:
			result = manager.move(type_='any', step=0)
			manager.caret.updateSelection()
			KeyboardInputGesture.fromName("control+x").send()
			ui.message(_("{data} cut from document").format(data=result["raw"]))

	def script_navigateDelete(self, gesture):
		with self.section_manager as manager:
			result = manager.move(type_='any', step=0)
			manager.caret.updateSelection()
			KeyboardInputGesture.fromName("delete").send()
			ui.message(_("{data} deleted from document").format(data=result["raw"]))
