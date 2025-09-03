from collections.abc import Iterable
import re
import os
import wx

import addonHandler
import api
import braille
import brailleTables
import config
import eventHandler
import globalVars
from logHandler import log
import louisHelper
from NVDAObjects.window import Window
import mathPres
from mathPres.mathPlayer import MathPlayer
from scriptHandler import script
import speech
from speech.commands import BreakCommand, PitchCommand
from speech.speech import _getSpellingCharAddCapNotification
from speechXml import SsmlParser
from synthDriverHandler import getSynth
import textInfos
import tones
import ui

import A8M_PM
from A8M_PM import MathContent
from lib.braille import display_braille
from lib.mathProcess import mathml2latex

addonHandler.initTranslation()

mathPlayer = None
try:
	mathPlayer = MathPlayer()
except BaseException:
	log.warning("MathPlayer 4 not available")

mathCAT = None
try:
	from globalPlugins.MathCAT import MathCAT
	mathCAT = MathCAT()
except BaseException:
	log.warning("MathCAT not available")

BRAILLE_UNICODE_PATTERNS_START = 0x2800


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


class A8MSsmlParser(SsmlParser):
	def _elementHandler(self, tagName: str, attrs: dict | None = None):
		processedTagName = "".join(tagName.title().split("-"))
		funcName = f"parse{processedTagName}"
		if (func := getattr(self, funcName, None)) is None:
			log.debugWarning(f"Unsupported tag: {tagName}")
			return
		for command in func(attrs):
			# If the last command in the sequence is of the same type, we can remove it.
			if self._speechSequence and type(self._speechSequence[-1]) is type(command):
				self._speechSequence.pop()
			self._speechSequence.append(command)


def translate_SpeechCommand(serializes):
	"""
	convert Access8Math serialize object to SpeechCommand
	@param lines: source serializes
	@type list
	@rtype SpeechCommand
	"""
	item = flatten(serializes)
	ssml = "<speak>" + "".join(item) + "</speak>"
	parser = A8MSsmlParser()
	speechSequence = parser.convertFromXml(ssml)
	return speechSequence


def translate_SpeechCommand_CapNotification(serializes):
	command = []
	for item in translate_SpeechCommand(serializes):
		if isinstance(item, str):
			seq = getCharAddCapNotification(item)
			command.extend(list(seq))
		else:
			command.append(item)
	return command


def translate_Unicode(serializes):
	"""
	convert Access8Math serialize object to SpeechCommand
	@param lines: source serializes
	@type list
	@rtype SpeechCommand
	"""
	pattern = re.compile(r'<break time="(?P<time>[\d]*)ms" />')
	result = ""

	for c in serializes:
		result += "\n"
		for item in flatten(c):
			ssml = "<speak>" + "".join(item) + "</speak>"
			parser = A8MSsmlParser()
			sequence = parser.convertFromXml(ssml)
			sequence = [command for command in sequence if type(command) is not BreakCommand]
			string = " ".join(sequence).strip()
			result = result + " " + string

	# replace mutiple blank to single blank
	pattern = re.compile(r'[ ]+')
	sequence = pattern.sub(lambda m: ' ', result)

	# replace blank line to none
	pattern = re.compile(r'\n\s*\n')
	sequence = pattern.sub(lambda m: '\n', sequence)

	# strip blank at start and end line
	temp = ''
	for i in sequence.split('\n'):
		temp = temp + i.strip() + '\n'
	sequence = temp
	return sequence.strip()

	speechSequence = []
	for s in serializes:
		for item in flatten(s):
			ssml = "<speak>" + "".join(item) + "</speak>"
			parser = A8MSsmlParser()
			sequence = parser.convertFromXml(ssml)
			sequence = [command for command in sequence if type(command) is not BreakCommand]
			string = " ".join(sequence).strip()
			if string != "":
				speechSequence.append(string)

		speechSequence = "\n".join(speechSequence)
	return speechSequence.strip()


def translate_Braille(serializes):
	"""
	convert Access8Math serialize object to braille
	@param lines: source serializes
	@type list
	@rtype str
	"""
	temp = ''
	for string in flatten(serializes):
		brailleCells, brailleToRawPos, rawToBraillePos, brailleCursorPos = louisHelper.translate([os.path.join(brailleTables.TABLES_DIR, config.conf["braille"]["translationTable"]), "braille-patterns.cti"], string, mode=4)
		temp += "".join([chr(BRAILLE_UNICODE_PATTERNS_START + cell) for cell in brailleCells])
	return "".join(temp)


def getCharAddCapNotification(text: str):
	if len(text) != 1 or not text.isupper():
		return [text]

	synth = getSynth()
	synthConfig = config.conf["speech"][synth.name]

	if PitchCommand in synth.supportedCommands:
		capPitchChange = synthConfig["capPitchChange"]
	else:
		capPitchChange = 0

	seq = list(_getSpellingCharAddCapNotification(
		text,
		sayCapForCapitals=synthConfig["sayCapForCapitals"],
		capPitchChange=capPitchChange,
		beepForCapitals=synthConfig["beepForCapitals"],
	))
	return seq


class GenericFrame(wx.Frame):
	def __init__(self, *args, **kwargs):
		super(GenericFrame, self).__init__(*args, **kwargs)
		self.buttons = []

		self.CreateStatusBar()  # A StatusBar in the bottom of the window
		self.createMenuBar()

		self.panel = wx.Panel(self, -1)
		self.createButtonBar(self.panel)

		mainSizer = wx.BoxSizer(wx.HORIZONTAL)
		for button in self.buttons:
			mainSizer.Add(button)

		self.panel.SetSizer(mainSizer)
		mainSizer.Fit(self)

	def menuData(self):
		return [
		]

	def createMenuBar(self):
		self.menuBar = wx.MenuBar()
		for eachMenuData in self.menuData():
			menuLabel = eachMenuData[0]
			menuItems = eachMenuData[1]
			self.menuBar.Append(self.createMenu(menuItems), menuLabel)

		self.SetMenuBar(self.menuBar)

	def createMenu(self, menuData):
		menu = wx.Menu()
		for eachItem in menuData:
			if len(eachItem) == 2:
				label = eachItem[0]
				subMenu = self.createMenu(eachItem[1])
				menu.AppendMenu(wx.NewId(), label, subMenu)

			else:
				self.createMenuItem(menu, *eachItem)
		return menu

	def createMenuItem(self, menu, label, status, handler, kind=wx.ITEM_NORMAL):
		if not label:
			menu.AppendSeparator()
			return
		menuItem = menu.Append(-1, label, status, kind)
		self.Bind(wx.EVT_MENU, handler, menuItem)

	def buttonData(self):
		return [
		]

	def createButtonBar(self, panel, yPos=0):
		xPos = 0
		for eachLabel, eachHandler in self.buttonData():
			pos = (xPos, yPos)
			button = self.buildOneButton(panel, eachLabel, eachHandler, pos)
			self.buttons.append(button)
			xPos += button.GetSize().width

	def buildOneButton(self, parent, label, handler, pos=(0, 0)):
		button = wx.Button(parent, -1, label, pos)
		self.Bind(wx.EVT_BUTTON, handler, button)
		return button


class A8MInteractionFrame(GenericFrame):
	def __init__(self):
		# Translators: The title of the interaction window
		title = _("Access8Math interaction window")
		super().__init__(wx.GetApp().TopWindow, title=title)
		self.Bind(wx.EVT_CHAR_HOOK, self.OnChar)

	def menuData(self):
		return [
			# Translators: A mmenu item in the Interaction window
			(_("&Menu"), (
				# Translators: A mmenu item in the Interaction window
				(_("&Exit"), _("Terminate the program"), self.OnExit),
			))
		]

	def buttonData(self):
		return (
			# Translators: A button label in the Interaction window
			(_("Interaction"), self.OnInteraction),
			# Translators: A button label in the Interaction window
			(_("Copy"), self.OnRawdataToClip),
		)

	def OnExit(self, event):
		self.Destroy()
		global main_frame
		main_frame = None

	def OnChar(self, event):
		keyCode = event.GetKeyCode()
		if keyCode == wx.WXK_ESCAPE:
			self.Destroy()
			global main_frame
			main_frame = None
			# self.Close()
		event.Skip()

	def set_mathcontent(self, mathcontent):
		self.mathcontent = mathcontent
		globalVars.mathcontent = mathcontent

	def OnInteraction(self, event):
		parent = api.getFocusObject()
		vw = A8MInteraction(parent=parent)
		vw.set(data=self.mathcontent, name="")
		vw.setFocus()

	def OnRawdataToClip(self, event):
		api.copyToClip(self.mathcontent.root.get_mathml())
		# Translators: A message reported to the user when copying data from the interaction window
		ui.message(_("Copied"))


class A8MProvider(mathPres.MathPresentationProvider):
	def getSpeechForMathMl(self, mathMl):
		"""Get speech output for specified MathML markup.
		@param mathMl: The MathML markup.
		@type mathMl: str
		@return: A speech sequence.
		@rtype: List[str, SpeechCommand]
		"""
		speechSequence = []
		if config.conf["Access8Math"]["settings"]["speech_source"] == "Access8Math":
			mathcontent = MathContent(config.conf["Access8Math"]["settings"]["language"], mathMl)
			speechSequence = translate_SpeechCommand_CapNotification(mathcontent.pointer.serialized())
		elif config.conf["Access8Math"]["settings"]["speech_source"] == "MathCAT":
			if mathCAT:
				speechSequence = mathCAT.getSpeechForMathMl(mathMl)
		else:
			if mathPlayer:
				speechSequence = mathPlayer.getSpeechForMathMl(mathMl)
		return speechSequence

	def getBrailleForMathMl(self, mathMl):
		"""Get braille output for specified MathML markup.
		@param mathMl: The MathML markup.
		@type mathMl: str
		@return: A string of Unicode braille.
		@rtype: unicode
		"""
		cells = ""
		if config.conf["Access8Math"]["settings"]["braille_source"] == "Access8Math":
			mathcontent = MathContent(config.conf["Access8Math"]["settings"]["language"], mathMl)
			cells = translate_Braille(mathcontent.root.brailleserialized())
		elif config.conf["Access8Math"]["settings"]["braille_source"] == "MathCAT":
			if mathCAT:
				cells = mathCAT.getBrailleForMathMl(mathMl)
		else:
			if mathPlayer:
				cells = mathPlayer.getBrailleForMathMl(mathMl)

		def inrange(cell):
			return ord(cell) >= BRAILLE_UNICODE_PATTERNS_START and ord(cell) < BRAILLE_UNICODE_PATTERNS_START + 256

		cells = [cell if inrange(cell) else chr(BRAILLE_UNICODE_PATTERNS_START) for cell in cells]
		return "".join(cells)

	def interactWithMathMl(self, mathMl):
		"""Begin interaction with specified MathML markup.
		@param mathMl: The MathML markup.
		"""
		if config.conf["Access8Math"]["settings"]["interact_source"] == "Access8Math":
			mathcontent = MathContent(config.conf["Access8Math"]["settings"]["language"], mathMl)
			vw = A8MInteraction(parent=api.getFocusObject())
			vw.set(data=mathcontent, name="")
			vw.setFocus()
		elif config.conf["Access8Math"]["settings"]["interact_source"] == "MathCAT":
			if mathCAT:
				mathCAT.interactWithMathMl(mathMl)
		else:
			if mathPlayer:
				mathPlayer.interactWithMathMl(mathMl)


class A8MInteraction(Window):
	# role = controlTypes.Role.MATH
	# Override the window name.
	# name = None
	# Any tree interceptor should not apply here.
	treeInterceptor = None

	def __init__(self, parent, root=None):
		self.parent = parent
		self.mathcontent = self.data = None
		super().__init__(windowHandle=self.parent.windowHandle)

	def set(self, name, data, *args, **kwargs):
		# self.name = name + " - math window"
		self.mathcontent = self.data = data

	def setFocus(self):
		eventHandler.executeEvent("gainFocus", self)

	@script(
		gestures=["kb:escape"]
	)
	def script_escape(self, gesture):
		eventHandler.executeEvent("gainFocus", self.parent)

	def _get_mathMl(self):
		return self.mathcontent.root.get_mathml()

	def makeTextInfo(self, position=textInfos.POSITION_FIRST):
		return A8MInteractionTextInfo(self, position)

	def event_gainFocus(self):
		# Translators: A message reported to the user when the interaction window takes the focus
		speech.speak([_("Entering interaction mode")])
		super().event_gainFocus()

	def reportFocus(self):
		speech.speak(translate_SpeechCommand_CapNotification(self.mathcontent.root.serialized()))

	def getBrailleRegions(self, review=False):
		yield braille.NVDAObjectRegion(self, appendText=" ")
		region = braille.Region()
		region.focusToHardLeft = True
		cells = ""
		if config.conf["Access8Math"]["settings"]["braille_source"] == "Access8Math":
			cells = translate_Braille(self.mathcontent.root.brailleserialized())
		else:
			if mathPlayer:
				cells = mathPlayer.getBrailleForMathMl(self.mathcontent.raw_mathMl)

		def inrange(cell):
			return ord(cell) >= BRAILLE_UNICODE_PATTERNS_START and ord(cell) < BRAILLE_UNICODE_PATTERNS_START + 256

		cells = [cell if inrange(cell) else chr(BRAILLE_UNICODE_PATTERNS_START) for cell in cells]
		region.rawText = "".join(cells)
		yield region

	@script(
		gestures=[
			"kb:downArrow", "kb:upArrow",
			"kb:leftArrow", "kb:rightArrow",
			"kb:control+alt+downArrow", "kb:control+alt+upArrow",
			"kb:control+alt+leftArrow", "kb:control+alt+rightArrow",
			"kb:home",
		],
	)
	def script_navigate(self, gesture):
		r = False
		if gesture.mainKeyName == "home":
			r = self.mathcontent.navigate(gesture.mainKeyName)
		elif gesture.mainKeyName in ["downArrow", "upArrow", "leftArrow", "rightArrow"] and "control" in gesture.modifierNames and "alt" in gesture.modifierNames:
			r = self.mathcontent.table_navigate(gesture.mainKeyName)
		elif gesture.mainKeyName in ["downArrow", "upArrow", "leftArrow", "rightArrow"]:
			r = self.mathcontent.navigate(gesture.mainKeyName)

		if not r:
			if config.conf["Access8Math"]["settings"]["no_move_beep"]:
				tones.beep(100, 100)
			else:
				# Translators: A message reported to the user when navigating in the math expression of the interaction window
				speech.speak([_("No move")])

		api.setReviewPosition(self.makeTextInfo(), clearNavigatorObject=False, isCaret=True)

		speech.speak(self.mathcontent.hint)
		if self.mathcontent.pointer.parent:
			if config.conf["Access8Math"]["settings"]["auto_generate"] and self.mathcontent.pointer.parent.role_level == A8M_PM.AUTO_GENERATE:
				speech.speak([self.mathcontent.pointer.des])
			elif self.mathcontent.pointer.parent.role_level == A8M_PM.DIC_GENERATE:
				speech.speak([self.mathcontent.pointer.des])
		else:
			speech.speak([self.mathcontent.pointer.des])

		command = translate_SpeechCommand_CapNotification(self.mathcontent.pointer.serialized())
		speech.speak(command)

		cells = translate_Braille(self.mathcontent.pointer.brailleserialized())
		brailleRegion = [braille.TextRegion(cells)]
		display_braille(brailleRegion)

	@script(
		gesture="kb:control+r",
	)
	def script_RawDataToClip(self, gesture):
		api.copyToClip(self.mathcontent.raw_mathMl)
		# Translators: A message reported to the user when copying data from the Interaction window
		ui.message(_("Copied raw MathML"))

	@script(
		gesture="kb:control+m",
	)
	def script_MathMLToClip(self, gesture):
		api.copyToClip(self.mathcontent.mathML)
		# Translators: A message reported to the user when copying data from the Interaction window
		ui.message(_("Copied MathML"))

	@script(
		gesture="kb:control+l",
	)
	def script_LaTeXToClip(self, gesture):
		try:
			latex = mathml2latex(self.mathcontent.mathML)
			api.copyToClip(latex)
		except BaseException:
			# Translators: A message reported to the user when copying LaTeX failed from the Interaction window
			ui.message(_("Copy LaTeX failed"))
		# Translators: A message reported to the user when copying LaTeX from the Interaction window
		ui.message(_("Copied LaTeX"))

	@script(
		gesture="kb:control+s",
	)
	def script_snapshot(self, gesture):
		# Translators: A message reported to the user when taking a snapshot of math data in the Interaction window
		ui.message(_("snapshot"))
		globalVars.mathcontent = self.mathcontent


class A8MInteractionTextInfo(textInfos.offsets.OffsetsTextInfo):
	def __init__(self, obj, position):
		super().__init__(obj, position)
		self.obj = obj

	def _getStoryLength(self):
		serializes = self.obj.mathcontent.pointer.serialized()
		return len(translate_Unicode(serializes))

	def _getStoryText(self):
		serializes = self.obj.mathcontent.pointer.serialized()
		return translate_Unicode(serializes)

	def _getTextRange(self, start, end):
		text = self._getStoryText()
		return text[start:end] if text else u""


def show_main_frame(mathcontent):
	global main_frame
	if not main_frame:
		main_frame = A8MInteractionFrame()
	main_frame.set_mathcontent(mathcontent=mathcontent)
	main_frame.Show()
	main_frame.Raise()


main_frame = None
