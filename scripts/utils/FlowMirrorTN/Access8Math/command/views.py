import api
import eventHandler
from keyboardHandler import KeyboardInputGesture
from NVDAObjects.window import Window
from scriptHandler import script
import textInfos
import textInfos.offsets
import tones
import ui


class MenuView(Window):
	name = _("command")

	def __init__(self, MenuModel, TextInfo):
		self.parent = api.getFocusObject()
		self.data = MenuModel()
		self.TextInfo = TextInfo
		super().__init__(windowHandle=self.parent.windowHandle)

	def getScript(self, gesture):
		if isinstance(gesture, KeyboardInputGesture):
			if (gesture.mainKeyName in ["NVDA", "enter", "escape", "leftArrow", "rightArrow", "upArrow", "downArrow", "home", "end", "numpadEnter"] or "NVDA" in gesture.modifierNames):
				return super().getScript(gesture)
			elif gesture.mainKeyName in ["numpad{}".format(i) for i in range(1, 10)] + ["numLock"]:
				return super().getScript(gesture)
		return lambda s: None

	def makeTextInfo(self, position=textInfos.POSITION_FIRST):
		return self.TextInfo(self, position)

	def syncTextInfoPosition(self):
		info = api.getReviewPosition().obj.makeTextInfo(textInfos.POSITION_FIRST)
		info.expand(textInfos.UNIT_CHARACTER)
		info.collapse()
		api.setReviewPosition(info)

	def setFocus(self):
		eventHandler.executeEvent("gainFocus", self)
		ui.message(self.data.pointer['name'])
		if "shortcut" in self.data.pointer and self.data.pointer["shortcut"] != "-1":
			ui.message(_("f{shortcut}").format(shortcut=self.data.pointer["shortcut"]))
		if self.data.pointer["type"] == "menu":
			ui.message(_("subMenu"))
		ui.message(_("{number} of {total}").format(
			number=self.data.path[-1] + 1,
			total=self.data.count,
		))

	@script(
		gestures=["kb:escape"]
	)
	def script_escape(self, gesture):
		eventHandler.executeEvent("gainFocus", self.parent)

	@script(
		gestures=[
			"kb:leftArrow", "kb:rightArrow",
			"kb:upArrow", "kb:downArrow",
			"kb:home", "kb:end",
		]
	)
	def script_arrow(self, gesture):
		if gesture.mainKeyName in ["upArrow", "downArrow", "leftArrow", "rightArrow"]:
			result = self.data.move(gesture.mainKeyName[:-5])
		elif gesture.mainKeyName in ["home", "end"]:
			result = self.data.move(gesture.mainKeyName)
		if result is not True:
			tones.beep(100, 100)

		self.syncTextInfoPosition()
		self.message()

	def message(self):
		text = ""
		brailleText = ""

		text += self.data.pointer["name"]
		brailleText += self.data.pointer["name"]

		if "shortcut" in self.data.pointer and self.data.pointer["shortcut"] != "-1":
			if self.data.pointer["shortcut"] in [str(i) for i in range(1, 13)]:
				shortcut = _("f{shortcut}").format(shortcut=self.data.pointer["shortcut"])
			else:
				shortcut = _("{shortcut}").format(shortcut=self.data.pointer["shortcut"])
			text += " " + shortcut
			brailleText += " " + shortcut

		if self.data.pointer["type"] == "menu":
			text += " " + _("subMenu")

		number = _("{number} of {total}").format(
			number=self.data.path[-1] + 1,
			total=self.data.count,
		)
		text += " " + number
		brailleText += " " + number

		try:
			ui.message(text=text, brailleText=brailleText)
		except TypeError:
			ui.message(text=text)


class MenuViewTextInfo(textInfos.offsets.OffsetsTextInfo):
	def __init__(self, obj, position):
		super().__init__(obj, position)
		self.obj = obj

	def _getStoryLength(self):
		data = self.obj.data.pointer["name"]
		serializes = data
		return len(serializes)

	def _getStoryText(self):
		data = self.obj.data.pointer["name"]
		serializes = data
		return serializes

	def _getTextRange(self, start, end):
		text = self._getStoryText()
		return text[start:end] if text else ""
