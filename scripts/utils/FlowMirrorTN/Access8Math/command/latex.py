import addonHandler
import api
import config
import eventHandler
from keyboardHandler import KeyboardInputGesture
from scriptHandler import script
import tones
import ui
import wx

from delimiter import LaTeX as LaTeX_delimiter, AsciiMath as AsciiMath_delimiter
delimiter_dict = {**AsciiMath_delimiter, **LaTeX_delimiter}

from lib.dataProcess import groupByField

from .clipboard import clearClipboard
from .models import MenuModel
from .views import MenuView, MenuViewTextInfo

addonHandler.initTranslation()

from lib.latex import latexData


def command(text, offset):
	try:
		temp = api.getClipData()
	except BaseException:
		temp = ''
	api.copyToClip(text)

	KeyboardInputGesture.fromName("control+v").send()

	leftArrow = KeyboardInputGesture.fromName("leftArrow")
	rightArrow = KeyboardInputGesture.fromName("rightArrow")
	if offset > 0:
		for i in range(abs(offset)):
			rightArrow.send()
	else:
		for i in range(abs(offset)):
			leftArrow.send()

	if temp != '':
		wx.CallLater(100, api.copyToClip, temp)
	else:
		wx.CallLater(100, clearClipboard)


class A8MLaTeXCommandModel(MenuModel):
	def __init__(self):
		super().__init__()
		self.data = [
			{
				"id": "shortcut",
				# Translators: LaTeX command category - shortcut
				"name": _("shortcut"),
				"type": "menu",
				"items": [latexData.latexShortcut[str(k)] for k in range(1, 13) if str(k) in latexData.latexShortcut] + [latexData.latexShortcut[k] for k in "abcdefghijklmnopqrstuvwxyz" if k in latexData.latexShortcut],
			},
			{
				"id": "common",
				# Translators: LaTeX command category - common
				"name": _("common"),
				"type": "menu",
				"items": latexData.latexMenu['common'],
			},
			{
				"id": "operator",
				# Translators: LaTeX command category - operator
				"name": _("operator"),
				"type": "menu",
				"items": latexData.latexMenu['operator'],
			},
			{
				"id": "relation",
				# Translators: LaTeX command category - relation
				"name": _("relation"),
				"type": "menu",
				"items": latexData.latexMenu['relation'],
			},
			{
				"id": "logic",
				# Translators: LaTeX command category - relation
				"name": _("logic"),
				"type": "menu",
				"items": latexData.latexMenu['logic'],
			},
			{
				"id": "arrow",
				# Translators: LaTeX command category - arrow
				"name": _("arrow"),
				"type": "menu",
				"items": latexData.latexMenu['arrow'],
			},
			{
				"id": "geometry",
				# Translators: LaTeX command category - geometry
				"name": _("geometry"),
				"type": "menu",
				"items": latexData.latexMenu['geometry'],
			},
			{
				"id": "combinatorics",
				# Translators: LaTeX command category - combinatorics
				"name": _("combinatorics"),
				"type": "menu",
				"items": latexData.latexMenu['combinatorics'],
			},
			{
				"id": "trigonometric",
				# Translators: LaTeX command category - trigonometric
				"name": _("trigonometric"),
				"type": "menu",
				"items": latexData.latexMenu['trigonometric'],
			},
			{
				"id": "2-dimension",
				# Translators: LaTeX command category - 2-dimension
				"name": _("2-dimension"),
				"type": "menu",
				"items": latexData.latexMenu['2-dimension'],
			},
			{
				"id": "set",
				# Translators: LaTeX command category - set
				"name": _("set"),
				"type": "menu",
				"items": latexData.latexMenu['set'],
			},
			{
				"id": "calculus",
				# Translators: LaTeX command category - calculus
				"name": _("calculus"),
				"type": "menu",
				"items": latexData.latexMenu['calculus'],
			},
			{
				"id": "other",
				# Translators: LaTeX command category - other
				"name": _("other"),
				"type": "menu",
				"items": latexData.latexMenu['other'],
			},
		]
		self.shortcut = latexData.latexShortcut
		self.greekAlphabet = latexData.greekAlphabetShortcut


class A8MLaTeXCommandView(MenuView):
	# Translators: alt+l window
	name = _("LaTeX command")

	def __init__(self, inSection=True):
		super().__init__(MenuModel=A8MLaTeXCommandModel, TextInfo=MenuViewTextInfo)
		self.inSection = inSection

	def update_menu(self):
		latexData.latexMenu = [{
			**i, **{
				"type": "item",
			}
		} for i in latexData.latexAll]
		latexData.latexMenu = groupByField(latexData.latexMenu, 'category', lambda i: i, lambda i: i)

		latexData.latexShortcut = latexData.data2shortcutMap(latexData.latexAll)

	def getScript(self, gesture):
		if isinstance(gesture, KeyboardInputGesture):
			if len(gesture.modifierNames) == 0 and gesture.mainKeyName in ["f{}".format(i) for i in range(1, 13)] + ["{}".format(c) for c in "abcdefghijklmnopqrstuvwxyz"]:
				return self.script_set_shortcut
			elif len(gesture.modifierNames) == 0 and gesture.mainKeyName in ["delete", "backspace"]:
				return self.script_reset_shortcut

		return super().getScript(gesture)

	@script(
		gestures=["kb:f{}".format(i) for i in range(1, 13)] + ["kb:f{}".format(c) for c in "abcdefghijklmnopqrstuvwxyz"]
	)
	def script_set_shortcut(self, gesture):
		if self.data.pointer['type'] == 'menu':
			# Translators: A message reported when setting the shortcut on an element of the LaTeX commands menu
			ui.message(_("Cannot set shortcut on menu"))
			return

		id_ = self.data.pointer['id']
		slot = gesture.mainKeyName[1:] if len(gesture.mainKeyName) > 1 else gesture.mainKeyName

		for item in latexData.latexAll:
			if item["id"] == id_:
				item["shortcut"] = slot
				break

		for item in latexData.latexAll:
			if item["id"] != id_ and item["shortcut"] == str(slot):
				item["shortcut"] = "-1"

		self.update_menu()
		# Translators: A message reported when setting the shortcut on an element of the LaTeX commands menu
		ui.message(_("Shortcut {slot} set").format(slot=slot))
		eventHandler.executeEvent("gainFocus", self.parent)

	@script(
		gestures=["kb:delete", "kb:backspace"]
	)
	def script_reset_shortcut(self, gesture):
		if self.data.pointer['type'] == 'menu':
			# Translators: A message reported when resetting the shortcut on an element of the LaTeX commands menu
			ui.message(_("Cannot clear shortcut on a menu"))
			return

		id_ = self.data.pointer['id']
		slot = self.data.pointer['shortcut']
		if slot == "-1":
			# Translators: A message reported when resetting the shortcut on an element of the LaTeX commands menu
			ui.message(_("No shortcut set on this item"))
			return

		for item in latexData.latexAll:
			if item["id"] == id_:
				item["shortcut"] = "-1"
				break

		self.update_menu()
		# Translators: A message reported when resetting the shortcut on an element of the LaTeX commands menu
		ui.message(_("Shortcut {slot} cleared").format(slot=slot))
		eventHandler.executeEvent("gainFocus", self.parent)

	@script(
		gestures=["kb:enter", "kb:numpadEnter"]
	)
	def script_enter(self, gesture):
		if self.data.pointer['type'] == 'menu':
			result = self.data.move("right")
			if result is not True:
				tones.beep(100, 100)
			self.syncTextInfoPosition()
			self.message()
		else:
			self.command(self.data.pointer['id'])

	def command(self, id_):
		try:
			kwargs = latexData.latexCommand[id_]
			if not self.inSection:
				delimiter = delimiter_dict[config.conf["Access8Math"]["settings"]["LaTeX_delimiter"]]
				offset = kwargs["offset"] - len(delimiter["end"])
				text = r"{start}{text}{end}".format(
					start=delimiter["start"],
					end=delimiter["end"],
					text=kwargs["text"]
				)
				command(text=text, offset=offset)
			else:
				command(**kwargs)
		except BaseException:
			tones.beep(100, 100)
			return
		eventHandler.executeEvent("gainFocus", self.parent)

	def greekAlphabetCommand(self, id_):
		try:
			kwargs = latexData.greekAlphabetCommand[id_]
			if not self.inSection:
				delimiter = delimiter_dict[config.conf["Access8Math"]["settings"]["LaTeX_delimiter"]]
				offset = kwargs["offset"] - len(delimiter["end"])
				text = r"{start}{text}{end}".format(
					start=delimiter["start"],
					end=delimiter["end"],
					text=kwargs["text"]
				)
				command(text=text, offset=offset)
			else:
				command(**kwargs)
		except BaseException:
			tones.beep(100, 100)
			return
		eventHandler.executeEvent("gainFocus", self.parent)
