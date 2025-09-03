import addonHandler
import api
import eventHandler
from keyboardHandler import KeyboardInputGesture
from scriptHandler import script
import wx

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


def A8MAutocompleteCommandModelFactory(command):
	latexMenu = [{
		**i, **{
			"type": "item",
		}
	} for i in latexData.latexAll if i['latex'].startswith(command)]

	class A8MAutocompleteCommandModel(MenuModel):
		def __init__(self):
			super().__init__()
			self.data = latexMenu

	return A8MAutocompleteCommandModel


class A8MAutocompleteCommandView(MenuView):
	# Translators: alt+m window
	name = _("autocomplete command")

	def __init__(self, section, command):
		super().__init__(MenuModel=A8MAutocompleteCommandModelFactory(command), TextInfo=MenuViewTextInfo)
		self._section = section

	@script(
		gestures=["kb:enter"]
	)
	def script_enter(self, gesture):
		self._section.commandSelection()
		self.command(self.data.pointer['id'])

	def command(self, id_):
		try:
			kwargs = latexData.latexCommand[id_]
			command(**kwargs)
		except BaseException:
			return
		eventHandler.executeEvent("gainFocus", self.parent)
