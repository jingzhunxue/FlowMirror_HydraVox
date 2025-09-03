import addonHandler
import api
import eventHandler
from keyboardHandler import KeyboardInputGesture
from scriptHandler import script
import textInfos
import wx

from .action import mark as _mark
from .clipboard import clearClipboard
from .models import MenuModel
from .views import MenuView, MenuViewTextInfo

addonHandler.initTranslation()


def mark(section, type_):
	try:
		temp = api.getClipData()
	except BaseException:
		temp = ''

	result = _mark(type_)(section.obj.makeTextInfo(textInfos.POSITION_SELECTION).text)
	api.copyToClip(result["text"])

	KeyboardInputGesture.fromName("control+v").send()

	leftArrow = KeyboardInputGesture.fromName("leftArrow")
	for i in range(result["end_offset"]):
		leftArrow.send()

	return
	if temp != '':
		wx.CallLater(100, api.copyToClip, temp)
	else:
		wx.CallLater(100, clearClipboard)


class A8MMarkCommandModel(MenuModel):
	def __init__(self):
		super().__init__()
		self.data = [
			{
				"id": "latex",
				# Translators: mark command category - LaTeX
				"name": _("LaTeX"),
				"type": "item",
			},
			{
				"id": "asciimath",
				# Translators: mark command category - AsciiMath
				"name": _("AsciiMath"),
				"type": "item",
			},
			{
				"id": "nemeth",
				# Translators: mark command category - AsciiMath
				"name": _("Nemeth"),
				"type": "item",
			},
		]


class A8MMarkCommandView(MenuView):
	# Translators: alt+m window
	name = _("mark command")

	def __init__(self, section):
		super().__init__(MenuModel=A8MMarkCommandModel, TextInfo=MenuViewTextInfo)
		self._section = section

	@script(
		gestures=["kb:enter"]
	)
	def script_enter(self, gesture):
		self.mark(self.data.pointer['id'])

	def mark(self, type_):
		mark(self._section, type_=type_)
		eventHandler.executeEvent("gainFocus", self.parent)
