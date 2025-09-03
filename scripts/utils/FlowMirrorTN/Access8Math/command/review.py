import os
import shutil

import addonHandler
import gui
from scriptHandler import script
import wx

from .models import MenuModel
from .views import MenuView, MenuViewTextInfo

addonHandler.initTranslation()


class A8MHTMLCommandModel(MenuModel):
	def __init__(self):
		super().__init__()
		self.data = [
			{
				"id": "preview",
				"name": _("preview"),
				"type": "item",
			},
			{
				"id": "export",
				"name": _("export"),
				"type": "item",
			},
		]


class A8MHTMLCommandView(MenuView):
	name = _("view command")

	def __init__(self, ad):
		super().__init__(MenuModel=A8MHTMLCommandModel, TextInfo=A8MHTMLCommandViewTextInfo)
		self.ad = ad

	@script(
		gestures=["kb:enter"]
	)
	def script_enter(self, gesture):
		if self.data.pointer['id'] == 'preview':
			self.OnPreview()
		elif self.data.pointer['id'] == 'export':
			self.OnExport()

	def OnPreview(self):
		def openfile():
			os.startfile(self.ad.review_entry)
		wx.CallAfter(openfile)

	def OnExport(self):
		def show():
			with wx.FileDialog(gui.mainFrame, message=_("Save file..."), wildcard="zip files (*.zip)|*.zip", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dialog:
				if dialog.ShowModal() != wx.ID_OK:
					return
				dst = dialog.GetPath()[:-4]
				shutil.make_archive(dst, 'zip', self.ad.review_folder)
		wx.CallAfter(show)


class A8MHTMLCommandViewTextInfo(MenuViewTextInfo):
	pass
