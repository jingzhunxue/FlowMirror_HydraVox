import gui
import winUser


def clearClipboard():
	with winUser.openClipboard(gui.mainFrame.Handle):
		winUser.emptyClipboard()
