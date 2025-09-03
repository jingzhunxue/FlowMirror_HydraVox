#Nao (NVDA Advanced OCR) is an addon that improves the standard OCR capabilities that NVDA provides on modern Windows versions.
#This file is covered by the GNU General Public License.
#See the file COPYING for more details.
#Last update 2022-01-29
#Copyright (C) 2021 Alessandro Albano, Davide De Carne and Simone Dal Maso

import api
import ui
import os
from comtypes.client import CreateObject as COMCreate
from .xplorer2Helper import Xplorer2Helper
from .totalCommanderHelper import TotalCommanderHelper

_shell = None

def is_explorer(obj=None):
	if obj is None: obj = api.getForegroundObject()
	#return obj and (obj.role == api.controlTypes.Role.PANE or obj.role == api.controlTypes.Role.WINDOW) and obj.appModule.appName == "explorer"
	return obj and obj.appModule and obj.appModule.appName and obj.appModule.appName == 'explorer'

def is_totalcommander(obj=None):
	return TotalCommanderHelper.is_totalcommander(obj)

def is_xplorer2(obj=None):
	return Xplorer2Helper.is_xplorer2(obj)

def get_selected_files_explorer_ps():
	import subprocess
	si = subprocess.STARTUPINFO()
	si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
	cmd = "$OutputEncoding = [console]::InputEncoding = [console]::OutputEncoding = New-Object System.Text.UTF8Encoding; (New-Object -ComObject 'Shell.Application').Windows() | ForEach-Object { echo \\\"$($_.HWND):$($_.Document.FocusedItem.Path)\\\" }"
	cmd = "powershell.exe \"{}\"".format(cmd)
	try:
		p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, startupinfo=si, encoding="utf-8", text=True)
		stdout, stderr = p.communicate()
		if p.returncode == 0 and stdout:
			ret = {}
			lines = stdout.splitlines()
			for line in lines:
				hwnd, name = line.split(':',1)
				ret[str(hwnd)] = name
			return ret
	except:
		pass
	return False

def get_selected_file_explorer(obj=None):
	if obj is None: obj = api.getForegroundObject()
	file_path = False
	# We check if we are in the Windows Explorer.
	if is_explorer(obj):
		desktop = False
		try:
			global _shell
			if not _shell:
				_shell = COMCreate("shell.application")
			# We go through the list of open Windows Explorers to find the one that has the focus.
			for window in _shell.Windows():
				if window.hwnd == obj.windowHandle:
					# Now that we have the current folder, we can explore the SelectedItems collection.
					file_path = str(window.Document.FocusedItem.path)
					break
			else: # loop exhausted
				desktop = True
		except:
			try:
				windows = get_selected_files_explorer_ps()
				if windows:
					if str(obj.windowHandle) in windows:
						file_path = windows[str(obj.windowHandle)]
					else:
						desktop = True
			except:
				pass
		if desktop:
			desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
			file_path = desktop_path + '\\' + api.getDesktopObject().objectWithFocus().name
			#if not os.path.isfile(file_path) and not os.path.isdir(file_path): file_path = False
	return file_path

def get_selected_file_total_commander(obj=None):
	# We check if we are in the Total Commander
	total_commander = TotalCommanderHelper(obj)
	if total_commander.is_valid():
		return total_commander.currentFileWithPath()
	return False

def get_selected_file_xplorer2(obj=None):
	# We check if we are in the xplorer2
	xplorer2 = Xplorer2Helper(obj)
	if xplorer2.is_valid():
		return xplorer2.currentFileWithPath()
	return False

def get_selected_file(obj=None):
	file_path = False
	if obj is None: obj = api.getForegroundObject()
	file_path = get_selected_file_explorer(obj)
	if not file_path: file_path = get_selected_file_total_commander(obj)
	if not file_path: file_path = get_selected_file_xplorer2(obj)
	return file_path
