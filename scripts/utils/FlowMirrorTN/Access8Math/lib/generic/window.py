#Nao (NVDA Advanced OCR) is an addon that improves the standard OCR capabilities that NVDA provides on modern Windows versions.
#This file is covered by the GNU General Public License.
#See the file COPYING for more details.
#Last update 2021-12-22
#Copyright (C) 2021 Alessandro Albano, Davide De Carne and Simone Dal Maso

def bring_handle_to_top(hwnd):
	import ctypes
	import threading
	if hwnd:
		user32 = ctypes.windll.user32
		hCurWnd = user32.GetForegroundWindow();
		if hwnd != hCurWnd:
			dwMyID = threading.get_ident();
			dwCurID = user32.GetWindowThreadProcessId(hCurWnd, 0);
			user32.AttachThreadInput(dwCurID, dwMyID, True);
			user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 1 | 2);	#HWND_TOPMOST, SWP_NOSIZE | SWP_NOMOVE
			user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, 0x40 | 1 | 2); #HWND_NOTOPMOST, SWP_SHOWWINDOW | SWP_NOSIZE | SWP_NOMOVE
			user32.SetForegroundWindow(hwnd)
			user32.SetFocus(hwnd)
			user32.SetActiveWindow(hwnd)
			user32.AttachThreadInput(dwCurID, dwMyID, False);

def bring_wx_to_top(wxctrl):
	bring_handle_to_top(wxctrl.GetHandle())