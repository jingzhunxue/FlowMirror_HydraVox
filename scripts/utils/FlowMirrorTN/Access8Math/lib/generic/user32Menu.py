#Nao (NVDA Advanced OCR) is an addon that improves the standard OCR capabilities that NVDA provides on modern Windows versions.
#This file is covered by the GNU General Public License.
#See the file COPYING for more details.
#Last update 2021-12-08
#Copyright (C) 2021 Alessandro Albano, Davide De Carne and Simone Dal Maso

import ctypes
from ctypes import c_bool, c_uint, c_long, c_size_t, c_wchar
import windowUtils
import re

class MENUITEMINFOW(ctypes.Structure):
    _fields_ = [
        ('cbSize', c_uint),
        ('fMask', c_uint),
        ('fType', c_uint),
        ('fState', c_uint),
        ('wID', c_uint),
        ('hSubMenu', c_size_t),
        ('hbmpChecked', c_size_t),
        ('hbmpUnchecked', c_size_t),
        ('dwItemData', c_size_t),
        ('dwTypeData', c_size_t),
        ('cch', c_uint),
        ('hbmpItem', c_size_t),
    ]

class User32MenuItem:
	def __init__(self, menu, index):
		self.info = None
		self.text = None
		self.is_default = False
		if menu:
			self.info = MENUITEMINFOW()
			self.info.cbSize = ctypes.sizeof(self.info)
			self.info.fMask = 31
			if ctypes.windll.user32.GetMenuItemInfoW(menu, c_long(index), True, ctypes.byref(self.info)):
				if self.info.cch:
					self.info.cch = self.info.cch + 1
					text = ctypes.create_unicode_buffer(self.info.cch)
					self.info.dwTypeData = ctypes.addressof(text)
					ctypes.windll.user32.GetMenuItemInfoW(menu, c_long(index), True, ctypes.byref(self.info))
					self.info.cch = self.info.cch - 1
					self.info.dwTypeData = 0
					if text.value:
						self.text = re.sub('&(.)', r'\1', text.value) # remove accelerators
				self.is_default = self.info.fState & 0x00001000 == 0x00001000
			else:
				self.info = None

class User32Menu:
	def __init__(self, menu):
		self.menu = menu
		self.items = []
		if menu > 0:
			count = ctypes.windll.user32.GetMenuItemCount(menu)
			for index in range(count):
				info = User32MenuItem(menu, index)
				if info.info:
					self.items.append(info)

	def get_context_menu():
		try:
			h = windowUtils.findDescendantWindow(parent=ctypes.windll.user32.GetDesktopWindow(),className="#32768")
			if h:
				ret = User32Menu(ctypes.windll.user32.SendMessageW(h, 0x01E1, 0, 0)) # MN_GETHMENU
				ret.hwnd = h
				return ret
		except LookupError:
			pass
		return None
