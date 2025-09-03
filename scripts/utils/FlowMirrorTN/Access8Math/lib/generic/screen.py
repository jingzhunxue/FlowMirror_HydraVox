#Nao (NVDA Advanced OCR) is an addon that improves the standard OCR capabilities that NVDA provides on modern Windows versions.
#This file is covered by the GNU General Public License.
#See the file COPYING for more details.
#Last update 2021-12-16
#Copyright (C) 2021 Alessandro Albano, Davide De Carne and Simone Dal Maso

import wx
import screenBitmap
import vision
from visionEnhancementProviders.screenCurtain import ScreenCurtainProvider

def get_size(dc=None):
	if dc is None:
		dc = wx.ScreenDC()
	return dc.Size.Get()

def take_snapshot_pixels():
	size = get_size()
	sb = screenBitmap.ScreenBitmap(size[0], size[1])
	return [sb.captureImage(0, 0, size[0], size[1]), size[0], size[1]]

def have_curtain():
	screenCurtainId = ScreenCurtainProvider.getSettings().getId()
	screenCurtainProviderInfo = vision.handler.getProviderInfo(screenCurtainId)
	return bool(vision.handler.getProviderInstance(screenCurtainProviderInfo))