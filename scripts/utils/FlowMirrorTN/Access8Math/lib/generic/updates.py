#Nao (NVDA Advanced OCR) is an addon that improves the standard OCR capabilities that NVDA provides on modern Windows versions.
#This file is covered by the GNU General Public License.
#See the file COPYING for more details.
#Last update 2022-01-27
#Copyright (C) 2021 Alessandro Albano, Davide De Carne and Simone Dal Maso

import threading
import time
import os
import wx
import gui
import stat
import addonHandler
from .http import json_post
from .. import language

language.initTranslation()

CHECK_INTERVAL = 86400 # 1 day
CHECK_INTERVAL_FAIL = 10800 # 3 hours

class UpdatesConfirmDialog(wx.Dialog):
	def __init__(self, parent, on_accept, on_cancel, version='', title=None, message=None):
		name = addonHandler.getCodeAddon().manifest["summary"]
		if not title:
			# Translators: The title of the update dialog
			title = _("{name} Update")
			title = title.format(name=name)
		if not message:
			if not version: version=''
			# Translators: The version in the message of the update dialog
			message = name + ' ' + _N("Version") + ' ' + version
			# Translators: The message of the update dialog
			message = message + ' ' + _("is available")
			# Translators: The message of the update dialog
			message = message + ' \n' + _("Do you want to download and install it?")
		
		super(UpdatesConfirmDialog, self).__init__(parent, title=title)
		mainSizer = wx.BoxSizer(wx.VERTICAL)
		sHelper = gui.guiHelper.BoxSizerHelper(self, orientation=wx.VERTICAL)
		
		sHelper.addItem(wx.StaticText(self, label=message))
		bHelper = sHelper.addDialogDismissButtons(gui.guiHelper.ButtonHelper(wx.HORIZONTAL))
		
		def _on_accept(evt):
			self.Destroy()
			if on_accept: on_accept()
		
		def _on_cancel(evt):
			self.Destroy()
			if on_cancel: on_cancel()
		
		confirmButton = bHelper.addButton(self, id=wx.ID_YES)
		cancelButton = bHelper.addButton(self, id=wx.ID_NO)
		cancelButton.Bind(wx.EVT_BUTTON, _on_cancel)
		
		self.Bind(wx.EVT_CLOSE, _on_cancel)
		
		confirmButton.SetDefault()
		confirmButton.Bind(wx.EVT_BUTTON, _on_accept)
		mainSizer.Add(sHelper.sizer, border=gui.guiHelper.BORDER_FOR_DIALOGS, flag=wx.ALL)
		mainSizer.Fit(self)
		self.SetSizer(mainSizer)
		self.Center(wx.BOTH|wx.Center)

	def Ask(on_accept, on_cancel, version='', title=None, message=None):
		import queueHandler
		def h():
			import winsound
			from . import window
			gui.mainFrame.prePopup()
			dialog = UpdatesConfirmDialog(gui.mainFrame, on_accept, on_cancel, version=version, title=title, message=message)
			winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
			dialog.Show()
			window.bring_wx_to_top(dialog)
			gui.mainFrame.postPopup()
		queueHandler.queueFunction(queueHandler.eventQueue, h)

class UpdatesCheckAndDownloadStatus:
	OK = 'ok'
	FAIL = 'fail'
	BUSY = 'busy'
	UPGRADE = 'upgrade'

	def __init__(self, status=None, installed=False):
		self._status = status
		self._installed = installed

	@property
	def Status(self):
		return self._status

	@property
	def Found(self):
		return True if self._status and self._status == UpdatesCheckAndDownloadStatus.UPGRADE else False

	@property
	def Failed(self):
		return False if self._status and (self._status == UpdatesCheckAndDownloadStatus.OK or self._status == UpdatesCheckAndDownloadStatus.UPGRADE) else True

	@property
	def Installed(self):
		return self._installed

PICKLE_UPDATES_DEFAULT_ROOT_NAME = "updates"

def pickle_updates_default_data():
	return {
		"last_check": 0,
		"last_status": UpdatesCheckAndDownloadStatus.FAIL,
		"since": time.time()
	}

class Updates:
	_instances = {}
	_instances_lock = threading.Lock()

	def __new__(cls, url):
		ret = None
		cls._instances_lock.acquire()
		if url in cls._instances:
			ret = cls._instances[url]()
		cls._instances_lock.release()
		if ret is None:
			ret = super(Updates, cls).__new__(cls)
		else:
			#singleton
			ret = None
		return ret

	def __init__(self, url):
		Updates._instances_lock.acquire()
		if url not in Updates._instances or Updates._instances[url]() is None:
			import weakref
			Updates._instances[url] = weakref.ref(self)
			self._request_data = None
			self._url = url
			self._thread = None
			self._progressDialog = None
		Updates._instances_lock.release()

	def _get_request_data(self, pickle, pickle_updates_root_name):
		if not self._request_data:
			from . import version
			self._request_data = version.composed_version()
			self._request_data["addon"]["update_version"] = 1
			if pickle:
				self._request_data["addon"]["since"] = pickle.cdata[pickle_updates_root_name]["since"]
		return self._request_data

	def check(self, cb, pickle=None, pickle_updates_root_name=PICKLE_UPDATES_DEFAULT_ROOT_NAME):
		def _check_proc():
			cb_data = self._get_request_data(pickle, pickle_updates_root_name)
			response = json_post(self._url, cb_data)
			status = UpdatesCheckAndDownloadStatus.FAIL
			if response:
				try:
					import json
					response = json.load(response)
					if response and "status" in response:
						status = response["status"]
						cb_data["update"] = response
						del cb_data["update"]["status"]
				except:
					del cb_data["update"]
					status = UpdatesCheckAndDownloadStatus.FAIL
			if pickle:
				data = pickle.start_write()
				data[pickle_updates_root_name]["last_check"] = time.time()
				data[pickle_updates_root_name]["last_status"] = status
				pickle.commit_write()
			self._thread = None
			if cb: wx.CallAfter(cb, self, status, cb_data)

		if not self._thread:
			self._thread = threading.Thread(target = _check_proc)
			self._thread.setDaemon(True)
			self._thread.start()
			return True
		return False

	def download(self, cb, data):
		url = None
		if data:
			if "update" in data:
				update = data["update"]
				if "url" in update:
					url = update["url"]
				elif "direct_url" in update:
					url = update["direct_url"]
			if not url and "url" in data:
				url = data["url"]
		if url:
			def _download_proc():
				response = json_post(url, data)
				self._thread = None
				if cb: wx.CallAfter(cb, self, response)
			
			if not self._thread:
				self._thread = threading.Thread(target = _download_proc)
				self._thread.setDaemon(True)
				self._thread.start()
				return True
		return False

	def check_and_download(self, cb, verbose=True, pickle=None):
		if verbose:
			self._progressDialog = gui.IndeterminateProgressDialog(gui.mainFrame,
				# Translators: The title of the dialog displayed while manually checking for an update.
				_N("Checking for Update"),
				# Translators: The progress message displayed while manually checking for an update.
				_N("Checking for update"))
		
		def check_cb(updates, status, data):
			def check_cb_later():
				if status == UpdatesCheckAndDownloadStatus.UPGRADE:
					last_version = None
					if data and "update" in data and "last_version" in data["update"] and data["update"]["last_version"]:
						last_version = data["update"]["last_version"]
					
					def do_update():
						import tempfile
						self._progressDialog = gui.IndeterminateProgressDialog(gui.mainFrame,
							# Translators: The title of the dialog displayed while downloading an update.
							_N("Downloading Update"),
							# Translators: The title of the dialog displayed while downloading an update.
							_N("Downloading Update"))
						
						if data and "addon" in data and "name" in data["addon"] and data["addon"]["name"]:
							tmp_file = data["addon"]["name"]
						else:
							tmp_file = addonHandler.getCodeAddon().manifest["summary"]
						if last_version:
							tmp_file = tmp_file + "-" + last_version
						else:
							tmp_file = tmp_file + "-update"
						tmp_file = os.path.join(tempfile.gettempdir(), tmp_file + ".nvda-addon")
						
						def remove_file():
							if os.path.isfile(tmp_file):
								try:
									os.chmod(tmp_file, stat.S_IWRITE)
									os.remove(tmp_file)
								except:
									pass
						
						def download_cb(updates, response):
							def download_cb_later():
								def install_cb(installed):
									remove_file()
									if cb: cb(updates, UpdatesCheckAndDownloadStatus(status=status, installed=installed))
								
								if response and response.status == 200:
									try:
										package = response.read()
									except:
										package = None
									if package:
										try:
											f = open(tmp_file, 'wb')
											f.write(package)
											f.close()
											self.install(tmp_file, install_cb)
											return
										except:
											remove_file()
								if cb: cb(updates, UpdatesCheckAndDownloadStatus(status=status))
							
							if self._progressDialog:
								self._progressDialog.done()
								self._progressDialog = None
								wx.CallLater(100, download_cb_later)
							else:
								download_cb_later()
						
						remove_file()
						if not self.download(download_cb, data):
							if cb: cb(updates, UpdatesCheckAndDownloadStatus(status=status))
					
					def cancel_update():
						if cb: cb(updates, UpdatesCheckAndDownloadStatus(status=status))
					
					UpdatesConfirmDialog.Ask(do_update, cancel_update, version=last_version)
				else:
					if cb: cb(updates, UpdatesCheckAndDownloadStatus(status=status))
			
			if self._progressDialog:
				self._progressDialog.done()
				self._progressDialog = None
				wx.CallLater(100, check_cb_later)
			else:
				check_cb_later()
		
		if not self.check(check_cb, pickle):
			if cb: cb(updates, UpdatesCheckAndDownloadStatus(status=UpdatesCheckAndDownloadStatus.BUSY))

	def install(self, addonPath, cb=None):
		from gui import addonGui
		def h():
			ret = addonGui.installAddon(gui.mainFrame, addonPath)
			if cb: cb(ret)
			if ret:
				addonGui.promptUserForRestart()
		wx.CallLater(100, h)

class AutoUpdates:
	def __init__(self, url, pickle, pickle_updates_root_name=PICKLE_UPDATES_DEFAULT_ROOT_NAME):
		from .. threading import ProgramTerminateHandler
		self._url = url
		self._pickle = pickle
		self._pickle_updates_root_name = pickle_updates_root_name
		self._timer = wx.PyTimer(self._check_remaining_time)
		wx.CallAfter(self._timer.Start, 10000, True)
		self._terminate_handler = ProgramTerminateHandler(self.terminate)

	def __del__(self):
		self.terminate()

	def terminate(self):
		if self._terminate_handler:
			self._terminate_handler.unregister()
			self._terminate_handler = None
		if self._timer and self._timer.IsRunning():
			self._timer.Stop()
		self._timer = None
		self._url = None
		self._pickle = None

	def _check_remaining_time(self):
		import globalVars
		if self._timer and self._timer.IsRunning():
			self._timer.Stop()
			self._timer = None
		if globalVars.appArgs.secure:
			self._timer = wx.PyTimer(self._check_remaining_time)
			wx.CallAfter(self._timer.Start, 10000, True)
		else:
			data = self._pickle.cdata[self._pickle_updates_root_name]
			secsSinceLast = max(time.time() - data["last_check"], 0)
			if data["last_status"] == UpdatesCheckAndDownloadStatus.OK or data["last_status"] == UpdatesCheckAndDownloadStatus.UPGRADE:
				secsTillNext = CHECK_INTERVAL - int(min(secsSinceLast, CHECK_INTERVAL))
			else:
				secsTillNext = CHECK_INTERVAL_FAIL - int(min(secsSinceLast, CHECK_INTERVAL_FAIL))
			if secsTillNext < 5:
				secsTillNext = 5
				self._timer = wx.PyTimer(self._updates_proc)
			else:
				self._timer = wx.PyTimer(self._check_remaining_time)
			wx.CallAfter(self._timer.Start, secsTillNext * 1000, True)

	def _updates_proc(self):
		def _end_proc(updates, status):
			if updates: del updates
			if not status.Installed:
				self._timer = wx.PyTimer(self._check_remaining_time)
				wx.CallAfter(self._timer.Start, 10000, True)
			else:
				self._timer = None
		updates = Updates(self._url)
		if updates:
			updates.check_and_download(_end_proc, verbose=False, pickle=self._pickle)
		else:
			_end_proc(updates, UpdatesCheckAndDownloadStatus(status=UpdatesCheckAndDownloadStatus.BUSY))

def ManualUpdatesCheck(url, pickle=None):
	def _end_proc(updates, status):
		if updates: del updates
		if status.Failed:
			# Translators: The title of the update dialog
			title = _("{name} Update")
			title = title.format(name=addonHandler.getCodeAddon().manifest["summary"]) + ' - '
			# Translators: The title of an error message dialog.
			title = title + _N("Error")
			gui.mainFrame.prePopup()
			gui.messageBox(
				# Translators: A message indicating that an error occurred while checking for an update.
				_N("Error checking for update."),
				title,
				wx.OK | wx.ICON_ERROR)
			gui.mainFrame.postPopup()
		elif not status.Found:
			# Translators: The title of the update dialog
			title = _("{name} Update")
			title = title.format(name=addonHandler.getCodeAddon().manifest["summary"])
			gui.mainFrame.prePopup()
			gui.messageBox(
				# Translators: A message indicating that no update is available.
				_N("No update available."),
				title,
				wx.OK)
			gui.mainFrame.postPopup()
	updates = Updates(url)
	if updates:
		updates.check_and_download(_end_proc, pickle=pickle)
	else:
		_end_proc(updates, UpdatesCheckAndDownloadStatus(status=UpdatesCheckAndDownloadStatus.BUSY))