#Nao (NVDA Advanced OCR) is an addon that improves the standard OCR capabilities that NVDA provides on modern Windows versions.
#This file is covered by the GNU General Public License.
#See the file COPYING for more details.
#Last update 2022-01-16
#Copyright (C) 2021 Alessandro Albano, Davide De Carne and Simone Dal Maso

import tones
from .. threading import Thread
from .. speech import speech
from .. import language

language.initTranslation()

class Announce:
	def __init__(self):
		self._threads = []

	def __del__(self):
		self.stop()

	def start(self, use_beep=True, beep_interval=1, first_beep_after=-1, use_text=False, text_interval=5, text=None, first_text_after=-1):
		if len(self._threads) == 0:
			if use_text and not text:
				# Translators: Announced periodically to indicate progress for an indeterminate progress bar.
				text = _N("Please wait")
			def beep_thread(wait):
				if first_beep_after >= 0 and not wait.must_terminate(timeout=first_beep_after):
					tones.beep(440, 40)
				while not wait.must_terminate(timeout=beep_interval):
					tones.beep(440, 40)
			def text_thread(wait):
				if first_text_after >= 0 and not wait.must_terminate(timeout=first_text_after):
					speech.queue_message(text)
				while not wait.must_terminate(timeout=text_interval):
					speech.queue_message(text)
			if use_beep and beep_interval > 0:
				self._threads.append(Thread(target=beep_thread, name='AnnounceBeepThread').start())
			if use_text and text_interval > 0:
				self._threads.append(Thread(target=text_thread, name='AnnounceTextThread').start())

	def stop(self):
		for t in self._threads:
			t.terminate()
		self._threads.clear()