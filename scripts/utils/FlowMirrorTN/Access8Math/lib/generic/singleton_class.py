#Nao (NVDA Advanced OCR) is an addon that improves the standard OCR capabilities that NVDA provides on modern Windows versions.
#This file is covered by the GNU General Public License.
#See the file COPYING for more details.
#Last update 2022-01-22
#Copyright (C) 2021 Alessandro Albano, Davide De Carne and Simone Dal Maso

from threading import Lock, RLock

class SingletonClass:
	Instances = dict()
	InstancesLock = Lock()

	class SingletonControlClass:
		def __init__(self):
			self.lock = RLock()

	def __new__(cls, *args, **kwargs):
		instance = None
		SingletonClass.InstancesLock.acquire()
		if cls in SingletonClass.Instances: instance = SingletonClass.Instances[cls]()
		SingletonClass.InstancesLock.release()
		if instance is None:
			instance = super(SingletonClass, cls).__new__(cls)
		return instance

	def __init__(self, *args, **kwargs):
		import weakref
		init = False
		selftype = type(self)
		SingletonClass.InstancesLock.acquire()
		if selftype not in SingletonClass.Instances or SingletonClass.Instances[selftype]() is None:
			self._singleton_control = SingletonClass.SingletonControlClass()
			SingletonClass.Instances[selftype] = weakref.ref(self)
			self.Lock.acquire()
			init = True
		SingletonClass.InstancesLock.release()
		if init: self.__singleton_init__(*args, **kwargs)

	def __del__(self):
		SingletonClass.InstancesLock.acquire()
		if type(self) in SingletonClass.Instances: del SingletonClass.Instances[type(self)]
		SingletonClass.InstancesLock.release()

	def __singleton_init__(self, *args, **kwargs):
		self.Lock.release()

	@property
	def Lock(self):
		return self._singleton_control.lock

	@property
	def Referrers(self):
		import gc
		return gc.get_referrers(self)