class MenuModel:
	def __init__(self):
		self.data = []
		self.path = [0]

	@property
	def pointer(self):
		ptr = self.data[self.path[0]]
		for index in self.path[1:]:
			ptr = ptr["items"][index]
		return ptr

	@property
	def pointer_parent(self):
		ptr = self.data[self.path[0]]
		for index in self.path[1:-1]:
			ptr = ptr["items"][index]
		return ptr

	@property
	def count(self):
		return len(self.pointer_parent["items"]) if len(self.path) > 1 else len(self.data)

	def move(self, direction):
		result = False
		if direction == "up" or direction == "down":
			if direction == "up":
				index = (self.path[-1] - 1) % self.count
			elif direction == "down":
				index = (self.path[-1] + 1) % self.count
			if index >= 0 and index < self.count:
				self.path[-1] = index
				result = True
			else:
				result = False
		elif direction == "left" or direction == "right":
			if direction == "left":
				if len(self.path) > 1:
					self.path = self.path[:-1]
					result = True
				else:
					result = False
			elif direction == "right":
				if self.pointer["type"] == "menu":
					self.path = self.path + [0]
					if self.count == 0:
						self.path = self.path[:-1]
						result = False
					else:
						result = True
				else:
					result = False
		elif direction == "home":
			index = 0
			self.path[-1] = index
			result = True
		elif direction == "end":
			index = self.count - 1
			self.path[-1] = index
			result = True

		return result
