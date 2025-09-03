import os

from lark import Transformer

from .utils import nemeth2symbol_with_priority


class Nemeth2TexTransformer(Transformer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		BASE_DIR = os.path.dirname(__file__)
		data_folder = os.path.join(BASE_DIR, "data")

		data_path = "number.csv"
		map = nemeth2symbol_with_priority(os.path.join(data_folder, data_path))
		self.number = sorted(list(map.items()), key=lambda i: -len(i[0]))

		data_path = "letter.csv"
		map = nemeth2symbol_with_priority(os.path.join(data_folder, data_path))
		self.letter = sorted(list(map.items()), key=lambda i: -len(i[0]))

		data_path = "symbol.csv"
		map = nemeth2symbol_with_priority(os.path.join(data_folder, data_path))
		self.symbol = sorted(list(map.items()), key=lambda i: -len(i[0]))

	def translate(self, string, map):
		for key, value in map:
			string = string.replace(key, value)
		return string

	def curly_brace(self, items):
		result = []
		for item in items:
			if not len(item) == 1:
				item = r"{" + item + r"}"
			result.append(item)
		return result

	def exp(self, items):
		return "".join(items)

	def exp_sup(self, items):
		items = self.curly_brace(items)
		return items[0] + r"^" + items[1]

	def exp_sub(self, items):
		items = self.curly_brace(items)
		return items[0] + r"_" + items[1]

	def exp_sup_simple(self, items):
		items = self.curly_brace(items)
		return items[0] + r"^" + items[1]

	def exp_sub_simple(self, items):
		items = self.curly_brace(items)
		return items[0] + r"_" + items[1]

	def exp_under(self, items):
		return r"\underset{" + items[1] + r"}{" + items[0] + r"}"

	def exp_over(self, items):
		return r"\overset{" + items[1] + r"}{" + items[0] + r"}"

	def exp_underover(self, items):
		return r"\overset{" + items[2] + r"}{\underset{" + items[1] + r"}{" + items[0] + r"}}"

	def exp_subsup_symbol(self, items):
		symbol = self.translate(items[0], self.symbol)
		return symbol.strip() + r"_{" + items[1] + r"}^{" + items[2] + r"}"

	def exp_underover_symbol(self, items):
		symbol = self.translate(items[0], self.symbol)
		return symbol.strip() + r"_{" + items[1] + r"}^{" + items[2] + r"}"

	def exp_frac(self, items):
		return r"\frac{" + items[0] + r"}{" + items[1] + r"}"

	def exp_mixed_number(self, items):
		return items[0] + r"\frac{" + items[1] + r"}{" + items[2] + r"}"

	def exp_sqrt(self, items):
		return r"\sqrt{" + items[0] + r"}"

	def exp_root(self, items):
		return r"\sqrt[" + items[0] + r"]{" + items[1] + r"}"

	def exp_line(self, items):
		return r"\overleftrightarrow{" + items[0] + r"}"

	def exp_line_segment(self, items):
		return r"\overline{" + items[0] + r"}"

	def exp_ray(self, items):
		return r"\overrightarrow{" + items[0] + r"}"

	def exp_arc(self, items):
		return r"\overset{\frown}{" + items[0] + r"}"

	def exp_vector(self, items):
		return r"\vec{" + items[0] + r"}"

	def exp_binom(self, items):
		return r"\binom{" + items[1] + r"}{" + items[0] + "}"

	def exp_limit(self, items):
		return r"\lim_{{" + items[0] + r"} \to {" + items[1] + r"}}"

	def exp_interm(self, items):
		return items[0]

	def exp_curly_brace(self, items):
		return r"\{" + "".join(items) + r"\}"

	def exp_square_bracket(self, items):
		return r"[" + "".join(items) + r"]"

	def exp_parenthesis(self, items):
		return r"(" + "".join(items) + r")"

	def const(self, items):
		return "".join(items)

	def EN_UPPERCASE(self, items):
		result = self.translate(items.replace("⠠", ""), self.letter).upper()
		return result

	def EN_LOWERCASE(self, items):
		result = self.translate(items, self.letter)
		return result

	def EN_UPPERCASE_CONTINUE(self, items):
		try:
			uppercases, lowercases = items.split("⠠⠄")
		except ValueError:
			uppercases = items
			lowercases = ""
		uppercases = uppercases.strip("⠠")
		result = self.translate(uppercases, self.letter).upper() + self.translate(lowercases, self.letter)
		return result

	def NUMBER(self, items):
		result = self.translate(items.replace("⠼", ""), self.number)
		return result

	def operand(self, items):
		return "".join(items)

	def OPERAND(self, items):
		result = self.translate(items, self.symbol)
		return result

	def other(self, items):
		return "".join(items)

	def OTHER(self, items):
		result = self.translate(items, self.symbol)
		return result
