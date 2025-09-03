import addonHandler

from lib.dataProcess import joinObjectArray, groupByField

import os
from python.csv import DictReader, DictWriter

addonHandler.initTranslation()


def load(patho):
	BASE_DIR = os.path.dirname(__file__)
	if not os.path.isfile(patho):
		path = os.path.join(BASE_DIR, 'latexs.csv')
	else:
		path = patho

	data = []
	with open(path, 'r', encoding='utf-8') as src_file:
		src_dict_csv = DictReader(src_file)
		for row in src_dict_csv:
			id_ = row.pop('id')
			latex = row.pop('latex')
			offset = int(row.pop('offset'))
			shortcut = row.pop('shortcut')
			category = row.pop('category')
			order = int(row.pop('order'))
			data.append({
				"id": id_,
				"latex": latex,
				"offset": offset,
				"shortcut": shortcut,
				"category": category,
				"order": order,
			})

	return data


def save(patho, latexAll):
	BASE_DIR = os.path.dirname(__file__)
	if not os.path.isfile(patho):
		path = os.path.join(BASE_DIR, 'latexs.csv')
	else:
		path = patho

	src_rows = []
	with open(path, 'r', encoding='utf-8') as src_file:
		src_dict_csv = DictReader(src_file)
		fields = src_dict_csv.fieldnames.copy()
		for row in src_dict_csv:
			id_ = row.pop('id')
			latex = row.pop('latex')
			offset = int(row.pop('offset'))
			shortcut = row.pop('shortcut')
			category = row.pop('category')
			order = int(row.pop('order'))
			src_rows.append({
				"id": id_,
				"latex": latex,
				"offset": offset,
				"shortcut": shortcut,
				"category": category,
				"order": order,
			})

	old_rows = []
	for row in src_rows:
		row.pop('shortcut')
		old_rows.append(row)

	shortcuts = [{"id": row["id"], "shortcut": row["shortcut"]} for row in latexAll]
	rows = joinObjectArray(shortcuts, old_rows, "id")

	path = patho
	with open(path, 'w', encoding='utf-8', newline='') as dst_file:
		dst_dict_csv = DictWriter(dst_file, fieldnames=fields)
		dst_dict_csv.writeheader()
		for row in rows:
			dst_dict_csv.writerow(row)

	return rows


def data2commandMap(latexAll):
	data = {}
	for row in latexAll:
		id_ = row['id']
		latex = row['latex']
		offset = int(row['offset'])
		data[id_] = {
			"text": latex,
			"offset": offset,
		}

	return data


def data2shortcutMap(latexAll):
	data = {}
	for row in latexAll:
		shortcut = row['shortcut']
		if shortcut != -1:
			id_ = row['id']
			name = row['name']
			data[shortcut] = {
				"id": id_,
				"name": name,
				"type": "item",
				"shortcut": shortcut,
			}

	return data


latexMenuData = [
	{
		"id": "matrix2X2",
		# Translators: LaTeX command - matrix (2X2)
		"name": _("matrix (2X2)"),
	},
	{
		"id": "matrix3X3",
		# Translators: LaTeX command - matrix (3X3)
		"name": _("matrix (3X3)"),
	},
	{
		"id": "determinant2X2",
		# Translators: LaTeX command - determinant (2X2)
		"name": _("determinant (2X2)"),
	},
	{
		"id": "determinant3X3",
		# Translators: LaTeX command - determinant (3X3)
		"name": _("determinant (3X3)"),
	},
	{
		"id": "leftarrow",
		# Translators: LaTeX command - left arrow
		"name": _("left arrow"),
	},
	{
		"id": "rightarrow",
		# Translators: LaTeX command - right arrow
		"name": _("right arrow"),
	},
	{
		"id": "leftrightarrow",
		# Translators: LaTeX command - left right arrow
		"name": _("left right arrow"),
	},
	{
		"id": "uparrow",
		# Translators: LaTeX command - up arrow
		"name": _("up arrow"),
	},
	{
		"id": "downarrow",
		# Translators: LaTeX command - down arrow
		"name": _("down arrow"),
	},
	{
		"id": "updownarrow",
		# Translators: LaTeX command - up down arrow
		"name": _("up down arrow"),
	},
	{
		"id": "dotproduct",
		# Translators: LaTeX command - dot product
		"name": _("dot product"),
	},
	{
		"id": "integral",
		# Translators: LaTeX command - integral
		"name": _("integral"),
	},
	{
		"id": "nabla",
		# Translators: LaTeX command - nabla
		"name": _("nabla"),
	},
	{
		"id": "partial",
		# Translators: LaTeX command - partial derivative
		"name": _("partial derivative"),
	},
	{
		"id": "prime",
		# Translators: LaTeX command - derivative
		"name": _("derivative"),
	},
	{
		"id": "differential",
		# Translators: LaTeX command - differential
		"name": _("differential"),
	},
	{
		"id": "combination",
		# Translators: LaTeX command - combination
		"name": _("combination"),
	},
	{
		"id": "permutation",
		# Translators: LaTeX command - permutation
		"name": _("permutation"),
	},
	{
		"id": "combination-with-repetition",
		# Translators: LaTeX command - combination with repetition
		"name": _("combination with repetition"),
	},
	{
		"id": "unordered-selection",
		# Translators: LaTeX command - unordered selection
		"name": _("unordered selection"),
	},
	{
		"id": "frac",
		# Translators: LaTeX command - fractions
		"name": _("fractions"),
	},
	{
		"id": "sqrt",
		# Translators: LaTeX command - square root
		"name": _("square root"),
	},
	{
		"id": "root",
		# Translators: LaTeX command - root
		"name": _("root"),
	},
	{
		"id": "sumupdown",
		# Translators: LaTeX command - summation
		"name": _("summation"),
	},
	{
		"id": "vector",
		# Translators: LaTeX command - vector
		"name": _("vector"),
	},
	{
		"id": "limit",
		# Translators: LaTeX command - limit
		"name": _("limit"),
	},
	{
		"id": "logarithm",
		# Translators: LaTeX command - logarithm
		"name": _("logarithm"),
	},
	{
		"id": "arc",
		# Translators: LaTeX command - arc
		"name": _("arc"),
	},
	{
		"id": "triangle",
		# Translators: LaTeX command - triangle
		"name": _("triangle"),
	},
	{
		"id": "angle",
		# Translators: LaTeX command - angle
		"name": _("angle"),
	},
	{
		"id": "degree",
		# Translators: LaTeX command - degree
		"name": _("degree"),
	},
	{
		"id": "circ",
		# Translators: LaTeX command - circle
		"name": _("circle"),
	},
	{
		"id": "parallel",
		# Translators: LaTeX command - parallel
		"name": _("parallel"),
	},
	{
		"id": "perp",
		# Translators: LaTeX command - perpendicular
		"name": _("perpendicular"),
	},
	{
		"id": "square",
		# Translators: LaTeX command - square
		"name": _("square"),
	},
	{
		"id": "small-diamond",
		# Translators: LaTeX command - small diamond
		"name": _("small diamond"),
	},
	{
		"id": "large-diamond",
		# Translators: LaTeX command - large diamond
		"name": _("large diamond"),
	},
	{
		"id": "because",
		# Translators: LaTeX command - because
		"name": _("because"),
	},
	{
		"id": "therefore",
		# Translators: LaTeX command - therefore
		"name": _("therefore"),
	},
	{
		"id": "iff",
		# Translators: LaTeX command - if and only if
		"name": _("if and only if"),
	},
	{
		"id": "implies",
		# Translators: LaTeX command - implies
		"name": _("implies"),
	},
	{
		"id": "impliedby",
		# Translators: LaTeX command - implied by
		"name": _("implied by"),
	},
	{
		"id": "times",
		# Translators: LaTeX command - times
		"name": _("times"),
	},
	{
		"id": "div",
		# Translators: LaTeX command - divide
		"name": _("divide"),
	},
	{
		"id": "pm",
		# Translators: LaTeX command - plus-minus sign
		"name": _("plus-minus sign"),
	},
	{
		"id": "modulus",
		# Translators: LaTeX command - modulus
		"name": _("modulus"),
	},
	{
		"id": "overline",
		# Translators: LaTeX command - line segment
		"name": _("line segment"),
	},
	{
		"id": "overleftrightarrow",
		# Translators: LaTeX command - line
		"name": _("line"),
	},
	{
		"id": "overrightarrow",
		# Translators: LaTeX command - ray
		"name": _("ray"),
	},
	{
		"id": "binom",
		# Translators: LaTeX command - binomial coefficient
		"name": _("binomial coefficient"),
	},
	{
		"id": "simultaneous-equations",
		# Translators: LaTeX command - simultaneous equations
		"name": _("simultaneous equations"),
	},
	{
		"id": "infty",
		# Translators: LaTeX command - infty
		"name": _("infty"),
	},
	{
		"id": "repeating-decimal",
		# Translators: LaTeX command - repeating decimal
		"name": _("repeating decimal"),
	},
	{
		"id": "ge",
		# Translators: LaTeX command - greater than or equal to
		"name": _("greater than or equal to"),
	},
	{
		"id": "le",
		# Translators: LaTeX command - less than or equal to
		"name": _("less than or equal to"),
	},
	{
		"id": "ne",
		# Translators: LaTeX command - not equal to
		"name": _("not equal to"),
	},
	{
		"id": "approx",
		# Translators: LaTeX command - approximate
		"name": _("approximate"),
	},
	{
		"id": "cong",
		# Translators: LaTeX command - full equal
		"name": _("full equal"),
	},
	{
		"id": "sim",
		# Translators: LaTeX command - similar
		"name": _("similar"),
	},
	{
		"id": "doteqdot",
		# Translators: LaTeX command - approximately equal to
		"name": _("approximately equal to"),
	},
	{
		"id": "propto",
		# Translators: LaTeX command - proportional to
		"name": _("proportional to"),
	},
	{
		"id": "in",
		# Translators: LaTeX command - belong to
		"name": _("belong to"),
	},
	{
		"id": "notin",
		# Translators: LaTeX command - not belong to
		"name": _("not belong to"),
	},
	{
		"id": "ni",
		# Translators: LaTeX command - include element
		"name": _("include element"),
	},
	{
		"id": "notni",
		# Translators: LaTeX command - not include element
		"name": _("not include element"),
	},
	{
		"id": "subset",
		# Translators: LaTeX command - lie in
		"name": _("lie in"),
	},
	{
		"id": "subsetneqq",
		# Translators: LaTeX command - properly lie in
		"name": _("properly lie in"),
	},
	{
		"id": "not-subset",
		# Translators: LaTeX command - not lie in
		"name": _("not lie in"),
	},
	{
		"id": "supset",
		# Translators: LaTeX command - include
		"name": _("include"),
	},
	{
		"id": "supsetneqq",
		# Translators: LaTeX command - properly include
		"name": _("properly include"),
	},
	{
		"id": "not-supset",
		# Translators: LaTeX command - not include
		"name": _("not include"),
	},
	{
		"id": "cap",
		# Translators: LaTeX command - intersection set
		"name": _("intersection set"),
	},
	{
		"id": "cup",
		# Translators: LaTeX command - union set
		"name": _("union set"),
	},
	{
		"id": "setminus",
		# Translators: LaTeX command - difference set
		"name": _("difference set"),
	},
	{
		"id": "complement",
		# Translators: LaTeX command - complement
		"name": _("complement"),
	},
	{
		"id": "emptyset",
		# Translators: LaTeX command - empty set
		"name": _("empty set"),
	},
	{
		"id": "natural-number",
		# Translators: LaTeX command - natural number
		"name": _("natural number"),
	},
	{
		"id": "real-number",
		# Translators: LaTeX command - real number
		"name": _("real number"),
	},
	{
		"id": "forall",
		# Translators: LaTeX command - for all
		"name": _("for all"),
	},
	{
		"id": "exists",
		# Translators: LaTeX command - exists
		"name": _("exists"),
	},
	{
		"id": "sine",
		# Translators: LaTeX command - sine
		"name": _("sine"),
	},
	{
		"id": "cosine",
		# Translators: LaTeX command - cosine
		"name": _("cosine"),
	},
	{
		"id": "tangent",
		# Translators: LaTeX command - tangent
		"name": _("tangent"),
	},
	{
		"id": "cotangent",
		# Translators: LaTeX command - cotangent
		"name": _("cotangent"),
	},
	{
		"id": "secant",
		# Translators: LaTeX command - secant
		"name": _("secant"),
	},
	{
		"id": "cosecant",
		# Translators: LaTeX command - cosecant
		"name": _("cosecant"),
	},
	{
		"id": "arcsine",
		# Translators: LaTeX command - arcsine
		"name": _("arcsine"),
	},
	{
		"id": "arccosine",
		# Translators: LaTeX command - arccosine
		"name": _("arccosine"),
	},
	{
		"id": "arctangent",
		# Translators: LaTeX command - arctangent
		"name": _("arctangent"),
	},
	{
		"id": "hyperbolic-sine",
		# Translators: LaTeX command - hyperbolic sine
		"name": _("hyperbolic sine"),
	},
	{
		"id": "hyperbolic-cosine",
		# Translators: LaTeX command - hyperbolic cosine
		"name": _("hyperbolic cosine"),
	},
	{
		"id": "hyperbolic-tangent",
		# Translators: LaTeX command - hyperbolic tangent
		"name": _("hyperbolic tangent"),
	},
	{
		"id": "hyperbolic-cotangent",
		# Translators: LaTeX command - hyperbolic cotangent
		"name": _("hyperbolic cotangent"),
	},
	{
		"id": "floor",
		# Translators: LaTeX command - floor
		"name": _("floor"),
	},
	{
		"id": "ceil",
		# Translators: LaTeX command - ceil
		"name": _("ceil"),
	},
]
latexData = []
latexAll = []
latexShortcut = {}
latexCommand = {}
latexMenu = {}

greekAlphabetMenuData = [
	{
		"id": "alpha",
		# Translators: greek alphabet command - alpha
		"name": _("alpha"),
	},
	{
		"id": "beta",
		# Translators: greek alphabet command - beta
		"name": _("beta"),
	},
	{
		"id": "gamma",
		# Translators: greek alphabet command - gamma
		"name": _("gamma"),
	},
	{
		"id": "delta",
		# Translators: greek alphabet command - delta
		"name": _("delta"),
	},
	{
		"id": "epsilon",
		# Translators: greek alphabet command - epsilon
		"name": _("epsilon"),
	},
	{
		"id": "zeta",
		# Translators: greek alphabet command - zeta
		"name": _("zeta"),
	},
	{
		"id": "eta",
		# Translators: greek alphabet command - eta
		"name": _("eta"),
	},
	{
		"id": "theta",
		# Translators: greek alphabet command - theta
		"name": _("theta"),
	},
	{
		"id": "iota",
		# Translators: greek alphabet command - iota
		"name": _("iota"),
	},
	{
		"id": "kappa",
		# Translators: greek alphabet command - kappa
		"name": _("kappa"),
	},
	{
		"id": "lambda",
		# Translators: greek alphabet command - lambda
		"name": _("lambda"),
	},
	{
		"id": "mu",
		# Translators: greek alphabet command - mu
		"name": _("mu"),
	},
	{
		"id": "nu",
		# Translators: greek alphabet command - nu
		"name": _("nu"),
	},
	{
		"id": "xi",
		# Translators: greek alphabet command - xi
		"name": _("xi"),
	},
	{
		"id": "omicron",
		# Translators: greek alphabet command - o
		"name": _("omicron"),
	},
	{
		"id": "pi",
		# Translators: greek alphabet command - pi
		"name": _("pi"),
	},
	{
		"id": "rho",
		# Translators: greek alphabet command - rho
		"name": _("rho"),
	},
	{
		"id": "sigma",
		# Translators: greek alphabet command - sigma
		"name": _("sigma"),
	},
	{
		"id": "tau",
		# Translators: greek alphabet command - tau
		"name": _("tau"),
	},
	{
		"id": "upsilon",
		# Translators: greek alphabet command - upsilon
		"name": _("upsilon"),
	},
	{
		"id": "phi",
		# Translators: greek alphabet command - phi
		"name": _("phi"),
	},
	{
		"id": "chi",
		# Translators: greek alphabet command - chi
		"name": _("chi"),
	},
	{
		"id": "psi",
		# Translators: greek alphabet command - psi
		"name": _("psi"),
	},
	{
		"id": "omega",
		# Translators: greek alphabet command - down omega
		"name": _("omega"),
	},
]
greekAlphabetData = []
greekAlphabetAll = []
greekAlphabetShortcut = {}
greekAlphabetCommand = {}
greekAlphabetMenu = {}


def initialize():
	global latexData, latexAll, latexCommand, latexShortcut, latexMenu
	BASE_DIR = os.path.dirname(__file__)
	path = os.path.join(BASE_DIR, 'latexs_user.csv')
	latexData = load(path)
	latexAll = joinObjectArray(latexData, latexMenuData, key="id")
	latexMenu = [{
		**i, **{
			"type": "item",
		}
	} for i in latexAll]
	latexMenu = groupByField(latexMenu, 'category', lambda i: i, lambda i: i)
	for key, value in latexMenu.items():
		latexMenu[key] = sorted(value, key=lambda i: i['order'])
	latexCommand = data2commandMap(latexAll)
	latexShortcut = data2shortcutMap(latexAll)

	global greekAlphabetData, greekAlphabetAll, greekAlphabetCommand, greekAlphabetShortcut, greekAlphabetMenu
	BASE_DIR = os.path.dirname(__file__)
	path = os.path.join(BASE_DIR, 'GreekAlphabets.csv')
	greekAlphabetData = load(path)
	greekAlphabetAll = joinObjectArray(greekAlphabetData, greekAlphabetMenuData, key="id")
	greekAlphabetMenu = [{
		**i, **{
			"type": "item",
		}
	} for i in greekAlphabetAll]
	greekAlphabetMenu = groupByField(greekAlphabetMenu, 'category', lambda i: i, lambda i: i)
	for key, value in greekAlphabetMenu.items():
		greekAlphabetMenu[key] = sorted(value, key=lambda i: i['order'])
	greekAlphabetCommand = data2commandMap(greekAlphabetAll)
	greekAlphabetShortcut = data2shortcutMap(greekAlphabetAll)


def terminate():
	BASE_DIR = os.path.dirname(__file__)
	path = os.path.join(BASE_DIR, 'latexs_user.csv')
	save(path, latexAll)
