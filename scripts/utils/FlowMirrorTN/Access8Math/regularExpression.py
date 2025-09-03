from delimiter import LaTeX as LaTeX_delimiter, AsciiMath as AsciiMath_delimiter, Nemeth as Nemeth_delimiter

translate_dict = {
	ord("$"): r"\$",
	ord("("): r"\(",
	ord(")"): r"\)",
	ord("\\"): r"\\",
}


def delimiterRegularExpression(delimiter):
	latex_restring = r"(?<=[^\\])(?P<ld_start>{delimiter_start})(?P<latex>[^\n]*?[^\\^\n])?(?P<ld_end>{delimiter_end})".format(
		delimiter_start=LaTeX_delimiter[delimiter["latex"]]["start"].translate(translate_dict),
		delimiter_end=LaTeX_delimiter[delimiter["latex"]]["end"].translate(translate_dict),
	)
	latex_restring_start = r"(?P<lsd_start>{delimiter_start})(?P<latex_start>[^\n]*?[^\\^\n])?(?P<lsd_end>{delimiter_end})".format(
		delimiter_start=LaTeX_delimiter[delimiter["latex"]]["start"].translate(translate_dict),
		delimiter_end=LaTeX_delimiter[delimiter["latex"]]["end"].translate(translate_dict),
	)

	asciimath_restring = r"(?<=[^\\])(?P<ad_start>{delimiter_start})(?P<asciimath>[^\n]*?[^\\^\n])?(?P<ad_end>{delimiter_end})".format(
		delimiter_start=AsciiMath_delimiter[delimiter["asciimath"]]["start"].translate(translate_dict),
		delimiter_end=AsciiMath_delimiter[delimiter["asciimath"]]["end"].translate(translate_dict),
	)
	asciimath_restring_start = r"(?P<asd_start>{delimiter_start})(?P<asciimath_start>[^\n]*?[^\\^\n])?(?P<asd_end>{delimiter_end})".format(
		delimiter_start=AsciiMath_delimiter[delimiter["asciimath"]]["start"].translate(translate_dict),
		delimiter_end=AsciiMath_delimiter[delimiter["asciimath"]]["end"].translate(translate_dict),
	)

	nemeth_restring = r"(?<=[^\\])(?P<nd_start>{delimiter_start})(?P<nemeth>[^\n]*?[^\\^\n])?(?P<nd_end>{delimiter_end})".format(
		delimiter_start=Nemeth_delimiter[delimiter["nemeth"]]["start"].translate(translate_dict),
		delimiter_end=Nemeth_delimiter[delimiter["nemeth"]]["end"].translate(translate_dict),
	)
	nemeth_restring_start = r"(?P<nsd_start>{delimiter_start})(?P<nemeth_start>[^\n]*?[^\\^\n])?(?P<nsd_end>{delimiter_end})".format(
		delimiter_start=Nemeth_delimiter[delimiter["nemeth"]]["start"].translate(translate_dict),
		delimiter_end=Nemeth_delimiter[delimiter["nemeth"]]["end"].translate(translate_dict),
	)

	return {
		"asciimath": asciimath_restring,
		"asciimath_start": asciimath_restring_start,
		"latex": latex_restring,
		"latex_start": latex_restring_start,
		"nemeth": nemeth_restring,
		"nemeth_start": nemeth_restring_start,
		"mathml": r"(?P<mathml><math.*?>.*?</math.*?>)"
	}
