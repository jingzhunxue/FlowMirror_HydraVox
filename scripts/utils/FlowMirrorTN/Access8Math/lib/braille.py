import braille


def display_braille(regions):
	if not braille.handler.enabled:
		return
	braille.handler.buffer = braille.handler.mainBuffer
	braille.handler.buffer.clear()

	for region in regions:
		region.obj = None
		region.update()
		braille.handler.buffer.regions.append(region)
	braille.handler.buffer.update()
	braille.handler.update()
	# braille.handler.buffer.focus(region)
	# braille.handler.scrollToCursorOrSelection(region)
	# braille.handler.update()
