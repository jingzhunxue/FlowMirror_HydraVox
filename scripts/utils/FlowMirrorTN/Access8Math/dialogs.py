import copy
from collections import OrderedDict
import os
import shutil
import wx

import addonHandler
import api
import config
import core
import gui
from gui import guiHelper, nvdaControls
from gui.contextHelp import ContextHelpMixin
from gui.settingsDialogs import MultiCategorySettingsDialog, SettingsDialog, SettingsPanel
import languageHandler
from logHandler import log
from mathPres.mathPlayer import MathPlayer
import queueHandler
import tones

import A8M_PM
from A8M_PM import MathContent

addonHandler.initTranslation()

base_path = os.path.dirname(os.path.abspath(__file__))
mathPlayer = None
try:
	mathPlayer = MathPlayer()
except BaseException:
	log.warning("MathPlayer 4 not available")

mathCAT = None
try:
	from globalPlugins.MathCAT import MathCAT
	mathCAT = MathCAT()
except BaseException:
	log.warning("MathCAT not available")


class A8MSettingsPanel(SettingsPanel):
	# Translators: Title of a setting dialog.
	title = _("Access8Math")
	settings = OrderedDict({})
	field = "Access8Math"

	def makeSettings(self, settingsSizer):
		sHelper = guiHelper.BoxSizerHelper(self, sizer=settingsSizer)
		for k, v in self.settings.items():
			if "options" in v:
				attr = k + "Selection"
				options = v["options"]
				if mathCAT and "Access8Math" in v["options"]:
					# Translators: A choice of a combobox in the writing settings dialog (not used)
					v["options"]["MathCAT"] = _("MathCAT")
				if mathPlayer and "Access8Math" in v["options"]:
					# Translators: A choice of a combobox in the writing settings dialog (not used)
					v["options"]["MathPlayer"] = _("Math Player")
				widget = sHelper.addLabeledControl(v["label"], wx.Choice, choices=list(options.values()))
				setattr(self, attr, widget)
				try:
					index = list(v["options"].keys()).index(str(config.conf["Access8Math"][self.field][k]))
				except BaseException:
					index = 0
					tones.beep(100, 100)
				widget.Selection = index
			else:
				setattr(self, k + "CheckBox", sHelper.addItem(wx.CheckBox(self, label=v["label"])))
				value = config.conf["Access8Math"][self.field][k]
				getattr(self, k + "CheckBox").SetValue(value)

	def onSave(self):
		try:
			for k, v in self.settings.items():
				if "options" in v:
					attr = k + "Selection"
					widget = getattr(self, attr)
					config.conf["Access8Math"][self.field][k] = list(v["options"].keys())[widget.GetSelection()]
				else:
					config.conf["Access8Math"][self.field][k] = getattr(self, k + "CheckBox").IsChecked()
		except BaseException:
			for k, v in self.settings.items():
				if "options" in v:
					config.conf["Access8Math"][self.field][k] = list(v["options"].keys())[0]
				else:
					config.conf["Access8Math"][self.field][k] = True
			tones.beep(100, 100)

		A8M_PM.initialize(config.conf["Access8Math"])


class MathReaderSettingsPanel(A8MSettingsPanel):
	# Translators: Title of a setting dialog.
	title = _("Math Reader")
	settings = OrderedDict({
		"speech_source": {
			# Translators: The label of an option in the reading settings dialog
			"label": _("&Speech source:"),
			"options": {
				# Translators: A choice of a combobox in the reading settings dialog
				"Access8Math": _("Access8Math"),
			}
		},
		"braille_source": {
			# Translators: The label of an option in the reading settings dialog
			"label": _("&Braille source:"),
			"options": {
				# Translators: A choice of a combobox in the reading settings dialog
				"Access8Math": _("Access8Math"),
			}
		},
		"interact_source": {
			# Translators: The label of an option in the reading settings dialog
			"label": _("Inter&act source:"),
			"options": {
				# Translators: A choice of a combobox in the reading settings dialog
				"Access8Math": _("Access8Math"),
			}
		},
	})
	field = "settings"


class ReadingSettingsPanel(A8MSettingsPanel):
	# Translators: Title of a setting dialog.
	title = _("Reading")
	settings = OrderedDict({
		# Translators: The label of an option in the reading settings dialog
		"language": {
			"label": _("&Language:"),
			"options": {}
		},
		"analyze_math_meaning": {
			# Translators: The label of an option in the reading settings dialog
			"label": _("Analyze mathematical meaning of content")
		},
		"auto_generate": {
			# Translators: The label of an option in the reading settings dialog
			"label": _("Reading of auto-generated meaning in interaction navigation mode")
		},
		"no_move_beep": {
			# Translators: The label of an option in the reading settings dialog
			"label": _("Use tone indicate to no move in interaction navigation mode")
		},
	})
	field = "settings"

	def makeSettings(self, settingsSizer):
		try:
			available_languages = A8M_PM.available_languages()
		except BaseException:
			available_languages = []
		map = dict(languageHandler.getAvailableLanguages())

		available_languages_dict = {}
		for k in available_languages:
			try:
				available_languages_dict[k] = map[k]
			except KeyError:
				available_languages_dict[k] = k

		self.settings["language"]["options"] = available_languages_dict

		sHelper = guiHelper.BoxSizerHelper(self, sizer=settingsSizer)
		for k, v in self.settings.items():
			if "options" in v:
				attr = k + "Selection"
				options = v["options"]
				if mathCAT and "Access8Math" in v["options"]:
					# Translators: A choice of a combobox in the writing settings dialog (not used)
					v["options"]["MathCAT"] = _("MathCAT")
				if mathPlayer and "Access8Math" in v["options"]:
					# Translators: A choice of a combobox in the writing settings dialog (not used)
					v["options"]["MathPlayer"] = _("Math Player")
				widget = sHelper.addLabeledControl(v["label"], wx.Choice, choices=list(options.values()))
				setattr(self, attr, widget)
				try:
					index = list(v["options"].keys()).index(str(config.conf["Access8Math"][self.field][k]))
				except BaseException:
					index = 0
					tones.beep(100, 100)
				widget.Selection = index
			else:
				setattr(self, k + "CheckBox", sHelper.addItem(wx.CheckBox(self, label=v["label"])))
				value = config.conf["Access8Math"][self.field][k]
				getattr(self, k + "CheckBox").SetValue(value)

		# Translators: The label of an option in the reading settings dialog
		item_interval_timeLabel = _("&Item interval time:")
		self.item_interval_timeChoices = [str(i) for i in range(1, 101)]
		self.item_interval_timeList = sHelper.addLabeledControl(item_interval_timeLabel, wx.Choice, choices=self.item_interval_timeChoices)
		try:
			index = self.item_interval_timeChoices.index(str(config.conf["Access8Math"][self.field]["item_interval_time"]))
		except BaseException:
			index = 0
			tones.beep(100, 100)
		self.item_interval_timeList.Selection = index

	def onSave(self):
		try:
			config.conf["Access8Math"][self.field]["item_interval_time"] = self.item_interval_timeChoices[self.item_interval_timeList.GetSelection()]
		except BaseException:
			config.conf["Access8Math"][self.field]["item_interval_time"] = 50

		try:
			for k, v in self.settings.items():
				if "options" in v:
					attr = k + "Selection"
					widget = getattr(self, attr)
					config.conf["Access8Math"][self.field][k] = list(v["options"].keys())[widget.GetSelection()]
				else:
					config.conf["Access8Math"][self.field][k] = getattr(self, k + "CheckBox").IsChecked()
		except BaseException:
			for k, v in self.settings.items():
				if "options" in v:
					config.conf["Access8Math"][self.field][k] = list(v["options"].keys())[0]
				else:
					config.conf["Access8Math"][self.field][k] = True
			tones.beep(100, 100)

		A8M_PM.initialize(config.conf["Access8Math"])


class WritingSettingsPanel(A8MSettingsPanel):
	# Translators: Title of a setting dialog.
	title = _("Writing")
	settings = OrderedDict({
		"command_mode": {
			# Translators: The label of an option in the writing settings dialog
			"label": _("Activate command gesture at startup")
		},
		"navigate_mode": {
			# Translators: The label of an option in the writing settings dialog
			"label": _("Activate block navigate gesture at startup")
		},
		"shortcut_mode": {
			# Translators: The label of an option in the writing settings dialog
			"label": _("Activate shortcut gesture at startup")
		},
		"writeNavAudioIndication": {
			# Translators: The label of an option in the writing settings dialog
			"label": _("Use audio indicate to switching of browse navigation mode")
		},
		"writeNavAcrossLine": {
			# Translators: The label of an option in the writing settings dialog
			"label": _("Left/Right arrow key allow of moving across line in browse navigation mode")
		},
	})
	field = "settings"


class DocumentSettingsPanel(A8MSettingsPanel):
	# Translators: Title of a setting dialog.
	title = _("Document")
	settings = OrderedDict({
		"LaTeX_delimiter": {
			# Translators: The label of an option in the writing settings dialog
			"label": _("&LaTeX delimiter:"),
			"options": {
				# Translators: A choice of a combobox in the writing settings dialog
				"bracket": _("bracket"),
				# Translators: A choice of a combobox in the writing settings dialog
				"dollar": _("dollar"),
			}
		},
		"Nemeth_delimiter": {
			# Translators: The label of an option in the writing settings dialog
			"label": _("&Nemeth delimiter:"),
			"options": {
				# Translators: A choice of a combobox in the writing settings dialog
				"at": _("at"),
				# Translators: A choice of a combobox in the writing settings dialog
				"ueb": _("UEB"),
			}
		},
		"HTML_document_display": {
			# Translators: The label of an option in the writing settings dialog
			"label": _("HTML &document display:"),
			"options": {
				# Translators: A choice of a combobox in the writing settings dialog
				"markdown": _("Markdown"),
				# Translators: A choice of a combobox in the writing settings dialog
				"text": _("text"),
			}
		},
		"HTML_math_display": {
			# Translators: The label of an option in the writing settings dialog
			"label": _("HTML &math display:"),
			"options": {
				# Translators: A choice of a combobox in the writing settings dialog
				"block": _("block"),
				# Translators: A choice of a combobox in the writing settings dialog
				"inline": _("inline"),
			}
		},
		"color": {
			# Translators: The label of an option in the writing settings dialog
			"label": _("HTML font color:"),
			"options": {
				# Translators: A choice of a combobox in the writing settings dialog
				"#ffffff": _("white"),
				# Translators: A choice of a combobox in the writing settings dialog
				"#000000": _("black"),
			}
		},
		"bg_color": {
			# Translators: The label of an option in the writing settings dialog
			"label": _("HTML background color:"),
			"options": {
				# Translators: A choice of a combobox in the writing settings dialog
				"#ffffff": _("white"),
				# Translators: A choice of a combobox in the writing settings dialog
				"#000000": _("black"),
			}
		},
	})
	field = "settings"


class RuleSettingsPanel(A8MSettingsPanel):
	# Translators: Title of a setting dialog.
	title = _("Rule")
	settings = OrderedDict({
		"SingleMsubsupType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Simplified subscript and superscript")
		},
		"SingleMsubType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Simplified subscript")
		},
		"SingleMsupType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Simplified superscript")
		},
		"SingleMunderoverType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Simplified underscript and overscript")
		},
		"SingleMunderType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Simplified underscript")
		},
		"SingleMoverType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Simplified overscript")
		},
		"SingleFractionType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Simplified fraction")
		},
		"SingleSqrtType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Simplified square root")
		},
		"PowerType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Power")
		},
		"SquarePowerType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Square power")
		},
		"CubePowerType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Cube power")
		},
		"SetType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Set")
		},
		"AbsoluteType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Absolute value")
		},
		"MatrixType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Matrix")
		},
		"DeterminantType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Determinant")
		},
		"AddIntegerFractionType": {
			# Translators: The label of an option in the Rule settings dialog
			"label": _("Integer and fraction")
		},
	})
	field = "rules"


NvdaSettingsDialogActiveConfigProfile = None
NvdaSettingsDialogWindowHandle = None


class Access8MathSettingsDialog(MultiCategorySettingsDialog):
	# Translators: This is the label for the Access8Math settings dialog.
	title = _("Access8Math Settings")
	INITIAL_SIZE = (1000, 480)
	MIN_SIZE = (470, 240)

	categoryClasses = [
		ReadingSettingsPanel,
		WritingSettingsPanel,
		DocumentSettingsPanel,
		RuleSettingsPanel,
		MathReaderSettingsPanel,
	]

	def __init__(self, parent, initialCategory=None):
		super().__init__(parent, initialCategory)

	def makeSettings(self, settingsSizer):
		# Ensure that after the settings dialog is created the name is set correctly
		super().makeSettings(settingsSizer)
		self._doOnCategoryChange()
		global NvdaSettingsDialogWindowHandle
		NvdaSettingsDialogWindowHandle = self.GetHandle()

	def _doOnCategoryChange(self):
		global NvdaSettingsDialogActiveConfigProfile
		NvdaSettingsDialogActiveConfigProfile = config.conf.profiles[-1].name
		if not NvdaSettingsDialogActiveConfigProfile:
			# Translators: The profile name for normal configuration
			NvdaSettingsDialogActiveConfigProfile = _("normal configuration")
		self.SetTitle(self._getDialogTitle())

	def _getDialogTitle(self):
		return u"{dialogTitle}: {panelTitle} ({configProfile})".format(
			dialogTitle=self.title,
			panelTitle=self.currentCategory.title,
			configProfile=NvdaSettingsDialogActiveConfigProfile
		)

	def onCategoryChange(self, evt):
		super().onCategoryChange(evt)
		if evt.Skipped:
			return
		self._doOnCategoryChange()

	def Destroy(self):
		global NvdaSettingsDialogActiveConfigProfile, NvdaSettingsDialogWindowHandle
		NvdaSettingsDialogActiveConfigProfile = None
		NvdaSettingsDialogWindowHandle = None
		super().Destroy()


class Symbol:
	def __init__(self, identifier, displayName="", replacement=""):
		self.identifier = identifier
		self.displayName = displayName
		self.replacement = replacement


class AddSymbolDialog(
		ContextHelpMixin,
		wx.Dialog  # wxPython does not seem to call base class initializer, put last in MRO
):

	helpId = "SymbolPronunciation"
	
	def __init__(self, parent):
		# Translators: This is the label for the add symbol dialog.
		super().__init__(parent, title=_("Add Symbol"))
		mainSizer = wx.BoxSizer(wx.VERTICAL)
		sHelper = guiHelper.BoxSizerHelper(self, orientation=wx.VERTICAL)

		# Translators: This is the label for the edit field in the add symbol dialog.
		symbolText = _("&Symbol:")
		self.identifierTextCtrl = sHelper.addLabeledControl(symbolText, wx.TextCtrl)

		sHelper.addDialogDismissButtons(self.CreateButtonSizer(wx.OK | wx.CANCEL))

		mainSizer.Add(sHelper.sizer, border=guiHelper.BORDER_FOR_DIALOGS, flag=wx.ALL)
		mainSizer.Fit(self)
		self.SetSizer(mainSizer)
		self.identifierTextCtrl.SetFocus()
		self.CentreOnScreen()


class UnicodeDicDialog(SettingsDialog):
	def __init__(self, parent, Access8MathConfig, language, category='speech'):
		self.Access8MathConfig = Access8MathConfig
		self.language = language
		self.category = category

		if category == "speech":
			title = _("Symbol Speech Dictionary")
		elif category == "braille":
			title = _("Symbol Braille Dictionary")

		self.title = _("{title} ({language})").format(
			language=languageHandler.getLanguageDescription(self.language),
			title=title,
		)

		self.A8M_symbol = A8M_PM.load_unicode_dic(language=self.language, category=self.category)

		super().__init__(
			parent,
			resizeable=True,
		)

	def makeSettings(self, settingsSizer):
		self.filteredSymbols = self.symbols = [Symbol(k, k, v) for k, v in self.A8M_symbol.items()]
		self.pendingRemovals = {}

		sHelper = guiHelper.BoxSizerHelper(self, sizer=settingsSizer)
		# Translators: The label of a text field to search for symbols in the speech symbols dialog.
		filterText = pgettext("speechSymbols", "&Filter by:")
		self.filterEdit = sHelper.addLabeledControl(
			labelText=filterText,
			wxCtrlClass=wx.TextCtrl,
			size=(self.scaleSize(310), -1),
		)
		self.filterEdit.Bind(wx.EVT_TEXT, self.onFilterEditTextChange)

		# Translators: The label for symbols list in symbol pronunciation dialog.
		symbolsText = _("&Symbols")
		self.symbolsList = sHelper.addLabeledControl(
			symbolsText,
			nvdaControls.AutoWidthColumnListCtrl,
			autoSizeColumn=2,  # The replacement column is likely to need the most space
			itemTextCallable=self.getItemTextForList,
			style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.LC_VIRTUAL
		)

		# Translators: The label for a column in symbols list used to identify a symbol.
		self.symbolsList.InsertColumn(0, _("Symbol"), width=self.scaleSize(150))
		# Translators: The label for a column in symbols list used to identify a replacement.
		self.symbolsList.InsertColumn(1, _("Replacement"))
		self.symbolsList.Bind(wx.EVT_LIST_ITEM_FOCUSED, self.onListItemFocused)

		# Translators: The label for the group of controls in symbol pronunciation dialog to change the pronunciation of a symbol.
		changeSymbolText = _("Change selected symbol")
		changeSymbolSizer = wx.StaticBoxSizer(wx.VERTICAL, self, label=changeSymbolText)
		changeSymbolGroup = guiHelper.BoxSizerHelper(self, sizer=changeSymbolSizer)
		changeSymbolHelper = sHelper.addItem(changeSymbolGroup)

		# Used to ensure that event handlers call Skip(). Not calling skip can cause focus problems for controls. More
		# generally the advice on the wx documentation is: "In general, it is recommended to skip all non-command events
		# to allow the default handling to take place. The command events are, however, normally not skipped as usually
		# a single command such as a button click or menu item selection must only be processed by one handler."
		def skipEventAndCall(handler):
			def wrapWithEventSkip(event):
				if event:
					event.Skip()
				return handler()
			return wrapWithEventSkip

		# Translators: The label for the edit field in symbol pronunciation dialog to change the replacement text of a symbol.
		replacementText = _("&Replacement")
		self.replacementEdit = changeSymbolHelper.addLabeledControl(
			labelText=replacementText,
			wxCtrlClass=wx.TextCtrl,
			size=(self.scaleSize(300), -1),
		)
		self.replacementEdit.Bind(wx.EVT_TEXT, skipEventAndCall(self.onSymbolEdited))

		bHelper = sHelper.addItem(guiHelper.ButtonHelper(orientation=wx.HORIZONTAL))
		# Translators: The label for a button in the Symbol Pronunciation dialog to add a new symbol.
		addButton = bHelper.addButton(self, label=_("&Add"))

		# Translators: The label for a button in the Symbol Pronunciation dialog to remove a symbol.
		self.removeButton = bHelper.addButton(self, label=_("Re&move"))
		self.removeButton.Disable()

		# Translators: The label for a button to restore default value.
		restoreDefaultButton = bHelper.addButton(self, label=_("&Restore default"))

		# Translators: The label for a button to import unicode.dic.
		importButton = bHelper.addButton(self, label=_("&Import"))

		# Translators: The label for a button to export unicode.dic.
		exportButton = bHelper.addButton(self, label=_("Exp&ort"))

		addButton.Bind(wx.EVT_BUTTON, self.OnAddClick)
		self.removeButton.Bind(wx.EVT_BUTTON, self.OnRemoveClick)
		restoreDefaultButton.Bind(wx.EVT_BUTTON, self.OnRestoreDefaultClick)
		importButton.Bind(wx.EVT_BUTTON, self.OnImportClick)
		exportButton.Bind(wx.EVT_BUTTON, self.OnExportClick)

		# Populate the unfiltered list with symbols.
		self.filter()

	def postInit(self):
		self.symbolsList.SetFocus()

	def filter(self, filterText=''):
		NONE_SELECTED = -1
		previousSelectionValue = None
		previousIndex = self.symbolsList.GetFirstSelected()  # may return NONE_SELECTED
		if previousIndex != NONE_SELECTED:
			previousSelectionValue = self.filteredSymbols[previousIndex]

		if not filterText:
			self.filteredSymbols = self.symbols
		else:
			# Do case-insensitive matching by lowering both filterText and each symbols's text.
			filterText = filterText.lower()
			self.filteredSymbols = [
				symbol for symbol in self.symbols
				if filterText in symbol.displayName.lower()
				or filterText in symbol.replacement.lower()
			]
		self.symbolsList.ItemCount = len(self.filteredSymbols)

		# sometimes filtering may result in an empty list.
		if not self.symbolsList.ItemCount:
			self.editingItem = None
			# disable the "change symbol" controls, since there are no items in the list.
			self.replacementEdit.Disable()
			self.removeButton.Disable()
			return  # exit early, no need to select an item.

		# If there was a selection before filtering, try to preserve it
		newIndex = 0  # select first item by default.
		if previousSelectionValue:
			try:
				newIndex = self.filteredSymbols.index(previousSelectionValue)
			except ValueError:
				pass

		# Change the selection
		self.symbolsList.Select(newIndex)
		self.symbolsList.Focus(newIndex)
		# We don't get a new focus event with the new index.
		self.symbolsList.sendListItemFocusedEvent(newIndex)

	def getItemTextForList(self, item, column):
		symbol = self.filteredSymbols[item]
		if column == 0:
			return symbol.displayName
		elif column == 1:
			return symbol.replacement
		else:
			raise ValueError("Unknown column: %d" % column)

	def onSymbolEdited(self):
		if self.editingItem is not None:
			# Update the symbol the user was just editing.
			item = self.editingItem
			symbol = self.filteredSymbols[item]
			symbol.replacement = self.replacementEdit.Value

	def onListItemFocused(self, evt):
		# Update the editing controls to reflect the newly selected symbol.
		item = evt.GetIndex()
		symbol = self.filteredSymbols[item]
		self.editingItem = item
		# ChangeValue and Selection property used because they do not cause EVNT_CHANGED to be fired.
		self.replacementEdit.ChangeValue(symbol.replacement)
		self.removeButton.Enable()
		self.replacementEdit.Enable()
		evt.Skip()

	def OnAddClick(self, evt):
		with AddSymbolDialog(self) as entryDialog:
			if entryDialog.ShowModal() != wx.ID_OK:
				return
			identifier = entryDialog.identifierTextCtrl.GetValue()
			if not identifier:
				return
		# Clean the filter, so we can select the new entry.
		self.filterEdit.Value = ""
		self.filter()
		for index, symbol in enumerate(self.symbols):
			if identifier == symbol.identifier:
				gui.messageBox(
					# Translators: An error reported in the Symbol Pronunciation dialog when adding a symbol that is already present.
					_('Symbol "%s" is already present.') % identifier,
					# Translators: The title of the error message window showing up when adding a symbol that is already present in the Symbol Pronunciation dialog .
					_("Error"),
					wx.OK | wx.ICON_ERROR
				)
				self.symbolsList.Select(index)
				self.symbolsList.Focus(index)
				self.symbolsList.SetFocus()
				return
		addedSymbol = Symbol(identifier)
		try:
			del self.pendingRemovals[identifier]
		except KeyError:
			pass
		addedSymbol.displayName = identifier
		addedSymbol.replacement = ""
		self.symbols.append(addedSymbol)
		self.symbolsList.ItemCount = len(self.symbols)
		index = self.symbolsList.ItemCount - 1
		self.symbolsList.Select(index)
		self.symbolsList.Focus(index)
		# We don't get a new focus event with the new index.
		self.symbolsList.sendListItemFocusedEvent(index)
		self.symbolsList.SetFocus()

	def OnRemoveClick(self, evt):
		index = self.symbolsList.GetFirstSelected()
		symbol = self.filteredSymbols[index]
		self.pendingRemovals[symbol.identifier] = symbol
		del self.filteredSymbols[index]
		if self.filteredSymbols is not self.symbols:
			self.symbols.remove(symbol)
		self.symbolsList.ItemCount = len(self.filteredSymbols)
		# sometimes removing may result in an empty list.
		if not self.symbolsList.ItemCount:
			self.editingItem = None
			# disable the "change symbol" controls, since there are no items in the list.
			self.replacementEdit.Disable()
			self.removeButton.Disable()
		else:
			index = min(index, self.symbolsList.ItemCount - 1)
			self.symbolsList.Select(index)
			self.symbolsList.Focus(index)
			# We don't get a new focus event with the new index.
			self.symbolsList.sendListItemFocusedEvent(index)
		self.symbolsList.SetFocus()

	def onOk(self, evt):
		self.onSymbolEdited()
		self.editingItem = None
		for symbol in self.symbols:
			if not symbol.replacement:
				continue

		self.A8M_symbol = {}
		for symbol in self.symbols:
			if not symbol.replacement:
				continue
			self.A8M_symbol[symbol.displayName] = symbol.replacement

		try:
			A8M_PM.save_unicode_dic(self.A8M_symbol, language=self.language, category=self.category)
		except IOError as e:
			log.error("Error saving user unicode dictionary: %s" % e)

		A8M_PM.initialize(self.Access8MathConfig)

		super().onOk(evt)

	def _refreshVisibleItems(self):
		count = self.symbolsList.GetCountPerPage()
		first = self.symbolsList.GetTopItem()
		self.symbolsList.RefreshItems(first, first + count)

	def onFilterEditTextChange(self, evt):
		self.filter(self.filterEdit.Value)
		self._refreshVisibleItems()
		evt.Skip()

	def load(self, path):
		data = A8M_PM.load_unicode_dic(path=path)
		if len(data) == 0:
			return
		self.A8M_symbol = data
		self.filteredSymbols = self.symbols = [Symbol(k, k, v) for k, v in self.A8M_symbol.items()]
		self.pendingRemovals = {}
		self.filter()

	def save(self, path, symbol):
		A8M_PM.save_unicode_dic(symbol, path=path)

	def OnRestoreDefaultClick(self, evt):
		self.onSymbolEdited()
		self.editingItem = None

		path = os.path.join(A8M_PM.LOCALE_DIR, self.language, self.category)

		self.load(os.path.join(path, "unicode.dic"))

	def OnImportClick(self, evt):
		self.onSymbolEdited()
		self.editingItem = None
		with wx.FileDialog(
			# Translators: The title of the import file dialog accessible from the Symbol Pronunciation dialog.
			self, message=_("Import file..."),
			defaultDir=base_path, wildcard="dictionary files (*.dic)|*.dic"
		) as entryDialog:
			if entryDialog.ShowModal() != wx.ID_OK:
				return
			pathname = entryDialog.GetPath()

		self.load(path=pathname)

	def OnExportClick(self, evt):
		with wx.FileDialog(
			self, message=_("Export file..."),
			defaultDir=base_path, defaultFile="export.dic", wildcard="dictionary files (*.dic)|*.dic",
			style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
		) as entryDialog:
			if entryDialog.ShowModal() != wx.ID_OK:
				return
			pathname = entryDialog.GetPath()

		self.A8M_symbol = {}
		for symbol in self.symbols:
			self.A8M_symbol[symbol.displayName] = symbol.replacement

		self.save(pathname, self.A8M_symbol)


class RuleEntryDialog(wx.Dialog):
	def __init__(self, parent, mathrule, title=_("Edit Math Rule Entry")):
		super().__init__(parent, title=title)
		self.mathrule = mathrule
		self.mathrule_child_count = len(self.mathrule[1].role)
		childChoices = [str(i) for i in range(self.mathrule_child_count)]
		mainSizer = wx.BoxSizer(wx.VERTICAL)
		sHelper = guiHelper.BoxSizerHelper(self, orientation=wx.VERTICAL)

		labelText = _("&Description")

		groupLabelText = _("Description")
		descriptionGroup = guiHelper.BoxSizerHelper(self, sizer=wx.StaticBoxSizer(wx.StaticBox(self, label=groupLabelText), wx.VERTICAL))

		labelText = _("&Description")
		self.descriptionWidget = descriptionGroup.addLabeledControl(labelText, wx.TextCtrl)
		self.descriptionWidget.SetValue(self.mathrule[1].description)
		sHelper.addItem(descriptionGroup)

		self.so_widgets = []
		groupLabelText = _("Serialized ordering")
		serializedOrderingGroup = guiHelper.BoxSizerHelper(self, sizer=wx.StaticBoxSizer(wx.StaticBox(self, label=groupLabelText), wx.VERTICAL))
		for index, item in enumerate(self.mathrule[1].serialized_order):
			groupLabelText = _("Ordering %d") % (index)
			itemGroup = guiHelper.BoxSizerHelper(self, sizer=wx.StaticBoxSizer(wx.StaticBox(self, label=groupLabelText), wx.HORIZONTAL))
			if isinstance(item, int):
				labelText = _("Child")
				widget = itemGroup.addLabeledControl(labelText, wx.Choice, choices=childChoices)
				widget.Selection = item
				self.so_widgets.append(widget)
			elif isinstance(item, tuple):
				# before item
				labelText = _("Before item text &%d") % (index)
				widget = itemGroup.addLabeledControl(labelText, wx.TextCtrl)
				widget.SetValue(item[0])
				self.so_widgets.append(widget)
				# after item
				labelText = _("After item text &%d") % (index)
				widget = itemGroup.addLabeledControl(labelText, wx.TextCtrl)
				widget.SetValue(item[1])
				self.so_widgets.append(widget)
			else:
				if index == 0:
					labelText = _("Start text &%d") % (index)
				elif index == len(self.mathrule[1].serialized_order) - 1:
					labelText = _("End text &%d") % (index)
				else:
					labelText = _("&Interval text &%d") % (index)
				widget = itemGroup.addLabeledControl(labelText, wx.TextCtrl)
				widget.SetValue(str(item))
				self.so_widgets.append(widget)
			serializedOrderingGroup.addItem(itemGroup)
		sHelper.addItem(serializedOrderingGroup)

		self.role_widgets = []
		groupLabelText = _("Child role")
		childRoleGroup = guiHelper.BoxSizerHelper(self, sizer=wx.StaticBoxSizer(wx.StaticBox(self, label=groupLabelText), wx.VERTICAL))
		for index, item in enumerate(self.mathrule[1].role):
			labelText = _("&Child %d meaning") % (index)
			widget = childRoleGroup.addLabeledControl(labelText, wx.TextCtrl)
			widget.SetValue(str(item))
			self.role_widgets.append(widget)
		sHelper.addItem(childRoleGroup)

		sHelper.addDialogDismissButtons(self.CreateButtonSizer(wx.OK | wx.CANCEL))

		mainSizer.Add(sHelper.sizer, border=20, flag=wx.ALL)
		mainSizer.Fit(self)
		self.SetSizer(mainSizer)

		try:
			self.descriptionWidget.SetFocus()
		except BaseException:
			pass

		self.Bind(wx.EVT_BUTTON, self.onOk, id=wx.ID_OK)

	def onOk(self, evt):

		self.mathrule[1].description = self.descriptionWidget.GetValue()

		for index, item in enumerate(self.mathrule[1].serialized_order):
			if isinstance(item, int):
				self.mathrule[1].serialized_order[index] = self.so_widgets[index].GetSelection()
			elif isinstance(item, tuple):
				self.mathrule[1].serialized_order[index] = (self.so_widgets[index].GetValue(), u'*')
			else:
				self.mathrule[1].serialized_order[index] = self.so_widgets[index].GetValue()

		for index, item in enumerate(self.mathrule[1].role):
			self.mathrule[1].role[index] = self.role_widgets[index].GetValue()

		try:
			self.ruleEntry = ''
		except Exception as e:
			log.debugWarning("Could not add dictionary entry due to (regex error) : %s" % e)
			# Translators: This is an error message to let the user know that the dictionary entry is not valid.
			gui.messageBox(_("Regular Expression error: \"%s\".") % e, _("Dictionary Entry Error"), wx.OK | wx.ICON_WARNING, self)
			return
		evt.Skip()


class MathRuleDialog(SettingsDialog):
	def __init__(self, parent, Access8MathConfig, language, category):
		self.Access8MathConfig = Access8MathConfig
		self.language = language
		self.category = category

		if category == "speech":
			title = _("Math Rules Speech Output")
		elif category == "braille":
			title = _("Math Rules Braille Output")

		self.title = _("{title} ({language})").format(
			language=languageHandler.getLanguageDescription(self.language),
			title=title,
		)

		self.A8M_mathrule = A8M_PM.load_math_rule(language=self.language, category=self.category)
		self.mathrules = [(k, v) for k, v in self.A8M_mathrule.items() if k not in ['node', 'none']]

		super().__init__(parent)

	def makeSettings(self, settingsSizer):
		sHelper = guiHelper.BoxSizerHelper(self, sizer=settingsSizer)
		# Translators: The label for math rule list in math rules dialog.
		mathrulesText = _("&Mathrules")

		self.mathrulesList = sHelper.addLabeledControl(mathrulesText, nvdaControls.AutoWidthColumnListCtrl, autoSizeColumn=0, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)

		# Translators: The label for a math rule list column header in the math rules dialog.
		self.mathrulesList.InsertColumn(0, _("Rule"))
		# Translators: The label for a math rule list column header in the math rules dialog.
		self.mathrulesList.InsertColumn(1, _("Description"))

		self.refresh()

		self.mathrulesList.Bind(wx.EVT_LIST_ITEM_FOCUSED, self.onListItemFocused)

		# Used to ensure that event handlers call Skip(). Not calling skip can cause focus problems for controls.
		# More generally the advice on the wx documentation is: "In general, it is recommended to skip all
		# non-command events to allow the default handling to take place. The command events are, however, normally
		# not skipped as usually a single command such as a button click or menu item selection must only be
		# processed by one handler."
		def skipEventAndCall(handler):
			def wrapWithEventSkip(event):
				if event:
					event.Skip()
				return handler()
			return wrapWithEventSkip

		bHelper = sHelper.addItem(guiHelper.ButtonHelper(orientation=wx.HORIZONTAL))

		# Translators: The label for a button to edit math rule.
		self.editButton = bHelper.addButton(self, label=_("&Edit"))
		self.editButton.Disable()

		# Translators: The label for a button to example math rule.
		self.exampleButton = bHelper.addButton(self, label=_("E&xample"))
		self.exampleButton.Disable()

		# Translators: The label for a button to restore default value.
		restoreDefaultButton = bHelper.addButton(self, label=_("&Restore default"))

		# Translators: The label for a button to import math.rule.
		importButton = bHelper.addButton(self, label=_("&Import"))

		# Translators: The label for a button to export math.rule.
		exportButton = bHelper.addButton(self, label=_("Exp&ort"))

		self.editButton.Bind(wx.EVT_BUTTON, self.OnEditClick)
		self.exampleButton.Bind(wx.EVT_BUTTON, self.OnExampleClick)
		restoreDefaultButton.Bind(wx.EVT_BUTTON, self.OnRestoreDefaultClick)
		importButton.Bind(wx.EVT_BUTTON, self.OnImportClick)
		exportButton.Bind(wx.EVT_BUTTON, self.OnExportClick)

	def postInit(self):
		self.mathrulesList.SetFocus()

	def load(self, path):
		self.A8M_mathrule = A8M_PM.load_math_rule(path)

		self.mathrules = [(k, v) for k, v in self.A8M_mathrule.items() if k not in ['node', 'none']]
		self.refresh()

		return True

	def save(self, path, mathrule):
		A8M_PM.save_math_rule(mathrule, path=path)
		return True

	def clear(self):
		self.mathrules = []
		self.refresh()

	def refresh(self):
		self.mathrulesList.DeleteAllItems()
		for item in self.mathrules:
			try:
				self.mathrulesList.Append((item[0], item[1].description,))
			except BaseException:
				pass

	def onListItemFocused(self, evt):
		# ChangeValue and Selection property used because they do not cause EVNT_CHANGED to be fired.
		self.editButton.Enable()
		self.exampleButton.Enable()
		evt.Skip()

	def OnEditClick(self, evt):
		index = self.mathrulesList.GetFirstSelected()
		mathrule = copy.deepcopy(self.mathrules[index])
		entryDialog = RuleEntryDialog(self, mathrule)
		if entryDialog.ShowModal() == wx.ID_OK:
			self.mathrules[index] = copy.deepcopy(entryDialog.mathrule)

			for key, mathrule in self.mathrules:
				self.A8M_mathrule[key] = mathrule

			self.refresh()
			self.mathrulesList.Select(index)
			self.mathrulesList.Focus(index)
			self.mathrulesList.SetFocus()
		entryDialog.Destroy()

	def OnExampleClick(self, evt):
		from .interaction import A8MInteraction
		index = self.mathrulesList.GetFirstSelected()
		mathrule = copy.deepcopy(self.mathrules[index])
		mathMl = mathrule[1].example
		mathcontent = MathContent(self.Access8MathConfig["settings"]["language"], mathMl)
		parent = api.getFocusObject()
		vw = A8MInteraction(parent=parent)
		vw.set(data=mathcontent, name="")
		vw.setFocus()

	def OnRestoreDefaultClick(self, evt):
		path = os.path.join(A8M_PM.LOCALE_DIR, self.language, self.category)
		self.clear()
		self.load(os.path.join(path, "math.rule"))

	def OnImportClick(self, evt):
		with wx.FileDialog(
			# Translators: The title of the file selection dialog to import rules
			self, message=_("Import file..."),
			defaultDir=base_path, wildcard="rule files (*.rule)|*.rule"
		) as entryDialog:
			if entryDialog.ShowModal() != wx.ID_OK:
				return
			pathname = entryDialog.GetPath()

		self.clear()
		self.load(pathname)

	def OnExportClick(self, evt):
		with wx.FileDialog(
			# Translators: The title of the file selection dialog to export rules
			self, message=_("Export file..."),
			defaultDir=base_path, defaultFile="export.rule", wildcard="rule files (*.rule)|*.rule",
			style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
		) as entryDialog:
			if entryDialog.ShowModal() != wx.ID_OK:
				return
			pathname = entryDialog.GetPath()

		temp = {}
		for key, mathrule in self.mathrules:
			temp[key] = mathrule
		temp['node'] = self.A8M_mathrule['node']
		temp['none'] = self.A8M_mathrule['none']

		try:
			A8M_PM.save_math_rule(temp, path=pathname)
		except IOError as e:
			log.error("Error saving user unicode dictionary: %s" % e)

	def onOk(self, evt):
		for key, mathrule in self.mathrules:
			self.A8M_mathrule[key] = mathrule

		try:
			A8M_PM.save_math_rule(self.A8M_mathrule, language=self.language, category=self.category)
		except IOError as e:
			log.error("Error saving user unicode dictionary: %s" % e)

		A8M_PM.initialize(self.Access8MathConfig)

		super().onOk(evt)


class NewLanguageAddingDialog(wx.Dialog):
	def __init__(self, parent):
		# Translators: The title of the Add new language dialog
		super().__init__(parent, title=_("New language adding"))
		self.mainSizer = wx.BoxSizer(wx.VERTICAL)
		self.sHelper = guiHelper.BoxSizerHelper(self, orientation=wx.HORIZONTAL)

		exist_languages = A8M_PM.available_languages()
		self.languageNames = languageHandler.getAvailableLanguages()
		self.languageNames = [x for x in self.languageNames if not x[0] in exist_languages and x[0] != "Windows"]
		languageChoices = [x[1] for x in self.languageNames]
		# Translators: The label of a setting in the Add new language dialog
		languageLabelText = _("&Language:")
		self.languageList = self.sHelper.addLabeledControl(languageLabelText, wx.Choice, choices=languageChoices)
		self.languageIndex = self.languageList.Selection = 0

		languageListSize = self.languageList.GetSize()
		self.certainLanguageList = self.sHelper.addItem(wx.Choice(self, size=languageListSize))
		self.certainLanguageList.Hide()

		# Translators: The label of a button in the Add new language dialog
		self.certainButton = self.sHelper.addItem(wx.Button(self, label=_("&Select")))
		self.certainButton.Bind(wx.EVT_BUTTON, self.OnCertainClick)
		self.certainLanguage = None

		# Translators: The label of a button in the Add new language dialog
		self.uncertainButton = self.sHelper.addItem(wx.Button(self, label=_("&Unselect")))
		self.uncertainButton.Bind(wx.EVT_BUTTON, self.OnUncertainClick)
		self.uncertainButton.Hide()

		# bHelper = sHelper.addItem(guiHelper.ButtonHelper(orientation=wx.HORIZONTAL))
		self.bHelper = guiHelper.ButtonHelper(orientation=wx.HORIZONTAL)

		# Add button
		# Translators: The label of a button in the Add new language dialog
		self.OkButton = self.bHelper.addButton(self, label=_("OK"), id=wx.ID_OK)

		# Bind button
		self.Bind(wx.EVT_BUTTON, self.onOk, id=wx.ID_OK)

		# Hide button
		self.OkButton.Disable()

		self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)

		self.mainSizer.Add(self.sHelper.sizer, border=guiHelper.BORDER_FOR_DIALOGS, flag=wx.ALL)
		self.mainSizer.Add(self.bHelper.sizer, border=guiHelper.BORDER_FOR_DIALOGS, flag=wx.ALL)

		self.mainSizer.Fit(self)
		self.SetSizer(self.mainSizer)

	def OnCertainClick(self, evt):
		self.languageIndex = self.languageList.Selection
		self.certainLanguage = self.languageNames[self.languageIndex][0]

		A8M_PM.add_language(self.certainLanguage)

		self.languageList.Clear()
		self.languageList.Append(self.languageNames[self.languageIndex][1])
		self.languageList.Selection = 0

		self.certainButton.Hide()
		self.uncertainButton.Show()
		self.OkButton.Enable()

		self.uncertainButton.SetFocus()

		self.mainSizer.Fit(self)
		# self.mainSizer.Layout()
		self.SetSizer(self.mainSizer)

	def OnUncertainClick(self, evt):
		A8M_PM.remove_language(self.certainLanguage)

		self.certainLanguage = None

		self.languageList.Clear()
		languageChoices = [x[1] for x in self.languageNames]
		self.languageList.AppendItems(languageChoices)
		self.languageList.Selection = self.languageIndex

		self.certainButton.Show()
		self.uncertainButton.Hide()
		self.OkButton.Disable()

		self.certainButton.SetFocus()

		self.mainSizer.Fit(self)
		# self.mainSizer.Layout()
		self.SetSizer(self.mainSizer)

	def OnCloseWindow(self, evt):
		if self.certainLanguage:
			dst = os.path.join(base_path, 'locale', self.certainLanguage)
			try:
				shutil.rmtree(dst)
			except BaseException:
				return
			self.certainLanguage = None
		self.Destroy()

	def onOk(self, evt):
		if gui.messageBox(
			# Translators: The message displayed
			_("For the new language to be added, NVDA must be restarted. Press enter to restart NVDA, or cancel to exit at a later time."),
			# Translators: The title of the dialog
			_("Add new language"), wx.OK | wx.CANCEL | wx.ICON_WARNING, self
		) == wx.OK:
			queueHandler.queueFunction(queueHandler.eventQueue, core.restart)
		self.Destroy()
