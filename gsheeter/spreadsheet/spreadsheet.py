from .base import SpreadsheetBase
from .sheet import Sheet

class Spreadsheet(SpreadsheetBase):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._sheet_identifiers = ('sheetId', 'title')

	@property
	def sheets(self):
		sheets = self.getattr('sheets')
		sheets = self.sheetify(sheets)
		self.setattr('sheets', sheets)
		return sheets

	@property
	def sheet_identifiers(self):
		return self._sheet_identifiers

	def sheetify(self, sheets: list):
		for i, sheet in enumerate(sheets):
			if type(sheet) == Sheet:
				continue
			sheets[i] = Sheet(**sheet)
		return sheets

	def find_sheet(
		self,
		target: str,
	) -> Sheet:
		for i, sheet in self.sheets:
			for id in self.sheet_identifiers:
				if sheet.getattr(id) == target:
					return sheet
		return None

	def get_sheet(
		self,
		target: str,
		delete_exist: bool = False,
		add: bool = True,
	) -> Sheet:
		pass