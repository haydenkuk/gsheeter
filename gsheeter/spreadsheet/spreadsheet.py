from .base import SpreadsheetBase
from typing import Generator
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

	def __iter__(self) -> Generator[Sheet, None, None]:
		yield from self.sheets

	def sheetify(self, sheets: list):
		from .sheet import Sheet
		for i, sheet in enumerate(sheets):
			if type(sheet) == Sheet:
				continue
			sheets[i] = Sheet(**sheet)
		return sheets

	def find_sheet(
		self,
		target: str,
	):
		for i, sheet in enumerate(self.sheets):
			for id in self.sheet_identifiers:
				if sheet.getattr(id) == target:
					sheet.setattr(
						'spreadsheetId',
						self.getattr('spreadsheetId')
					)
					self.sheets[i] = sheet
					return sheet
		return None

	def get_sheet(
		self,
		target: str,
		delete_exist: bool = False,
		add: bool = True,
	):
		sheet = self.find_sheet(target)

		if sheet is not None and delete_exist:
			self.delete_sheet(sheet, True)

		if add and sheet is None:
			sheet = self.add_sheet(sheetname=target)

		return sheet

	def delete_sheet(
		self,
		target,
		exec: bool=True,
	) -> None:
		from .sheet import Sheet
		sheetId = target.getattr('sheetId') if type(target) == Sheet else None

		if sheetId is None:
			sheet = self.find_sheet(target)

			if sheet is not None:
				sheetId = sheet.getattr('sheetId')

		if sheetId is None:
			return False

		req = {
			'deleteSheet': {
				'sheetId': sheetId
			}
		}
		self.requests.append(req)

		for i, sht in enumerate(self.sheets):
			if sheetId == sht.getattr('sheetId'):
				self.sheets.pop(i)
				break

		if exec:
			self.batchUpdate(self.requests)

		return True

	def add_sheet(
		self,
		sheetname: str='Sheet1',
		rowCount: int=1000,
		columnCount: int=26,
		index: int=0,
	):
		from .sheet import Sheet
		sheet = self.find_sheet(sheetname)

		if sheet is not None:
			return sheet

		req = {
			'addSheet': {
				'properties': {
					'title': sheetname,
					'gridProperties': {
						'rowCount': rowCount,
						'columnCount': columnCount
					},
					'index': index
				}
			}
		}
		self.requests.append(req)
		result: dict= self.batchUpdate(self.requests)
		sheet = result.get('replies')[0].get('addSheet')
		sheet = Sheet(**sheet)
		sheet.setattr(
			'spreadsheetId',
			self.getattr('spreadsheetId')
		)
		self.sheets.append(sheet)
		return sheet
