from ..lego.lego import Lego
from typing import Iterable
from ..lego.api import GoogleAPI
from .sheets_endpoints import SHEETS_ENDPOINTS
import numpy as np, pandas as pd
from ..lego.bamboo import Bamboo

class SpreadsheetBase(Lego, GoogleAPI):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._requests:list = []

	@property
	def spreadsheetId(self):
		return self.getattr('spreadsheetId')

	@property
	def requests(self) -> list:
		return self._requests

	def batchUpdate(
		self,
		requests: list = None
	) -> None:
		if requests is None:
			requests = self.requests

		if type(requests) != Iterable:
			requests = [requests]

		if len(requests) == 0:
			return

		endpoint_items = SHEETS_ENDPOINTS['spreadsheets']['batchUpdate']
		endpoint_items['data'] = {
			'requests': requests
		}
		result = self.request(**endpoint_items)
		return result


class SheetBase(SpreadsheetBase):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	@property
	def values(self) -> np.ndarray:
		return self.getattr('values')

	def get_column_char(
			self,
			column_index: int
		) -> str:
			alph = ''
			while column_index > 0:
				column_index, remainder = divmod(column_index, 26)
				alph = chr(65 + remainder) + alph
				column_index -= 1
			return alph

	
