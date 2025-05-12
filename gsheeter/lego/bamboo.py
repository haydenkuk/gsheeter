import pandas as pd
from . import types
from ..spreadsheet.sheet_utils import (
	has_digit_index
)

@pd.api.extensions.register_dataframe_accessor('bamboo')
class Bamboo:

	def __init__(self, pandas_obj:pd.DataFrame):
		self._validate(pandas_obj)
		self._df:pd.DataFrame= pandas_obj

	@staticmethod
	def _validate(pandas_obj):
		if not isinstance(pandas_obj, pd.DataFrame):
			raise ValueError('Input must be a pandas DataFrame')

	@property
	def df(self) -> pd.DataFrame:
		return self._df

	@df.setter
	def df(self, value):
		self._df = value

	@property
	def column_height(self):
		return self.df.columns.nlevels

	@property
	def index_width(self):
		if has_digit_index(self.df.index.tolist()):
			return 0
		return self.df.index.nlevels

	def is_num(self, value):
		if type(value) == str:
			num_check = pd.to_numeric(value, errors='coerce')
			return True if pd.notna(num_check) else False
		checkers = (
			pd.isna,
			pd.api.types.is_numeric_dtype
		)

		if any([func(value) for func in checkers]):
			return True

		return False

	def is_datetime(self, value):
		if type(value) == str:
			dt_check = pd.to_datetime(value, format='mixed', errors='coerce')
			return True if pd.notna(dt_check) else False

		checkers = (
			pd.isna,
			pd.api.types.is_datetime64_any_dtype
		)

		if any([func(value) for func in checkers]):
			return True

		return False

	def autodtype(self, values):
		if all([self.is_num(val) for val in values]):
			return pd.to_numeric(values, errors='coerce')
		if all([self.is_datetime(val) for val in values]):
			return pd.to_datetime(values, format='mixed', errors='coerce')
		return values

	def autodtypes(self):
		for col in self.df.columns:
			self.df[col] = self.autodtype(self.df[col])
		return self.df