from .base import SheetBase
import pandas as pd, numpy as np
import datetime as dt, sys
from .sheets_enum import Dimension
from typing import (
	Union, Iterable, Generator, List
)
from .sheet_objects import (
	Table, ValueLayers,
	SheetSquared
)
from .sheet_types import (
	FRAME_TYPES,
	ARRAY_TYPES,
	DATA_TYPES,
)
from icecream import ic
from .sheet_utils import (
	get_value_layers,
	has_digit_index,
	to_ndarray
)
from gsheeter.environ.environ import (
	TABLE_BUFFER,
	TABLE_FILLER,
)
ic.configureOutput(includeContext=True)


class Sheet(SheetBase):

	def __init__(
		self,
		df_idx: int = 0,
		**kwargs,
	) -> None:
		super().__init__(**kwargs)
		self.df_idx = df_idx
		self._tables = []

	@property
	def tables(self) -> List[Table]:
		self._tables = self.ndarray_to_tables(self.values)
		return self._tables

	@property
	def table(self) -> Table:
		if len(self.tables) == 0:
			return None
		return self.tables[self.df_idx]

	@property
	def df(self) -> pd.DataFrame:
		if len(self.tables) == 0:
			self._df = pd.DataFrame()
		else:
			self._df = self.table.df
		return self._df

	@df.setter
	def df(self, value):
		self._df = value

	def get_table(
		self,
		anchor: tuple,
		outer_height: int,
		outer_width: int,
	) -> Table:
		table_items = self.get_table_items(
			anchor,
			outer_height,
			outer_width)
		return Table(**table_items)

	def get_table_items(
		self,
		anchor: tuple,
		outer_height: int,
		outer_width: int,
	) -> dict:
		return {
			'spreadsheetId': self.spreadsheetId,
			'sheetId': self.sheetId,
			'title': self.title,
			'anchor': anchor,
			'values': self.values,
			'outer_height': outer_height,
			'outer_width': outer_width,
			'rowCount': self.rowCount,
			'columnCount': self.columnCount,
			'parent': self,
		}

	def get_nth_table(self, idx: int) -> Table:
		if idx >=  len(self.tables):
			return None
		return self.tables[idx]

	def ndarray_to_tables(
		self,
		values: np.ndarray
	) -> List[Table]:
		if self.has_single_table(values):
			return [self.get_single_table(values)]

		tables = []
		layers = get_value_layers(values)

		for i in range(0, values.shape[0]):
			for j in range(0, values.shape[1]):
				anchor = (i, j)
				ver = layers.ver_layer[anchor]
				bin = layers.bin_layer[anchor]
				square = None

				if ver != -1 and bin == 1:
					square, layers = SheetSquared.get_square(
						anchor, layers)

				if square is not None:
					table = self.get_table(anchor, *square.shape)
					tables.append(table)

		return tables

	def get_single_table(
		self,
		values: np.ndarray
	) -> Table:
		vl = get_value_layers(values)
		first_fill_idx = vl.first_fill_idx
		last_fill_idx = vl.last_fill_idx
		width = vl.width
		height = last_fill_idx - first_fill_idx + 1
		table = self.get_table(
			anchor=(0, 0),
			outer_height=height,
			outer_width=width)
		return table

	def has_single_table(
		self,
		values: np.ndarray
	) -> bool:
		vl = get_value_layers(values=values)
		first_fill_idx = vl.first_fill_idx
		max_fill_idx = vl.max_fill_idx

		if first_fill_idx is None and max_fill_idx is None:
			return False

		first_fill_row = values[first_fill_idx]
		if any(
			[
				TABLE_FILLER in first_fill_row,
				TABLE_BUFFER in first_fill_row
			]
		):
			return False
		return first_fill_idx == max_fill_idx

	def set_values(
		self,
		data: DATA_TYPES,
		x_offset: int = 0,
		y_offset: int = 0,
		append: bool = False,
	) -> None:
		if len(data) == 0:
			return

		data = self.convert_input(data)
		appended = False

		if append:
			if y_offset == 0:
				appended = self.append_to_table(
					data=data,
					x_offset=x_offset)

			if appended:
				return True

		if not appended:
			self.paste_data(
				data=data,
				x_offset=x_offset,
				y_offset=y_offset,
				append=append)

	def paste_data(
		self,
		data: DATA_TYPES,
		x_offset: int,
		y_offset: int,
		append: bool,
	) -> bool:
		input_values = to_ndarray(
			data=data, keep_columns=True)
		x_anchor = x_offset
		y_anchor = y_offset

		if append:
			y_anchor += self.get_last_filled_y(
				self.values,
				x_offset=x_offset,
				width=input_values.shape[1]
			) + 1

		self.update_sheet(
			values=input_values,
			x_offset=x_anchor,
			y_offset=y_anchor)

	def append_to_table(
		self,
		data: DATA_TYPES,
		x_offset: int,
	) -> bool:
		result = False

		for table in self.tables:
			table: Table = table

			if x_offset == table.x_anchor:
				result = table.append(data)

			if result:
				break

		return result

	def table_append_condition(
		self,
		data: DATA_TYPES,
		x_offset: int,
		table: Table,
	) -> bool:
		result = False

		if x_offset != table.x_anchor:
			return result

		if isinstance(data, ARRAY_TYPES):

			if table.outer_width == data.shape[1]:
				result = True

		elif isinstance(data, FRAME_TYPES):
			data_digit_indexed = has_digit_index(data.index.tolist())
			table_digit_indexed = has_digit_index(table.df.index.tolist())
			data_cols = data.columns
			table_cols = table.df.columns
			index_test = data_digit_indexed == table_digit_indexed

			if index_test:
				comm_cols = list(set(data_cols) & set(table_cols))

				if len(comm_cols) > 0:
					result = True

		return result

	def format(self):
		pass
