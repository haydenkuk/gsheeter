from .base import SheetBase
import pandas as pd, numpy as np
from pandas import RangeIndex
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
from .sheet_utils import (
	get_value_layers,
	has_digit_index,
	to_ndarray
)
from ..environ.environ import (
	TABLE_BUFFER,
	TABLE_FILLER,
)


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
					columns = np.array([
						'-'.join(c)
						for c in table.df.columns
					])
					unique_cols = np.unique(columns)
					add_table = len(columns) == len(unique_cols)

					if not isinstance((index := table.df.index), RangeIndex) and add_table:
						index = np.array([str(r) for r in index])
						unique_idx = np.unique(index)
						add_table = len(index) == len(unique_idx)

					if add_table:
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

		if (first_fill_idx is None
			and max_fill_idx is None):
			return False

		first_fill_row = values[first_fill_idx]
		if any(
			[
				TABLE_FILLER in first_fill_row,
				TABLE_BUFFER in first_fill_row
			]
		):
			return False

		first_row = values[first_fill_idx]
		unique_arr = np.unique(first_row[pd.notna(first_row)])
		return (
			first_fill_idx == max_fill_idx and
			len(first_row) == len(unique_arr)
		)

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

	def format(self):
		pass
