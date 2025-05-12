from .base import SheetBase
from ..lego.lego import Lego
from ..lego.bamboo import Bamboo
import numpy as np, pandas as pd
from .sheets_enum import IndexType
from typing import (
	Tuple, Union, Generator
)
from .sheet_types import (
	DIMENSIONS,
	DATA_TYPES,
	ARRAY_TYPES
)
from copy import deepcopy
from icecream import ic
from .sheet_utils import (
	ndarray_to_df,
	get_index_width,
	get_column_height,
	get_value_layers,
	to_ndarray,
	make_frame_edges
)
ic.configureOutput(includeContext=True)


class Table(SheetBase):

	def __init__(
		self,
		anchor: tuple,
		outer_height: int,
		outer_width: int,
		parent: SheetBase,
		**kwargs,
	):
		kwargs['anchor'] = anchor
		kwargs['parent'] = parent
		kwargs['properties'] = deepcopy(DIMENSIONS)
		kwargs['properties']['dimensions']['outer_height'] = outer_height
		kwargs['properties']['dimensions']['outer_width'] = outer_width
		super().__init__(**kwargs)
		self._df = None
		self._layers = None
		self.set_dims()

	@property
	def spreadsheetId(self) -> str:
		return self.parent.spreadsheetId

	@property
	def sheetId(self) -> str:
		return self.parent.sheetId

	@property
	def title(self) -> str:
		return self.parent.title

	@property
	def x_anchor(self) -> int:
		return self.getattr('anchor')[1]

	@property
	def y_anchor(self) -> int:
		return self.getattr('anchor')[0]

	@property
	def anchor(self) -> tuple:
		return self.getattr('anchor')

	@property
	def properties(self) -> dict:
		return self.getattr('properties')

	@property
	def layers(self):
		if self._layers is None:
			self._layers = get_value_layers(self.range_values)
		return self._layers

	@property
	def range_values(self) -> np.ndarray:
		ver_range = slice(self.y_anchor, self.y_anchor+self.outer_height)
		hor_range = slice(self.x_anchor, self.x_anchor+self.outer_width)
		return self.values[ver_range, hor_range]

	@property
	def df(self) -> pd.DataFrame:
		if self._df is None:
			self._df = ndarray_to_df(self.range_values)
		return self._df

	@df.setter
	def df(self, value):
		self._df = value

	@property
	def index_width(self) -> int:
		index_width = self.getattr('index_width')
		if pd.isna(index_width):
			index_width = get_index_width(self.layers)
			self.setattr('index_width', index_width)
		return index_width

	@property
	def column_height(self) -> int:
		column_height = self.getattr('column_height')
		if pd.isna(column_height):
			column_height = get_column_height(self.layers)
			self.setattr('column_height', column_height)
		return column_height

	@property
	def inner_height(self) -> int:
		self.setattr('inner_height', self.df.shape[0])
		return self.df.shape[0]

	@property
	def inner_width(self) -> int:
		self.setattr('inner_width', self.df.shape[1])
		return self.df.shape[1]

	@property
	def outer_height(self) -> int:
		return self.getattr('outer_height')

	@property
	def outer_width(self) -> int:
		return self.getattr('outer_width')

	@property
	def parent(self) -> SheetBase:
		return self.getattr('parent')

	@property
	def values(self) -> np.ndarray:
		return self.parent.values

	@values.setter
	def values(self, value):
		self.parent.values = value

	@property
	def rowCount(self) -> int:
		return self.parent.rowCount

	@property
	def columnCount(self) -> int:
		return self.parent.columnCount

	def __iter__(self) -> Generator[pd.Series, None, None]:
		yield from self.df.iterrows()

	def update(self):
		new_values = to_ndarray(self.df, keep_columns=True)
		self.update_sheet(
			values=new_values,
			x_offset=self.x_anchor,
			y_offset=self.y_anchor)

	def detonate(self):
		self.clear_range(
			x_offset=self.x_anchor,
			y_offset=self.y_anchor,
			width=self.outer_width,
			height=self.outer_height)

	def reanchor(self, y_anchor, x_anchor):
		curr_values = self.range_values.copy()
		self.detonate()
		self.update_sheet(
			values=curr_values,
			x_offset=x_anchor,
			y_offset=y_anchor)
		self.setattr('anchor', (y_anchor, x_anchor))

	def update_row(
		self,
		row:pd.Series,
	):
		y_offset = self.find_y(row)
		x_offset = self.x_anchor + self.index_width
		input_values = to_ndarray(row, False)
		self.update_sheet(
			values=input_values,
			x_offset=x_offset,
			y_offset=y_offset)

	def find_y(self, row: pd.Series) -> int:
		index = row.name

		for i, idx in enumerate(self.df.index):
			if idx == index:
				return i + self.y_anchor + self.column_height

		return None

	def set_dims(self):
		self.df
		self.column_height
		self.index_width
		self.inner_height
		self.inner_width

	def delete_row(
		self,
		row: pd.Series
	):
		y_offset = self.find_y(row)
		x_offset = self.x_anchor + self.index_width
		input_values = to_ndarray(row, False)
		empty_arr = np.ndarray(shape=input_values.shape, dtype='object')
		self.update_sheet(
			values=empty_arr,
			x_offset=x_offset,
			y_offset=y_offset)

	def append(
		self,
		data: DATA_TYPES,
	):
		data = self.convert_input(data)
		validated = self.validate_shape(data)

		if not validated:
			return False

		data = self.rearrange(data)
		input_values = to_ndarray(data, False)
		x_offset = self.x_anchor
		y_offset = self.y_anchor + self.outer_height
		self.update_sheet(
			values=input_values,
			x_offset=x_offset,
			y_offset=y_offset)
		new_outer_height = self.outer_height + input_values.shape[0]
		self.setattr(
			'outer_height',
			new_outer_height)
		appendee = data

		if isinstance(data, ARRAY_TYPES):
			if self.index_width > 0:
				idx = make_frame_edges(
					data[:, 0:self.index_width],
					'index')
				body = data[:, self.index_width:]
				appendee = pd.DataFrame(
					data=body,
					columns=self.df.columns,
					index=idx)
			else:
				appendee = pd.DataFrame(
					data=data,
					columns=self.df.columns)

		self.df = pd.concat(
			[self.df, appendee],
			axis=0)
		self.set_dims()
		return True

	def rearrange(
		self,
		data: Union[pd.DataFrame, np.ndarray, pd.Series]
	) -> Union[pd.DataFrame, np.ndarray]:
		if isinstance(data, (np.ndarray, pd.Series)):
			return data

		missing_cols = list(set(self.df.columns) - set(data.columns))

		if len(missing_cols) > 0:
			data.loc[:, missing_cols] = None

		data.columns = self.df.columns
		return data

	def validate_shape(
		self,
		data: Union[pd.DataFrame, np.ndarray]
	) -> bool:
		if not isinstance(data, (pd.DataFrame, np.ndarray)):
			raise Exception(
				'Only pd.DataFrame and np,ndarray can be appended to Table')

		if isinstance(data, np.ndarray):
			if data.shape[1] == self.outer_width:
				return True
		elif isinstance(data, pd.DataFrame):
			if self.index_width == data.bamboo.index_width:
				if len(list(set(self.df.columns) & set(data.columns))) > 1:
					return True

		return False

class ValueLayers(Lego):

	def __init__(
		self,
		values: np.ndarray,
		**kwargs
	):
		kwargs['values'] = values
		kwargs['width'] = None
		super().__init__(**kwargs)
		self.bin_layer = self.make_layer(values)
		self.ver_layer = self.make_layer(values)

	@property
	def values(self) -> np.ndarray:
		return self.getattr('values')

	@property
	def first_fill_idx(self):
		idx = None

		for i, row in enumerate(self.bin_layer):
			if row.sum() > 0:
				idx = i
				break

		return idx

	@property
	def max_fill_idx(self) -> int | None:
		return self.get_max_fill_idx()

	@property
	def width(self) -> int:
		width = self.getattr('width')

		if width is None:
			self.get_max_fill_idx()
			width = self.getattr('width')

		return width

	@property
	def last_fill_idx(self):
		idx = None

		for i, row in enumerate(self.bin_layer):
			row_sum = row.sum()

			if row_sum > 0:
				idx = i

		return idx

	def get_fill_idx(self, idx_type: str) -> int | None:
		if idx_type not in IndexType:
			raise Exception('INVALID IDX_TYPE')

		idx = None

		for i, row in enumerate(self.bin_layer):
			if row.sum() > 0:
				idx = i
				break

		return idx

	def get_max_fill_idx(self) -> int | None:
		idx = None
		max_sum = None

		for i, row in enumerate(self.bin_layer):
			row_len = row.shape[0]
			row_sum = row.sum()

			if row_len == row_sum:
				if max_sum is None:
					idx = i
					max_sum = row_sum
					self.setattr('width', row_sum)
				else:
					if max_sum < row_sum:
						idx = i
						max_sum = row_sum
						self.setattr('width', row_sum)

		return idx

	def make_layer(
		self,
		values: np.ndarray,
		none_val: int = 0,
		fill_val: int = 1,
	) -> np.ndarray:
		layer = values.copy()
		layer = np.where(
			(layer == None),
			none_val,
			fill_val
		)
		return layer

class SheetSquared:

	@classmethod
	def get_square(
		cls,
		anchor: tuple,
		layers: ValueLayers
	) -> Tuple[np.ndarray, ValueLayers]:
		width = cls.get_width(anchor, layers)

		if width < 1:
			return None, layers

		height = cls.get_height(anchor, width, layers)

		if height < 2:
			return None, layers

		region = (
			slice(anchor[0], anchor[0]+height),
			slice(anchor[1], anchor[1]+width)
		)
		square = layers.values[region]
		layers.ver_layer[region] = -1
		return square, layers

	@classmethod
	def get_height(
		cls,
		anchor: tuple,
		width: int,
		layers: ValueLayers
	) -> int:
		val_range = layers.bin_layer[anchor[0]:, anchor[1]:anchor[1]+width]
		height = 0

		for i, row in enumerate(val_range):
			if row.sum() == 0:
				break
			else:
				height += 1

		return height

	@classmethod
	def get_width(
		cls,
		anchor: tuple,
		layers: ValueLayers
	) -> int:
		row = layers.bin_layer[anchor[0], anchor[1]:]
		width = 0

		for i, val in enumerate(row):
			if val == 1:
				width += val
			else:
				break

		return width