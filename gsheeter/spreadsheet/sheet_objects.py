from ..lego.lego import Lego
import numpy as np, pandas as pd
from .sheets_enum import IndexType
from typing import Any, Iterable, Mapping
from numbers import Number


class SheetUpdater:
	y_offset = None
	x_offset = None
	data = None

class Table(Lego):
	def __init__(
		self,
		values: np.ndarray,
		**kwargs,
	):
		kwargs['values'] = values
		super().__init__(**kwargs)

	@property
	def values(self) -> np.ndarray:
		return self.getattr('values')

	@property
	def df(self) -> pd.DataFrame:
		pass

	@property
	def column_height(self) -> int:
		return self.getattr('column_height')

	@property
	def index_width(self) -> int:
		return self.getattr('index_width')

	def to_df(self, values: np.ndarray = None) -> pd.DataFrame:
		column_height = None
		index_width = None

	def get_column_height(self, values: np.ndarray) -> int:
		pass

	def get_index_width(self, values: np.ndarray) -> int:
		pass

class ValueLayers(Lego):

	def __init__(
		self,
		values: np.ndarray,
		**kwargs
	):
		kwargs['values'] = values
		kwargs['width'] = None
		super().__init__(**kwargs)

	@property
	def values(self) -> np.ndarray:
		return self.getattr('values')

	@property
	def bin_layer(self) -> np.ndarray:
		bin_layer = self.getattr('bin_layer')

		if bin_layer is None:
			bin_layer = self.make_layer(self.values)
			self.setattr('bin_layer', bin_layer)

		return bin_layer

	@property
	def ver_layer(self) -> np.ndarray:
		ver_layer = self.getattr('ver_layer')
		if ver_layer is None:
			ver_layer = self.make_layer(self.values)
			self.setattr('ver_ayer', ver_layer)

		return ver_layer

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
	def last_fill_col(self):
		pass

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
				else:
					if max_sum < row_sum:
						idx = i
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
	) -> Iterable[np.ndarray, ValueLayers]:
		width = cls.get_width(anchor, layers)

		if width == 0:
			return None, layers

		height = cls.get_height(anchor, width, layers)

		if height == 0:
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