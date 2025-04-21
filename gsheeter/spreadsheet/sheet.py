from .base import SpreadsheetBase
from .sheets_endpoints import SHEETS_ENDPOINTS
import pandas as pd, numpy as np
import datetime as dt, sys
from .sheets_enum import Dimension
from ..lego.bamboo import Bamboo
from typing import (
	Union, Mapping, Any, Iterable
)
from .sheet_objects import (
	SheetUpdater, Table, ValueLayers,
	SheetSquared,
)


class Sheet(SpreadsheetBase):

	def __init__(
		self,
		df_idx: int = 0,
		**kwargs,
	) -> None:
		super().__init__(**kwargs)
		self._max_packet_size = 1.9
		self.df_idx = df_idx
		self.tables = []

	@property
	def sheetId(self) -> str:
		return self.getattr('sheetId')

	@property
	def max_packet_size(self) -> float:
		return self._max_packet_size

	@property
	def values(self) -> np.ndarray:
		values = self.getattr('values')

		if values is None:
			values = self.get_values()
			self.setattr('values', values)

		return values

	@property
	def rowCount(self) -> int:
		return self.getattr('rowCount')

	@property
	def columnCount(self) -> int:
		return self.getattr('columnCount')

	@property
	def df(self) -> pd.DataFrame:
		if len(self.tables) == 0:
			self._df = pd.DataFrame()
		else:
			self._df = self.tables[self.df_idx]
		return self._df

	def rectanglize(self, values: list) -> list:
		width = max([len(row) for row in values])

		for i, row in enumerate(values):
			diff = width - len(row)

			for j in range(0, diff):
				values[i].append(None)

		return values

	def jsonify_values(
		self,
		values: np.ndarray,
	) -> list:
		output: np.ndarray= values.copy()
		output: np.ndarray = np.where(
			pd.isna(values),
			'',
			output
		)

		def format_datetime(val):
			if isinstance(val, np.datetime64):
				return dt.datetime.fromtimestamp(
					val.astype('datetime64[s]')).strftime('%Y-%m-%d %H:%M:%S')
			return val

		def format_numeric(val):
			if np.issubdtype(type(val), np.number):
				return str(val)
			return str(val)

		datetime_formatter = np.vectorize(format_datetime)
		numeric_formatter = np.vectorize(format_numeric)
		output: np.ndarray = datetime_formatter(output)
		output: np.ndarray = numeric_formatter(output)
		return output.tolist()

	def append_dimension(
		self,
		dimension: str,
		length: int,
		send: bool = True,
	):
		if dimension not in Dimension:
			raise Exception('INVALID DIMENSION VAR')

		req = {
			'appendDimension': {
				'sheetId': self.getattr('sheetId'),
				'dimension': dimension,
				'length': length
			}
		}

		if send:
			self.batchUpdate(requests=[req])
		else:
			self.requests.append(req)

		return req

	def send_packets(
		self,
		packets: list,
	) -> None:
		endpoint_items = SHEETS_ENDPOINTS['values']['batchUpdate']

		for packet in packets:
			endpoint_items['data'] = {
				'valueInputOption': 'USER_ENTERED',
				'data': packet
			}
			result = self.request(**endpoint_items)

	def make_packets(
		self,
		values: np.ndarray,
		x_offset: int,
		y_offset: int,
	) -> list:
		packets = []
		start = 0
		end = len(values)

		while start < end:
			batch = values[start:end]
			data_size = sys.getsizeof(batch) / (1024 * 1024)

			if data_size > self.max_packet_size:
				end -= 1
			else:
				rng = self.get_range(
					x_offset=x_offset,
					y_offset=y_offset,
					width=len(batch[0]),
					height=len(batch)
				)
				data: list = self.jsonify_values(batch)
				packet = {
					'range': rng,
					'values': data
				}
				packets.append(packet)
				start += len(batch)

		return packet

	def get_range(
		self,
		x_offset: int,
		y_offset: int,
		width: int,
		height: int,
	) -> str:
		start_col = self.get_column_char(x_offset)
		end_col = self.get_column_char(x_offset + width - 1)
		start_row = y_offset + 1
		end_row = y_offset + height
		sheetname = self.getattr('title')
		return f'{sheetname}!{start_col}{start_row}:{end_col}{end_row}'

	def get_values(self) -> np.ndarray:
		spreadsheetId = self.getattr('spreadsheetId')
		title = self.getattr('title')
		endpoint_items = SHEETS_ENDPOINTS['values']['get']
		endpoint_items['endpoint'] = self.add_query(
			endpoint=endpoint_items['endpoint'],
			spreadsheetId=spreadsheetId,
			range=title)
		result = self.request(**endpoint_items)
		rowCount = self.getattr('rowCount')
		columnCount = self.getattr('columnCount')
		values = result.get('values', np.empty((rowCount, columnCount), dtype='object'))

		if type(values) == list:
			values = self.rectanglize(values)
			values = np.array(values, dtype='object')
			values = np.where(values == '', None, values)

		return values

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

	def df_to_ndarray(self, data: pd.DataFrame) -> np.ndarray:
		values = data.values
		column_array = self.column_array(data)
		values = np.concatenate([column_array, values], axis=0)

		if not data.bamboo.has_digit_index(data):
			index_array = self.get_index_array(data)
			values = np.concatenate((index_array, values), axis=1)

		return values

	def ndarray_to_tables(
		self,
		values: np.ndarray
	) -> Iterable:
		if self.has_single_table(values):
			return [self.get_single_table(values)]

		tables = []
		layers = ValueLayers(values=values)

		for i in range(0, values.shape[0]):
			for j in range(0, values.shape[1]):
				anchor = (i, j)
				ver = layers.ver_layer[anchor]
				bin = layers.bin_layer[anchor]
				square = None

				if ver != -1 and bin == 1:
					square, layers = SheetSquared.get_square(anchor, layers)

				if square is not None:
					table = Table(values=square)
					tables.append(table)

		return tables

	def get_single_table(
		self,
		values: np.ndarray
	) -> Table:
		vl = ValueLayers(values=values)
		first_fill_idx = vl.first_fill_idx
		last_fill_idx = vl.last_fill_idx
		width = vl.width
		table_vals = values[first_fill_idx:last_fill_idx+1, 0:width]
		return Table(values=table_vals)

	def has_single_table(
		self,
		values: np.ndarray
	) -> bool:
		vl = ValueLayers(values=values,)
		first_fill_idx = vl.first_fill_idx
		max_fill_idx = vl.max_fill_idx

		if first_fill_idx is None and max_fill_idx is None:
			return False

		return vl.first_fill_idx == vl.max_fill_idx

	def get_index_array(self, data:pd.DataFrame) -> np.ndarray:
		indexes = data.index
		index_width = indexes.nlevels
		column_height = data.columns.nlevels
		index_array = np.empty(
			shape=(indexes.shape[0], index_width),
			dtype='object')

		for i, idx_tup in enumerate(indexes):
			for j, idx in enumerate(idx_tup):
				if i > 0:
					if index_array[i-1, j] == idx:
						index_array[i, j] = '-'
					else:
						index_array[i, j] = idx
				else:
					index_array[i, j] = idx
		full_array= np.full(
			shape=(column_height, index_width),
			fill_value='-')
		index_array = np.concatenate(
			(full_array, index_array),
			axis=0)
		return index_array

	def column_array(self, data:pd.DataFrame) -> np.ndarray:
		columns = data.columns
		col_height = columns.nlevels
		column_array = np.empty(shape=(col_height, columns.shape[0]), dtype='object')

		if col_height == 1:
			for i, col in enumerate(columns):
				column_array[0, i] = col
		else:
			for i, col_tup in enumerate(columns):
				for j, col in enumerate(col_tup):
					if i > 0:
						if column_array[j, i-1] == col:
							column_array[j, i] = '-'
						else:
							column_array[j, i] = col
					else:
						column_array[j, i] = col
		return column_array

	def set_values(
		self,
		data: Union[pd.DataFrame, np.ndarray, list, pd.Series],
		x_offset: int = 0,
		y_offset: int = 0,
		append: bool = False,
		format: bool = True
	) -> None:
		if len(data) == 0:
			return

		data = self.convert_input(data=data)
		updater = None

		if append:
			if y_offset == 0:
				updater = self.append_to_table(
					data=data,
					x_offset=x_offset
				)

			if updater is None:
				updater = self.append_data(
					data=data,
					x_offset=x_offset,
					y_offset=y_offset
				)
		else:
			updater = self.get_updater(data=data)

		if updater is None:
			return

		self.update_values(updater)

	def update_values(
		self,
		updater: SheetUpdater,
	) -> None:
		pass

	def append_data(
		self,
		data: Union[pd.DataFrame, np.ndarray],
		x_offset: int,
		y_offset: int,
	) -> SheetUpdater:
		values = self.values
		bin_layer = self.binary_layer(values)
		bin_layer = bin_layer[x_offset:x_offset+data.shape[1], :]
		y_index = 0

		for i, row in enumerate(bin_layer):
			row: np.ndarray = row

			if row.sum() > 0:
				y_index = i

		y_index += y_offset
		updater = SheetUpdater
		updater.y_offset = y_index
		updater.x_offset = x_offset
		updater.data = data
		return updater

	def get_updater(
		self,
		data: Union[pd.DataFrame, np.ndarray],
		x_offset: int,
		y_offset: int,
	) -> SheetUpdater:
		pass

	def append_to_table(
		self,
		data: Union[pd.DataFrame, np.ndarray],
		x_offset: int,
	) -> SheetUpdater:
		pass

	def convert_input(
		self,
		data: Union[pd.DataFrame, np.ndarray, list, pd.Series]
	) -> Union[pd.DataFrame, np.ndarray]:
		if type(data) == pd.Series:
			data = data.to_frame().T
		elif isinstance(data, (np.ndarray, list)):
			data = self.two_dimensionalize(data=data)
		return data

	def two_dimensionalize(
		self,
		data: Union[np.ndarray, list],
	) -> np.ndarray:
		if not isinstance(data[0], (np.ndarray, list)):
			if isinstance(data, list):
				data = np.array(data, dtype='object')

			data = np.array(data)
		return data