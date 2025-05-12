import numpy as np, pandas as pd
import datetime as dt
from ..environ.environ import (
	TABLE_BUFFER,
	TABLE_FILLER,
	FLOAT_FORMAT,
)
from .sheet_types import (
	DATA_TYPES,
	ARRAY_TYPES,
	FRAME_TYPES,
)
from ..lego import types
from typing import Iterable
from pandas import RangeIndex
from icecream import ic
ic.configureOutput(includeContext=True)


def rectanglize(values: list) -> list:
	width = max([len(row) for row in values])

	for i, row in enumerate(values):
		diff = width - len(row)

		for j in range(0, diff):
			values[i].append(None)

	return values

def jsonify_values(values: np.ndarray,) -> list:
	output: np.ndarray= values.copy()
	output: np.ndarray = np.where(
		pd.isna(values),
		'',
		output)

	def format_datetime(val):
		if isinstance(val, np.datetime64):
			return dt.datetime.fromtimestamp(
				val.astype('datetime64[s]')).strftime('%Y-%m-%d %H:%M:%S')
		return val

	def format_numeric(val):
		if np.issubdtype(type(val), np.number):
			if np.issubdtype(type(val), np.floating):
				return FLOAT_FORMAT.format(val)
			return str(val)
		return str(val)

	def format_nan(val):
		if pd.isna(val):
			return ''
		return str(val)

	nan_formmater = np.vectorize(format_nan)
	datetime_formatter = np.vectorize(format_datetime)
	numeric_formatter = np.vectorize(format_numeric)
	output: np.ndarray = nan_formmater(output)
	output: np.ndarray = datetime_formatter(output)
	output: np.ndarray = numeric_formatter(output)
	return output.tolist()

def ndarray_to_df(values: np.ndarray) -> pd.DataFrame:
	value_layers = get_value_layers(values)
	column_height = get_column_height(value_layers)
	index_width = get_index_width(value_layers)
	columns = make_frame_edges(
		values[0:column_height:, index_width:],
		'column'
	)
	data = values[column_height:, index_width:]
	frame_args = {
		'data': data,
		'columns': columns
	}

	if index_width > 0:
		indexes = make_frame_edges(
			values[column_height:, 0:index_width],
			'index'
		)
		frame_args['index'] = indexes

	df = pd.DataFrame(**frame_args)
	return df

def get_value_layers(values: np.ndarray):
	from .sheet_objects import ValueLayers
	return ValueLayers(values=values)

def get_column_height(value_layers) -> int:
	column_height = 1

	for i in range(0, len(value_layers.values)):
		row = value_layers.values[i]
		buffers = row[row == TABLE_BUFFER]
		bin_row = value_layers.bin_layer[i]
		row_sum = bin_row.sum() - len(buffers)
		row_len = len(row) - len(buffers)
		fillers = row[row == TABLE_FILLER]

		if row_sum == row_len and len(fillers) > 0:
			column_height += 1
		else:
			break

	return column_height

def get_index_width(value_layers) -> int:
	if has_digit_index(value_layers.values[:, 0]):
		return 0

	index_width = 0

	for i in range(0, len(value_layers.values[0])):
		column_values = value_layers.values[:, i]
		buffers = column_values[column_values == TABLE_BUFFER]
		fillers = column_values[column_values == TABLE_FILLER]

		if len(buffers) > 0 or len(fillers) > 0:
			index_width += 1
		else:
			break

	return index_width

def has_digit_index(values: DATA_TYPES) -> bool:
	if isinstance(values, pd.DataFrame):
		if isinstance(values.index, RangeIndex):
			return True
	return all([types.DIGIT_REGEX.match(str(val)) for val in values])

def make_frame_edges(
	edges: np.ndarray,
	edge_type: str,
) -> np.ndarray | pd.MultiIndex:
	if any([True if dim == 0 else False for dim in edges.shape]):
		return edges

	if edge_type not in ('column', 'index'):
		raise Exception(f'INVALID EDGE TYPE:{edge_type}')

	output = edges.copy() if edge_type == 'column' else edges.copy().transpose()

	for i, row in enumerate(output):
		for j, val in enumerate(row):
			if val == TABLE_FILLER:
				output[i, j] = output[i, j-1]

	if output.shape[0] == 1:
		return output[0]

	return pd.MultiIndex.from_arrays(output)

def to_ndarray(
	data: DATA_TYPES,
	keep_columns: bool,
) -> np.ndarray:
	if not isinstance(data, pd.DataFrame):
		if isinstance(data, pd.Series):
			return data.to_frame().T.values
		elif isinstance(data, (list, tuple)):
			return np.array(data)
		return data

	values = data.values

	if keep_columns:
		column_array = get_column_array(data)
		values = np.concatenate([column_array, values], axis=0)

	if not has_digit_index(data):
		index_array = get_index_array(data)
		values = np.concatenate((index_array, values), axis=1)

	return values

def get_column_char(column_index: int) -> str:
	alph = ''
	while column_index > -1:
		column_index, remainder = divmod(column_index, 26)
		alph = chr(65 + remainder) + alph
		column_index -= 1
	return alph

def get_index_array(data:pd.DataFrame) -> np.ndarray:
	indexes = data.index
	index_width = indexes.nlevels
	column_height = data.columns.nlevels
	index_array = np.empty(
		shape=(indexes.shape[0], index_width),
		dtype='object')

	for i, idx_row in enumerate(indexes):
		if isinstance(idx_row, tuple):
			for j, idx in enumerate(idx_row):
				if i > 0:
					if index_array[i-1, j] == idx:
						index_array[i, j] = TABLE_FILLER
					else:
						index_array[i, j] = idx
				else:
					index_array[i, j] = idx
		else:
			index_array[i] = idx_row

	full_array= np.full(
		shape=(column_height, index_width),
		fill_value=TABLE_BUFFER)
	index_array = np.concatenate(
		(full_array, index_array),
		axis=0)
	return index_array

def get_column_array(data:pd.DataFrame) -> np.ndarray:
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
						column_array[j, i] = TABLE_FILLER
					else:
						column_array[j, i] = col
				else:
					column_array[j, i] = col
	return column_array

def width(data: DATA_TYPES) -> int:
	if isinstance(data, ARRAY_TYPES):
		pass
	elif isinstance(data, FRAME_TYPES):
		pass
	raise Exception(f'INVALID DATA TYPE:{type(data)}')

def height(data: DATA_TYPES) -> int:
	if isinstance(data, ARRAY_TYPES):
		pass
	elif isinstance(data, FRAME_TYPES):
		pass
	raise Exception(f'INVALID DATA TYPE:{type(data)}')