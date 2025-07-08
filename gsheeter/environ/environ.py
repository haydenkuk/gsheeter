TABLE_FILLER = '-'
TABLE_BUFFER = '/'
FLOAT_FORMAT = '{:.2f}'
AUTOTYPING = True


def set_table_filler(val: str) -> None:
	TABLE_FILLER = val

def set_table_buffer(val: str) -> None:
	TABLE_BUFFER = val

def set_float_format(val: str) -> None:
	FLOAT_FORMAT = val

def set_autotype(val: bool) -> None:
  AUTOTYPING = val
