import pandas as pd
import numpy as np
import datetime as dt
import re

DATETIME_TYPES = (
	dt.datetime,
	dt.timedelta,
	dt.date,
	dt.time,
	pd.Timestamp,
	np.datetime64,
)

DATETIME_REGEX = re.compile(
	r'\d{2,}[\-\.\/]+\d{2,}[\-\.\/]+\d{2,}|\d{2}\:\d{2}\:\d{2}'
)
DIGIT_REGEX = re.compile(
	r'^\d+$'
)
NUM_REGEX = re.compile(
	r'^[\d\,]+[\d\.]*$'
)