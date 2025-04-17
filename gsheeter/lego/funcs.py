import string, random
from dateparser import parse
from . import types
import re
import datetime as dt


def nested_recursive_search(
  d:dict,
  target_keys : str | list
):
  if type(target_keys) == str:
    return recursive_search(d, target_keys)

  result = None
  current_dict = d

  for i in range(0, len(target_keys)):
    key = target_keys[i]
    curr_result = recursive_search(current_dict, key)

    if isinstance(curr_result, dict):
      current_dict = curr_result
    else:
      result = curr_result

  return result

def recursive_search(d, target_key):
  if isinstance(d, dict):
    for k, v in d.items():
      if k == target_key:
        return v
      elif isinstance(v, dict):
        result = recursive_search(v, target_key)

        if result is not None:
          return result
  return None

def recursive_set(d, key, new_value):
  value_set = False
  current = d

  if isinstance(current, dict):
    for k, v in current.items():
      if k == key:
        current[k] = new_value
        value_set = True
        break
      elif isinstance(v, dict):
        current = v
        recursive_set(current, key, new_value)
  return value_set


def set_nested_value(d:dict, keys: str | list, new_value):
  if type(keys) == str:
    keys = [keys]

  value_set = update_key(d, keys[-1], new_value)

  if not value_set:
    current = d

    for k in keys[:-1]:
      if isinstance(current, dict):
        current_keys = current.keys()

        if k in current_keys:
          current = current[k]
        else:
          current[k] = {}
          current = current[k]

    current[keys[-1]] = new_value


def update_key(d, target_key, new_value):
  completed = False
  if isinstance(d, dict):
    for k, v in d.items():
      if k == target_key:
        d[k] = new_value
        completed = True
        break
      elif isinstance(v, dict):
        update_key(v, target_key, new_value)
      elif isinstance(v, list):
        for item in v:
          update_key(item, target_key, new_value)
  return completed


def get_all_keys(d):
  if isinstance(d, dict):
    for k, v in d.items():
      yield k
      if isinstance(v, dict):
        yield from get_all_keys(v)
  return None


def generate_random_string(length:int=12) -> str:
  chars = string.ascii_letters + string.digits
  random_string = ''.join(random.choice(chars) for i in range(length))
  return random_string


def generate_random_float(min, max) -> float:
  return random.uniform(min, max)


def is_datetime(value):
	return isinstance(value, types.DATETIME_TYPES)


def parse_date(input) -> dt.datetime:
  if type(input) == str:
    pattern_match = re.match(
      r'^(?=\d{2,4}\D\d{2,4})[\d\-\/\:\s]{6,}$',
      input
    )

    if pattern_match is not None:
      return parse(input)
  elif is_datetime(input):
    return parse(input.strftime('%Y-%m-%d %H:%M:%S'))
  return input