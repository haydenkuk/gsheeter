from ..lego.api import GoogleAPI
from .file import File
from .folder import Folder
from . import drive_types
import os
from ..cache.cache import cache
from googleapiclient.http import MediaFileUpload
from ..spreadsheet.spreadsheet import Spreadsheet
import pandas as pd
from ..spreadsheet.sheets_endpoints import (
	SHEETS_ENDPOINTS,
)
from .drive_endpoints import (
	DRIVE_ENDPOINTS
)

class Drive(GoogleAPI):

	@classmethod
	def get_files(
		cls,
		q: str,
		files: str = 'files(name, webViewLink, id, mimeType), nextPageToken',
		pageToken: str = None,
		supportsAllDrives: str = 'true',
		includeItemsFromAllDrives: str = 'true',
	) -> pd.DataFrame:
		endpoint_items = DRIVE_ENDPOINTS['files']['list']
		endpoint_items['endpoint'] = cls.add_query(
			endpoint=endpoint_items['endpoint'],
			q=q,
			files=files,
			pageToken=pageToken,
			supportsAllDrives=supportsAllDrives,
			includeItemsFromAllDrives=includeItemsFromAllDrives)
		result = cls.request(**endpoint_items)
		pageToken = result.get('pageToken')
		output = pd.DataFrame(result.get('files', []))

		while pageToken is not None:
			endpoint_items['data']['pageToken'] = pageToken
			result = cls.request(**endpoint_items)
			pageToken = result.get('pageToken')
			files = result.get('files', [])

			if len(files) > 0:
				output = pd.concat([output, pd.DataFrame(files)])

		return output

	@classmethod
	def get_file(
		cls,
		target: str,
		folderId: str = None,
	) -> File:
		file = cache.get_item(target)

		if file is not None:
			return file

		if folderId is not None:
			q = f"'{folderId}' in parents and trashed = false"
			result = cls.get_files(q=q)
			file: pd.DataFrame= result[
				(result['name'] == target) |
				(result['id'] == target)
			]

			if len(file) == 0:
				raise Exception('FILE NOT FOUND')

			file = File(**file.iloc[0].to_dict())
		else:
			endpoint_items = DRIVE_ENDPOINTS['files']['get']
			endpoint_items['endpoint'] = endpoint_items['endpoint'].format(fileId=target)
			result = cls.request(**endpoint_items)
			file = File(**result)

		cache.set_item(file)
		return file

	@classmethod
	def get_spreadsheet(
		cls,
		target: str,
		folderId: str = None
	) -> Spreadsheet:
		spreadsheet: Spreadsheet = cache.get_item(target)

		if spreadsheet is not None:
			return spreadsheet

		if folderId is not None:
			file = cls.get_file(target=target, folderId=folderId)
			spreadsheetId = file.getattr('id')
		else:
			spreadsheetId = target

		endpoint_items = SHEETS_ENDPOINTS['spreadsheets']['get']
		endpoint_items['endpoint'] = endpoint_items['endpoint'].format(
			spreadsheetId=spreadsheetId)
		spreadsheet = cls.request(**endpoint_items)
		spreadsheet = Spreadsheet(**spreadsheet)
		cache.set_item(spreadsheet)
		return spreadsheet

	@classmethod
	def move_file(
		cls,
		fileId: str,
		parentId: str,
		removeParents: str = 'root',
		supportsAllDrives: str = 'true',
	) -> None:
		payload = DRIVE_ENDPOINTS['files']['update']
		payload['endpoint'] = payload['endpoint'].format(
			fileId=fileId
		)
		payload['endpoint'] = cls.add_query(
			endpoint=payload['endpoint'],
			addParents=parentId,
			removeParents=removeParents,
			supportsAllDrives=supportsAllDrives
		)
		res = cls.request(**payload)
		return res

	@classmethod
	def create_spreadsheet(
		cls,
		filename: str,
		sheetname: str = None,
		parentId: str = None,
	) -> Spreadsheet:
		req = {
			'properties': {
				'title': filename
			},
			'sheets': [
				{
					'properties': {
						'title': sheetname,
						'sheetId': 0,
						'index': 0
					}
				}
			]
		}
		payload = SHEETS_ENDPOINTS['spreadsheets']['create']
		payload['data'] = req
		result = cls.request(**payload)
		ss = Spreadsheet(**result)

		if parentId is not None:
			res = cls.move_file(
				ss.getattr('spreadsheetId'),
				parentId=parentId)
		cache.set_item(ss)
		return ss

	@classmethod
	def create(
		cls,
		folder_id: str,
		name: str = None,
		filepath: str = None,
		filetype: str = None,
		**kwargs
	) -> None:
		name, filepath = cls.get_filename(name=name, filepath=filepath)

	@classmethod
	def parse_mimetype(cls, string:str):
		mimetype = drive_types.MIME_TYPES.get(string)

		if mimetype is not None:
			return mimetype

		for k, v in drive_types.MIME_TYPES.items():
			if k in string or string in k:
				return v

		return None

	@classmethod
	def get_filename(
		cls,
		name:str=None,
		filepath:str=None
	):
		if filepath is None and name is None:
			raise Exception('FILEPATH OR NAME REQUIRED')

		if name is None and filepath is not None:
			name = os.path.basename(filepath)

		return name, filepath

	@classmethod
	def get_filetype(
		cls,
		name:str=None,
		filetype:str=None
	):
		if filetype is not None:
			return filetype

		if name is not None:
			splt_name = name.split('.')

			if len(splt_name) > 1:
				return splt_name[-1]

		raise Exception('INVALID FILENAME')
