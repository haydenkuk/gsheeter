from .base import SheetBase

class Sheet(SheetBase):

	def __init__(
		self,
		**kwargs,
	) -> None:
		super().__init__(**kwargs)
		