from pydantic import BaseModel, Field


class PageText(BaseModel):
    page_number: int = Field(..., ge=1)
    method: str  # "text" or "ocr"
    text: str
    char_count: int
    non_whitespace_ratio: float


class DocumentPreview(BaseModel):
    path: str
    pages_total: int
    pages_previewed: int
    chosen_strategy: str  # "text" or "ocr"
    reason: str
    pages: list[PageText]
