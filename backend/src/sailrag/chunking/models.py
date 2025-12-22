from pydantic import BaseModel, Field


class Chunk(BaseModel):
    doc_id: str
    page_number: int = Field(..., ge=1)
    chunk_id: str
    text: str
    char_count: int
    tags: list[str] = Field(default_factory=list)