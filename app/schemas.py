from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = None

class Citation(BaseModel):
    source: str
    chunk_id: int
    score: float
    text_preview: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    meta: Dict[str, Any]
