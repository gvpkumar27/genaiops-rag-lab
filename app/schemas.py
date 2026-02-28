"""Pydantic schemas for API request and response models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


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
