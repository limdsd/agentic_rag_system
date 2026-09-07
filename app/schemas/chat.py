from typing import List, Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., description="用户输入的提问文本", example="NexusRAG 的技术架构是什么？")

class DocumentCitation(BaseModel):
    source: str
    child_content: str
    score: float

class ChatResponse(BaseModel):
    query: str
    transformed_query: Optional[str] = None
    answer: str
    citations: List[DocumentCitation] = []
    