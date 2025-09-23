from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class QueryRequest(BaseModel):
    query: str = Field(..., description="Вопрос для поиска ответа")
    collection_name: str = Field(..., description="Название коллекции в Qdrant")
    top_k: int = Field(default=5, description="Количество документов для поиска", ge=1, le=20)
    score_threshold: float = Field(default=0.5, description="Минимальный порог релевантности", ge=0.0, le=1.0)
    temperature: float = Field(default=0.1, description="Температура для генерации ответа", ge=0.0, le=1.0)

class SourceDocument(BaseModel):
    file_name: str
    chunk_text: str
    relevance_score: float
    chunk_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query: str
    collection_name: str
    processing_time: float
    status: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, bool]