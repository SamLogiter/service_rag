from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Dict
import logging
from .models import QueryRequest, QueryResponse, HealthResponse, SourceDocument
from .rag.pipeline import EnhancedRagPipeline
from .rag.base_pipeline import RagPipelineOllamaQdrant
from config.settings import QDRANT_HOST, QDRANT_PORT

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Service API",
    description="API для работы с RAG системой на базе Ollama и Qdrant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipelines: Dict[str, EnhancedRagPipeline] = {}

def get_or_create_rag_pipeline(collection_name: str) -> EnhancedRagPipeline:
    if collection_name not in rag_pipelines:
        logger.info(f"Создание нового RAG пайплайна для коллекции: {collection_name}")
        rag_pipelines[collection_name] = EnhancedRagPipeline(
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            collection_name=collection_name
        )
    return rag_pipelines[collection_name]

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "RAG Service API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    services_status = {
        "api": True,
        "qdrant": False,
        "ollama": False
    }
    
    try:
        test_pipeline = get_or_create_rag_pipeline("test")
        if test_pipeline._base_pipeline._qdrant_client:
            services_status["qdrant"] = True
    except:
        pass
    
    try:
        test_pipeline = get_or_create_rag_pipeline("test")
        response = test_pipeline._base_pipeline.invoke("test")
        if response.content:
            services_status["ollama"] = True
    except:
        pass
    
    overall_status = "healthy" if all(services_status.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        services=services_status
    )

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def process_query(request: QueryRequest):
    start_time = datetime.now()
    
    try:
        logger.info(f"Обработка запроса: '{request.query}' для коллекции '{request.collection_name}'")
        rag_pipeline = get_or_create_rag_pipeline(request.collection_name)
        
        answer, sources = rag_pipeline.retrieve_context_enhanced(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            temperature=request.temperature
        )
        
        formatted_sources = [
            SourceDocument(
                file_name=source["payload"].get("file_name", "Unknown"),
                chunk_text=source["payload"].get("chunk_text", ""),
                relevance_score=source["score"],
                chunk_id=str(source["id"]) if source.get("id") else None
            ) for source in sources
        ]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Запрос обработан успешно за {processing_time:.2f} секунд")
        
        return QueryResponse(
            answer=answer,
            sources=formatted_sources,
            query=request.query,
            collection_name=request.collection_name,
            processing_time=processing_time,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "query": request.query,
                "collection_name": request.collection_name,
                "processing_time": processing_time,
                "status": "error"
            }
        )

@app.post("/search", tags=["RAG"])
async def search_documents(request: QueryRequest):
    try:
        logger.info(f"Поиск документов: '{request.query}' в коллекции '{request.collection_name}'")
        rag_pipeline = get_or_create_rag_pipeline(request.collection_name)
        
        search_results = rag_pipeline.search_knn_by_query_enhanced(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )
        
        documents = [
            {
                "file_name": result["payload"].get("file_name", "Unknown"),
                "chunk_text": result["payload"].get("chunk_text", ""),
                "relevance_score": result["score"],
                "chunk_id": str(result["id"]) if result.get("id") else None
            } for result in search_results
        ]
        
        return {
            "query": request.query,
            "collection_name": request.collection_name,
            "documents": documents,
            "total_found": len(documents),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Ошибка при поиске документов: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "status": "error"
            }
        )

@app.get("/collections", tags=["Management"])
async def list_collections():
    return {
        "collections": list(rag_pipelines.keys()),
        "total": len(rag_pipelines)
    }