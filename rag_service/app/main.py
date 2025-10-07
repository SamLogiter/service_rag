from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Dict
import logging
import pandas as pd
from .models import (
    QueryRequest, QueryResponse, HealthResponse, SourceDocument, BencmarkResponse
)
from .rag.pipeline import EnhancedRagPipeline
from ..config.settings import (
    QDRANT_HOST, QDRANT_PORT, VLLM_GEN_HOST, VLLM_GEN_PORT, VLLM_GEN_MODEL,
    VLLM_EMBED_HOST, VLLM_EMBED_PORT, VLLM_EMBED_MODEL
)
# ragas (for benchmark, optional)
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    NonLLMContextRecall, NonLLMContextPrecisionWithReference,
    LLMContextPrecisionWithReference, LLMContextRecall, Faithfulness,
    FactualCorrectness, ContextEntityRecall, NoiseSensitivity, ResponseRelevancy
)
from langchain_ollama import OllamaLLM  # Fallback for ragas if needed
from ragas.run_config import RunConfig
import numpy as np  # For benchmark if needed

my_run_config = RunConfig(max_workers=64, timeout=600000)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Service API", description="RAG with vLLM and Qdrant", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

rag_pipelines: Dict[str, EnhancedRagPipeline] = {}

def get_or_create_rag_pipeline(collection_name: str) -> EnhancedRagPipeline:
    if collection_name not in rag_pipelines:
        logger.info(f"Creating pipeline for {collection_name}")
        rag_pipelines[collection_name] = EnhancedRagPipeline(
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            collection_name=collection_name,
            vllm_gen_host=VLLM_GEN_HOST,
            vllm_gen_port=VLLM_GEN_PORT,
            vllm_gen_model=VLLM_GEN_MODEL,
            vllm_embed_host=VLLM_EMBED_HOST,
            vllm_embed_port=VLLM_EMBED_PORT,
            vllm_embed_model=VLLM_EMBED_MODEL,
        )
    return rag_pipelines[collection_name]

@app.get("/", tags=["General"])
async def root():
    return {"message": "RAG Service API vLLM", "version": "1.1.0"}

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    services = {"api": True, "qdrant": False, "vllm_gen": False, "vllm_embed": False}
    try:
        pipeline = get_or_create_rag_pipeline("test")
        services["qdrant"] = True
        services["vllm_gen"] = True
        services["vllm_embed"] = True
    except Exception:
        pass
    status = "healthy" if all(services.values()) else "degraded"
    return HealthResponse(status=status, timestamp=datetime.now().isoformat(), services=services)

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def process_query(request: QueryRequest):
    start = datetime.now()
    try:
        logger.info(f"Query: '{request.query}' (hybrid: {request.use_hybrid})")
        pipeline = get_or_create_rag_pipeline(request.collection_name)
        answer, sources = pipeline.retrieve_context_enhanced(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            temperature=request.temperature,
            use_hybrid=request.use_hybrid,
            bm25_weight=request.bm25_weight,
            use_rerank=request.use_rerank,
        )
        formatted_sources = [
            SourceDocument(
                file_name=s["payload"].get("file_name", "Unknown"),
                chunk_text=s["payload"].get("pages_text", ""),  # pages_text
                relevance_score=s["score"],
                chunk_id=s["payload"].get("chunk_id"),
                search_type=s.get("search_type", "semantic"),
            ) for s in sources
        ]
        time = (datetime.now() - start).total_seconds()
        search_type = "hybrid" if request.use_hybrid else "semantic"
        return QueryResponse(
            answer=answer, sources=formatted_sources, query=request.query,
            collection_name=request.collection_name, processing_time=time,
            status="success", search_type=search_type
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e), "status": "error"})

@app.get("/benchmark", response_model=BencmarkResponse, tags=["RAG"])
async def process_bench():
    # Your benchmark code (adapted)
    file_path = "./rag_service/app/datasets/ru_rag_test_dataset.pkl"
    df = pd.read_pickle(file_path)
    # ollama_llm = ... (use vLLM wrapper if needed, but fallback to Ollama for ragas)
    ollama_llm = OllamaLLM(model="qwen2.5:7b", temperature=0.1, base_url="http://localhost:11434")  # Adjust
    start_time = datetime.now()

    sample_queries = df["Вопрос"].tolist()[:10]
    expected_responses = df["Правильный ответ"].tolist()[:10]

    dataset = []
    for query, reference in zip(sample_queries, expected_responses):
        request = QueryRequest(
            query=query,
            collection_name="dataset_ru_chunker_token_bf3907a5-3ab3-4444-b963-66ad41eab5ed",
            top_k=5,
            score_threshold=0.2,
            temperature=0.1,
        )
        pipeline = get_or_create_rag_pipeline(request.collection_name)
        answer, sources = pipeline.retrieve_context_enhanced(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            temperature=request.temperature,
        )
        formatted_sources = [
            SourceDocument(
                file_name=s["payload"].get("file_name", "Unknown"),
                chunk_text=s["payload"].get("pages_text", ""),
                relevance_score=s["score"],
                chunk_id=s["payload"].get("chunk_id"),
            ) for s in sources
        ]
        response = QueryResponse(
            answer=answer,
            sources=formatted_sources,
            query=request.query,
            collection_name=request.collection_name,
            processing_time=(datetime.now() - start_time).total_seconds(),
            status="success",
        )
        dataset.append({
            "user_input": query,
            "retrieved_contexts": [answer],
            "response": answer,
            "reference": reference,
            "reference_contexts": [reference],
        })

    evaluation_dataset = EvaluationDataset.from_list(dataset)  # Assume import
    evaluator_llm = LangchainLLMWrapper(langchain_llm=ollama_llm)
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            NonLLMContextRecall(),
            NonLLMContextPrecisionWithReference(),
            LLMContextPrecisionWithReference(),
            ContextEntityRecall(),
            NoiseSensitivity(),
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness(),
        ],
        llm=evaluator_llm,
        run_config=my_run_config,
    )

    return BencmarkResponse(scores=result.scores)

@app.post("/search", tags=["RAG"])
async def search_documents(request: QueryRequest):
    try:
        pipeline = get_or_create_rag_pipeline(request.collection_name)
        results = pipeline.search_knn_by_query_enhanced(
            query=request.query, top_k=request.top_k, score_threshold=request.score_threshold,
            use_hybrid=request.use_hybrid, bm25_weight=request.bm25_weight, use_rerank=request.use_rerank
        )
        docs = [
            {
                "file_name": r["payload"].get("file_name"),
                "chunk_text": r["payload"].get("pages_text"),  # pages_text
                "relevance_score": r["score"],
                "chunk_id": r["payload"].get("chunk_id"),
            } for r in results
        ]
        return {"query": request.query, "documents": docs, "total_found": len(docs), "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/collections", tags=["Management"])
async def list_collections():
    return {"collections": list(rag_pipelines.keys()), "total": len(rag_pipelines)}