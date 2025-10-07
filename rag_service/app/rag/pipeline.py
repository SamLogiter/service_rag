from typing import List, Dict, Any
import logging
from .base_pipeline import RagPipelineVLLM

logger = logging.getLogger(__name__)

class EnhancedRagPipeline:
    def __init__(self, qdrant_host: str, qdrant_port: int, collection_name: str, **vllm_kwargs):
        self._base_pipeline = RagPipelineVLLM(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection_name=collection_name,
            **vllm_kwargs,
        )
        self._collection_name = collection_name

        self._prompt = """
Ты - экспертная система для ответов на вопросы на основе предоставленной документации.

ВОПРОС: {query}

КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{context}

ИНСТРУКЦИИ:
1. Внимательно проанализируй контекст и найди ТОЧНО релевантную информацию для ответа на вопрос
2. Если в контексте есть прямой ответ - предоставь его полностью и точно
3. Если информация частичная - укажи, что именно известно, а что отсутствует
4. Если контекст не содержит ответа - честно скажи об этом
5. НЕ придумывай и не домысливай информацию
6. Структурируй ответ логично и последовательно
7. При наличии нескольких аспектов вопроса - освети каждый
8. Используй точные формулировки из документов

ОТВЕТ (на русском языке):
"""

    def vectorize_query(self, *, text: str) -> List[float]:
        return self._base_pipeline.vectorize_query(text=text)

    def search_knn_by_query_enhanced(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.4,
        use_hybrid: bool = True,
        bm25_weight: float = 1.2,
        use_rerank: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Enhanced search with filtering on pages_text"""
        if use_hybrid:
            results = self._base_pipeline.search_hybrid(
                query=query, top_k=top_k * 2, bm25_weight=bm25_weight, use_rerank=use_rerank
            )
            search_type = "hybrid"
        else:
            semantic = self._base_pipeline.search_knn_by_query(query=query, top_k=top_k * 2, score_threshold=score_threshold)
            results = [{"id": h.id, "score": h.score, "payload": h.payload, "search_type": "semantic"} for h in semantic]
            search_type = "semantic"

        # Dedup by file_name + chunk_id
        seen = set()
        filtered = []
        for r in results:
            key = (r["payload"].get("file_name", ""), r["payload"].get("chunk_id", ""))
            if key not in seen:
                seen.add(key)
                filtered.append(r)

        # Filter by pages_text: non-empty, min length >50 chars
        filtered = [
            r for r in filtered
            if len(r["payload"].get("pages_text", "")) > 50
        ]

        # Example metadata filter: only PDF files
        filtered = [
            r for r in filtered
            if r["payload"].get("file_name", "").endswith('.pdf')
        ]

        filtered.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Filtered to {len(filtered)} results after pages_text filter (type: {search_type})")
        return filtered[:top_k]

    def retrieve_context_enhanced(self, query: str, top_k: int = 5, temperature: float = 0.1, **kwargs):
        results = self.search_knn_by_query_enhanced(query, top_k, **kwargs)
        # Context from pages_text, limit 3, truncate
        contexts = [r["payload"].get("pages_text", "") for r in results]
        context = "\n\n".join(contexts[:3])
        if len(context) > 8000:
            context = context[:8000] + "..."

        prompt = self._prompt.format(query=query, context=context)
        answer = self._base_pipeline.invoke(prompt).content
        return answer, results