from typing import List, Dict, Any
import logging
from .interface import RagInterface
from .base_pipeline import RagPipelineOllamaQdrant

logger = logging.getLogger(__name__)

class EnhancedRagPipeline(RagInterface):
    def __init__(self, qdrant_host: str, qdrant_port: int, collection_name: str):
        self._base_pipeline = RagPipelineOllamaQdrant(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection_name=collection_name
        )
        self._collection_name = collection_name
        self._qdrant_host = qdrant_host
        self._qdrant_port = qdrant_port
        
        self._prompt = """
Ты - экспертная система для ответов на вопросы на основе предоставленной документации.

ВОПРОС: 
{query}

КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{context}

ИНСТРУКЦИИ:
1. Внимательно проанализируй контекст и найди ТОЛЬКО релевантную информацию для ответа на вопрос
2. Если в контексте есть прямой ответ - предоставь его полностью и точно
3. Если информация частичная - укажи, что именно известно, а что отсутствует
4. Если контекст не содержит ответа - честно скажи об этом
5. НЕ придумывай и не домысливай информацию
6. Структурируй ответ логично и последовательно
7. При наличии нескольких аспектов вопроса - освети каждый
8. Используй точные формулировки из документов
9. В конце укажи источники в формате: [Источник: имя_файла]

ОТВЕТ (на русском языке):
"""
    
    def vectorize_query(self, *, text: str) -> List[float]:
        return self._base_pipeline.vectorize_query(text=text)
    
    def search_knn_by_query(self, *, query: str, top_k: int = 5) -> Any:
        return self.search_knn_by_query_enhanced(query=query, top_k=top_k)
    
    def search_knn_by_query_enhanced(self, query: str, top_k: int = 5, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        try:
            query_vector = self.vectorize_query(text=query)
            
            search_result = self._base_pipeline._qdrant_client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=min(top_k * 2, 20),
                with_payload=True,
                score_threshold=score_threshold
            )
            
            results = [
                {"id": hit.id, "score": hit.score, "payload": hit.payload}
                for hit in search_result if hit.score >= score_threshold
            ]
            
            results = results[:top_k]
            logger.info(f"Найдено {len(results)} релевантных документов для запроса")
            return results
            
        except Exception as error:
            logger.error(f"Ошибка при поиске в Qdrant: {error}")
            raise Exception(f"Error in searching in Qdrant's DB: {error}")
    
    def retrieve_context(self) -> str:
        raise NotImplementedError("Use retrieve_context_enhanced instead")
    
    def retrieve_context_enhanced(self, query: str, top_k: int = 5, 
                                 score_threshold: float = 0.5, 
                                 temperature: float = 0.1) -> tuple[str, List[Dict[str, Any]]]:
        try:
            self._base_pipeline.temperature = temperature
            search_results = self.search_knn_by_query_enhanced(
                query=query, 
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            if not search_results:
                return "По вашему запросу не найдено релевантной информации в базе знаний.", []
            
            highly_relevant = [r for r in search_results if r["score"] > 0.7]
            moderately_relevant = [r for r in search_results if 0.5 <= r["score"] <= 0.7]
            context = self._format_context_by_relevance(highly_relevant, moderately_relevant)
            
            if not context:
                return "Найденная информация недостаточно релевантна для точного ответа на ваш вопрос.", []
            
            formatted_prompt = self._prompt.format(query=query, context=context)
            response = self._base_pipeline.invoke(formatted_prompt)
            
            return response.content, search_results
            
        except Exception as error:
            logger.error(f"Ошибка при извлечении контекста: {error}")
            raise Exception(f"Error in retrieving context: {error}")
    
    def _format_context_by_relevance(self, highly_relevant: List[Dict], 
                                    moderately_relevant: List[Dict]) -> str:
        context_parts = []
        
        if highly_relevant:
            context_parts.append("=== ВЫСОКОРЕЛЕВАНТНЫЕ ДОКУМЕНТЫ ===")
            for result in highly_relevant:
                filename = result["payload"].get("file_name", "Unknown")
                chunk = result["payload"].get("chunk_text", "")
                score = result["score"]
                context_parts.append(
                    f"\nИсточник: {filename} (релевантность: {score:.3f})\n"
                    f"Содержание:\n{chunk}\n"
                )
        
        if moderately_relevant:
            context_parts.append("\n=== ДОПОЛНИТЕЛЬНЫЕ ДОКУМЕНТЫ ===")
            for result in moderately_relevant:
                filename = result["payload"].get("file_name", "Unknown")
                chunk = result["payload"].get("chunk_text", "")
                score = result["score"]
                context_parts.append(
                    f"\nИсточник: {filename} (релевантность: {score:.3f})\n"
                    f"Содержание:\n{chunk}\n"
                )
        
        return "\n".join(context_parts) if context_parts else ""