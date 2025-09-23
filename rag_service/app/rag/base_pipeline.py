from typing import List, Any
import requests
from qdrant_client import QdrantClient
from .interface import RagInterface
import logging

logger = logging.getLogger(__name__)

class RagPipelineOllamaQdrant(RagInterface):
    def __init__(
        self,
        qdrant_host: str,
        qdrant_port: int,
        collection_name: str,
        qdrant_api_key: str = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama2:7b",
        embed_model: str = "mxbai-embed-large:latest",
    ):
        self._qdrant_client = QdrantClient(
            url=f"{qdrant_host}:{qdrant_port}",
            api_key=qdrant_api_key,
            timeout=60
        )
        self._collection_name = collection_name
        self._ollama_url = ollama_url
        self._ollama_model = ollama_model
        self._embed_model = embed_model
        self._ollama_generator = self  # Mock generator for compatibility
        self._ollama_embedder = self   # Mock embedder for compatibility
        self._check_connections()

    def _check_connections(self):
        """Проверка соединений с Qdrant и Ollama"""
        try:
            collections = self._qdrant_client.get_collections()
            logger.info(f"✅ Qdrant подключен: {collections}")
            collection_names = [col.name for col in collections.collections]
            if self._collection_name in collection_names:
                logger.info(f"✅ Коллекция '{self._collection_name}' найдена")
            else:
                logger.warning(f"❌ Коллекция '{self._collection_name}' не найдена. Доступные: {collection_names}")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Qdrant: {e}")

        try:
            response = requests.get(f"{self._ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                logger.info(f"✅ Ollama подключен. Модели: {[m['name'] for m in models]}")
            else:
                logger.error(f"❌ Ошибка подключения к Ollama: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Не удалось подключиться к Ollama: {e}")

    def vectorize_query(self, *, text: str) -> List[float]:
        """Получение эмбеддинга из Ollama"""
        payload = {"model": self._embed_model, "prompt": text}
        try:
            response = requests.post(
                f"{self._ollama_url}/api/embeddings", json=payload, timeout=30
            )
            if response.status_code == 200:
                return response.json().get("embedding", [])
            logger.error(f"Ошибка получения эмбеддинга: {response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Ошибка подключения к Ollama для эмбеддингов: {e}")
            return []

    def search_knn_by_query(self, *, query: str, top_k: int = 5) -> Any:
        """Поиск похожих документов в Qdrant"""
        query_embedding = self.vectorize_query(text=query)
        if not query_embedding:
            logger.error("❌ Не удалось получить эмбеддинг для запроса")
            return []

        try:
            search_result = self._qdrant_client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=0.5,
                with_payload=True
            )
            return search_result
        except Exception as e:
            logger.error(f"❌ Ошибка поиска в Qdrant: {e}")
            return []

    def retrieve_context(self) -> str:
        """Заглушка для retrieve_context (используется retrieve_context_enhanced)"""
        raise NotImplementedError("Use retrieve_context_enhanced in EnhancedRagPipeline")

    def invoke(self, prompt: str) -> Any:
        """Эмуляция вызова генератора (для совместимости с EnhancedRagPipeline)"""
        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9, "num_ctx": 4096},
        }
        try:
            response = requests.post(
                f"{self._ollama_url}/api/generate", json=payload, timeout=120
            )
            if response.status_code == 200:
                class Response:
                    def __init__(self, content):
                        self.content = content
                return Response(response.json().get("response", "Ошибка генерации ответа"))
            logger.error(f"Ошибка Ollama: {response.status_code}")
            return Response(f"Ошибка Ollama: {response.status_code}")
        except Exception as e:
            logger.error(f"Ошибка подключения к Ollama: {e}")
            return Response(f"Ошибка подключения к Ollama: {e}")

    @property
    def temperature(self):
        """Mock temperature property for compatibility"""
        return 0.1

    @temperature.setter
    def temperature(self, value: float):
        """Mock temperature setter"""
        pass