from typing import List, Any, Dict
import requests
from qdrant_client import QdrantClient
import numpy as np
from sklearn.preprocessing import normalize
import logging
import re
from collections import Counter
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import pymorphy2  # Для русского

logger = logging.getLogger(__name__)

# NLTK setup
nltk_data_path = "/app/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Russian stopwords (extended list from your original)
RUSSIAN_STOPWORDS = {
            "и",
            "в",
            "во",
            "не",
            "что",
            "он",
            "на",
            "я",
            "с",
            "со",
            "как",
            "а",
            "то",
            "все",
            "она",
            "так",
            "его",
            "но",
            "да",
            "ты",
            "к",
            "у",
            "же",
            "вы",
            "за",
            "бы",
            "по",
            "только",
            "ее",
            "мне",
            "было",
            "вот",
            "от",
            "меня",
            "еще",
            "нет",
            "о",
            "из",
            "ему",
            "теперь",
            "когда",
            "даже",
            "ну",
            "вдруг",
            "ли",
            "если",
            "уже",
            "или",
            "ни",
            "быть",
            "был",
            "него",
            "до",
            "вас",
            "нибудь",
            "опять",
            "уж",
            "вам",
            "ведь",
            "там",
            "потом",
            "себя",
            "ничего",
            "ей",
            "может",
            "они",
            "тут",
            "где",
            "есть",
            "надо",
            "ней",
            "для",
            "мы",
            "тебя",
            "их",
            "чем",
            "была",
            "сам",
            "чтоб",
            "без",
            "будто",
            "чего",
            "раз",
            "тоже",
            "себе",
            "под",
            "будет",
            "ж",
            "тогда",
            "кто",
            "этот",
            "того",
            "потому",
            "этого",
            "какой",
            "совсем",
            "ним",
            "здесь",
            "этом",
            "один",
            "почти",
            "мой",
            "тем",
            "чтобы",
            "нее",
            "сейчас",
            "были",
            "куда",
            "зачем",
            "всех",
            "никогда",
            "можно",
            "при",
            "наконец",
            "два",
            "об",
            "другой",
            "хоть",
            "после",
            "над",
            "больше",
            "тот",
            "через",
            "эти",
            "нас",
            "про",
            "всего",
            "них",
            "какая",
            "много",
            "разве",
            "три",
            "эту",
            "моя",
            "впрочем",
            "хорошо",
            "свою",
            "этой",
            "перед",
            "иногда",
            "лучше",
            "чуть",
            "том",
            "нельзя",
            "такой",
            "им",
            "более",
            "всегда",
            "конечно",
            "всю",
            "между",
            "это",
            "как",
            "так",
            "и",
            "в",
            "над",
            "к",
            "до",
            "не",
            "на",
            "но",
            "за",
            "то",
            "с",
            "ли",
            "а",
            "во",
            "от",
            "со",
            "для",
            "о",
            "же",
            "ну",
            "вы",
            "бы",
            "что",
            "кто",
            "он",
            "она",
            "что",
            "где",
            "когда",
            "какой",
            "какая",
            "какое",
            "какие",
            "почему",
            "зачем",
            "сколько",
            "как",
            "на",
            "по",
            "из",
            "от",
            "до",
            "у",
            "без",
            "под",
            "над",
            "при",
            "после",
            "в",
            "течение",
            "или",
            "либо",
            "ни",
            "нет",
            "да",
            "но",
            "однако",
            "хотя",
            "пусть",
            "если",
            "то",
            "так",
            "как",
            "будто",
            "точно",
            "словно",
            "чем",
            "нежели",
            "чтобы",
            "кабы",
            "дабы",
            "пока",
            "едва",
            "лишь",
            "только",
            "ли",
            "же",
            "ведь",
            "вот",
            "мол",
            "дескать",
            "типа",
            "например",
            "так",
            "итак",
            "следовательно",
            "поэтому",
            "затем",
            "потом",
            "вдобавок",
            "кроме",
            "сверх",
            "вместо",
            "около",
            "возле",
            "вокруг",
            "перед",
            "за",
            "из-за",
            "из-под",
            "через",
            "сквозь",
            "внутри",
            "снаружи",
            "среди",
            "между",
            "наверху",
            "внизу",
            "спереди",
            "сзади",
            "сбоку",
            "вперед",
            "назад",
            "вверх",
            "вниз",
            "вправо",
            "влево",
            "далеко",
            "близко",
            "высоко",
            "низко",
            "глубоко",
            "мелко",
            "рано",
            "поздно",
            "долго",
            "скоро",
            "сразу",
            "сейчас",
            "теперь",
            "тогда",
            "иногда",
            "часто",
            "редко",
            "всегда",
            "никогда",
            "уже",
            "еще",
            "тоже",
            "также",
            "причем",
            "притом",
            "впрочем",
            "однако",
            "зато",
            "только",
            "лишь",
            "исключительно",
            "особенно",
            "даже",
            "уж",
            "вовсе",
            "отнюдь",
            "совсем",
            "абсолютно",
            "полностью",
            "целиком",
            "почти",
            "примерно",
            "приблизительно",
            "ровно",
            "точно",
            "прямо",
            "как",
            "так",
            "столько",
            "сколько",
            "настолько",
            "до",
            "после",
            "перед",
            "за",
            "из-за",
            "из-под",
            "через",
            "сквозь",
            "внутри",
            "снаружи",
            "среди",
            "между",
            "наверху",
            "внизу",
            "спереди",
            "сзади",
            "сбоку",
            "вперед",
            "назад",
            "вверх",
            "вниз",
            "вправо",
            "влево",
            "далеко",
            "близко",
            "высоко",
            "низко",
            "глубоко",
            "мелко",
            "рано",
            "поздно",
            "долго",
            "скоро",
            "сразу",
            "сейчас",
            "теперь",
            "тогда",
            "иногда",
            "часто",
            "редко",
            "всегда",
            "никогда",
            "уже",
            "еще",
            "тоже",
            "также",
            "причем",
            "притом",
            "впрочем",
            "однако",
            "зато",
            "только",
            "лишь",
            "исключительно",
            "особенно",
            "даже",
            "уж",
            "вовсе",
            "отнюдь",
            "совсем",
            "абсолютно",
            "полностью",
            "целиком",
            "почти",
            "примерно",
            "приблизительно",
            "ровно",
            "точно",
            "прямо",
            "как",
            "так",
            "столько",
            "сколько",
            "настолько",
            "этом",
            "всем",
            "своих",
            "ними",
            "вами",
            "нами",
            "ими",
            "вами",
            "нами",
            "ими",
            "вами",
            "нами",
        }


class RagPipelineVLLM:
    def __init__(
        self,
        qdrant_host: str,
        qdrant_port: int,
        collection_name: str,
        qdrant_api_key: str = None,
        vllm_gen_host: str = None,
        vllm_gen_port: int = None,
        vllm_gen_model: str = None,
        vllm_embed_host: str = None,
        vllm_embed_port: int = None,
        vllm_embed_model: str = None,
        vllm_rerank_host: str = None,
    ):
        self._qdrant_client = QdrantClient(
            url=f"{qdrant_host}:{qdrant_port}", api_key=qdrant_api_key, timeout=6000
        )
        self._collection_name = collection_name

        # vLLM params
        self._vllm_gen_host = vllm_gen_host or "localhost"
        self._vllm_gen_port = vllm_gen_port or 8001
        self._vllm_gen_model = vllm_gen_model or "Qwen/Qwen2.5-7B-Instruct"
        self._vllm_embed_host = vllm_embed_host or "localhost"
        self._vllm_embed_port = vllm_embed_port or 8002
        self._vllm_embed_model = vllm_embed_model or "BAAI/bge-m3"
        self._vllm_rerank_host = vllm_rerank_host or "BAAI/bge-reranker-base"

        # BM25 cache
        self._bm25_cache = {}
        self._documents_cache = {}
        self._id_cache = {}  
        self._payload_cache = {} 

        # Russian analyzer
        self.morph = pymorphy2.MorphAnalyzer()
        self.russian_stopwords = RUSSIAN_STOPWORDS

        self._check_connections()

    def _russian_tokenize(self, text: str) -> List[str]:
        """Improved tokenization: lemmatization + stopwords"""
        tokens = word_tokenize(text.lower())
        lemmas = [self.morph.parse(token)[0].normal_form for token in tokens if token.isalpha()]
        return [lemma for lemma in lemmas if lemma not in self.russian_stopwords]

    def _get_documents(self, collection_name: str) -> List[str]:
        """Fetch pages_text for BM25 cache + ID/payload mapping"""
        if collection_name not in self._documents_cache:
            try:
                all_points = self._qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=10000,  # Adjust for your data size
                    with_payload=True,
                )[0]
                docs = [point.payload.get("pages_text", "") for point in all_points]
                self._documents_cache[collection_name] = docs
                
                # New: Cache for real IDs and payloads
                self._id_cache[collection_name] = [point.id for point in all_points]
                self._payload_cache[collection_name] = [point.payload for point in all_points]
                
                tokenized_docs = [self._russian_tokenize(doc) for doc in docs]
                self._bm25_cache[collection_name] = BM25Okapi(tokenized_docs)
                logger.info(f"BM25 indexed {len(docs)} pages_text for {collection_name} with ID mapping")
            except Exception as e:
                logger.error(f"Error indexing BM25: {e}")
                return []
        return self._documents_cache.get(collection_name, [])

    def search_bm25(self, query: str, collection_name: str, top_k: int = 5, bm25_weight: float = 1.2) -> List[Dict]:
        """BM25 on pages_text with real ID/payload mapping"""
        docs = self._get_documents(collection_name)
        if not docs:
            return []

        tokenized_query = self._russian_tokenize(query)
        bm25 = self._bm25_cache.get(collection_name)
        if bm25 is None:
            return []

        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k * 2]

        # New: Real caches
        id_cache = self._id_cache.get(collection_name, [])
        payload_cache = self._payload_cache.get(collection_name, [])

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "id": id_cache[idx],  # Real UUID
                    "score": float(scores[idx]) * bm25_weight,
                    "payload": payload_cache[idx],  # Full real payload (pages_text, file_name, chunk_id)
                    "search_type": "bm25"
                })
        return results[:top_k]

    def search_hybrid(self, query: str, top_k: int = 5, score_threshold: float = 0.4, bm25_weight: float = 1.2, use_rerank: bool = True):
        """Hybrid with RRF"""
        semantic_results = self.search_knn_by_query(query=query, top_k=top_k * 2, score_threshold=score_threshold)

        bm25_results = self.search_bm25(query=query, collection_name=self._collection_name, top_k=top_k * 2, bm25_weight=bm25_weight)

        # RRF
        all_docs = {}
        k = 60
        for i, hit in enumerate(semantic_results):
            doc_id = hit.id
            all_docs[doc_id] = all_docs.get(doc_id, 0) + 1 / (k + i + 1)
        for i, hit in enumerate(bm25_results):
            doc_id = hit["id"]
            all_docs[doc_id] = all_docs.get(doc_id, 0) + bm25_weight / (k + i + 1)

        rrf_sorted = sorted(all_docs.items(), key=lambda x: x[1], reverse=True)
        hybrid_hits = []
        for doc_id, _ in rrf_sorted[:top_k * 2]:
            hit = next((h for h in semantic_results if h.id == doc_id), next((h for h in bm25_results if h["id"] == doc_id), None))
            if hit:
                hybrid_hits.append(hit)

        if use_rerank:
            hybrid_hits = self._rerank(hybrid_hits, query)

        return [h for h in hybrid_hits if getattr(h, 'score', 0) >= score_threshold][:top_k]

    def _rerank(self, hits: List[Any], query: str) -> List[Any]:
        """Rerank with vLLM bge-reranker-base"""
        try:
            url = f"http://{self._vllm_rerank_host}:{self._vllm_rerank_port}/v1/chat/completions"
            texts = [hit.payload.get('pages_text', '') for hit in hits]
            scores = []
            for doc in texts:
                payload = {
                    "model": self._vllm_rerank_model,
                    "messages": [
                        {"role": "user", "content": f"Query: {query}\nDocument: {doc}"},  # Pair format for reranker
                    ],
                    "temperature": 0.0,  # Deterministic
                    "max_tokens": 1,  # Just score
                    "stream": False,
                }
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    score = float(resp.json()["choices"][0]["message"]["content"].strip())  # Assume model outputs score
                    scores.append(score)
                else:
                    scores.append(0.0)
            
            for hit, score in zip(hits, scores):
                hit.score = score
            return sorted(hits, key=lambda h: h.score, reverse=True)[:5]
        except Exception as e:
            logger.error(f"Rerank error: {e}")
            return hits

    def _check_connections(self):
        """Check vLLM and Qdrant"""
        # vLLM Gen
        try:
            resp = requests.get(f"http://{self._vllm_gen_host}:{self._vllm_gen_port}/v1/models", timeout=10)
            if resp.status_code == 200 and self._vllm_gen_model in [m['id'] for m in resp.json()['data']]:
                logger.info(f"vLLM Gen '{self._vllm_gen_model}' OK")
            else:
                logger.error(f"vLLM Gen model not found")
        except Exception as e:
            logger.error(f"vLLM Gen check failed: {e}")

        # vLLM Embed
        try:
            resp = requests.get(f"http://{self._vllm_embed_host}:{self._vllm_embed_port}/v1/models", timeout=10)
            if resp.status_code == 200 and self._vllm_embed_model in [m['id'] for m in resp.json()['data']]:
                logger.info(f"vLLM Embed '{self._vllm_embed_model}' OK")
            else:
                logger.error(f"vLLM Embed model not found")
        except Exception as e:
            logger.error(f"vLLM Embed check failed: {e}")

        # Qdrant: Assume OK if client init

    def _format_text_for_embedding(self, text: str, is_query: bool = True) -> str:
        if "bge-m3" in self._vllm_embed_model.lower():
            prefix = "Represent the query for retrieval of relevant documents: " if is_query else "Represent the document for retrieval of relevant queries: "
            return prefix + text
        return text

    def vectorize_query(self, *, text: str) -> List[float]:
        formatted = self._format_text_for_embedding(text, is_query=True)
        payload = {"model": self._vllm_embed_model, "input": formatted}

        try:
            url = f"http://{self._vllm_embed_host}:{self._vllm_embed_port}/v1/embeddings"
            resp = requests.post(url, json=payload, timeout=300)
            if resp.status_code == 200:
                embedding = resp.json()["data"][0]["embedding"]
                logger.info(f"Embedding dim: {len(embedding)}")

                if "bge-m3" in self._vllm_embed_model.lower():
                    emb_array = np.array(embedding).reshape(1, -1)
                    normalized = normalize(emb_array, norm="l2")[0]
                    embedding = normalized.tolist()

                return embedding
            else:
                logger.error(f"vLLM Embed error: {resp.status_code}")
                return []
        except Exception as e:
            logger.error(f"vLLM Embed error: {e}")
            return []

    def search_knn_by_query(self, *, query: str, top_k: int = 5, score_threshold: float = 0.3) -> List[Any]:
        embedding = self.vectorize_query(text=query)
        if not embedding:
            return []

        try:
            results = self._qdrant_client.search(
                collection_name=self._collection_name,
                query_vector=embedding,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
            logger.info(f"Found {len(results)} semantic docs")
            return results
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return []

    def invoke(self, prompt: str) -> Any:
        url = f"http://{self._vllm_gen_host}:{self._vllm_gen_port}/v1/chat/completions"
        payload = {
            "model": self._vllm_gen_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 1024,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=600)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                class Response:
                    def __init__(self, content):
                        self.content = content
                return Response(content)
            else:
                logger.error(f"vLLM Gen error: {resp.status_code}")
                return Response("Generation error")
        except Exception as e:
            logger.error(f"vLLM Gen error: {e}")
            return Response(f"vLLM error: {e}")

    @property
    def temperature(self):
        return 0.1

    @temperature.setter
    def temperature(self, value: float):
        pass