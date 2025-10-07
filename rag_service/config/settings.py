from dotenv import load_dotenv
import os

load_dotenv()

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# vLLM Generation (Qwen)
VLLM_GEN_HOST = os.getenv("VLLM_GEN_HOST", "localhost")
VLLM_GEN_PORT = int(os.getenv("VLLM_GEN_PORT", "8001"))
VLLM_GEN_MODEL = os.getenv("VLLM_GEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# vLLM Embeddings (BGE-M3)
VLLM_EMBED_HOST = os.getenv("VLLM_EMBED_HOST", "localhost")
VLLM_EMBED_PORT = int(os.getenv("VLLM_EMBED_PORT", "8002"))
VLLM_EMBED_MODEL = os.getenv("VLLM_EMBED_MODEL", "BAAI/bge-m3")

# vLLM Reranker (BGE-RERANKER-BASE)
VLLM_RERANK_HOST = os.getenv("VLLM_RERANK_HOST", "localhost")
VLLM_RERANK_PORT=int(os.getenv("VLLM_EMBED_PORT", "8003"))
VLLM_RERANK_MODEL =  os.getenv("VLLM_RERANK_MODEL", "BAAI/bge-reranker-base")