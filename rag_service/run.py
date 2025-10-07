import uvicorn
from rag_service.config.settings import API_HOST, API_PORT
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info(f"Starting RAG Service API on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "rag_service.app.main:app",
        host=API_HOST,
         port=(API_PORT),
        reload=True,
        log_level="info",
    )