import uvicorn
from config.settings import API_HOST, API_PORT
from logs.logger import setup_logger

logger = setup_logger()

if __name__ == "__main__":
    logger.info(f"Запуск RAG Service API на {API_HOST}:{API_PORT}")
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )