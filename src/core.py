"""
SnipAI v2.0 — core.py
"""
import os
import logging
from typing import List
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import openai

load_dotenv()
logger = logging.getLogger(__name__)

LLM_MODEL   = os.getenv("LLM_MODEL",       "qwen/qwen-2.5-72b-instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL",      "intfloat/multilingual-e5-small")
QDRANT_HOST = os.getenv("QDRANT_HOST",      "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT",  "6333"))
COLLECTION  = os.getenv("COLLECTION_NAME",  "snips_rk")

openai_client: openai.OpenAI = None


class E5Embedding(BaseEmbedding):
    _model: SentenceTransformer = None

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        object.__setattr__(self, '_model', SentenceTransformer(model_name))

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(f"query: {str(query)}").tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(f"passage: {str(text)}").tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._model.encode([f"passage: {str(t)}" for t in texts]).tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


def get_qdrant_client() -> QdrantClient:
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
        client.get_collections()
        return client
    except Exception as e:
        raise ConnectionError(
            f"Qdrant недоступен на {QDRANT_HOST}:{QDRANT_PORT}.\n"
            f"Запустите: docker compose up -d\nОшибка: {e}"
        )


def call_llm(prompt: str, retries: int = 3, delay: float = 2.0) -> str:
    """Прямой вызов OpenRouter через openai SDK с автоматическим retry."""
    import time

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Вы — SnipAI, экспертная система по СНиП РК."},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048,
            )

            # Защита от None
            if response is None or not response.choices:
                raise ValueError("OpenRouter вернул пустой ответ (choices=None)")

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("OpenRouter вернул message.content=None")

            return content.strip()

        except (ValueError, Exception) as e:
            last_error = e
            logger.warning(f"[call_llm] попытка {attempt}/{retries} — ошибка: {e}")
            if attempt < retries:
                time.sleep(delay)

    raise RuntimeError(f"[call_llm] все {retries} попытки завершились ошибкой: {last_error}")


def init_settings() -> bool:
    global openai_client

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY не установлен в .env файле")
        return False

    try:
        openai_client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://snipai.kz",
                "X-Title": "SnipAI v2.0",
            }
        )

        print(f"⏳ Загрузка модели эмбеддингов: {EMBED_MODEL}")
        Settings.embed_model = E5Embedding(model_name=EMBED_MODEL)
        Settings.chunk_size    = 512
        Settings.chunk_overlap = 100

        print(f"✅ LLM (OpenRouter): {LLM_MODEL}")
        print(f"✅ Embeddings (E5) : {EMBED_MODEL}")
        print(f"✅ Qdrant          : {QDRANT_HOST}:{QDRANT_PORT} / {COLLECTION}")
        return True

    except Exception as e:
        logger.exception("init_settings failed")
        print(f"❌ Ошибка инициализации: {e}")
        return False


if __name__ == "__main__":
    if init_settings():
        client = get_qdrant_client()
        cols = [c.name for c in client.get_collections().collections]
        print(f"\n📦 Коллекции в Qdrant: {cols or ['(пусто)']}")
        print("\n🚀 Ядро работает корректно!")
        