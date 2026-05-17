"""
SnipAI v2.0 — core.py
"""
import os
import logging
from dotenv import load_dotenv

import openai
from llama_index.core import Settings

load_dotenv()
logger = logging.getLogger(__name__)

LLM_MODEL   = os.getenv("LLM_MODEL",      "qwen/qwen3-8b")
COLLECTION  = os.getenv("COLLECTION_NAME", "snips_rk")

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

openai_client: openai.OpenAI = None
_settings_initialized = False


def get_qdrant_client():
    from qdrant_client import QdrantClient
    qdrant_url = os.getenv("QDRANT_URL")
    api_key    = os.getenv("QDRANT_API_KEY")
    try:
        if qdrant_url:
            client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=60)
        else:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
        client.get_collections()
        return client
    except Exception as e:
        target = qdrant_url or f"{QDRANT_HOST}:{QDRANT_PORT}"
        raise ConnectionError(f"Qdrant недоступен: {target}\nОшибка: {e}")


def init_settings() -> bool:
    global openai_client, _settings_initialized

    if _settings_initialized:
        return True

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY не задан")
        return False

    try:
        openai_client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://snipai.kz",
                "X-Title":      "SnipAI v2.0",
            },
        )

        # ✅ OpenAIEmbedding не принимает кастомные имена моделей.
        # Используем прямой openai.Embedding через OpenRouter
        from llama_index.core.embeddings import BaseEmbedding
        from typing import List

        class OpenRouterEmbedding(BaseEmbedding):
            def _get_query_embedding(self, query: str) -> List[float]:
                return self._embed(query)

            def _get_text_embedding(self, text: str) -> List[float]:
                return self._embed(text)

            def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                resp = openai_client.embeddings.create(
                    model="openai/text-embedding-3-small",
                    input=texts,
                )
                return [d.embedding for d in resp.data]

            def _embed(self, text: str) -> List[float]:
                resp = openai_client.embeddings.create(
                    model="openai/text-embedding-3-small",
                    input=[text],
                )
                return resp.data[0].embedding

            async def _aget_query_embedding(self, query: str) -> List[float]:
                return self._get_query_embedding(query)

            async def _aget_text_embedding(self, text: str) -> List[float]:
                return self._get_text_embedding(text)

        Settings.embed_model   = OpenRouterEmbedding(model_name="openai/text-embedding-3-small")
        Settings.chunk_size    = 512
        Settings.chunk_overlap = 100

        qdrant_url = os.getenv("QDRANT_URL")
        mode = f"☁️  {qdrant_url}" if qdrant_url else f"🐳 {QDRANT_HOST}:{QDRANT_PORT}"
        print(f"✅ LLM      : {LLM_MODEL}")
        print(f"✅ Embeddings: openai/text-embedding-3-small (OpenRouter)")
        print(f"✅ Qdrant   : {mode} / {COLLECTION}")

        _settings_initialized = True
        return True

    except Exception as e:
        logger.exception("init_settings failed")
        print(f"❌ Ошибка инициализации: {e}")
        return False


def call_llm(prompt: str, retries: int = 3, delay: float = 2.0) -> str:
    import time
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Вы — SnipAI, экспертная система по СНиП РК."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            if not response or not response.choices:
                raise ValueError("Пустой ответ от OpenRouter")
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("message.content=None")
            return content.strip()
        except Exception as e:
            last_error = e
            logger.warning(f"[call_llm] попытка {attempt}/{retries}: {e}")
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError(f"[call_llm] все {retries} попытки провалились: {last_error}")


if __name__ == "__main__":
    if init_settings():
        client = get_qdrant_client()
        cols = [c.name for c in client.get_collections().collections]
        print(f"\n📦 Коллекции: {cols or ['(пусто)']}")
        print("\n🚀 Ядро работает!")
        