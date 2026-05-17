"""
SnipAI v2.0 — query.py
"""
import os
import json
import logging
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from core import get_qdrant_client, COLLECTION, call_llm

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.qdrant import QdrantVectorStore

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Вы — SnipAI, экспертная система по нормативно-технической документации Республики Казахстан (СНиП РК, СП РК, РДС, ГОСТ).

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
1. Отвечайте ИСКЛЮЧИТЕЛЬНО на основе предоставленного контекста.
2. Точно воспроизводите цифры, допуски и нормы из документов.
3. Указывайте название документа и номер пункта/таблицы.
4. Если данных нет в базе — прямо скажите: "По данному вопросу информация в базе СНиП РК не найдена."
5. Структурируйте ответ: используйте нумерованные списки для нескольких требований.

КОНТЕКСТ ИЗ БАЗЫ СНиП РК:
─────────────────────────────────────
{context}
─────────────────────────────────────

ВОПРОС СПЕЦИАЛИСТА: {question}

ОТВЕТ ЭКСПЕРТА SnipAI:"""

NO_INFO_MSG = "По данному вопросу информация в базе СНиП РК не найдена."


def _extract_text(node: NodeWithScore) -> str:
    c = node.node.get_content()
    if c and c.strip():
        return c.strip()

    t = getattr(node.node, "text", None)
    if t and str(t).strip():
        return str(t).strip()

    meta = getattr(node.node, "metadata", {}) or {}

    raw = meta.get("_node_content")
    if raw:
        try:
            parsed = json.loads(raw)
            t2 = parsed.get("text") or parsed.get("content") or ""
            if str(t2).strip():
                return str(t2).strip()
        except Exception:
            if str(raw).strip():
                return str(raw).strip()

    for key in ("text", "content", "page_content", "chunk_text"):
        val = meta.get(key)
        if val and str(val).strip():
            return str(val).strip()

    return ""


def _get_source(node: NodeWithScore) -> str:
    meta = getattr(node.node, "metadata", {}) or {}
    return meta.get("source_file") or meta.get("file_name") or ""


def _init_embed_model():
    """
    Ленивая инициализация модели эмбеддингов.
    Вызывается только при первом реальном запросе — НЕ при импорте модуля.
    Это предотвращает OOM на Render Free (512MB лимит).
    """
    from llama_index.core import Settings
    from core import E5Embedding, EMBED_MODEL

    if Settings.embed_model is None or isinstance(Settings.embed_model, type):
        logger.info("Загрузка модели эмбеддингов: %s", EMBED_MODEL)
        Settings.embed_model  = E5Embedding(model_name=EMBED_MODEL)
        Settings.chunk_size   = 512
        Settings.chunk_overlap = 100
        logger.info("Модель эмбеддингов загружена.")


def _init_llm_client():
    """
    Ленивая инициализация OpenRouter клиента.
    Вызывается только при первом реальном запросе.
    """
    import core
    if core.openai_client is None:
        import openai
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY не задан в переменных окружения")
        core.openai_client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://snipai.kz",
                "X-Title":      "SnipAI v2.0",
            },
        )
        logger.info("OpenRouter клиент инициализирован.")


def build_retriever() -> Optional[VectorIndexRetriever]:
    """
    Строит ретривер. Модель эмбеддингов инициализируется здесь лениво —
    только при первом вызове, не при старте приложения.
    """
    # ✅ Ленивая загрузка — грузим модели только сейчас, не при импорте
    try:
        _init_embed_model()
        _init_llm_client()
    except Exception as e:
        logger.error("Ошибка инициализации: %s", e)
        return None

    try:
        client = get_qdrant_client()
    except ConnectionError as e:
        logger.error("Qdrant недоступен: %s", e)
        return None

    try:
        info  = client.get_collection(COLLECTION)
        count = info.points_count or 0
        if count == 0:
            logger.warning("Коллекция «%s» пуста.", COLLECTION)
            return None
        logger.info("Документов в базе: %d векторов", count)
    except Exception:
        logger.warning("Коллекция «%s» не найдена.", COLLECTION)
        return None

    vector_store    = QdrantVectorStore(client=client, collection_name=COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index           = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )
    return VectorIndexRetriever(index=index, similarity_top_k=6)


def ask(retriever: VectorIndexRetriever, question: str) -> str:
    nodes: List[NodeWithScore] = retriever.retrieve(question)

    chunks = []
    for n in nodes:
        text = _extract_text(n)
        if text:
            src = _get_source(n)
            chunks.append(f"[{src}]\n{text}" if src else text)

    if not chunks:
        logger.warning(
            "Все %d нод вернули пустой текст. Payload первой ноды: %s",
            len(nodes),
            json.dumps(
                getattr(nodes[0].node, "metadata", {}) if nodes else {},
                ensure_ascii=False
            )[:400],
        )
        return NO_INFO_MSG

    context = "\n\n---\n\n".join(chunks)
    prompt  = SYSTEM_PROMPT.format(context=context, question=question)
    return call_llm(prompt)


# ── CLI режим (python query.py) ───────────────────
def main() -> None:
    import sys

    print("=" * 65)
    print(f"  🏗️  SnipAI v2.0  |  {datetime.now():%d.%m.%Y}")
    print("=" * 65)

    retriever = build_retriever()
    if retriever is None:
        print("❌ Не удалось инициализировать ретривер.")
        sys.exit(1)

    print("  Введите вопрос по СНиП РК. Для выхода: \'выход\'")
    print("=" * 65 + "\n")

    while True:
        try:
            question = input("❓ Вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Завершение работы SnipAI.")
            break

        if not question:
            continue
        if question.lower() in ("выход", "exit", "quit", "q"):
            print("\n👋 До свидания!")
            break

        print("\n⏳ Поиск в базе СНиП РК...\n")
        try:
            answer = ask(retriever, question)
            print("─" * 65)
            print("📋 ОТВЕТ SnipAI:\n")
            print(answer)
            print("─" * 65 + "\n")
        except Exception as e:
            logger.exception("Ошибка запроса.")
            print(f"❌ Ошибка: {e}\n")


if __name__ == "__main__":
    main()
    