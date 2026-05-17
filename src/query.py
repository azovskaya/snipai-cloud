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

from core import get_qdrant_client, COLLECTION, call_llm, init_settings

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


def build_retriever() -> Optional[VectorIndexRetriever]:
    """Ленивая инициализация — вызывается при первом запросе."""
    if not init_settings():
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
        logger.warning("Все %d нод вернули пустой текст.", len(nodes))
        return NO_INFO_MSG

    context = "\n\n---\n\n".join(chunks)
    prompt  = SYSTEM_PROMPT.format(context=context, question=question)
    return call_llm(prompt)


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
            print("\n👋 Завершение.")
            break
        if not question:
            continue
        if question.lower() in ("выход", "exit", "quit", "q"):
            break
        print("\n⏳ Поиск...\n")
        try:
            print("─" * 65)
            print(ask(retriever, question))
            print("─" * 65 + "\n")
        except Exception as e:
            print(f"❌ Ошибка: {e}\n")

if __name__ == "__main__":
    main()
    