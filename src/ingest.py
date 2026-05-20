"""
SnipAI v2.0 — ingest.py
Умная индексация: PDF, DOCX, DOC → Qdrant Cloud
Без torch/sentence-transformers — эмбеддинги через OpenRouter API
 
v2.1: source of truth перенесён в Qdrant payload (убран .indexed_files.json)
      Поддерживает обновление файлов (тот же name, другой hash)
"""
from __future__ import annotations
 
import os
import sys
import time
import hashlib
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional
 
import pdfplumber
from docx import Document as DocxDocument
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
 
from core import get_qdrant_client, COLLECTION, init_settings
 
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document as LlamaDocument
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue
 
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
 
DATA_DIR   = Path(os.getenv("DATA_DIR", "/app/data"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
SUPPORTED_EXT = {".pdf", ".docx", ".doc"}
 
 
# ── Хэш файла ────────────────────────────────────────────────────────────────
 
def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:20]
 
 
# ── Парсеры ───────────────────────────────────────────────────────────────────
 
def parse_pdf(path: Path) -> Optional[str]:
    try:
        parts = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    parts.append(f"\n[Страница {page_num}]\n{text.strip()}")
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables, 1):
                    if not table:
                        continue
                    rows = []
                    for row in table:
                        cells = [(cell or "").strip() for cell in row]
                        row_text = " | ".join(cells)
                        if row_text.replace("|", "").strip():
                            rows.append(row_text)
                    if rows:
                        parts.append(f"\n[Таблица {t_idx}, стр.{page_num}]\n" + "\n".join(rows))
        return "\n".join(parts) if parts else None
    except Exception as e:
        logger.warning(f"Ошибка парсинга PDF {path.name}: {e}")
        return None
 
 
def parse_docx(path: Path) -> Optional[str]:
    try:
        doc = DocxDocument(str(path))
        parts = []
        W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
            if tag == "p":
                text = "".join(node.text or "" for node in element.iter(f"{{{W}}}t"))
                if text.strip():
                    parts.append(text.strip())
            elif tag == "tbl":
                parts.append("\n[ТАБЛИЦА]")
                for row in element.iter(f"{{{W}}}tr"):
                    cells = []
                    for cell in row.iter(f"{{{W}}}tc"):
                        cell_text = "".join(n.text or "" for n in cell.iter(f"{{{W}}}t")).strip()
                        cells.append(cell_text)
                    if any(cells):
                        parts.append(" | ".join(cells))
                parts.append("[/ТАБЛИЦА]\n")
        return "\n".join(parts) if parts else None
    except Exception as e:
        logger.warning(f"Ошибка парсинга DOCX {path.name}: {e}")
        return None
 
 
def parse_doc(path: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["antiword", "-w", "0", str(path)],
            capture_output=True, text=True, timeout=30, encoding="utf-8"
        )
        return result.stdout if result.returncode == 0 and result.stdout.strip() else None
    except FileNotFoundError:
        logger.warning("antiword не найден")
        return None
    except subprocess.TimeoutExpired:
        logger.warning(f"Таймаут: {path.name}")
        return None
 
 
def parse_file(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext == ".pdf":   return parse_pdf(path)
    if ext == ".docx":  return parse_docx(path)
    if ext == ".doc":   return parse_doc(path)
    return None
 
 
# ── Qdrant как источник истины ────────────────────────────────────────────────
 
def get_indexed_from_qdrant(client) -> dict[str, str]:
    """
    Возвращает {file_hash: filename} для всех уже проиндексированных файлов.
    Читает прямо из Qdrant payload — никакого локального JSON не нужно.
    """
    indexed: dict[str, str] = {}
    try:
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION not in collections:
            return indexed  # коллекция ещё не создана — первый запуск
 
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=COLLECTION,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                payload = point.payload or {}
                fh = payload.get("file_hash")
                fn = payload.get("source_file")
                if fh and fn:
                    indexed[fh] = fn
            if offset is None:
                break
    except Exception as e:
        logger.warning(f"Не удалось прочитать метаданные из Qdrant: {e}")
    return indexed
 
 
def delete_file_vectors(client, filename: str) -> None:
    """Удаляет все векторы указанного файла из Qdrant."""
    try:
        client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=filename))]
            )
        )
        logger.info(f"Удалены старые векторы: {filename}")
    except Exception as e:
        logger.warning(f"Ошибка удаления векторов {filename}: {e}")
 
 
# ── Основная логика ───────────────────────────────────────────────────────────
 
def ingest() -> None:
    print("\n" + "=" * 60)
    print("  SnipAI v2.0 — Индексация документов")
    print("=" * 60)
 
    if not init_settings():
        sys.exit(1)
 
    if not DATA_DIR.exists():
        print(f"❌ Папка не найдена: {DATA_DIR}")
        sys.exit(1)
 
    all_files = sorted([
        f for f in DATA_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT and not f.name.startswith(".")
    ])
 
    if not all_files:
        print(f"❌ Нет PDF/DOCX/DOC файлов в {DATA_DIR}")
        sys.exit(1)
 
    try:
        qdrant = get_qdrant_client()
    except ConnectionError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
 
    # Читаем что уже есть в Qdrant
    indexed = get_indexed_from_qdrant(qdrant)           # {hash: filename}
    indexed_hashes = set(indexed.keys())
    indexed_by_name = {v: k for k, v in indexed.items()}  # {filename: hash}
 
    new_files:     list[Path] = []
    updated_files: list[Path] = []
 
    for f in all_files:
        fh = file_hash(f)
        if fh in indexed_hashes:
            continue                            # этот хэш уже есть → пропускаем
        if f.name in indexed_by_name:
            updated_files.append(f)             # имя знакомо, но хэш другой → файл обновился
        else:
            new_files.append(f)                 # совсем новый файл
 
    to_process = new_files + updated_files
    skip_count = len(all_files) - len(to_process)
 
    print(f"\n📂 Всего файлов         : {len(all_files)}")
    print(f"⏭️  Уже актуальны        : {skip_count}")
    print(f"🆕 Новых файлов          : {len(new_files)}")
    print(f"🔄 Обновлённых файлов    : {len(updated_files)}")
 
    if not to_process:
        print("\n✅ База актуальна.")
        return
 
    # Удаляем старые векторы обновлённых файлов
    if updated_files:
        print("\n🗑️  Удаляем устаревшие векторы...")
        for f in updated_files:
            delete_file_vectors(qdrant, f.name)
            print(f"   ✓ {f.name}")
 
    # Индексируем
    vector_store    = QdrantVectorStore(client=qdrant, collection_name=COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=[], storage_context=storage_context, show_progress=False)
 
    success = 0
    errors  = 0
    total_batches = (len(to_process) + BATCH_SIZE - 1) // BATCH_SIZE
 
    for batch_num, batch_start in enumerate(range(0, len(to_process), BATCH_SIZE), start=1):
        batch = to_process[batch_start : batch_start + BATCH_SIZE]
        print(f"\n📦 Пакет {batch_num}/{total_batches}  ({len(batch)} файлов)")
 
        parsed = []
        for path in tqdm(batch, desc="  Парсинг", unit="файл", ncols=60):
            text = parse_file(path)
            if not text or len(text.strip()) < 30:
                print(f"  ⚠️  Пустой: {path.name}")
                errors += 1
                continue
            doc = LlamaDocument(
                text=text,
                metadata={
                    "source_file"  : path.name,
                    "file_hash"    : file_hash(path),   # ← ключевое поле
                    "file_type"    : path.suffix.lower(),
                    "document_type": "СНиП/НПА",
                    "indexed_at"   : datetime.now().isoformat(),
                }
            )
            parsed.append((path, doc))
 
        if not parsed:
            continue
 
        try:
            for path, doc in tqdm(parsed, desc="  Индексация", unit="doc", ncols=60):
                index.insert(doc)
                success += 1
            print(f"  💾 Прогресс: {success}/{len(to_process)}")
        except Exception as e:
            print(f"  ❌ Ошибка вставки: {e}")
            errors += 1
 
        if batch_start + BATCH_SIZE < len(to_process):
            time.sleep(1)
 
    print("\n" + "=" * 60)
    print(f"✅ Проиндексировано: {success}")
    print(f"❌ Ошибок          : {errors}")
    print(f"📦 Коллекция       : {COLLECTION}")
    print("=" * 60)
 
 
if __name__ == "__main__":
    ingest()
 
