"""
SnipAI v2.0 — ingest.py
Умная индексация: PDF, DOCX, DOC → Qdrant Cloud
Без torch/sentence-transformers — эмбеддинги через OpenRouter API
"""
from __future__ import annotations

import os
import sys
import json
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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DATA_DIR      = Path(os.getenv("DATA_DIR", "/app/data"))
INDEX_TRACKER = DATA_DIR / ".indexed_files.json"
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "20"))
SUPPORTED_EXT = {".pdf", ".docx", ".doc"}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:20]


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


def load_tracker() -> dict:
    if INDEX_TRACKER.exists():
        try:
            return json.loads(INDEX_TRACKER.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_tracker(tracker: dict) -> None:
    INDEX_TRACKER.write_text(
        json.dumps(tracker, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def ingest() -> None:
    print("\n" + "=" * 60)
    print("  SnipAI v2.0 — Индексация документов")
    print("=" * 60)

    # ✅ Ленивая инициализация — только здесь, не при импорте
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

    tracker   = load_tracker()
    new_files = [f for f in all_files if file_hash(f) not in tracker]
    skip_count = len(all_files) - len(new_files)

    print(f"\n📂 Всего файлов      : {len(all_files)}")
    print(f"⏭️  Уже проиндексировано: {skip_count}")
    print(f"🆕 Новых для обработки: {len(new_files)}")

    if not new_files:
        print("\n✅ База актуальна.")
        return

    try:
        qdrant = get_qdrant_client()
    except ConnectionError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    vector_store    = QdrantVectorStore(client=qdrant, collection_name=COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=[], storage_context=storage_context, show_progress=False)

    success = 0
    errors  = 0
    total_batches = (len(new_files) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, batch_start in enumerate(range(0, len(new_files), BATCH_SIZE), start=1):
        batch = new_files[batch_start : batch_start + BATCH_SIZE]
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
                tracker[file_hash(path)] = {
                    "name"      : path.name,
                    "indexed_at": datetime.now().isoformat(),
                }
                success += 1
            save_tracker(tracker)
            print(f"  💾 Прогресс сохранён ({success}/{len(new_files)})")
        except Exception as e:
            print(f"  ❌ Ошибка вставки: {e}")
            errors += 1

        if batch_start + BATCH_SIZE < len(new_files):
            time.sleep(1)

    print("\n" + "=" * 60)
    print(f"✅ Проиндексировано: {success}")
    print(f"❌ Ошибок          : {errors}")
    print(f"📦 Коллекция       : {COLLECTION}")
    print("=" * 60)


if __name__ == "__main__":
    ingest()
    