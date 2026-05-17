"""
SnipAI v2.0 — app.py
FastAPI backend + встроенный фронтенд
"""
import os
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from core import get_qdrant_client, COLLECTION
from query import build_retriever, ask

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Глобальное состояние ──────────────────────────
retriever = None
current_model = os.getenv("LLM_MODEL", "qwen/qwen3-8b")

AVAILABLE_MODELS = [
    {"id": "qwen/qwen3-8b",                     "name": "Qwen3 8B"},
    {"id": "qwen/qwen3-14b",                     "name": "Qwen3 14B"},
    {"id": "qwen/qwen3-32b",                     "name": "Qwen3 32B"},
    {"id": "deepseek/deepseek-chat-v3-0324",     "name": "DeepSeek V3"},
    {"id": "meta-llama/llama-4-maverick",        "name": "Llama 4 Maverick"},
    {"id": "meta-llama/llama-3.3-70b-instruct",  "name": "Llama 3.3 70B"},
    {"id": "google/gemma-3-27b-it",              "name": "Gemma 3 27B"},
]


# ── Lifespan ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ НЕ грузим модель при старте — экономим память (Free план = 512MB)
    # Ретривер инициализируется лениво при первом запросе в /api/ask
    logger.info("SnipAI v2.0 запущен.")
    yield
    logger.info("SnipAI остановлен.")


# ── Приложение ────────────────────────────────────
app = FastAPI(title="SnipAI", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Модели запросов ───────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    model: Optional[str] = None


class ModelRequest(BaseModel):
    model_id: str


# ── API эндпоинты ─────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Статус системы: сколько документов в базе."""
    try:
        client = get_qdrant_client()
        info   = client.get_collection(COLLECTION)
        count  = info.points_count or 0
        data_dir = Path(os.getenv("DATA_DIR", "/app/data"))
        files = [
            f.name for f in data_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".pdf", ".docx", ".doc"}
        ] if data_dir.exists() else []
        return {
            "status": "ok",
            "vectors": count,
            "documents": files,
            "model": current_model,
            "collection": COLLECTION,
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "vectors": 0, "documents": []}


@app.get("/api/models")
async def get_models():
    """Список доступных моделей."""
    return {"models": AVAILABLE_MODELS, "current": current_model}


@app.post("/api/model")
async def set_model(req: ModelRequest):
    """Смена модели без перезапуска."""
    global current_model
    ids = [m["id"] for m in AVAILABLE_MODELS]
    if req.model_id not in ids:
        raise HTTPException(status_code=400, detail="Модель не найдена")
    current_model = req.model_id
    os.environ["LLM_MODEL"] = current_model
    import core
    core.LLM_MODEL = current_model
    return {"ok": True, "model": current_model}


@app.post("/api/ask")
async def ask_question(req: QuestionRequest):
    """Основной эндпоинт — задать вопрос."""
    global retriever, current_model

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    # Если модель передана в запросе — переключаем
    if req.model and req.model != current_model:
        current_model = req.model
        import core
        core.LLM_MODEL = current_model

    # Ленивая инициализация ретривера при первом запросе
    if retriever is None:
        retriever = build_retriever()
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="База документов пуста. Сначала проиндексируйте файлы."
        )

    try:
        answer = ask(retriever, req.question)
        return {
            "answer": answer,
            "model": current_model,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.exception("Ошибка при обработке вопроса")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reindex")
async def reindex():
    """Переиндексация документов."""
    global retriever
    import subprocess, sys
    try:
        result = subprocess.run(
            [sys.executable, "src/ingest.py"],
            capture_output=True, text=True, timeout=300
        )
        retriever = build_retriever()
        return {
            "ok": True,
            "output": result.stdout[-2000:] if result.stdout else "",
            "errors": result.stderr[-500:]  if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "Таймаут индексации (5 мин)"}
    except Exception as e:
        return {"ok": False, "output": str(e)}


@app.get("/", response_class=HTMLResponse)
async def frontend():
    """Встроенный фронтенд."""
    html_path = Path("/app/src/frontend.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>frontend.html не найден</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
    