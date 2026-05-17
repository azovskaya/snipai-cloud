# =====================================================
# SnipAI v2.0 — Makefile
# Удобные команды вместо длинных docker-compose команд
# Использование: make <команда>
# =====================================================

.PHONY: help build start stop ingest query logs status clean

# Цвета для вывода
GREEN  = \033[0;32m
YELLOW = \033[1;33m
RESET  = \033[0m

help:  ## Показать список команд
	@echo ""
	@echo "$(GREEN)SnipAI v2.0 — Доступные команды:$(RESET)"
	@echo "──────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)make %-12s$(RESET) %s\n", $$1, $$2}'
	@echo ""

build:  ## Собрать Docker образ (первый раз ~5-10 мин)
	docker-compose build

start:  ## Запустить Qdrant в фоне
	docker-compose up -d qdrant
	@echo "$(GREEN)✅ Qdrant запущен. Ждём готовности...$(RESET)"
	@sleep 5
	@echo "$(GREEN)✅ Готово! Теперь: make ingest$(RESET)"

ingest:  ## Проиндексировать документы из ./data/
	docker-compose run --rm snipai python -u src/ingest.py

query:  ## Запустить интерфейс вопросов
	docker-compose run --rm snipai python -u src/query.py

logs:  ## Показать логи (Ctrl+C для выхода)
	docker-compose logs -f

status:  ## Статус контейнеров
	docker-compose ps

stop:  ## Остановить все контейнеры
	docker-compose down

clean:  ## ⚠️  Удалить ВСЕ данные Qdrant (нужна переиндексация!)
	@echo "$(YELLOW)⚠️  Это удалит все векторные данные!$(RESET)"
	@read -p "Вы уверены? (yes/no): " confirm && [ "$$confirm" = "yes" ]
	docker-compose down -v
	rm -f data/.indexed_files.json
	@echo "$(GREEN)✅ Очищено. Запустите: make start && make ingest$(RESET)"
