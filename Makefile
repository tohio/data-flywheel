.PHONY: up down build seed run-flywheel resume-flywheel status-flywheel test test-all lint reset logs install install-dev

COMPOSE = docker compose -f infra/docker/docker-compose.yml

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

build:
	$(COMPOSE) build

logs:
	$(COMPOSE) logs -f orchestrator worker beat

seed:
	python infra/scripts/seed_logs.py

run-flywheel:
	curl -s -X POST http://localhost:8000/flywheel/run | python -m json.tool

resume-flywheel:
	@LAST_RUN=$$(curl -s http://localhost:8000/flywheel/runs?limit=1\&status=failed | python -m json.tool | grep '"run_id"' | head -1 | awk -F'"' '{print $$4}'); \
	if [ -z "$$LAST_RUN" ]; then \
		echo "No failed run found to resume."; \
	else \
		echo "Resuming run: $$LAST_RUN"; \
		curl -s -X POST http://localhost:8000/flywheel/resume/$$LAST_RUN | python -m json.tool; \
	fi

status-flywheel:
	@LAST_RUN=$$(curl -s "http://localhost:8000/flywheel/runs?limit=1" | python -m json.tool | grep '"run_id"' | head -1 | awk -F'"' '{print $$4}'); \
	if [ -z "$$LAST_RUN" ]; then \
		echo "No runs found."; \
	else \
		curl -s http://localhost:8000/flywheel/status/$$LAST_RUN | python -m json.tool; \
	fi

reset:
	bash infra/scripts/reset.sh

test:
	pytest tests/unit/ -v

test-all:
	pytest tests/ -v

lint:
	ruff check orchestrator/
	mypy orchestrator/

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt