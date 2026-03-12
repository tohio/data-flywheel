.PHONY: up down build seed run-flywheel test test-all lint reset logs install install-dev

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
