ARGS ?=
TEST_DATABASE_URL_PG ?= postgresql+asyncpg://recipe:recipe@localhost:5433/recipe_test

BACKEND = cd backend &&
FRONTEND = cd frontend &&
PY = uv run --project backend python -c
# `make up GPU=1` gives Ollama access to NVIDIA GPUs.
COMPOSE = docker compose $(if $(GPU),-f docker-compose.yml -f docker-compose.gpu.yml)
CLI = python -m app.cli

# A literal space, so $(info) lines in `help` can start indented.
EMPTY :=
SP := $(EMPTY) $(EMPTY)

.DEFAULT_GOAL := help

.PHONY: help install up up-d down reset logs db models dev fe-dev \
        seed reindex seed-local reindex-local \
        lint format typecheck test test-pg check fe-lint fe-build clean

help:
	$(info Usage: make TARGET [ARGS=...] [GPU=1])
	$(info Setup)
	$(info $(SP)$(SP)install       install backend (uv) and frontend (npm) dependencies, create .env)
	$(info Docker)
	$(info $(SP)$(SP)up            build and start db + ollama + backend + frontend (foreground))
	$(info $(SP)$(SP)up-d          same, in the background)
	$(info $(SP)$(SP)down          stop the stack)
	$(info $(SP)$(SP)reset         stop the stack and delete the database volume (keeps models))
	$(info $(SP)$(SP)logs          follow container logs)
	$(info $(SP)$(SP)db            start only the database)
	$(info $(SP)$(SP)models        start Ollama and pull the configured models)
	$(info Knowledge base)
	$(info $(SP)$(SP)seed          import + index TheMealDB recipes in the running stack, e.g. ARGS="--letters ab")
	$(info $(SP)$(SP)reindex       re-embed stale recipes in the running stack (ARGS=--force for all))
	$(info $(SP)$(SP)seed-local    like seed, but with the local backend (for `make dev`))
	$(info $(SP)$(SP)reindex-local like reindex, but with the local backend)
	$(info Local dev)
	$(info $(SP)$(SP)dev           backend with auto-reload on :8000 (starts db + ollama containers))
	$(info $(SP)$(SP)fe-dev        frontend Vite dev server on :5173)
	$(info Backend quality)
	$(info $(SP)$(SP)lint          ruff check)
	$(info $(SP)$(SP)format        ruff format + ruff check --fix)
	$(info $(SP)$(SP)typecheck     mypy (strict))
	$(info $(SP)$(SP)test          pytest on in-memory SQLite (vector search tests are skipped))
	$(info $(SP)$(SP)test-pg       pytest on the compose Postgres (database recipe_test), all tests)
	$(info $(SP)$(SP)check         lint + format check + typecheck + test + frontend lint/build)
	$(info Frontend)
	$(info $(SP)$(SP)fe-lint       oxlint)
	$(info $(SP)$(SP)fe-build      type-check and production build)
	$(info Misc)
	$(info $(SP)$(SP)clean         remove caches and build output)
	@:

# --- Setup -----------------------------------------------------------------------------

# Only runs when .env does not exist yet, so local edits are never overwritten.
.env:
	$(PY) "import shutil; shutil.copyfile('.env.example', '.env'); print('Created .env from .env.example')"

install: .env
	$(BACKEND) uv sync
	$(FRONTEND) npm ci

# --- Docker ----------------------------------------------------------------------------

up: .env
	$(COMPOSE) up --build

up-d: .env
	$(COMPOSE) up --build -d

down:
	$(COMPOSE) down

# Deletes only the database volume; `down -v` would also delete the (large) Ollama models.
reset:
	$(COMPOSE) down
	docker volume rm -f recipe-planner_pgdata

logs:
	$(COMPOSE) logs -f

db: .env
	$(COMPOSE) up -d --wait db

# Starts the ollama service as a dependency and exits once the models are present.
models: .env
	$(COMPOSE) run --rm ollama-pull

# --- Knowledge base --------------------------------------------------------------------

seed:
	$(COMPOSE) exec backend $(CLI) import-themealdb $(ARGS)

reindex:
	$(COMPOSE) exec backend $(CLI) reindex $(ARGS)

seed-local: db models
	$(BACKEND) uv run $(CLI) import-themealdb $(ARGS)

reindex-local: db models
	$(BACKEND) uv run $(CLI) reindex $(ARGS)

# --- Local dev -------------------------------------------------------------------------

dev: db models
	$(BACKEND) uv run fastapi dev $(ARGS)

fe-dev:
	$(FRONTEND) npm run dev

# --- Backend quality -------------------------------------------------------------------

lint:
	$(BACKEND) uv run ruff check . $(ARGS)

format:
	$(BACKEND) uv run ruff format .
	$(BACKEND) uv run ruff check --fix .

typecheck:
	$(BACKEND) uv run mypy

test:
	$(BACKEND) uv run pytest $(ARGS)

# Tables are dropped after every test, hence a separate throw-away database.
# The leading "-" ignores the error when recipe_test already exists.
test-pg: export TEST_DATABASE_URL := $(TEST_DATABASE_URL_PG)
test-pg: db
	-$(COMPOSE) exec db createdb -U recipe recipe_test
	$(BACKEND) uv run pytest $(ARGS)

check:
	$(BACKEND) uv run ruff check .
	$(BACKEND) uv run ruff format --check .
	$(BACKEND) uv run mypy
	$(BACKEND) uv run pytest
	$(FRONTEND) npm run lint
	$(FRONTEND) npm run build

# --- Frontend --------------------------------------------------------------------------

fe-lint:
	$(FRONTEND) npm run lint

fe-build:
	$(FRONTEND) npm run build

# --- Misc ------------------------------------------------------------------------------

clean:
	$(PY) "import pathlib, shutil; b = pathlib.Path('backend'); dirs = [p for p in b.rglob('__pycache__') if '.venv' not in p.parts]; dirs += [b / n for n in ('.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov')]; dirs.append(pathlib.Path('frontend/dist')); [shutil.rmtree(d, ignore_errors=True) for d in dirs]; print('Cleaned')"
