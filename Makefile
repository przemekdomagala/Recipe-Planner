ARGS ?=
TEST_DATABASE_URL_PG ?= postgresql+asyncpg://recipe:recipe@localhost:5433/recipe_test

BACKEND = cd backend &&
FRONTEND = cd frontend &&
PY = uv run --project backend python -c

# A literal space, so $(info) lines in `help` can start indented.
EMPTY :=
SP := $(EMPTY) $(EMPTY)

.DEFAULT_GOAL := help

.PHONY: help install up up-d down reset logs db dev fe-dev seed reindex llm-status \
        lint format typecheck test test-pg check fe-lint fe-build clean

help:
	$(info Usage: make TARGET [ARGS=...])
	$(info Setup)
	$(info $(SP)$(SP)install     install backend (uv) and frontend (npm) dependencies, create .env)
	$(info Docker)
	$(info $(SP)$(SP)up          build and start db + backend + frontend (foreground))
	$(info $(SP)$(SP)up-d        same, in the background)
	$(info $(SP)$(SP)down        stop the stack)
	$(info $(SP)$(SP)reset       stop the stack and delete the database volume)
	$(info $(SP)$(SP)logs        follow container logs)
	$(info $(SP)$(SP)db          start only the database)
	$(info Local dev)
	$(info $(SP)$(SP)dev         backend with auto-reload on :8000 (starts the db container))
	$(info $(SP)$(SP)fe-dev      frontend Vite dev server on :5173)
	$(info Knowledge base (needs the Ollama app running))
	$(info $(SP)$(SP)seed        import + embed all TheMealDB recipes (ARGS="--limit 20" etc.))
	$(info $(SP)$(SP)reindex     embed recipes with missing/stale chunks (ARGS="--force" for all))
	$(info $(SP)$(SP)llm-status  check the Ollama connection and models)
	$(info Backend quality)
	$(info $(SP)$(SP)lint        ruff check)
	$(info $(SP)$(SP)format      ruff format + ruff check --fix)
	$(info $(SP)$(SP)typecheck   mypy (strict))
	$(info $(SP)$(SP)test        pytest on in-memory SQLite)
	$(info $(SP)$(SP)test-pg     pytest on the compose Postgres (database recipe_test))
	$(info $(SP)$(SP)check       lint + format check + typecheck + test + frontend lint/build)
	$(info Frontend)
	$(info $(SP)$(SP)fe-lint     oxlint)
	$(info $(SP)$(SP)fe-build    type-check and production build)
	$(info Misc)
	$(info $(SP)$(SP)clean       remove caches and build output)
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
	docker compose up --build

up-d: .env
	docker compose up --build -d

down:
	docker compose down

reset:
	docker compose down -v

logs:
	docker compose logs -f

db: .env
	docker compose up -d --wait db

# --- Local dev -------------------------------------------------------------------------

dev: db
	$(BACKEND) uv run fastapi dev $(ARGS)

fe-dev:
	$(FRONTEND) npm run dev

# --- Knowledge base --------------------------------------------------------------------

seed: db
	$(BACKEND) uv run python -m app.cli ingest-mealdb $(ARGS)

reindex: db
	$(BACKEND) uv run python -m app.cli reindex $(ARGS)

llm-status:
	$(BACKEND) uv run python -m app.cli llm-status

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
	-docker compose exec db createdb -U recipe recipe_test
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
	$(PY) "import pathlib, shutil; b = pathlib.Path('backend'); dirs = [p for p in b.rglob('__pycache__') if '.venv' not in p.parts]; dirs += [b / n for n in ('.artifacts', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov')]; dirs.append(pathlib.Path('frontend/dist')); [shutil.rmtree(d, ignore_errors=True) for d in dirs]; print('Cleaned')"
