# Recipe Planner

Web application focused on personalized meal planning, dietary management, and smart recipe generation, utilizing AI.

## Current Project stage

**Early stage: end-to-end skeleton.** A FastAPI + async SQLAlchemy backend, a React (Vite + TypeScript) frontend and a PostgreSQL database with the **pgvector** extension are wired together through Docker Compose.

| Area                      | State                                                                  |
| ------------------------- | ---------------------------------------------------------------------- |
| Ingredients and recipes   | Implemented: create and list via the API and the UI                    |
| Database schema           | Implemented: `Ingredient`, `Recipe`, `RecipeIngredient`; created with `create_all` at start-up (no migrations yet) |
| Vector storage            | Groundwork only: `recipes.embedding` column exists but is never populated |
| RAG search                | Placeholder: `POST /api/recipes/search` returns a mock response        |
| Meal planning, dietary management, AI recipe generation | Not started                              |
| Backend tests             | CRUD endpoints covered; run on SQLite by default, optionally on Postgres |
| Frontend tests            | None yet                                                               |

```
.
├── backend/              FastAPI app (uv-managed, Python 3.12)
│   ├── app/
│   │   ├── api/          routers (ingredients, recipes, health) and dependencies
│   │   ├── core/         settings (env) and domain exceptions
│   │   ├── crud/         database access functions
│   │   ├── db/           declarative Base, async engine/session, `get_db` dependency
│   │   ├── models/       Ingredient, Recipe (with `embedding vector(1536)`), RecipeIngredient
│   │   ├── schemas/      Pydantic request/response models
│   │   └── services/     business logic (RAG search placeholder here)
│   └── tests/            pytest suite
├── frontend/             React + Vite + TypeScript
├── db/init/              SQL run on first DB start (enables pgvector)
├── docker-compose.yml    db + backend + frontend
└── .env.example          configuration template (copy to `.env`)
```

## Run everything with Docker Compose

Requires Docker with the Compose plugin.

```bash
# create your local config, then edit .env and change the credentials if needed
cp .env.example .env

docker compose up --build
```

| Service  | URL                                                      |
| -------- | -------------------------------------------------------- |
| Frontend | http://localhost:5173                                    |
| Backend  | http://localhost:8000 (interactive docs: `/docs`)        |
| Postgres | `localhost:5433` (host port set by `POSTGRES_PORT`; user/password/db: `recipe`/`recipe`/`recipe_planner`) |

On start-up the backend runs `CREATE EXTENSION IF NOT EXISTS vector` and creates the tables.

Stop the stack with `docker compose down` (add `-v` to also delete the database volume).

> The frontend calls the backend directly from the browser at `VITE_API_URL` (default `http://localhost:8000`); this value is baked in at build time, so re-run `docker compose up --build` after changing it. The backend's CORS allow-list is `CORS_ORIGINS`.

Verify the pgvector extension is active:

```bash
docker compose exec db psql -U recipe -d recipe_planner -c "\dx vector" -c "\d recipes"
```

## Linting, formatting and type checking (backend)

Requires [uv](https://docs.astral.sh/uv/); `uv run` installs the pinned Python 3.12 and the dependencies on first use.

```bash
cd backend
uv run ruff check .             # lint (add --fix to auto-fix)
uv run ruff format --check .    # check formatting (drop --check to reformat)
uv run mypy                     # strict type checking
```

## Tests (backend)

```bash
cd backend
uv run pytest
```

By default the tests use an in-memory SQLite database (via `aiosqlite`), so they need **no running services**; the `embedding` column transparently falls back to JSON there. FastAPI's `get_db` dependency is overridden with a per-test database, so every test starts from an empty schema.

To run the same suite against real PostgreSQL + pgvector instead (e.g. the Compose database; use a throw-away database since tables are dropped after each test):

```bash
docker compose up -d db
docker compose exec db createdb -U recipe recipe_test
cd backend
TEST_DATABASE_URL=postgresql+asyncpg://recipe:recipe@localhost:5433/recipe_test uv run pytest
```

Running tests inside the backend container is not supported because the image deliberately excludes dev dependencies and tests.

## API overview

| Method | Path                    | Description                                              |
| ------ | ----------------------- | -------------------------------------------------------- |
| GET    | `/health`               | Liveness check                                           |
| POST   | `/api/ingredients`      | Create an ingredient (names are trimmed + lowercased; `409` on duplicates) |
| GET    | `/api/ingredients`      | List ingredients                                         |
| POST   | `/api/recipes`          | Create a recipe with optional `ingredients: [{ingredient_id, quantity, unit}]` |
| GET    | `/api/recipes`          | List recipes (`skip`, `limit`)                           |
| GET    | `/api/recipes/{id}`     | Get one recipe                                           |
| POST   | `/api/recipes/search`   | **Placeholder** for RAG vector search; returns a mock response (`"mock": true`) |

## RAG groundwork

- `recipes.embedding` is a nullable `vector(1536)` column (dimension configurable via `EMBEDDING_DIM`, which must be set before the table is first created).
- Queries such as `Recipe.embedding.cosine_distance(query_vector)` are available through the `pgvector` SQLAlchemy integration, and the asyncpg codec for the `vector` type is registered on every connection.
- `app/services/recipe_search.py` is where embedding the query + nearest-neighbour retrieval should be implemented.
- Schema changes are currently applied with `create_all` at start-up; introduce Alembic migrations before the schema evolves (including adding a vector index such as HNSW).
