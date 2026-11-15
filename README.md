# Recipe Planner

Web application focused on personalized meal planning, dietary management, and smart recipe generation, utilizing AI.

## Current Project stage

**Milestone 2: recipe knowledge base and RAG foundation.** On top of the end-to-end skeleton (FastAPI + async SQLAlchemy, React, PostgreSQL + pgvector, Docker Compose), the backend now has its own retrieval pipeline over recipes imported from [TheMealDB](https://www.themealdb.com/) and a first LLM call through a local [Ollama](https://ollama.com/) server.

| Area                      | State                                                                  |
| ------------------------- | ---------------------------------------------------------------------- |
| Ingredients and recipes   | Implemented: create and list via the API and the UI                    |
| Recipe knowledge base     | Implemented: TheMealDB importer (~800 recipes) with measure parsing, ingredient merging and idempotent re-imports |
| RAG pipeline              | Implemented: chunking, embeddings (Ollama), pgvector store with HNSW index, retrieval API, grounded answers (Ollama chat) |
| Database schema           | `Ingredient`, `Recipe`, `RecipeIngredient`, `RecipeChunk`; created with `create_all` at start-up (no migrations yet) |
| Meal planning, dietary management, AI recipe modification | Not started                       |
| Backend tests             | Unit + API tests with a fake LLM; SQLite by default, Postgres for vector search |
| Frontend                  | Minimal: forms, recipe list and a search / "Ask AI" box; no tests yet   |

```
.
├── backend/              FastAPI app (uv-managed, Python 3.12)
│   ├── app/
│   │   ├── api/          routers (ingredients, recipes, rag, health), dependencies, response mappers
│   │   ├── core/         settings (env) and domain exceptions
│   │   ├── crud/         database access functions
│   │   ├── db/           declarative Base, async engine/session, `get_db` dependency
│   │   ├── llm/          provider-agnostic Embedder/ChatModel protocols + Ollama implementation
│   │   ├── models/       Ingredient, Recipe, RecipeIngredient, RecipeChunk (`embedding vector(768)`)
│   │   ├── rag/          chunking, indexing, retrieval, prompts, answer generation
│   │   ├── schemas/      Pydantic request/response models
│   │   ├── services/     business logic (recipe import)
│   │   ├── sources/      TheMealDB client/parser, ingredient measure parser
│   │   └── cli.py        `python -m app.cli` (import / reindex)
│   └── tests/            pytest suite (+ fakes for the LLM)
├── frontend/             React + Vite + TypeScript
├── db/init/              SQL run on first DB start (enables pgvector)
├── docker-compose.yml    db + ollama (+ model pull job) + backend + frontend
├── docker-compose.gpu.yml optional override: NVIDIA GPU for Ollama
└── .env.example          configuration template (copy to `.env`)
```

## Quick start (Docker Compose)

Requires Docker with the Compose plugin (and `make`; every target is a thin wrapper, see the `Makefile` for the raw commands).

```bash
make up-d          # creates .env from .env.example on first run; add GPU=1 for an NVIDIA GPU
make seed          # import + index all TheMealDB recipes (~2 min on a GPU); ARGS="--letters ab" for a quick subset
```

The **first** `make up-d` downloads the models (`nomic-embed-text` ≈ 0.3 GB, `llama3.2:3b` ≈ 2 GB) before the backend starts; follow it with `make logs`. Models live in the `ollama` volume and survive `make reset`.

| Service  | URL                                                      |
| -------- | -------------------------------------------------------- |
| Frontend | http://localhost:5173                                    |
| Backend  | http://localhost:8000 (interactive docs: `/docs`)        |
| Ollama   | http://localhost:11434 (host port set by `OLLAMA_PORT`)  |
| Postgres | `localhost:5433` (host port set by `POSTGRES_PORT`; user/password/db: `recipe`/`recipe`/`recipe_planner`) |

Stop the stack with `make down`; `make reset` also deletes the database (not the models).

> **Upgrading from Milestone 1:** the schema changed (new recipe columns, `recipe_chunks` replaces `recipes.embedding`) and `create_all` does not alter existing tables, so run `make reset` once.

> The frontend calls the backend directly from the browser at `VITE_API_URL` (default `http://localhost:8000`); this value is baked in at build time, so re-run `make up` after changing it. The backend's CORS allow-list is `CORS_ORIGINS`.

> **Already running Ollama on the host** (e.g. the Windows app)? Either change `OLLAMA_PORT` to avoid the port clash, or point the backend at it with `OLLAMA_BASE_URL=http://host.docker.internal:11434` and pull the models there.

## The RAG pipeline

```
TheMealDB ─► sources/ (parse) ─► services/recipe_import (upsert recipes + ingredients)
                                          │
                                          ▼
                   rag/chunking ─► rag/indexing ─► recipe_chunks (pgvector, HNSW)
                                   (Ollama embed)          │
question ─► Ollama embed ─► rag/retrieval (cosine kNN, best chunk per recipe, filters)
                                          │
                                          ▼
                 rag/prompts + rag/generation ─► Ollama chat ─► answer + sources
```

- **Import** (`app/sources`, `app/services/recipe_import.py`): meals are fetched letter by letter. Ingredient names are normalized like user input. Measures such as `"1 1/2 cups (360 ml)"` are parsed into a quantity and a canonical unit (`1.5 cup`), and the raw text is kept in `measure`. Ingredients listed twice in a meal are merged (same unit: quantities summed). Areas are normalized to cuisine names ("France" → "French"). Recipes are upserted by `(source, external_id)`, so re-imports update rather than duplicate.
- **Chunking** (`app/rag/chunking.py`): each recipe becomes one *summary* chunk (title, category, cuisine, tags and the full ingredient list), which matches "what can I cook with X and Y" questions. The instructions are packed sentence by sentence into chunks of `CHUNK_MAX_CHARS` with `CHUNK_OVERLAP_CHARS` of overlap; each chunk gets a header naming the recipe, so it stays meaningful on its own.
- **Indexing** (`app/rag/indexing.py`): chunks are embedded with `nomic-embed-text` (768 dimensions, using its `search_document:` / `search_query:` prefixes). Unchanged recipes are skipped (same chunk texts and model), so re-seeding is cheap. If embedding fails, the old chunks are kept. Recipes created through the API are indexed immediately (best-effort: creation still succeeds when Ollama is down).
- **Retrieval** (`app/rag/retrieval.py`): cosine nearest-neighbour search over the chunks, with optional case-insensitive `category` / `area` filters. It fetches `limit × RAG_CANDIDATE_MULTIPLIER` chunks and keeps the best chunk per recipe, so one long recipe can't fill all the results. `hnsw.ef_search` is raised so the HNSW index can return that many candidates.
- **Generation** (`app/rag/generation.py`, `prompts.py`): the top recipes (summary chunk plus matched instruction chunks) are passed to the chat model with instructions to answer only from them. If nothing is retrieved, the LLM is not called at all.

The RAG code depends only on the `Embedder` / `ChatModel` protocols in `app/llm/base.py`, so the provider can be swapped and tests use deterministic fakes.

### Knowledge base CLI

```bash
python -m app.cli import-themealdb [--letters abc] [--limit N] [--no-index]
python -m app.cli reindex [--force]
```

`make seed` / `make reindex` run these inside the backend container; `make seed-local` / `make reindex-local` run them with the local backend (see below). Pass options via `ARGS="..."`.

## Local development

```bash
make install       # uv sync + npm ci, creates .env
make dev           # starts db + ollama containers (pulls models), backend with auto-reload on :8000
make seed-local    # in another terminal
make fe-dev        # Vite dev server on :5173
```

The local backend reads its settings from the environment or `backend/.env`; the defaults point at the Compose database (`localhost:5433`) and Ollama (`localhost:11434`). Useful settings (see `app/core/config.py`):

| Variable                  | Default              | Notes |
| ------------------------- | -------------------- | ----- |
| `OLLAMA_BASE_URL`         | `http://localhost:11434` | |
| `OLLAMA_CHAT_MODEL`       | `llama3.2:3b`        | any Ollama chat model |
| `OLLAMA_EMBEDDING_MODEL`  | `nomic-embed-text`   | changing it requires `EMBEDDING_DIM` to match, `make reset` and re-seeding |
| `EMBEDDING_DIM`           | `768`                | size of the `vector` column |
| `EMBEDDING_DOCUMENT_PREFIX` / `EMBEDDING_QUERY_PREFIX` | `search_document: ` / `search_query: ` | set to empty strings for models without task prefixes |
| `CHUNK_MAX_CHARS` / `CHUNK_OVERLAP_CHARS` | `1000` / `150` | |
| `OLLAMA_TIMEOUT_SECONDS`  | `180`                | the first request after start-up also loads the model |

## Linting, formatting and type checking

```bash
make lint          # ruff check
make format        # ruff format + ruff check --fix
make typecheck     # mypy --strict
make check         # all of the above + tests + frontend lint/build
```

## Tests

```bash
make test          # SQLite in memory, no services needed
make test-pg       # against the Compose Postgres (database recipe_test)
```

The LLM is never called in tests: the embedding and chat models are replaced by fakes (`tests/fakes.py`; the embedder is a hashed bag-of-words, so similarity ranking still means something), and the Ollama server is replaced by `httpx.MockTransport`.

Vector similarity search needs PostgreSQL + pgvector, so tests marked `@pytest.mark.postgres` are skipped on SQLite and run with `make test-pg` (or any `TEST_DATABASE_URL=postgresql+asyncpg://...`). Running tests inside the backend container is not supported because the image deliberately excludes dev dependencies and tests.

## API overview

| Method | Path                    | Description                                              |
| ------ | ----------------------- | -------------------------------------------------------- |
| GET    | `/health`               | Liveness check                                           |
| POST   | `/api/ingredients`      | Create an ingredient (names are trimmed + lowercased; `409` on duplicates) |
| GET    | `/api/ingredients`      | List ingredients                                         |
| POST   | `/api/recipes`          | Create a recipe with optional `ingredients: [{ingredient_id, quantity, unit}]`; indexed for search |
| GET    | `/api/recipes`          | List recipes (`skip`, `limit`)                           |
| GET    | `/api/recipes/{id}`     | Get one recipe                                           |
| POST   | `/api/recipes/search`   | Semantic search: `{query, limit?, category?, area?}` → recipes with score and snippet |
| POST   | `/api/rag/ask`          | Grounded answer: `{question, top_k?, category?, area?}` → `{answer, model, sources}` |
| GET    | `/api/rag/status`       | Indexed recipe/chunk counts and whether Ollama has the configured models |

Search and ask return `503` when Ollama is unreachable or a model is missing.

## Known limitations / next steps

- Schema changes still go through `create_all`; Alembic migrations should be introduced before the next schema change.
- Retrieval is pure vector search. Hybrid search (adding Postgres full-text) and re-ranking would help exact-term queries.
- Filtered queries on very large tables can return fewer results with HNSW; pgvector ≥ 0.8 iterative index scans would fix this once the dataset grows.
