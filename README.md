# Recipe Planner

Web application focused on personalized meal planning, dietary management, and smart recipe generation, utilizing AI.

## Current Project stage

**Milestone 3: fridge suggestions via RAG + LLM.** A FastAPI + async SQLAlchemy backend, a React (Vite + TypeScript) frontend and a PostgreSQL database with the **pgvector** extension are wired together through Docker Compose. Recipes from [TheMealDB](https://www.themealdb.com/api.php) are chunked, embedded and searched semantically. A RAG assistant answers questions with **qwen3.5:4b** in the local Ollama app. "What's in the fridge" suggests recipes for the ingredients at hand: retrieval and a deterministic diet/allergen checker pick and adapt the candidates, the LLM ranks them and proposes swaps, and every swap is validated again by the checker.

| Area                      | State                                                                  |
| ------------------------- | ---------------------------------------------------------------------- |
| Ingredients and recipes   | Implemented: create and list via the API and the UI                    |
| Database schema           | Implemented: `Ingredient`, `Recipe`, `RecipeIngredient`, `RecipeChunk`, `FridgeItem`, `DietProfile`; created with `create_all` at start-up (no migrations yet) |
| Knowledge base            | Implemented: TheMealDB import (`make seed`), idempotent upserts, chunking + embeddings in `recipe_chunks` (pgvector, HNSW) |
| RAG search                | Implemented: `POST /api/recipes/search` (semantic search with category/cuisine filters) |
| LLM                       | Implemented: `POST /api/assistant/ask`, answers grounded in retrieved recipes (local Ollama app) |
| What's in the fridge      | Implemented: fridge items with expiry dates, `POST /api/fridge/suggestions` (hybrid retrieval, rule-based adaptation, LLM ranking with validated swaps, fallback without the LLM) |
| Dietary management        | Implemented: diet profile (6 diets, the 14 EU allergens, excluded ingredients), keyword taxonomy over ingredient names, substitution table |
| Meal planning, nutrition, AI recipe generation | Not started (milestones 4 and 5)                  |
| Backend tests             | CRUD, chunking, ingestion, search, assistant, taxonomy, diet checker, fridge ranking and API, HTTP clients; SQLite by default, optionally Postgres |
| Frontend tests            | None yet                                                               |

```
.
├── backend/              FastAPI app (uv-managed, Python 3.12)
│   ├── app/
│   │   ├── api/          routers (ingredients, recipes, assistant, fridge, profile, health), deps, error handlers
│   │   ├── clients/      HTTP clients: Ollama (chat + embeddings), TheMealDB
│   │   ├── core/         settings (env) and domain exceptions
│   │   ├── crud/         database access, incl. the vector store query (`crud/chunk.py`)
│   │   ├── db/           declarative Base, async engine/session, `get_db` dependency
│   │   ├── models/       Ingredient, Recipe, RecipeIngredient, RecipeChunk (`embedding vector(768)`), FridgeItem, DietProfile
│   │   ├── schemas/      Pydantic request/response models
│   │   ├── services/     RAG (chunking, indexing, ingestion, recipe_search, assistant), llm protocols,
│   │   │                 dietary rules (taxonomy, dietary, substitutions), fridge (fridge_ranking, fridge)
│   │   └── cli.py        `python -m app.cli ingest-mealdb | reindex | llm-status | unclassified`
│   └── tests/            pytest suite
├── frontend/             React + Vite + TypeScript
├── db/init/              SQL run on first DB start (enables pgvector)
├── docker-compose.yml    db + backend + frontend
└── .env.example          configuration template (copy to `.env`)
```

## LLM setup (Ollama)

The backend talks only to the **local Ollama app** (`http://localhost:11434`; from Docker `http://host.docker.internal:11434`). No API key is needed.

| Role       | Default model                | Setting |
| ---------- | ---------------------------- | ------- |
| Chat       | `qwen3.5:4b`, thinking off   | `LLM_MODEL`, `LLM_THINK` |
| Embeddings | `nomic-embed-text` (768-dim) | `EMBEDDING_MODEL`, `EMBEDDING_DIM` |

```bash
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
make llm-status                 # or GET /health/llm
```

Any Ollama model works as `LLM_MODEL`. That includes `-cloud` models such as `gemma4:31b-cloud` (after `ollama signin`), but those need usage credits on the ollama.com account; without them chat fails with `402`, which the API reports as `503`. `LLM_THINK=true` turns on reasoning for models that support it. It's much slower on CPU.

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

On start-up the backend runs `CREATE EXTENSION IF NOT EXISTS vector` and creates the tables. Then load the knowledge base (about 300 recipes, takes a few minutes):

```bash
make seed                    # or: docker compose exec backend python -m app.cli ingest-mealdb
make seed ARGS="--limit 30"  # quicker sample
```

Re-running `make seed` is cheap. Recipes are upserted by `(source, external_id)` and only re-embedded when their text or the embedding model changed.

> **Upgrading from milestone 1:** the schema changed (new `recipe_chunks` table and new `recipes` columns) and there are no migrations yet. Run `make reset` (it deletes the database volume), then `make up` and `make seed`. Milestone 3 only adds tables (`fridge_items`, `diet_profiles`), which are created automatically on start-up.

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
| GET    | `/health/llm`           | Reachability of the chat/embedding servers and whether the models are installed |
| POST   | `/api/ingredients`      | Create an ingredient (names are trimmed + lowercased; `409` on duplicates) |
| GET    | `/api/ingredients`      | List ingredients                                         |
| POST   | `/api/recipes`          | Create a recipe (`ingredients: [{ingredient_id, quantity, unit}]`, optional `category`, `area`, `tags`); indexed right away (best effort) |
| GET    | `/api/recipes`          | List recipes (`skip`, `limit`)                           |
| GET    | `/api/recipes/{id}`     | Get one recipe                                           |
| POST   | `/api/recipes/search`   | Semantic search: `{query, limit, filters: {category, area}, min_score}` → recipes with their matching chunks |
| POST   | `/api/assistant/ask`    | RAG answer: `{question, max_recipes, filters}` → `{answer, model, sources}` |
| GET    | `/api/fridge/items`     | Fridge contents, soonest expiry first                    |
| POST   | `/api/fridge/items`     | Add `{name, quantity, unit, expires_on}`; unknown ingredients are created (`409` if already in the fridge) |
| PATCH  | `/api/fridge/items/{id}`| Change amount or expiry date                             |
| DELETE | `/api/fridge/items/{id}`| Remove an item                                           |
| POST   | `/api/fridge/suggestions` | Recipes for the fridge: `{query, limit, max_missing, ingredients?, profile?, use_llm}` → `{suggestions, model, llm_used, notes}` |
| GET    | `/api/profile`          | Diet profile: `{diets, allergens, excluded_ingredients}` |
| PUT    | `/api/profile`          | Replace the diet profile                                 |
| GET    | `/api/profile/options`  | Supported diets (with what they rule out) and allergens  |

LLM failures map to `503` (Ollama unreachable, model missing, no cloud credits) or `502` (invalid upstream response). The fridge suggestions are the exception: they fall back to the rule-based ranking and explain why in `notes`.

## RAG pipeline

```
TheMealDB ──► ingestion ──► chunking ──► embeddings (Ollama) ──► recipe_chunks (pgvector, HNSW)
                                                                          │
question ──► embed query ──► top-k chunks (cosine) ──► group by recipe ──►│ sources
                                    └──► full recipes as numbered context ──► qwen3.5:4b ──► answer
```

- **Ingestion** (`clients/themealdb.py`, `services/ingestion.py`): one `search.php?f=<letter>` request per letter. Free-text measures such as `1 1/2 cups` are parsed into quantity + unit, and ingredients are normalized and shared across recipes.
- **Chunking** (`services/chunking.py`): one *overview* chunk (title, category, cuisine, tags, ingredients) plus *instructions* chunks. The instructions are split on steps, packed up to `CHUNK_MAX_CHARS` and overlap by one step. Every chunk starts with the recipe title.
- **Indexing** (`services/indexing.py`): a recipe's chunks are rebuilt only when its text or the embedding model changed. `make reindex` catches up on recipes created while the embedder was down.
- **Retrieval** (`crud/chunk.py`, `services/recipe_search.py`): pgvector `<=>` on PostgreSQL. A recipe's score is its best chunk's cosine similarity. The SQLite test suite uses an exact numpy fallback.
- **Generation** (`services/assistant.py`): the retrieved recipes go into the prompt as numbered context. The model answers only from them, cites `[n]` and replies in the question's language. If nothing is retrieved, the LLM isn't called.
- The services depend on the `Embedder` / `ChatModel` protocols (`services/llm.py`), so the provider can be swapped and the tests use deterministic fakes.
- `nomic-embed-text` is English-centric, and TheMealDB is English. For non-English queries, consider a multilingual embedder such as `bge-m3` (`EMBEDDING_DIM=1024`, then `make reset && make seed`).

## Fridge suggestions ("what's in the fridge")

```
fridge items ──┬──► ingredient overlap (SQL) ──┐
   + wish      └──► semantic search (RAG) ─────┴──► candidates ──► adapt to profile ──► score ──► shortlist
                                                                   (substitution table)             │
                       validated suggestions ◄── re-check swaps, recompute coverage ◄── LLM ranking (JSON) ◄┘
```

- **Taxonomy** (`services/taxonomy.py`): ingredient names are tokenized and matched against curated phrases, longest first. A longer phrase claims its words, so "peanut butter" isn't dairy and "coconut milk" isn't milk; qualifiers such as "gluten free" or "vegan" remove tags. The rules were curated against TheMealDB's ~1000 ingredient names; `python -m app.cli unclassified` lists the stored ingredients no rule recognises, and suggestions report such ingredients as `unverified_ingredients` instead of assuming they're safe.
- **Diet checker** (`services/dietary.py`): diets map to forbidden tags (vegan forbids meat, fish, shellfish, animal by-products, milk, eggs and honey; keto forbids high-carb ingredients). Allergens are the 14 EU allergens. Excluded ingredients match as phrases ("mushroom" excludes "wild mushrooms").
- **Substitutions** (`services/substitutions.py`): a curated table of replacements, best first. Only replacements that pass the whole profile are offered, and the ones already in the fridge come first.
- **Shortlist** (`services/fridge_ranking.py`): candidates come from ingredient overlap and from semantic search over the fridge contents plus the optional wish. Each recipe is adapted (recipes with a violation the table can't fix are dropped), and recipes that use nothing from the fridge or need more than `max_missing` purchases are dropped. The score combines coverage (45%), share of the fridge used (20%), semantic similarity (20%) and share of soon-expiring items used (15%). Salt, pepper, oil and water count as staples; expired items are skipped.
- **LLM step** (`services/fridge.py`): the best `FRIDGE_LLM_CANDIDATES` recipes go into the prompt, with every ingredient annotated as in the fridge, missing, staple or *forbidden, with its allowed swaps*. The reply is constrained to a JSON schema (Ollama's `format`). Invented recipe numbers are ignored. A proposed swap is applied only if the replacement passes the checker, is recognised by the taxonomy, and either fixes a violation or uses something from the fridge; rule-based swaps cover whatever is left. Rejected output is listed in `notes`. If the LLM is unreachable or its JSON is invalid, the deterministic ranking is returned.
