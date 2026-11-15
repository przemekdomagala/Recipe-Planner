/**
 * Typed client for the Recipe Planner backend API.
 *
 * The interfaces below mirror the backend's Pydantic schemas; keep them in sync.
 */

/** Ingredient as returned by the API (`IngredientRead`). */
export interface Ingredient {
  id: number
  name: string
}

/** One ingredient line of a recipe, with an optional amount (`RecipeIngredientRead`). */
export interface RecipeIngredient {
  ingredient: Ingredient
  quantity: number | null
  unit: string | null
  /** Original free-text amount of imported recipes, e.g. "2 cloves minced". */
  measure: string | null
}

/** Recipe as returned by the API (`RecipeRead`). */
export interface Recipe {
  id: number
  title: string
  instructions: string
  /** `"user"` or the external source, e.g. `"themealdb"`. */
  source: string
  external_id: string | null
  category: string | null
  area: string | null
  tags: string[]
  image_url: string | null
  source_url: string | null
  ingredients: RecipeIngredient[]
}

/** Payload for creating a recipe (`RecipeCreate`). */
export interface NewRecipe {
  title: string
  instructions: string
  ingredients: { ingredient_id: number }[]
}

/** A recipe matched by semantic search (`RecipeSearchHit`). */
export interface SearchHit {
  recipe_id: number
  title: string
  category: string | null
  area: string | null
  image_url: string | null
  /** Cosine similarity of the best-matching chunk (higher is better). */
  score: number
  snippet: string
}

/** Semantic search results (`RecipeSearchResponse`). */
export interface SearchResponse {
  query: string
  results: SearchHit[]
}

/** A generated answer with the recipes it is grounded on (`AskResponse`). */
export interface AskResponse {
  question: string
  answer: string
  /** Chat model used, or `null` when no recipe matched and no model was called. */
  model: string | null
  sources: SearchHit[]
}

/** Readiness of the knowledge base and the LLM (`RagStatus`). */
export interface RagStatus {
  total_recipes: number
  indexed_recipes: number
  chunks: number
  embedding_model: string
  chat_model: string
  llm_available: boolean
  missing_models: string[]
  detail: string | null
}

/** Backend base URL; Vite inlines `VITE_API_URL` at build time. */
const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

/**
 * Send a JSON request to the backend and parse the JSON response.
 *
 * @param path - API path starting with `/`, e.g. `/api/recipes`.
 * @param init - Extra `fetch` options (method, body, headers).
 * @returns The parsed response body.
 * @throws Error carrying the backend's `detail` message (or the status text) on non-2xx responses.
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body: unknown = await response.json()
      if (typeof body === 'object' && body !== null && 'detail' in body) {
        const raw = (body as { detail: unknown }).detail
        detail = typeof raw === 'string' ? raw : JSON.stringify(raw)
      }
    } catch {
      // Response had no JSON body; keep the status text.
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

/** Fetch all ingredients, sorted by name. */
export const listIngredients = (): Promise<Ingredient[]> => request('/api/ingredients')

/** Create an ingredient; the backend normalizes the name and rejects duplicates. */
export const createIngredient = (name: string): Promise<Ingredient> =>
  request('/api/ingredients', { method: 'POST', body: JSON.stringify({ name }) })

/** Fetch recipes in creation order. */
export const listRecipes = (): Promise<Recipe[]> => request('/api/recipes')

/** Create a recipe with links to existing ingredients. */
export const createRecipe = (recipe: NewRecipe): Promise<Recipe> =>
  request('/api/recipes', { method: 'POST', body: JSON.stringify(recipe) })

/** Find the recipes semantically closest to a natural-language query. */
export const searchRecipes = (query: string, limit = 5): Promise<SearchResponse> =>
  request('/api/recipes/search', { method: 'POST', body: JSON.stringify({ query, limit }) })

/** Ask the LLM a question answered from the recipe knowledge base. */
export const askAssistant = (question: string): Promise<AskResponse> =>
  request('/api/rag/ask', { method: 'POST', body: JSON.stringify({ question }) })

/** Fetch how many recipes are indexed and whether the LLM is ready. */
export const getRagStatus = (): Promise<RagStatus> => request('/api/rag/status')
