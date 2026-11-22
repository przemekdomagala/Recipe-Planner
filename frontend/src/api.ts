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
}

/** Recipe as returned by the API (`RecipeRead`). */
export interface Recipe {
  id: number
  title: string
  instructions: string
  /** `"user"` or the dataset the recipe was imported from, e.g. `"themealdb"`. */
  source: string
  category: string | null
  /** Cuisine, e.g. `"Italian"`. */
  area: string | null
  tags: string[]
  thumbnail_url: string | null
  source_url: string | null
  ingredients: RecipeIngredient[]
}

/** Payload for creating a recipe (`RecipeCreate`). */
export interface NewRecipe {
  title: string
  instructions: string
  ingredients: { ingredient_id: number }[]
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

/** Optional metadata filters for search and the assistant (`RecipeSearchFilters`). */
export interface SearchFilters {
  category?: string | null
  area?: string | null
}

/** A recipe chunk that matched a query (`MatchedChunk`). */
export interface MatchedChunk {
  kind: 'overview' | 'instructions'
  content: string
  /** Cosine similarity to the query (higher is better). */
  score: number
}

/** One recipe found by semantic search (`RecipeSearchHit`). */
export interface SearchHit {
  recipe_id: number
  title: string
  category: string | null
  area: string | null
  thumbnail_url: string | null
  /** Similarity of the best-matching chunk. */
  score: number
  matches: MatchedChunk[]
}

/** Semantic search results (`RecipeSearchResponse`). */
export interface SearchResponse {
  query: string
  results: SearchHit[]
}

/** The assistant's answer and the recipes it was grounded on (`AskResponse`). */
export interface AskResponse {
  question: string
  answer: string
  /** Chat model that wrote the answer; `null` when no recipe matched and no LLM ran. */
  model: string | null
  sources: SearchHit[]
}

/** Semantic recipe search over the knowledge base. */
export const searchRecipes = (
  query: string,
  limit = 6,
  filters: SearchFilters = {},
): Promise<SearchResponse> =>
  request('/api/recipes/search', {
    method: 'POST',
    body: JSON.stringify({ query, limit, filters }),
  })

/** Ask the RAG assistant a cooking question. */
export const askAssistant = (
  question: string,
  filters: SearchFilters = {},
): Promise<AskResponse> =>
  request('/api/assistant/ask', { method: 'POST', body: JSON.stringify({ question, filters }) })

/** The 14 major EU allergens (`Allergen`); milk is called `dairy`. */
export type Allergen =
  | 'gluten'
  | 'crustaceans'
  | 'eggs'
  | 'fish'
  | 'peanuts'
  | 'soy'
  | 'dairy'
  | 'tree_nuts'
  | 'celery'
  | 'mustard'
  | 'sesame'
  | 'sulphites'
  | 'lupin'
  | 'molluscs'

/** Dietary patterns that rule out whole food groups (`Diet`). */
export type Diet = 'vegetarian' | 'vegan' | 'pescatarian'

/** What's in the fridge plus the restrictions every suggestion must respect (`FridgeRequest`). */
export interface FridgeRequest {
  ingredients: string[]
  allergens?: Allergen[]
  diet?: Diet | null
  /** Other ingredients to keep out (dislikes, intolerances). */
  avoid?: string[]
  /** Free-text wish, e.g. "quick weeknight dinner". */
  note?: string | null
  /** 1–5, default 3. */
  max_suggestions?: number
  /** Count salt, pepper, oil, water and sugar as available (default true). */
  assume_staples?: boolean
  filters?: SearchFilters
}

/** A change to a recipe: an ingredient swap or removal, or a plain tip (`Adaptation`). */
export interface Adaptation {
  /** Recipe ingredient replaced; `null` for a plain tip. */
  replace: string | null
  /** What to use instead; `null` means leave it out. */
  substitute: string | null
  note: string
}

/** A constraint-checked recipe suggestion (`FridgeSuggestion`). */
export interface FridgeSuggestion {
  rank: number
  recipe_id: number
  title: string
  category: string | null
  area: string | null
  thumbnail_url: string | null
  /** Why it fits, written by the LLM; `null` when ranked without it. */
  reason: string | null
  used_ingredients: string[]
  missing_ingredients: string[]
  adaptations: Adaptation[]
  /** Share (0–1) of the adapted recipe's non-staple ingredients you have. */
  coverage: number
  similarity: number
}

/** A retrieved recipe dropped for breaking the restrictions (`ExcludedRecipe`). */
export interface ExcludedRecipe {
  recipe_id: number
  title: string
  /** E.g. `"butter: contains dairy"`. */
  reasons: string[]
}

/** Ranked fridge suggestions (`FridgeResponse`). */
export interface FridgeResponse {
  /** The normalized fridge contents. */
  ingredients: string[]
  suggestions: FridgeSuggestion[]
  excluded: ExcludedRecipe[]
  /** Chat model that ranked the suggestions; `null` if none did. */
  model: string | null
}

/** Ask for recipes that use what's in the fridge and respect the given restrictions. */
export const suggestFromFridge = (fridge: FridgeRequest): Promise<FridgeResponse> =>
  request('/api/fridge/suggestions', { method: 'POST', body: JSON.stringify(fridge) })
