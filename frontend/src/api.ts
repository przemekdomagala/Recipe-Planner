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

/** An ingredient in the fridge (`FridgeItemRead`). */
export interface FridgeItem {
  id: number
  ingredient: Ingredient
  quantity: number | null
  unit: string | null
  /** ISO date (`YYYY-MM-DD`). */
  expires_on: string | null
  added_at: string
}

/** Payload for adding an ingredient to the fridge (`FridgeItemCreate`). */
export interface NewFridgeItem {
  name: string
  quantity?: number | null
  unit?: string | null
  expires_on?: string | null
}

/** Dietary constraints (`DietProfileData`). */
export interface DietProfile {
  diets: string[]
  allergens: string[]
  excluded_ingredients: string[]
}

/** A selectable diet or allergen (`LabeledOption` / `DietOption`). */
export interface ProfileOption {
  value: string
  label: string
  /** For diets: the ingredient properties the diet rules out. */
  forbids?: string[]
}

/** The diets and allergens a profile can contain (`ProfileOptions`). */
export interface ProfileOptions {
  diets: ProfileOption[]
  allergens: ProfileOption[]
}

/** An ingredient swap applied to a suggestion (`Substitution`). */
export interface Substitution {
  original: string
  replacement: string
  reason: string
  /** `"llm"` if the model proposed it (and it passed validation), else `"rules"`. */
  source: 'llm' | 'rules'
}

/** A recipe suggested for the fridge contents (`FridgeSuggestion`). */
export interface FridgeSuggestion {
  recipe_id: number
  title: string
  category: string | null
  area: string | null
  thumbnail_url: string | null
  score: number
  /** Share of the non-staple ingredients already at hand, 0 to 1. */
  coverage: number
  used_ingredients: string[]
  missing_ingredients: string[]
  expiring_used: string[]
  substitutions: Substitution[]
  unverified_ingredients: string[]
  reason: string | null
}

/** Fridge suggestions and how they were produced (`SuggestionResponse`). */
export interface SuggestionResponse {
  suggestions: FridgeSuggestion[]
  model: string | null
  llm_used: boolean
  notes: string[]
}

/** Options for {@link suggestRecipes} (`SuggestionRequest`). */
export interface SuggestionOptions {
  query?: string | null
  limit?: number
  max_missing?: number
  use_llm?: boolean
}

/** Fetch the fridge contents, soonest expiry first. */
export const listFridgeItems = (): Promise<FridgeItem[]> => request('/api/fridge/items')

/** Put an ingredient into the fridge; unknown ingredients are created. */
export const addFridgeItem = (item: NewFridgeItem): Promise<FridgeItem> =>
  request('/api/fridge/items', { method: 'POST', body: JSON.stringify(item) })

/** Remove an item from the fridge. */
export async function deleteFridgeItem(id: number): Promise<void> {
  const response = await fetch(`${API_URL}/api/fridge/items/${id}`, { method: 'DELETE' })
  if (!response.ok) throw new Error(response.statusText)
}

/** Fetch the diet profile. */
export const getProfile = (): Promise<DietProfile> => request('/api/profile')

/** Replace the diet profile. */
export const saveProfile = (profile: DietProfile): Promise<DietProfile> =>
  request('/api/profile', { method: 'PUT', body: JSON.stringify(profile) })

/** Fetch the supported diets and allergens. */
export const getProfileOptions = (): Promise<ProfileOptions> => request('/api/profile/options')

/** Suggest recipes for the stored fridge and diet profile. */
export const suggestRecipes = (options: SuggestionOptions = {}): Promise<SuggestionResponse> =>
  request('/api/fridge/suggestions', { method: 'POST', body: JSON.stringify(options) })

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
