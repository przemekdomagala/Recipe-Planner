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
