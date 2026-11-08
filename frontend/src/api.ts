export interface Ingredient {
  id: number
  name: string
}

export interface RecipeIngredient {
  ingredient: Ingredient
  quantity: number | null
  unit: string | null
}

export interface Recipe {
  id: number
  title: string
  instructions: string
  ingredients: RecipeIngredient[]
}

export interface NewRecipe {
  title: string
  instructions: string
  ingredients: { ingredient_id: number }[]
}

const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

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

export const listIngredients = (): Promise<Ingredient[]> => request('/api/ingredients')

export const createIngredient = (name: string): Promise<Ingredient> =>
  request('/api/ingredients', { method: 'POST', body: JSON.stringify({ name }) })

export const listRecipes = (): Promise<Recipe[]> => request('/api/recipes')

export const createRecipe = (recipe: NewRecipe): Promise<Recipe> =>
  request('/api/recipes', { method: 'POST', body: JSON.stringify(recipe) })
