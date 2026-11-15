import { useCallback, useEffect, useState } from 'react'
import { listIngredients, listRecipes, type Ingredient, type Recipe } from './api'
import { IngredientForm } from './components/IngredientForm'
import { RecipeForm } from './components/RecipeForm'
import { RecipeList } from './components/RecipeList'

/** Root component: loads ingredients and recipes and renders the forms and the list. */
function App() {
  const [ingredients, setIngredients] = useState<Ingredient[]>([])
  const [recipes, setRecipes] = useState<Recipe[]>([])
  const [error, setError] = useState<string | null>(null)

  /** Reload ingredients and recipes from the backend, surfacing any error. */
  const refresh = useCallback(async () => {
    try {
      const [loadedIngredients, loadedRecipes] = await Promise.all([
        listIngredients(),
        listRecipes(),
      ])
      setIngredients(loadedIngredients)
      setRecipes(loadedRecipes)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not reach the backend')
    }
  }, [])

  useEffect(() => {
    // State is only set after the awaited fetch resolves, not synchronously.
    // oxlint-disable-next-line react/set-state-in-effect
    void refresh()
  }, [refresh])

  return (
    <main>
      <h1>Recipe Planner</h1>
      {error && <p className="error">Backend error: {error}</p>}
      <div className="grid">
        <div>
          <IngredientForm ingredients={ingredients} onCreated={refresh} />
          <RecipeForm ingredients={ingredients} onCreated={refresh} />
        </div>
        <RecipeList recipes={recipes} />
      </div>
    </main>
  )
}

export default App
