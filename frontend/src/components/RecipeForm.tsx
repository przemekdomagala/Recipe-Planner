import { useState, type SubmitEventHandler } from 'react'
import { createRecipe, type Ingredient } from '../api'

/** Props for {@link RecipeForm}. */
interface Props {
  /** Ingredients the user can attach to the new recipe. */
  ingredients: Ingredient[]
  /** Called after a recipe was created, so the parent can refresh. */
  onCreated: () => void
}

/** Form for creating a recipe and picking its ingredients. */
export function RecipeForm({ ingredients, onCreated }: Props) {
  const [title, setTitle] = useState('')
  const [instructions, setInstructions] = useState('')
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  /** Add or remove an ingredient id from the selection. */
  const toggle = (id: number) =>
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  /** Create the recipe and reset the form, or show the backend error. */
  const handleSubmit: SubmitEventHandler<HTMLFormElement> = async (event) => {
    event.preventDefault()
    setSubmitting(true)
    setError(null)
    try {
      await createRecipe({
        title,
        instructions,
        ingredients: [...selected].map((ingredient_id) => ({ ingredient_id })),
      })
      setTitle('')
      setInstructions('')
      setSelected(new Set())
      onCreated()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section>
      <h2>Add a recipe</h2>
      <form onSubmit={handleSubmit} className="column">
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Title"
          aria-label="Recipe title"
          required
        />
        <textarea
          value={instructions}
          onChange={(e) => setInstructions(e.target.value)}
          placeholder="Instructions"
          aria-label="Recipe instructions"
          rows={3}
          required
        />
        {ingredients.length > 0 && (
          <fieldset>
            <legend>Ingredients</legend>
            {ingredients.map((ingredient) => (
              <label key={ingredient.id} className="check">
                <input
                  type="checkbox"
                  checked={selected.has(ingredient.id)}
                  onChange={() => toggle(ingredient.id)}
                />
                {ingredient.name}
              </label>
            ))}
          </fieldset>
        )}
        <button type="submit" disabled={submitting}>
          Create recipe
        </button>
      </form>
      {error && <p className="error">{error}</p>}
    </section>
  )
}
