import { useState, type SubmitEventHandler } from 'react'
import { createIngredient, type Ingredient } from '../api'

/** Props for {@link IngredientForm}. */
interface Props {
  /** Existing ingredients, shown as tags below the form. */
  ingredients: Ingredient[]
  /** Called after an ingredient was created, so the parent can refresh. */
  onCreated: () => void
}

/** Form for adding an ingredient, plus the list of existing ingredients. */
export function IngredientForm({ ingredients, onCreated }: Props) {
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  /** Create the ingredient and reset the form, or show the backend error. */
  const handleSubmit: SubmitEventHandler<HTMLFormElement> = async (event) => {
    event.preventDefault()
    setSubmitting(true)
    setError(null)
    try {
      await createIngredient(name)
      setName('')
      onCreated()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section>
      <h2>Ingredients</h2>
      <form onSubmit={handleSubmit} className="row">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. tomato"
          aria-label="Ingredient name"
          required
        />
        <button type="submit" disabled={submitting || name.trim() === ''}>
          Add ingredient
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {ingredients.length === 0 ? (
        <p className="muted">No ingredients yet.</p>
      ) : (
        <ul className="tags">
          {ingredients.map((ingredient) => (
            <li key={ingredient.id}>{ingredient.name}</li>
          ))}
        </ul>
      )}
    </section>
  )
}
