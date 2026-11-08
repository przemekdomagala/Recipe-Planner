import { useState, type SubmitEventHandler } from 'react'
import { createIngredient, type Ingredient } from '../api'

interface Props {
  ingredients: Ingredient[]
  onCreated: () => void
}

export function IngredientForm({ ingredients, onCreated }: Props) {
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

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
