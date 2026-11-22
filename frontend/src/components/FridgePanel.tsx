import { useState, type SubmitEventHandler } from 'react'
import {
  suggestFromFridge,
  type Adaptation,
  type Allergen,
  type Diet,
  type FridgeResponse,
  type FridgeSuggestion,
} from '../api'

/** Allergen options in display order, with their labels. */
const ALLERGENS: { value: Allergen; label: string }[] = [
  { value: 'gluten', label: 'Gluten' },
  { value: 'dairy', label: 'Dairy' },
  { value: 'eggs', label: 'Eggs' },
  { value: 'peanuts', label: 'Peanuts' },
  { value: 'tree_nuts', label: 'Tree nuts' },
  { value: 'soy', label: 'Soy' },
  { value: 'fish', label: 'Fish' },
  { value: 'crustaceans', label: 'Crustaceans' },
  { value: 'molluscs', label: 'Molluscs' },
  { value: 'sesame', label: 'Sesame' },
  { value: 'celery', label: 'Celery' },
  { value: 'mustard', label: 'Mustard' },
  { value: 'sulphites', label: 'Sulphites' },
  { value: 'lupin', label: 'Lupin' },
]

/** Diet options; the empty value means no diet. */
const DIETS: { value: Diet | ''; label: string }[] = [
  { value: '', label: 'No diet' },
  { value: 'vegetarian', label: 'Vegetarian' },
  { value: 'vegan', label: 'Vegan' },
  { value: 'pescatarian', label: 'Pescatarian' },
]

/** Split a comma-separated list into trimmed, non-empty items. */
const splitList = (text: string): string[] =>
  text
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)

/** One adaptation as a sentence, e.g. "butter → olive oil: Sear in oil." */
function AdaptationLine({ adaptation }: { adaptation: Adaptation }) {
  const { replace, substitute, note } = adaptation
  const change = replace ? `${replace} → ${substitute ?? 'leave out'}` : null
  return (
    <li>
      {change && <strong>{change}</strong>}
      {change && note && ': '}
      {note}
    </li>
  )
}

/** Card for one suggestion: why it fits, what you have, what's missing and how to adapt it. */
function SuggestionCard({ suggestion }: { suggestion: FridgeSuggestion }) {
  const meta = [suggestion.category, suggestion.area].filter(Boolean).join(' · ')
  return (
    <li className="suggestion">
      {suggestion.thumbnail_url && (
        <img src={`${suggestion.thumbnail_url}/small`} alt="" loading="lazy" />
      )}
      <div>
        <h3>
          <span className="muted">#{suggestion.rank} </span>
          {suggestion.title}
        </h3>
        <p className="muted">
          {meta && `${meta} · `}you have {Math.round(suggestion.coverage * 100)}% of it
        </p>
        {suggestion.reason && <p>{suggestion.reason}</p>}
        {suggestion.used_ingredients.length > 0 && (
          <p>
            <span className="muted">Have: </span>
            {suggestion.used_ingredients.join(', ')}
          </p>
        )}
        {suggestion.missing_ingredients.length > 0 && (
          <p>
            <span className="muted">Need: </span>
            {suggestion.missing_ingredients.join(', ')}
          </p>
        )}
        {suggestion.adaptations.length > 0 && (
          <ul className="adaptations">
            {suggestion.adaptations.map((adaptation, i) => (
              <AdaptationLine key={i} adaptation={adaptation} />
            ))}
          </ul>
        )}
      </div>
    </li>
  )
}

/**
 * "What's in the fridge": ranked recipe suggestions for the listed ingredients, adapted by
 * the LLM and checked against the selected allergens, diet and avoided ingredients.
 */
export function FridgePanel() {
  const [fridge, setFridge] = useState('')
  const [allergens, setAllergens] = useState<Set<Allergen>>(new Set())
  const [diet, setDiet] = useState<Diet | ''>('')
  const [avoid, setAvoid] = useState('')
  const [note, setNote] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<FridgeResponse | null>(null)

  /** Add or remove an allergen from the selection. */
  const toggleAllergen = (allergen: Allergen) =>
    setAllergens((prev) => {
      const next = new Set(prev)
      if (next.has(allergen)) next.delete(allergen)
      else next.add(allergen)
      return next
    })

  /** Request suggestions for the current fridge and restrictions. */
  const handleSubmit: SubmitEventHandler<HTMLFormElement> = async (event) => {
    event.preventDefault()
    const ingredients = splitList(fridge)
    if (ingredients.length === 0) return
    setLoading(true)
    setError(null)
    try {
      setResult(
        await suggestFromFridge({
          ingredients,
          allergens: [...allergens],
          diet: diet || null,
          avoid: splitList(avoid),
          note: note.trim() || null,
        }),
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="assistant">
      <h2>What's in the fridge?</h2>
      <form className="column" onSubmit={handleSubmit}>
        <input
          value={fridge}
          onChange={(event) => setFridge(event.target.value)}
          placeholder="Ingredients, comma-separated: chicken, rice, spinach, garlic"
          aria-label="Fridge ingredients"
        />
        <fieldset className="allergens">
          <legend className="muted">Allergies</legend>
          {ALLERGENS.map(({ value, label }) => (
            <label key={value} className="check">
              <input
                type="checkbox"
                checked={allergens.has(value)}
                onChange={() => toggleAllergen(value)}
              />
              {label}
            </label>
          ))}
        </fieldset>
        <div className="row">
          <select
            value={diet}
            onChange={(event) => setDiet(event.target.value as Diet | '')}
            aria-label="Diet"
          >
            {DIETS.map(({ value, label }) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
          <input
            value={avoid}
            onChange={(event) => setAvoid(event.target.value)}
            placeholder="Avoid (comma-separated): mushroom, coriander"
            aria-label="Ingredients to avoid"
          />
        </div>
        <div className="row">
          <input
            value={note}
            onChange={(event) => setNote(event.target.value)}
            placeholder="Anything else? e.g. quick weeknight dinner"
            maxLength={300}
            aria-label="Note"
          />
          <button type="submit" disabled={loading || splitList(fridge).length === 0}>
            Suggest recipes
          </button>
        </div>
      </form>

      {loading && <p className="muted">Finding recipes and asking the LLM…</p>}
      {error && <p className="error">{error}</p>}

      {!loading && result && (
        <>
          {result.suggestions.length === 0 ? (
            <p className="muted">
              No recipe fits this fridge and these restrictions. Did you run `make seed`?
            </p>
          ) : (
            <ul className="suggestions">
              {result.suggestions.map((suggestion) => (
                <SuggestionCard key={suggestion.recipe_id} suggestion={suggestion} />
              ))}
            </ul>
          )}
          <p className="muted small">
            {result.model ? `Ranked by ${result.model}. ` : 'Ranked without the LLM. '}
            Restrictions are checked against ingredient names; always read the labels.
          </p>
          {result.excluded.length > 0 && (
            <details>
              <summary className="muted">
                {result.excluded.length} recipe(s) left out for your restrictions
              </summary>
              <ul>
                {result.excluded.map((recipe) => (
                  <li key={recipe.recipe_id}>
                    {recipe.title} <span className="muted">— {recipe.reasons.join('; ')}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}
    </section>
  )
}
