import { useCallback, useEffect, useState, type SubmitEventHandler } from 'react'
import {
  addFridgeItem,
  deleteFridgeItem,
  getProfile,
  getProfileOptions,
  listFridgeItems,
  saveProfile,
  suggestRecipes,
  type DietProfile,
  type FridgeItem,
  type FridgeSuggestion,
  type ProfileOptions,
  type SuggestionResponse,
} from '../api'

const EMPTY_PROFILE: DietProfile = { diets: [], allergens: [], excluded_ingredients: [] }

/** Days from today until an ISO date (negative once it has passed). */
function daysUntil(isoDate: string): number {
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  return Math.round((new Date(`${isoDate}T00:00:00`).getTime() - today.getTime()) / 86_400_000)
}

/** Short label for an expiry date, e.g. "expires in 2 days" or "expired". */
function expiryLabel(isoDate: string): string {
  const days = daysUntil(isoDate)
  if (days < 0) return 'expired'
  if (days === 0) return 'expires today'
  return days === 1 ? 'expires tomorrow' : `expires in ${days} days`
}

/** Return `values` with `value` added or removed. */
function toggle(values: string[], value: string): string[] {
  return values.includes(value) ? values.filter((v) => v !== value) : [...values, value]
}

/** Result card for one fridge suggestion. */
function SuggestionCard({ suggestion }: { suggestion: FridgeSuggestion }) {
  const meta = [suggestion.category, suggestion.area].filter(Boolean).join(' · ')
  return (
    <li className="hit suggestion">
      {suggestion.thumbnail_url && (
        <img src={`${suggestion.thumbnail_url}/small`} alt="" loading="lazy" />
      )}
      <div>
        <h3>{suggestion.title}</h3>
        <p className="muted">
          {meta && `${meta} · `}
          {Math.round(suggestion.coverage * 100)}% at hand
        </p>
        {suggestion.reason && <p>{suggestion.reason}</p>}
        <p>
          <strong>Have:</strong> {suggestion.used_ingredients.join(', ') || '–'}
        </p>
        {suggestion.missing_ingredients.length > 0 && (
          <p>
            <strong>Buy:</strong> {suggestion.missing_ingredients.join(', ')}
          </p>
        )}
        {suggestion.expiring_used.length > 0 && (
          <p className="soon">Uses up soon: {suggestion.expiring_used.join(', ')}</p>
        )}
        {suggestion.substitutions.length > 0 && (
          <ul className="swaps">
            {suggestion.substitutions.map((swap) => (
              <li key={swap.original}>
                {swap.original} → <strong>{swap.replacement}</strong>{' '}
                <span className="muted">
                  ({swap.reason}
                  {swap.source === 'llm' ? ', AI' : ''})
                </span>
              </li>
            ))}
          </ul>
        )}
        {suggestion.unverified_ingredients.length > 0 && (
          <p className="error">
            Check yourself (unknown to the diet rules): {suggestion.unverified_ingredients.join(', ')}
          </p>
        )}
      </div>
    </li>
  )
}

/** "What's in the fridge": fridge contents, diet profile and recipe suggestions. */
export function FridgePanel() {
  const [items, setItems] = useState<FridgeItem[]>([])
  const [profile, setProfile] = useState<DietProfile>(EMPTY_PROFILE)
  const [options, setOptions] = useState<ProfileOptions | null>(null)
  const [name, setName] = useState('')
  const [expiresOn, setExpiresOn] = useState('')
  const [excluded, setExcluded] = useState('')
  const [wish, setWish] = useState('')
  const [useLlm, setUseLlm] = useState(true)
  const [result, setResult] = useState<SuggestionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [profileSaved, setProfileSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)

  /** Load the fridge, the profile and the profile options. */
  const load = useCallback(async () => {
    try {
      const [loadedItems, loadedProfile, loadedOptions] = await Promise.all([
        listFridgeItems(),
        getProfile(),
        getProfileOptions(),
      ])
      setItems(loadedItems)
      setProfile(loadedProfile)
      setExcluded(loadedProfile.excluded_ingredients.join(', '))
      setOptions(loadedOptions)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load the fridge')
    }
  }, [])

  useEffect(() => {
    // State is only set after the awaited fetch resolves, not synchronously.
    // oxlint-disable-next-line react/set-state-in-effect
    void load()
  }, [load])

  /** Run an action, showing its error and reloading the fridge afterwards. */
  const run = async (action: () => Promise<unknown>) => {
    setError(null)
    try {
      await action()
      setItems(await listFridgeItems())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed')
    }
  }

  /** Add the typed ingredient to the fridge. */
  const handleAdd: SubmitEventHandler<HTMLFormElement> = async (event) => {
    event.preventDefault()
    await run(() => addFridgeItem({ name, expires_on: expiresOn || null }))
    setName('')
    setExpiresOn('')
  }

  /** Save the profile with the excluded ingredients from the text field. */
  const handleSaveProfile = async () => {
    const next = {
      ...profile,
      excluded_ingredients: excluded.split(',').map((s) => s.trim()).filter(Boolean),
    }
    await run(async () => {
      setProfile(await saveProfile(next))
      setProfileSaved(true)
    })
  }

  /** Ask for suggestions for the stored fridge and profile. */
  const handleSuggest = async () => {
    setLoading(true)
    setError(null)
    try {
      setResult(await suggestRecipes({ query: wish || null, use_llm: useLlm }))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="assistant fridge">
      <h2>What's in the fridge</h2>

      <div className="grid">
        <div>
          <h3>Fridge</h3>
          <form className="row" onSubmit={handleAdd}>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. chicken thighs"
              aria-label="Ingredient"
            />
            <input
              type="date"
              value={expiresOn}
              onChange={(e) => setExpiresOn(e.target.value)}
              aria-label="Expires on"
              className="date"
            />
            <button type="submit" disabled={!name.trim()}>
              Add
            </button>
          </form>
          {items.length === 0 ? (
            <p className="muted">The fridge is empty.</p>
          ) : (
            <ul className="tags fridge-items">
              {items.map((item) => (
                <li
                  key={item.id}
                  className={
                    item.expires_on && daysUntil(item.expires_on) <= 3 ? 'expiring' : undefined
                  }
                >
                  {item.ingredient.name}
                  {item.expires_on && (
                    <span className="muted"> · {expiryLabel(item.expires_on)}</span>
                  )}
                  <button
                    type="button"
                    className="remove"
                    aria-label={`Remove ${item.ingredient.name}`}
                    onClick={() => void run(() => deleteFridgeItem(item.id))}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div>
          <h3>Diet profile</h3>
          {options && (
            <>
              <p className="muted">Diets</p>
              <div>
                {options.diets.map((diet) => (
                  <label key={diet.value} className="check" title={`Rules out: ${diet.forbids?.join(', ')}`}>
                    <input
                      type="checkbox"
                      checked={profile.diets.includes(diet.value)}
                      onChange={() => {
                        setProfile({ ...profile, diets: toggle(profile.diets, diet.value) })
                        setProfileSaved(false)
                      }}
                    />
                    {diet.label}
                  </label>
                ))}
              </div>
              <p className="muted">Allergies</p>
              <div>
                {options.allergens.map((allergen) => (
                  <label key={allergen.value} className="check">
                    <input
                      type="checkbox"
                      checked={profile.allergens.includes(allergen.value)}
                      onChange={() => {
                        setProfile({
                          ...profile,
                          allergens: toggle(profile.allergens, allergen.value),
                        })
                        setProfileSaved(false)
                      }}
                    />
                    {allergen.label}
                  </label>
                ))}
              </div>
            </>
          )}
          <div className="row">
            <input
              value={excluded}
              onChange={(e) => {
                setExcluded(e.target.value)
                setProfileSaved(false)
              }}
              placeholder="Never use, e.g. mushroom, coriander"
              aria-label="Excluded ingredients"
            />
            <button type="button" className="secondary" onClick={() => void handleSaveProfile()}>
              {profileSaved ? 'Saved' : 'Save profile'}
            </button>
          </div>
        </div>
      </div>

      <div className="row">
        <input
          value={wish}
          onChange={(e) => setWish(e.target.value)}
          placeholder="Optional wish, e.g. something quick and spicy"
          maxLength={300}
          aria-label="Wish"
        />
        <label className="check">
          <input type="checkbox" checked={useLlm} onChange={() => setUseLlm(!useLlm)} />
          AI ranking
        </label>
        <button type="button" disabled={loading || items.length === 0} onClick={() => void handleSuggest()}>
          Suggest recipes
        </button>
      </div>

      {error && <p className="error">{error}</p>}
      {loading && <p className="muted">Matching recipes to your fridge…</p>}
      {!loading && result && (
        <>
          {result.notes.map((note) => (
            <p key={note} className="muted">
              {note}
            </p>
          ))}
          {result.model && <p className="muted">Ranked by {result.model}</p>}
          <ul className="hits">
            {result.suggestions.map((suggestion) => (
              <SuggestionCard key={suggestion.recipe_id} suggestion={suggestion} />
            ))}
          </ul>
        </>
      )}
    </section>
  )
}
