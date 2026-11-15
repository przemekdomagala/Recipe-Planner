import { useState, type SubmitEventHandler } from 'react'
import { askAssistant, searchRecipes, type AskResponse, type SearchHit } from '../api'

/** What the panel last ran: a plain search or an assistant question. */
type Mode = 'search' | 'ask'

/** Result card for one retrieved recipe, with its best-matching chunk. */
function HitCard({ hit, index }: { hit: SearchHit; index?: number }) {
  const meta = [hit.category, hit.area].filter(Boolean).join(' · ')
  const best = hit.matches[0]
  return (
    <li className="hit">
      {hit.thumbnail_url && <img src={`${hit.thumbnail_url}/small`} alt="" loading="lazy" />}
      <div>
        <h3>
          {index !== undefined && <span className="muted">[{index}] </span>}
          {hit.title}
        </h3>
        <p className="muted">
          {meta && `${meta} · `}score {hit.score.toFixed(3)}
        </p>
        {best && <p className="snippet">{best.content.split('\n').slice(1).join(' ')}</p>}
      </div>
    </li>
  )
}

/** Natural-language recipe search and RAG question answering over the knowledge base. */
export function AssistantPanel() {
  const [text, setText] = useState('')
  const [mode, setMode] = useState<Mode>('ask')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hits, setHits] = useState<SearchHit[] | null>(null)
  const [answer, setAnswer] = useState<AskResponse | null>(null)

  /** Run the chosen mode for the current text. */
  const run = async (next: Mode) => {
    const query = text.trim()
    if (!query) return
    setMode(next)
    setLoading(true)
    setError(null)
    try {
      if (next === 'search') {
        setHits((await searchRecipes(query)).results)
        setAnswer(null)
      } else {
        setAnswer(await askAssistant(query))
        setHits(null)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  /** Enter in the input asks the assistant. */
  const handleSubmit: SubmitEventHandler<HTMLFormElement> = (event) => {
    event.preventDefault()
    void run('ask')
  }

  return (
    <section className="assistant">
      <h2>Ask the kitchen</h2>
      <form className="row" onSubmit={handleSubmit}>
        <input
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="e.g. Something quick with chicken and rice?"
          maxLength={500}
        />
        <button type="submit" disabled={loading || !text.trim()}>
          Ask AI
        </button>
        <button
          type="button"
          className="secondary"
          disabled={loading || !text.trim()}
          onClick={() => void run('search')}
        >
          Search
        </button>
      </form>

      {loading && (
        <p className="muted">{mode === 'ask' ? 'Retrieving recipes and asking the LLM…' : 'Searching…'}</p>
      )}
      {error && <p className="error">{error}</p>}

      {!loading && answer && (
        <div className="answer">
          <p>{answer.answer}</p>
          {answer.model && <p className="muted">Answered by {answer.model}</p>}
          {answer.sources.length > 0 && (
            <>
              <h3>Sources</h3>
              <ul className="hits">
                {answer.sources.map((hit, i) => (
                  <HitCard key={hit.recipe_id} hit={hit} index={i + 1} />
                ))}
              </ul>
            </>
          )}
        </div>
      )}

      {!loading && hits && (
        hits.length === 0 ? (
          <p className="muted">No matching recipes. Did you run `make seed`?</p>
        ) : (
          <ul className="hits">
            {hits.map((hit) => (
              <HitCard key={hit.recipe_id} hit={hit} />
            ))}
          </ul>
        )
      )}
    </section>
  )
}
