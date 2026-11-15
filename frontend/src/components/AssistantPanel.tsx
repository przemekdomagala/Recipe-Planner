import { useEffect, useState, type SubmitEventHandler } from 'react'
import {
  askAssistant,
  getRagStatus,
  searchRecipes,
  type AskResponse,
  type RagStatus,
  type SearchHit,
} from '../api'

/** What the last submitted query produced: search hits or a generated answer. */
type Result = { kind: 'search'; hits: SearchHit[] } | { kind: 'ask'; response: AskResponse }

/** One-line summary of the knowledge base and LLM readiness. */
function StatusLine({ status }: { status: RagStatus | null }) {
  if (status === null) return <p className="muted">Loading knowledge base status…</p>
  return (
    <p className="muted">
      {status.indexed_recipes}/{status.total_recipes} recipes indexed ({status.chunks} chunks) ·
      models {status.embedding_model} + {status.chat_model} ·{' '}
      {status.llm_available ? 'LLM ready' : <span className="error">{status.detail}</span>}
    </p>
  )
}

/** List of retrieved recipes with their similarity scores. */
function HitList({ hits }: { hits: SearchHit[] }) {
  if (hits.length === 0) return <p className="muted">No matching recipes.</p>
  return (
    <ul className="cards">
      {hits.map((hit) => (
        <li key={hit.recipe_id}>
          <h3>
            {hit.title} <span className="muted">({hit.score.toFixed(2)})</span>
          </h3>
          <p className="muted">{[hit.category, hit.area].filter(Boolean).join(' · ')}</p>
        </li>
      ))}
    </ul>
  )
}

/**
 * Minimal UI for the RAG pipeline: one query box that either runs a semantic search
 * or asks the LLM, which answers from the retrieved recipes.
 */
export function AssistantPanel() {
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState<RagStatus | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    getRagStatus()
      .then(setStatus)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : 'Unknown error'))
  }, [])

  /** Run the action of the clicked submit button (`search` or `ask`) for the query. */
  const handleSubmit: SubmitEventHandler<HTMLFormElement> = async (event) => {
    event.preventDefault()
    const submitter = (event.nativeEvent as SubmitEvent).submitter as HTMLButtonElement | null
    setBusy(true)
    setError(null)
    try {
      if (submitter?.value === 'ask') {
        setResult({ kind: 'ask', response: await askAssistant(query) })
      } else {
        setResult({ kind: 'search', hits: (await searchRecipes(query)).results })
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <section>
      <h2>Recipe assistant</h2>
      <StatusLine status={status} />
      <form onSubmit={handleSubmit} className="row">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. What can I cook with chicken and rice?"
          aria-label="Question or search query"
          required
        />
        <button type="submit" value="search" disabled={busy || query.trim() === ''}>
          Search
        </button>
        <button type="submit" value="ask" disabled={busy || query.trim() === ''}>
          Ask AI
        </button>
      </form>
      {busy && <p className="muted">Working… (the first AI answer can take a minute)</p>}
      {error && <p className="error">{error}</p>}
      {result?.kind === 'search' && <HitList hits={result.hits} />}
      {result?.kind === 'ask' && (
        <>
          <p className="answer">{result.response.answer}</p>
          {result.response.model && (
            <p className="muted">Answered by {result.response.model}, based on:</p>
          )}
          <HitList hits={result.response.sources} />
        </>
      )}
    </section>
  )
}
