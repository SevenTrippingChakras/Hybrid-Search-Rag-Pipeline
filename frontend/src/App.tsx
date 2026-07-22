import { Database } from "lucide-react"
import { Documents } from "./pages/Documents"
import { Ask } from "./pages/Ask"

function App() {
  return (
    <div className="mx-auto flex h-screen max-w-6xl flex-col p-6">
      <header className="mb-6 flex items-center gap-3">
        <div className="flex size-9 items-center justify-center rounded-lg bg-primary/15 text-primary">
          <Database className="size-5" />
        </div>
        <div>
          <h1 className="text-xl font-bold leading-tight">Hybrid RAG</h1>
          <p className="text-xs text-muted">Upload documents · ask grounded questions</p>
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-6 md:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
        <section className="min-h-0 rounded-xl border border-border bg-card/40 p-5">
          <Documents />
        </section>
        <section className="min-h-0 rounded-xl border border-border bg-card/40 p-5">
          <Ask />
        </section>
      </div>
    </div>
  )
}

export default App
