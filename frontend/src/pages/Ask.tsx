import { useState } from "react"
import { Loader2, Search, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import type { AskResponse, Confidence } from "../types"
import { askQuestion, splitAnswer } from "./Ask.helper"

const CONFIDENCE_DIMS: { key: keyof Confidence; label: string }[] = [
  { key: "retrieval", label: "Retrieval" },
  { key: "citation_coverage", label: "Citations" },
  { key: "completeness", label: "Completeness" },
  { key: "score", label: "Overall" },
]

function ConfidenceBars({ confidence }: { confidence: Confidence }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      {CONFIDENCE_DIMS.map(({ key, label }) => {
        const pct = Math.round(confidence[key] * 100)
        return (
          <div key={key}>
            <div className="flex justify-between text-xs">
              <span className="text-muted">{label}</span>
              <span className="text-faint">{pct}%</span>
            </div>
            <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-card2">
              <div
                className={cn("h-full rounded-full", key === "score" ? "bg-primary" : "bg-accent")}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function Ask() {
  const [question, setQuestion] = useState("")
  const [result, setResult] = useState<AskResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeCite, setActiveCite] = useState<number | null>(null)

  const onAsk = async () => {
    if (!question.trim() || loading) return
    setError(null)
    setLoading(true)
    setActiveCite(null)
    try {
      setResult(await askQuestion(question.trim()))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-full flex-col gap-4">
      <h2 className="text-lg font-semibold">Ask</h2>

      <div className="flex gap-2">
        <Input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onAsk()}
          placeholder="Ask a question about your documents…"
        />
        <Button onClick={onAsk} disabled={loading || !question.trim()}>
          {loading ? <Loader2 className="size-4 animate-spin" /> : <Search className="size-4" />}
          Ask
        </Button>
      </div>

      {error && <p className="text-sm text-danger">{error}</p>}

      {!result && !loading && (
        <div className="flex flex-1 flex-col items-center justify-center gap-2 text-faint">
          <Sparkles className="size-6" />
          <p className="text-sm">Answers are grounded in your indexed documents, with citations.</p>
        </div>
      )}

      {result && (
        <div className="flex flex-col gap-4 overflow-y-auto pr-1">
          {result.abstained ? (
            <Card className="border-amber/30 p-4">
              <p className="text-sm font-medium text-amber">No grounded answer</p>
              <p className="mt-1 text-sm text-muted">{result.message}</p>
            </Card>
          ) : (
            <>
              <Card className="p-4">
                <p className="text-[15px] leading-relaxed">
                  {splitAnswer(result.answer ?? "").map((part, i) =>
                    "text" in part ? (
                      <span key={i}>{part.text}</span>
                    ) : (
                      <button
                        key={i}
                        onClick={() => setActiveCite(part.cite)}
                        className={cn(
                          "mx-0.5 rounded border px-1 align-super text-[10px] font-semibold transition",
                          activeCite === part.cite
                            ? "border-primary bg-primary/20 text-primary"
                            : "border-primary/30 text-primary hover:bg-primary/10",
                        )}
                      >
                        {part.cite}
                      </button>
                    ),
                  )}
                </p>
              </Card>

              {result.confidence && (
                <Card className="p-4">
                  <p className="mb-3 text-xs font-medium uppercase tracking-wide text-muted">Confidence</p>
                  <ConfidenceBars confidence={result.confidence} />
                </Card>
              )}

              {result.citations.length > 0 && (
                <div className="flex flex-col gap-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted">Citations</p>
                  {result.citations.map((c) => (
                    <Card
                      key={c.number}
                      className={cn(
                        "p-3 transition",
                        activeCite === c.number && "border-primary/60 bg-primary/5",
                      )}
                    >
                      <div className="flex items-center gap-2 text-xs text-faint">
                        <span className="rounded border border-primary/30 px-1.5 text-primary">
                          {c.number}
                        </span>
                        <span>{c.source}</span>
                        {c.heading && <span>· {c.heading}</span>}
                        {c.page != null && <span>· p.{c.page}</span>}
                      </div>
                      <p className="mt-2 text-sm text-muted">{c.text}</p>
                    </Card>
                  ))}
                </div>
              )}
            </>
          )}

          {result.sources.length > 0 && (
            <div className="flex flex-col gap-1.5">
              <p className="text-xs font-medium uppercase tracking-wide text-muted">
                Retrieved sources
              </p>
              {result.sources.map((s, i) => (
                <div key={i} className="flex items-center justify-between text-xs text-faint">
                  <span className="truncate">{s.source ?? "unknown"}</span>
                  <span>score {s.score.toFixed(3)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
