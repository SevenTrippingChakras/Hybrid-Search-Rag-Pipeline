import { useEffect, useRef, useState } from "react"
import { ChevronDown, Loader2, Send, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import type { AskResponse, Confidence } from "../types"
import { askQuestion, splitAnswer } from "./Ask.helper"

type ChatMessage =
  | { id: number; role: "user"; text: string }
  | { id: number; role: "assistant"; data: AskResponse }
  | { id: number; role: "error"; text: string }

const CONFIDENCE_DIMS: { key: keyof Confidence; label: string }[] = [
  { key: "retrieval", label: "Retrieval" },
  { key: "citation_coverage", label: "Citations" },
  { key: "completeness", label: "Completeness" },
  { key: "score", label: "Overall" },
]

function ConfidenceBars({ confidence }: { confidence: Confidence }) {
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-2">
      {CONFIDENCE_DIMS.map(({ key, label }) => {
        const pct = Math.round(confidence[key] * 100)
        return (
          <div key={key}>
            <div className="flex justify-between text-[11px]">
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

// One assistant turn: the grounded answer (or abstention), with its citations
// and confidence tucked into a collapsed panel the reader can open on demand.
function AssistantMessage({ data }: { data: AskResponse }) {
  const [open, setOpen] = useState(false)
  const [activeCite, setActiveCite] = useState<number | null>(null)

  if (data.abstained) {
    return (
      <div className="rounded-2xl rounded-bl-sm border border-amber/30 bg-amber/5 px-4 py-3">
        <p className="text-sm font-medium text-amber">No grounded answer</p>
        <p className="mt-1 text-sm text-muted">{data.message}</p>
      </div>
    )
  }

  const openCite = (n: number) => {
    setActiveCite(n)
    setOpen(true)
  }
  const score = data.confidence ? Math.round(data.confidence.score * 100) : null
  const count = data.citations.length

  return (
    <div className="rounded-2xl rounded-bl-sm border border-border bg-card px-4 py-3">
      <p className="text-[15px] leading-relaxed">
        {splitAnswer(data.answer ?? "").map((part, i) =>
          "text" in part ? (
            <span key={i}>{part.text}</span>
          ) : (
            <button
              key={i}
              onClick={() => openCite(part.cite)}
              className="mx-0.5 rounded border border-primary/40 px-1 align-super text-[10px] font-semibold text-primary transition hover:bg-primary/15"
            >
              {part.cite}
            </button>
          ),
        )}
      </p>

      <div className="mt-2.5 flex items-center gap-3 text-xs text-faint">
        {score !== null && (
          <span className="rounded-full bg-card2 px-2 py-0.5">confidence {score}%</span>
        )}
        {count > 0 && (
          <button
            onClick={() => setOpen((o) => !o)}
            className="inline-flex items-center gap-1 transition hover:text-foreground"
          >
            <ChevronDown className={cn("size-3.5 transition", open && "rotate-180")} />
            {count} source{count > 1 ? "s" : ""}
          </button>
        )}
      </div>

      {open && (
        <div className="mt-3 flex flex-col gap-3 border-t border-border pt-3">
          {data.confidence && <ConfidenceBars confidence={data.confidence} />}
          {data.citations.map((c) => (
            <div
              key={c.number}
              className={cn(
                "rounded-lg border bg-card2 p-2.5 transition",
                activeCite === c.number ? "border-primary/60" : "border-border",
              )}
            >
              <div className="flex items-center gap-1.5 text-[11px] text-faint">
                <span className="rounded border border-primary/30 px-1 text-primary">{c.number}</span>
                <span>{c.source}</span>
                {c.heading && <span>· {c.heading}</span>}
                {c.page != null && <span>· p.{c.page}</span>}
              </div>
              <p className="mt-1.5 text-[13px] text-muted">{c.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export function Ask() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState("")
  const [loading, setLoading] = useState(false)
  const nextId = useRef(1)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading])

  const onSend = async () => {
    const q = question.trim()
    if (!q || loading) return
    setQuestion("")
    setLoading(true)
    setMessages((m) => [...m, { id: nextId.current++, role: "user", text: q }])
    try {
      const data = await askQuestion(q)
      setMessages((m) => [...m, { id: nextId.current++, role: "assistant", data }])
    } catch (e) {
      setMessages((m) => [...m, { id: nextId.current++, role: "error", text: (e as Error).message }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto pr-1">
        {messages.length === 0 && !loading && (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-faint">
            <Sparkles className="size-6" />
            <p className="max-w-xs text-sm">
              Ask anything about your indexed documents. Answers are grounded, with citations.
            </p>
          </div>
        )}

        {messages.map((m) => {
          if (m.role === "user") {
            return (
              <div key={m.id} className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-primary/15 px-4 py-2.5 text-sm">
                  {m.text}
                </div>
              </div>
            )
          }
          if (m.role === "error") {
            return (
              <div key={m.id} className="max-w-[92%] rounded-2xl rounded-bl-sm border border-danger/30 bg-danger/5 px-4 py-3 text-sm text-danger">
                {m.text}
              </div>
            )
          }
          return (
            <div key={m.id} className="max-w-[92%]">
              <AssistantMessage data={m.data} />
            </div>
          )
        })}

        {loading && (
          <div className="flex max-w-[92%] items-center gap-2 rounded-2xl rounded-bl-sm border border-border bg-card px-4 py-3 text-sm text-muted">
            <Loader2 className="size-4 animate-spin" /> Thinking…
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="flex gap-2 border-t border-border pt-3">
        <Input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onSend()}
          placeholder="Ask a question…"
        />
        <Button onClick={onSend} disabled={loading || !question.trim()}>
          <Send className="size-4" />
        </Button>
      </div>
    </div>
  )
}
