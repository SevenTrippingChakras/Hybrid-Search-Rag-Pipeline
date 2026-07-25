import { API_URL } from "../config"
import type { Citation, Confidence, Source } from "../types"

// Callbacks the streaming client fires as SSE frames arrive. The answer prose
// comes in via onDelta; citations and confidence resolve at the end (onFinal).
export interface StreamHandlers {
  onSources?: (sources: Source[]) => void
  onDelta?: (text: string) => void
  onFinal?: (data: {
    answer: string
    citations: Citation[]
    confidence: Confidence | null
  }) => void
  onAbstain?: (data: { message: string | null; sources: Source[] }) => void
  onError?: (message: string) => void
}

// Ask a question over SSE, dispatching each event to the matching handler.
// axios can't read a streaming body in the browser, so this uses fetch and
// parses the text/event-stream frames by hand.
export async function askStream(
  question: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  })
  if (!res.ok || !res.body) {
    throw new Error(`Request failed (${res.status})`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  // SSE frames are separated by a blank line; keep the trailing partial frame
  // in the buffer until its terminator arrives in a later chunk.
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let sep: number
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      dispatchFrame(frame, handlers)
    }
  }
}

function dispatchFrame(frame: string, handlers: StreamHandlers): void {
  let event = "message"
  const dataLines: string[] = []
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim()
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim())
  }
  if (dataLines.length === 0) return
  const data = JSON.parse(dataLines.join("\n"))

  switch (event) {
    case "sources":
      handlers.onSources?.(data.sources)
      break
    case "delta":
      handlers.onDelta?.(data.text)
      break
    case "final":
      handlers.onFinal?.(data)
      break
    case "abstain":
      handlers.onAbstain?.(data)
      break
    case "error":
      handlers.onError?.(data.message)
      break
  }
}

// A run of plain text, or a single [n] citation marker.
export type AnswerPart = { text: string } | { cite: number }

// Split an answer into text runs and [n] markers so the view can render the
// markers as clickable chips instead of leaving raw "[1]" in the prose.
export function splitAnswer(text: string): AnswerPart[] {
  const parts: AnswerPart[] = []
  const re = /\[(\d+)\]/g
  let last = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push({ text: text.slice(last, m.index) })
    parts.push({ cite: Number(m[1]) })
    last = m.index + m[0].length
  }
  if (last < text.length) parts.push({ text: text.slice(last) })
  return parts
}
