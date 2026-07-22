import { ask } from "../api"
import type { AskResponse } from "../types"

// Ask a question and hand the component the pipeline's result.
export async function askQuestion(question: string): Promise<AskResponse> {
  const res = await ask(question)
  return res.data
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
