import { useEffect, useRef, useState } from "react"
import { FileText, Loader2, Trash2, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { StatusBadge } from "@/components/ui/badge"
import { POLL_INTERVAL_MS } from "../config"
import type { DocumentRecord } from "../types"
import { hasActive, loadDocuments, removeDocument, uploadFile } from "./Documents.helper"

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export function Documents() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  // Load once, then keep polling while any document is still ingesting.
  useEffect(() => {
    let timer: number | undefined
    let cancelled = false

    const tick = async () => {
      try {
        const docs = await loadDocuments()
        if (cancelled) return
        setDocuments(docs)
        if (hasActive(docs)) timer = window.setTimeout(tick, POLL_INTERVAL_MS)
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      }
    }

    tick()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [])

  const refresh = async () => setDocuments(await loadDocuments())

  const onPick = async (file: File | undefined) => {
    if (!file) return
    setError(null)
    setUploading(true)
    try {
      await uploadFile(file)
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ""
    }
  }

  const onDelete = async (id: string) => {
    try {
      await removeDocument(id)
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Documents</h2>
        <span className="text-xs text-muted">{documents.length} total</span>
      </div>

      <input
        ref={fileInput}
        type="file"
        accept=".md,.markdown,.txt,.text,.html,.htm,.pdf"
        className="hidden"
        onChange={(e) => onPick(e.target.files?.[0])}
      />
      <Button
        onClick={() => fileInput.current?.click()}
        disabled={uploading}
        className="w-full"
      >
        {uploading ? (
          <>
            <Loader2 className="size-4 animate-spin" /> Uploading &amp; ingesting…
          </>
        ) : (
          <>
            <Upload className="size-4" /> Upload document
          </>
        )}
      </Button>

      {error && <p className="text-sm text-danger">{error}</p>}

      <div className="flex flex-col gap-2 overflow-y-auto pr-1">
        {documents.length === 0 && (
          <p className="text-sm text-faint">No documents yet. Upload one to get started.</p>
        )}
        {documents.map((d) => (
          <Card key={d.document_id} className="flex items-center gap-3 p-3">
            <FileText className="size-4 shrink-0 text-muted" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{d.filename}</p>
              <p className="text-xs text-faint">
                {formatSize(d.size)}
                {d.status === "indexed" && d.chunk_count != null && ` · ${d.chunk_count} chunks`}
                {d.status === "failed" && d.error && ` · ${d.error}`}
              </p>
            </div>
            <StatusBadge status={d.status} />
            <button
              onClick={() => onDelete(d.document_id)}
              className="text-faint transition hover:text-danger"
              title="Delete"
            >
              <Trash2 className="size-4" />
            </button>
          </Card>
        ))}
      </div>
    </div>
  )
}
