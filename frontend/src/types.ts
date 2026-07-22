export type DocumentStatus =
  | "pending"
  | "uploaded"
  | "processing"
  | "indexed"
  | "failed"

export interface DocumentRecord {
  document_id: string
  filename: string
  content_type: string
  size: number
  storage_key: string
  status: DocumentStatus
  chunk_count: number | null
  error: string | null
  created_at: string
  updated_at: string
}

export interface InitiateUploadResponse {
  document_id: string
  upload_url: string
}

export interface Citation {
  number: number
  source: string
  text: string
  heading: string | null
  page: number | null
}

export interface Confidence {
  retrieval: number
  citation_coverage: number
  completeness: number
  score: number
}

export interface Source {
  source: string | null
  document_id: string | null
  score: number
}

export interface AskResponse {
  query: string
  answer: string | null
  citations: Citation[]
  confidence: Confidence | null
  abstained: boolean
  message: string | null
  sources: Source[]
}
