import {
  completeUpload,
  deleteDocument,
  initiateUpload,
  listDocuments,
  uploadToStorage,
} from "../api"
import type { DocumentRecord } from "../types"

// Data-loading + actions for the Documents pane: call the services and hand the
// component exactly what it needs, keeping request wiring out of the view.

// The documents list (backend returns newest-first).
export async function loadDocuments(): Promise<DocumentRecord[]> {
  const res = await listDocuments()
  return res.data
}

// True while any document is still moving toward a terminal state, so the
// component knows whether it needs to keep polling.
export function hasActive(documents: DocumentRecord[]): boolean {
  return documents.some(
    (d) => d.status === "pending" || d.status === "uploaded" || d.status === "processing",
  )
}

// The full presigned upload flow, start to finish:
//   1. initiate  -> get a record id + a signed URL
//   2. PUT       -> the browser uploads the bytes straight to storage
//   3. complete  -> tell the backend it landed; ingestion starts in the background
export async function uploadFile(file: File): Promise<void> {
  const contentType = file.type || "application/octet-stream"
  const { data } = await initiateUpload({
    filename: file.name,
    content_type: contentType,
    size: file.size,
  })
  await uploadToStorage(data.upload_url, file, contentType)
  await completeUpload(data.document_id)
}

export async function removeDocument(id: string): Promise<void> {
  await deleteDocument(id)
}
