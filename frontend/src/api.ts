import axios, { type AxiosError } from "axios"
import { API_URL } from "./config"
import type { AskResponse, DocumentRecord, InitiateUploadResponse } from "./types"

const api = axios.create({ baseURL: API_URL })

// Normalise the backend's { error: { code, message } } envelope into a plain
// Error carrying the message, so callers can just catch and show err.message.
api.interceptors.response.use(
  (res) => res,
  (error: AxiosError<{ error?: { message?: string } }>) => {
    const message = error.response?.data?.error?.message ?? error.message
    return Promise.reject(new Error(message))
  },
)

// --- Documents ---

export function initiateUpload(body: {
  filename: string
  content_type: string
  size: number
}) {
  return api.post<InitiateUploadResponse>("/documents", body)
}

// Uploads the file straight to storage's presigned URL. Bare axios on purpose:
// no baseURL and no interceptors, since this request goes to Supabase, not us.
export function uploadToStorage(url: string, file: File, contentType: string) {
  return axios.put(url, file, { headers: { "Content-Type": contentType } })
}

export function completeUpload(id: string) {
  return api.post<DocumentRecord>(`/documents/${id}/complete`)
}

export function listDocuments() {
  return api.get<DocumentRecord[]>("/documents")
}

export function getDocument(id: string) {
  return api.get<DocumentRecord>(`/documents/${id}`)
}

export function deleteDocument(id: string) {
  return api.delete(`/documents/${id}`)
}

// --- Q&A ---

export function ask(question: string) {
  return api.post<AskResponse>("/ask", { question })
}
