// Base URL of the backend API. Override with VITE_API_URL in a .env file.
export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"

// How often to re-poll a document's status while it is still ingesting.
export const POLL_INTERVAL_MS = 2000
