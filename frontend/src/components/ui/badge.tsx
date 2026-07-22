import { cn } from "@/lib/utils"
import type { DocumentStatus } from "@/types"

const styles: Record<DocumentStatus, string> = {
  pending: "text-faint border-border",
  uploaded: "text-primary border-primary/30",
  processing: "text-amber border-amber/30",
  indexed: "text-accent border-accent/30",
  failed: "text-danger border-danger/30",
}

export function StatusBadge({ status }: { status: DocumentStatus }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium",
        styles[status],
      )}
    >
      {status}
    </span>
  )
}
