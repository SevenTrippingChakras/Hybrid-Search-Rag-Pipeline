import type { InputHTMLAttributes } from "react"
import { cn } from "@/lib/utils"

export function Input({
  className,
  ...props
}: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "w-full rounded-lg border border-border bg-card2 px-3.5 py-2 text-sm text-foreground placeholder:text-faint outline-none focus:border-primary/50",
        className,
      )}
      {...props}
    />
  )
}
