import type { ButtonHTMLAttributes } from "react"
import { cn } from "@/lib/utils"

type Variant = "primary" | "ghost" | "danger"

const variants: Record<Variant, string> = {
  primary: "bg-primary/15 text-primary border-primary/30 hover:bg-primary/25",
  ghost: "bg-transparent text-muted border-border hover:text-foreground",
  danger: "bg-transparent text-danger border-danger/30 hover:bg-danger/10",
}

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
}

export function Button({ className, variant = "primary", ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg border px-3.5 py-2 text-sm font-medium transition disabled:opacity-50 disabled:pointer-events-none",
        variants[variant],
        className,
      )}
      {...props}
    />
  )
}
