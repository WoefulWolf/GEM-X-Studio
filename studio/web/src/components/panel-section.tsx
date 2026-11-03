import { ChevronDown, Loader2, type LucideIcon } from "lucide-react"
import { Collapsible } from "radix-ui"
import { useState, type ReactNode } from "react"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const STORAGE_PREFIX = "gem-studio:section:"

function readOpen(id: string, fallback: boolean) {
  try {
    const value = localStorage.getItem(STORAGE_PREFIX + id)
    return value == null ? fallback : value === "1"
  } catch {
    return fallback
  }
}

/**
 * One collapsible card per side-panel section (Export, Refine). The header shows
 * the title and, while collapsed, a short summary of the section's settings.
 * Open/closed state is remembered per section id.
 */
export function PanelSection(props: {
  id: string
  title: string
  summary?: ReactNode
  defaultOpen?: boolean
  children: ReactNode
}) {
  const [open, setOpen] = useState(() => readOpen(props.id, props.defaultOpen ?? true))
  const toggle = (next: boolean) => {
    setOpen(next)
    try {
      localStorage.setItem(STORAGE_PREFIX + props.id, next ? "1" : "0")
    } catch {
      /* storage unavailable */
    }
  }
  return (
    <Collapsible.Root open={open} onOpenChange={toggle} className="rounded-lg border bg-card">
      <Collapsible.Trigger className="group flex w-full items-center gap-2 rounded-lg px-4 py-3 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring">
        <h3 className="shrink-0 text-[11px] font-medium tracking-wide text-muted-foreground uppercase group-hover:text-foreground">
          {props.title}
        </h3>
        {!open && props.summary && (
          <span className="min-w-0 flex-1 truncate text-right text-[11px] text-muted-foreground/80">{props.summary}</span>
        )}
        <ChevronDown
          className={cn(
            "ml-auto size-4 shrink-0 text-muted-foreground transition-transform duration-200",
            !open && "-rotate-90",
            !open && props.summary && "ml-0",
          )}
        />
      </Collapsible.Trigger>
      <Collapsible.Content className="flex flex-col gap-4 border-t px-4 py-4">{props.children}</Collapsible.Content>
    </Collapsible.Root>
  )
}

/** Divided list of actions; sits flush with the section card's edges. */
export function ActionList({ children }: { children: ReactNode }) {
  return (
    <div className="-mx-4 -mb-4 flex flex-col divide-y not-first:border-t first:-mt-4">{children}</div>
  )
}

export function ActionRow(props: {
  icon: LucideIcon
  title: string
  hint?: string
  action?: string
  busy?: boolean
  disabled?: boolean
  onClick: () => void
  /** Replaces the default action button (e.g. Render / Stop). */
  control?: ReactNode
  /** Extra content under the row (e.g. a progress bar). */
  children?: ReactNode
}) {
  const Icon = props.icon
  return (
    <div className="flex flex-col gap-2.5 px-4 py-3">
      <div className="flex items-center gap-3">
        <Icon className="size-4 shrink-0 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <div className="truncate text-[13px] font-medium">{props.title}</div>
          {props.hint && <div className="truncate text-[11px] text-muted-foreground">{props.hint}</div>}
        </div>
        {props.control ?? (
          <Button
            size="sm"
            variant="secondary"
            disabled={props.busy || props.disabled}
            onClick={props.onClick}
            className={cn(props.busy && "w-[72px]")}
          >
            {props.busy ? <Loader2 className="animate-spin" /> : (props.action ?? "Export")}
          </Button>
        )}
      </div>
      {props.children}
    </div>
  )
}

/** Small label + control + optional hint, as used for section options. */
export function OptionField(props: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[13px] font-medium">{props.label}</span>
      {props.children}
      {props.hint && <p className="text-[11px] leading-relaxed text-muted-foreground">{props.hint}</p>}
    </div>
  )
}
