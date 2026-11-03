import { CheckCircle2, CircleDashed, CircleSlash, Clock, Loader2, XCircle } from "lucide-react"

import { Progress } from "@/components/ui/progress"
import type { Job, JobStatus } from "@/lib/api"
import { cn } from "@/lib/utils"

const STYLES: Record<JobStatus, { label: string; className: string; icon: typeof Clock }> = {
  queued: { label: "Queued", className: "bg-muted text-muted-foreground", icon: Clock },
  running: { label: "Running", className: "bg-primary/15 text-primary", icon: Loader2 },
  done: { label: "Done", className: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400", icon: CheckCircle2 },
  failed: { label: "Failed", className: "bg-destructive/15 text-destructive", icon: XCircle },
  cancelled: { label: "Cancelled", className: "bg-muted text-muted-foreground", icon: CircleSlash },
}

export function StatusBadge({ status, className }: { status: JobStatus; className?: string }) {
  const style = STYLES[status] ?? { label: status, className: "", icon: CircleDashed }
  const Icon = style.icon
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium",
        style.className,
        className,
      )}
    >
      <Icon className={cn("size-3", status === "running" && "animate-spin")} />
      {style.label}
    </span>
  )
}

export function JobProgress({ job, className }: { job: Job; className?: string }) {
  const fraction = job.progress ? job.progress[0] / Math.max(job.progress[1], 1) : null
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <div className="flex items-baseline justify-between gap-3 text-xs">
        <span className="truncate text-muted-foreground">
          {job.stage}
          {job.detail ? ` · ${job.detail}` : ""}
        </span>
        {job.progress && (
          <span className="shrink-0 text-muted-foreground tabular-nums">
            {job.progress[0]} / {job.progress[1]}
          </span>
        )}
      </div>
      {fraction == null ? (
        <div className="relative h-1.5 overflow-hidden rounded-full bg-primary/15">
          <div className="absolute inset-y-0 w-1/3 animate-[indeterminate_1.4s_ease-in-out_infinite] rounded-full bg-primary" />
        </div>
      ) : (
        <Progress value={fraction * 100} className="h-1.5" />
      )}
    </div>
  )
}
