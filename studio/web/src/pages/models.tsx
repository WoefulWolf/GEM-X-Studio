import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { CheckCircle2, CircleAlert, Download, ExternalLink, Footprints, Hand, Loader2, PersonStanding } from "lucide-react"
import { toast } from "sonner"

import { PageHeader } from "@/components/logo"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { api, type DownloadState, type ModelComponent, type TaskId } from "@/lib/api"
import { formatBytes } from "@/lib/format"
import { openExternal } from "@/lib/native"
import { cn } from "@/lib/utils"

const TASK_ICONS: Record<TaskId, typeof Hand> = {
  body_pose: PersonStanding,
  hand_pose: Hand,
  body_motion: Footprints,
}

export function ModelsPage() {
  const queryClient = useQueryClient()
  const models = useQuery({
    queryKey: ["models"],
    queryFn: api.models,
    refetchInterval: (query) =>
      Object.values(query.state.data?.downloads ?? {}).some((d) => d.status === "queued" || d.status === "downloading")
        ? 800
        : 5000,
  })
  const download = useMutation({
    mutationFn: api.download,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["models"] }),
    onError: (error) => toast.error("Download failed to start", { description: (error as Error).message }),
  })
  const data = models.data
  const components = new Map((data?.components ?? []).map((component) => [component.id, component]))
  const downloads = data?.downloads ?? {}

  return (
    <div>
      <PageHeader
        title="Models"
        description="Checkpoints and body-model assets are downloaded once from Hugging Face and stored in the repository."
        actions={
          <Button
            variant="outline"
            disabled={!data || Object.values(data.tasks).every((task) => task.ready)}
            onClick={() => download.mutate({ tasks: ["body_pose", "hand_pose", "body_motion"] })}
          >
            <Download /> Download everything
          </Button>
        }
      />
      <div className="flex flex-col gap-8 p-8">
        <div className="grid gap-4 lg:grid-cols-3">
          {data &&
            (Object.entries(data.tasks) as [TaskId, (typeof data.tasks)[TaskId]][]).map(([id, task]) => {
              const Icon = TASK_ICONS[id]
              const parts = task.components.map((c) => components.get(c)!).filter(Boolean)
              const missing = parts.filter((part) => !part.ready)
              const size = missing.reduce((total, part) => total + part.approx_mb, 0)
              const states = task.components.map((c) => downloads[c]).filter(Boolean) as DownloadState[]
              const active = states.some((s) => s.status === "queued" || s.status === "downloading")
              const loaded = states.reduce((total, s) => total + (s.status === "done" ? s.total : s.downloaded), 0)
              const total = states.reduce((total, s) => total + (s.total || 0), 0)
              return (
                <div
                  key={id}
                  className={cn(
                    "flex flex-col gap-4 rounded-xl border bg-card p-5",
                    task.ready && "border-emerald-500/30",
                  )}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={cn(
                        "flex size-10 items-center justify-center rounded-lg",
                        task.ready ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400" : "bg-muted text-muted-foreground",
                      )}
                    >
                      <Icon className="size-5" />
                    </div>
                    <div>
                      <div className="font-medium">{task.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {task.ready ? "Ready to run" : `${missing.length} of ${parts.length} parts missing`}
                      </div>
                    </div>
                    {task.ready && <CheckCircle2 className="ml-auto size-5 text-emerald-500" />}
                  </div>
                  <ul className="flex flex-col gap-1 text-xs">
                    {parts.map((part) => (
                      <li key={part.id} className="flex items-center gap-2">
                        {part.ready ? (
                          <CheckCircle2 className="size-3.5 text-emerald-500" />
                        ) : downloads[part.id]?.status === "downloading" ? (
                          <Loader2 className="size-3.5 animate-spin text-primary" />
                        ) : (
                          <span className="size-3.5 rounded-full border border-dashed border-muted-foreground/50" />
                        )}
                        <span className={cn(!part.ready && "text-muted-foreground")}>{part.name}</span>
                      </li>
                    ))}
                  </ul>
                  <div className="mt-auto">
                    {active ? (
                      <div className="flex flex-col gap-1.5">
                        <Progress value={total ? (100 * loaded) / total : 2} className="h-1.5" />
                        <div className="flex justify-between text-[11px] text-muted-foreground tabular-nums">
                          <span>Downloading…</span>
                          <span>{total ? `${formatBytes(loaded)} / ${formatBytes(total)}` : "starting…"}</span>
                        </div>
                      </div>
                    ) : task.ready ? (
                      <Button variant="secondary" className="w-full" disabled>
                        Installed
                      </Button>
                    ) : (
                      <Button className="w-full" onClick={() => download.mutate({ tasks: [id] })}>
                        <Download /> Download · ~{(size / 1000).toFixed(1)} GB
                      </Button>
                    )}
                  </div>
                </div>
              )
            })}
        </div>

        <div>
          <h2 className="mb-3 text-sm font-medium">All components</h2>
          <div className="overflow-hidden rounded-xl border">
            {(data?.components ?? []).map((component) => (
              <ComponentRow
                key={component.id}
                component={component}
                state={downloads[component.id]}
                onDownload={() => download.mutate({ components: [component.id] })}
              />
            ))}
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            Each model is subject to its own license on Hugging Face (linked above). By downloading you accept those
            terms.
          </p>
        </div>
      </div>
    </div>
  )
}

function ComponentRow(props: { component: ModelComponent; state?: DownloadState; onDownload: () => void }) {
  const { component, state } = props
  const active = state && (state.status === "queued" || state.status === "downloading")
  return (
    <div className="flex items-center gap-4 border-b bg-card px-4 py-3 last:border-b-0">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{component.name}</span>
          {component.ready ? (
            <Badge variant="outline" className="border-emerald-500/40 text-[10px] text-emerald-600 dark:text-emerald-400">
              installed
            </Badge>
          ) : state?.status === "error" ? (
            <Badge variant="destructive" className="text-[10px]">
              failed
            </Badge>
          ) : null}
          {component.license_url && (
            <button
              onClick={() => openExternal(component.license_url!)}
              className="text-muted-foreground hover:text-foreground"
              aria-label="Open model page"
            >
              <ExternalLink className="size-3" />
            </button>
          )}
        </div>
        <div className="truncate text-xs text-muted-foreground">{component.description}</div>
        {state?.status === "error" && (
          <div className="mt-1 flex items-center gap-1 text-xs text-destructive">
            <CircleAlert className="size-3" /> {state.error}
          </div>
        )}
        {active && (
          <div className="mt-2 flex items-center gap-3">
            <Progress value={state.total ? (100 * state.downloaded) / state.total : 2} className="h-1" />
            <span className="shrink-0 text-[11px] text-muted-foreground tabular-nums">
              {state.status === "queued"
                ? "queued"
                : state.total
                  ? `${formatBytes(state.downloaded)} / ${formatBytes(state.total)}`
                  : "starting…"}
            </span>
          </div>
        )}
      </div>
      <span className="w-16 text-right text-xs text-muted-foreground tabular-nums">
        {component.approx_mb >= 1000 ? `${(component.approx_mb / 1000).toFixed(1)} GB` : `${component.approx_mb} MB`}
      </span>
      <Button
        size="sm"
        variant={component.ready ? "ghost" : "secondary"}
        disabled={component.ready || active}
        onClick={props.onDownload}
        className="w-24"
      >
        {component.ready ? "Ready" : active ? <Loader2 className="animate-spin" /> : "Download"}
      </Button>
    </div>
  )
}
