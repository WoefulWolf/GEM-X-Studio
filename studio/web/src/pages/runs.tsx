import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { FolderOpen, MoreHorizontal, Plus, RotateCcw, Square, Trash2 } from "lucide-react"
import { useState } from "react"
import { toast } from "sonner"

import { PageHeader } from "@/components/logo"
import { JobProgress, StatusBadge } from "@/components/status"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { api, urls, type Job } from "@/lib/api"
import { formatDuration, jobElapsed, modeLabel, timeAgo } from "@/lib/format"
import { navigate } from "@/lib/router"

type Filter = "all" | "active" | "done" | "failed"

export function RunsPage() {
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 1500 })
  const [filter, setFilter] = useState<Filter>("all")
  const list = (jobs.data ?? []).filter((job) => {
    if (filter === "active") return job.status === "running" || job.status === "queued"
    if (filter === "done") return job.status === "done"
    if (filter === "failed") return job.status === "failed" || job.status === "cancelled"
    return true
  })
  return (
    <div>
      <PageHeader
        title="Runs"
        description="Every run is kept on disk with its predictions, preview and exports."
        actions={
          <>
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={filter}
              onValueChange={(value) => value && setFilter(value as Filter)}
            >
              <ToggleGroupItem value="all">All</ToggleGroupItem>
              <ToggleGroupItem value="active">Active</ToggleGroupItem>
              <ToggleGroupItem value="done">Done</ToggleGroupItem>
              <ToggleGroupItem value="failed">Failed</ToggleGroupItem>
            </ToggleGroup>
            <Button onClick={() => navigate("/new")}>
              <Plus /> New run
            </Button>
          </>
        }
      />
      <div className="p-8">
        {jobs.isLoading ? (
          <div className="h-48 animate-pulse rounded-xl bg-muted/50" />
        ) : list.length === 0 ? (
          <div className="rounded-xl border border-dashed px-6 py-16 text-center">
            <div className="font-medium">No runs {filter === "all" ? "yet" : "here"}</div>
            <p className="mt-1 text-sm text-muted-foreground">Start one from New run.</p>
          </div>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(300px,1fr))] gap-4">
            {list.map((job) => (
              <RunCard key={job.id} job={job} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function RunCard({ job }: { job: Job }) {
  const label = modeLabel(job.kind, job.task)
  const active = job.status === "running" || job.status === "queued"
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => navigate(`/runs/${job.id}`)}
      onKeyDown={(event) => event.key === "Enter" && navigate(`/runs/${job.id}`)}
      className="group overflow-hidden rounded-xl border bg-card text-left transition-all outline-none hover:border-foreground/20 hover:shadow-lg hover:shadow-black/5 focus-visible:ring-2 focus-visible:ring-ring"
    >
      <div className="relative aspect-[16/8] overflow-hidden bg-muted">
        <img
          src={urls.jobThumb(job.id)}
          alt=""
          loading="lazy"
          className="size-full object-cover transition-transform duration-500 group-hover:scale-[1.03]"
        />
        <div className="absolute top-2.5 left-2.5">
          <StatusBadge status={job.status} className="bg-background/85 backdrop-blur" />
        </div>
        <div className="absolute top-2 right-2" onClick={(event) => event.stopPropagation()}>
          <RunMenu job={job} />
        </div>
      </div>
      <div className="flex flex-col gap-2 p-3.5">
        <div className="flex items-center gap-2">
          <span className="min-w-0 flex-1 truncate text-sm font-medium">{job.title}</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="font-medium text-foreground/80">{label.title}</span>
          <span>·</span>
          <span>{job.kind.endsWith("image") ? "image" : "video"}</span>
          <span className="ml-auto">{timeAgo(job.created)}</span>
        </div>
        {active ? (
          <JobProgress job={job} />
        ) : job.status === "failed" ? (
          <p className="line-clamp-2 text-xs text-destructive">{job.error}</p>
        ) : job.status === "done" ? (
          <p className="text-xs text-muted-foreground">
            Finished in {formatDuration(jobElapsed(job))}
            {job.device ? ` on ${job.device.toUpperCase()}` : ""}
          </p>
        ) : null}
      </div>
    </div>
  )
}

export function RunMenu({ job, onDeleted }: { job: Job; onDeleted?: () => void }) {
  const queryClient = useQueryClient()
  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ["jobs"] })
    queryClient.invalidateQueries({ queryKey: ["job", job.id] })
  }
  const cancel = useMutation({ mutationFn: () => api.cancel(job.id), onSuccess: refresh })
  const retry = useMutation({
    mutationFn: () => api.retry(job.id),
    onSuccess: (created) => {
      refresh()
      navigate(`/runs/${created.id}`)
    },
  })
  const remove = useMutation({
    mutationFn: () => api.deleteJob(job.id),
    onSuccess: () => {
      refresh()
      toast.success("Run deleted")
      onDeleted?.()
    },
  })
  const active = job.status === "running" || job.status === "queued"
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="secondary" size="icon" className="size-7 bg-background/85 backdrop-blur">
          <MoreHorizontal />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => api.reveal(job.id)}>
          <FolderOpen /> Open output folder
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => retry.mutate()}>
          <RotateCcw /> Run again
        </DropdownMenuItem>
        {active && (
          <DropdownMenuItem onClick={() => cancel.mutate()}>
            <Square /> Cancel
          </DropdownMenuItem>
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem variant="destructive" onClick={() => remove.mutate()}>
          <Trash2 /> Delete run
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
