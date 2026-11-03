import { useQuery } from "@tanstack/react-query"
import {
  Boxes,
  Cpu,
  History,
  Monitor,
  Moon,
  Plus,
  Settings2,
  Sun,
  type LucideIcon,
} from "lucide-react"
import type { ReactNode } from "react"

import { Logo } from "@/components/logo"
import { Progress } from "@/components/ui/progress"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { api } from "@/lib/api"
import { modeLabel } from "@/lib/format"
import { navigate, useRoute } from "@/lib/router"
import { useTheme, type ThemeChoice } from "@/lib/theme"
import { cn } from "@/lib/utils"
import { ModelsPage } from "@/pages/models"
import { NewRunPage } from "@/pages/new-run"
import { RunDetailPage } from "@/pages/run-detail"
import { RunsPage } from "@/pages/runs"
import { SettingsPage } from "@/pages/settings"

export function App() {
  const [section, id] = useRoute()
  let page: ReactNode
  if (section === "runs" && id) page = <RunDetailPage key={id} jobId={id} />
  else if (section === "runs") page = <RunsPage />
  else if (section === "models") page = <ModelsPage />
  else if (section === "settings") page = <SettingsPage />
  else page = <NewRunPage />
  return (
    <div className="flex h-full overflow-hidden">
      <Sidebar section={section ?? "new"} />
      <main className="min-w-0 flex-1 overflow-y-auto">{page}</main>
    </div>
  )
}

function Sidebar({ section }: { section: string }) {
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 2000 })
  const models = useQuery({ queryKey: ["models"], queryFn: api.models, refetchInterval: 5000 })
  const active = (jobs.data ?? []).filter((job) => job.status === "running" || job.status === "queued")
  const running = active.find((job) => job.status === "running")
  const anyTaskReady = models.data && Object.values(models.data.tasks).some((task) => task.ready)
  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground">
      <div className="flex items-center gap-2.5 px-4 pt-5 pb-6">
        <Logo className="size-8" />
        <div className="leading-tight">
          <div className="text-[15px] font-semibold tracking-tight">GEM-X Studio</div>
          <div className="text-[11px] text-muted-foreground">Pose · Hands · Motion</div>
        </div>
      </div>
      <nav className="flex flex-col gap-0.5 px-2.5">
        <NavItem icon={Plus} label="New run" href="/new" active={section === "new"} />
        <NavItem
          icon={History}
          label="Runs"
          href="/runs"
          active={section === "runs"}
          badge={active.length ? String(active.length) : undefined}
        />
        <NavItem
          icon={Boxes}
          label="Models"
          href="/models"
          active={section === "models"}
          dot={models.data && !anyTaskReady}
        />
        <NavItem icon={Settings2} label="Settings" href="/settings" active={section === "settings"} />
      </nav>

      <div className="mt-auto flex flex-col gap-3 p-3">
        {running && (
          <button
            onClick={() => navigate(`/runs/${running.id}`)}
            className="rounded-lg border border-sidebar-border bg-background/40 p-3 text-left transition-colors hover:bg-sidebar-accent"
          >
            <div className="mb-1 flex items-center gap-2 text-xs font-medium">
              <span className="relative flex size-2">
                <span className="absolute inline-flex size-full animate-ping rounded-full bg-primary opacity-60" />
                <span className="relative inline-flex size-2 rounded-full bg-primary" />
              </span>
              {modeLabel(running.kind, running.task).title}
            </div>
            <div className="truncate text-xs text-muted-foreground">{running.stage}</div>
            <Progress
              className="mt-2 h-1"
              value={running.progress ? (100 * running.progress[0]) / running.progress[1] : null}
            />
          </button>
        )}
        <DeviceChip />
        <ThemeSwitch />
      </div>
    </aside>
  )
}

function NavItem(props: {
  icon: LucideIcon
  label: string
  href: string
  active: boolean
  badge?: string
  dot?: boolean
}) {
  const Icon = props.icon
  return (
    <button
      onClick={() => navigate(props.href)}
      className={cn(
        "group flex h-9 items-center gap-2.5 rounded-md px-2.5 text-sm transition-colors",
        props.active
          ? "bg-sidebar-accent font-medium text-sidebar-accent-foreground"
          : "text-sidebar-foreground/75 hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground",
      )}
    >
      <Icon className={cn("size-4", props.active && "text-primary")} />
      <span className="flex-1 text-left">{props.label}</span>
      {props.badge && (
        <span className="rounded-full bg-primary/15 px-1.5 text-[11px] font-semibold text-primary">
          {props.badge}
        </span>
      )}
      {props.dot && <span className="size-1.5 rounded-full bg-amber-500" />}
    </button>
  )
}

function DeviceChip() {
  const system = useQuery({
    queryKey: ["system"],
    queryFn: api.system,
    refetchInterval: (query) => (query.state.data?.runtime ? false : 1500),
  })
  const runtime = system.data?.runtime
  const gpu = runtime?.gpus?.[0]
  const label = !runtime
    ? "Checking hardware..."
    : gpu
      ? gpu.name.replace(/^NVIDIA (GeForce )?/, "")
      : runtime.mps_available
        ? "Apple GPU (MPS)"
        : "CPU only"
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          onClick={() => navigate("/settings")}
          className="flex items-center gap-2 rounded-md px-2 py-1.5 text-xs text-muted-foreground hover:bg-sidebar-accent/60"
        >
          <Cpu className={cn("size-3.5", gpu && "text-primary")} />
          <span className="truncate">{label}</span>
          {gpu && <span className="ml-auto tabular-nums">{gpu.memory_gb} GB</span>}
        </button>
      </TooltipTrigger>
      <TooltipContent side="right">
        {runtime?.torch ? `PyTorch ${runtime.torch}` : "Probing the Python runtime"}
        {runtime?.cuda_version ? ` · CUDA ${runtime.cuda_version}` : ""}
      </TooltipContent>
    </Tooltip>
  )
}

function ThemeSwitch() {
  const { theme, setTheme } = useTheme()
  const options: [ThemeChoice, LucideIcon][] = [
    ["light", Sun],
    ["dark", Moon],
    ["system", Monitor],
  ]
  return (
    <div className="flex rounded-md border border-sidebar-border p-0.5">
      {options.map(([value, Icon]) => (
        <button
          key={value}
          onClick={() => setTheme(value)}
          aria-label={`${value} theme`}
          className={cn(
            "flex h-7 flex-1 items-center justify-center rounded-[5px] text-muted-foreground transition-colors",
            theme === value ? "bg-sidebar-accent text-foreground" : "hover:text-foreground",
          )}
        >
          <Icon className="size-3.5" />
        </button>
      ))}
    </div>
  )
}
