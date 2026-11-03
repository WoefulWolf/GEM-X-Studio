import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  AlertTriangle,
  Check,
  Film,
  Footprints,
  Hand,
  ImageIcon,
  Loader2,
  PersonStanding,
  Play,
  Trash2,
  Upload,
  X,
  type LucideIcon,
} from "lucide-react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { toast } from "sonner"

import { PageHeader } from "@/components/logo"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import {
  api,
  uploadInput,
  urls,
  type InputRecord,
  type JobKind,
  type JobOptions,
  type TaskId,
} from "@/lib/api"
import { formatBytes, formatDuration } from "@/lib/format"
import { nativeApi } from "@/lib/native"
import { navigate } from "@/lib/router"
import { cn } from "@/lib/utils"

const TASKS: {
  id: TaskId
  title: string
  icon: LucideIcon
  blurb: string
  accepts: ("image" | "video")[]
}[] = [
  {
    id: "body_pose",
    title: "Body Pose",
    icon: PersonStanding,
    blurb: "Full-body 3D mesh for the main person, per image or video frame.",
    accepts: ["image", "video"],
  },
  {
    id: "hand_pose",
    title: "Hand Pose",
    icon: Hand,
    blurb: "Detailed left and right hand meshes with 25 keypoints each.",
    accepts: ["image", "video"],
  },
  {
    id: "body_motion",
    title: "Body Motion",
    icon: Footprints,
    blurb: "Temporally smooth motion in camera and world space, with foot contact.",
    accepts: ["video"],
  },
]

type Upload = { name: string; progress: number }

export function NewRunPage() {
  const queryClient = useQueryClient()
  const library = useQuery({ queryKey: ["inputs"], queryFn: api.inputs })
  const models = useQuery({ queryKey: ["models"], queryFn: api.models })
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settings })
  const [selected, setSelected] = useState<string[]>([])
  const [uploads, setUploads] = useState<Upload[]>([])
  const [task, setTask] = useState<TaskId>("body_pose")
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const inputs = library.data ?? []
  const byId = useMemo(() => new Map(inputs.map((input) => [input.id, input])), [inputs])
  const chosen = selected.map((id) => byId.get(id)).filter(Boolean) as InputRecord[]

  const addInputs = useCallback(
    (records: InputRecord[]) => {
      queryClient.setQueryData<InputRecord[]>(["inputs"], (old = []) => [
        ...records,
        ...old.filter((item) => !records.some((record) => record.id === item.id)),
      ])
      setSelected((old) => [...new Set([...old, ...records.map((record) => record.id)])])
    },
    [queryClient],
  )

  const uploadFiles = useCallback(
    async (files: File[]) => {
      for (const file of files) {
        setUploads((old) => [...old, { name: file.name, progress: 0 }])
        try {
          const record = await uploadInput(file, (progress) =>
            setUploads((old) => old.map((u) => (u.name === file.name ? { ...u, progress } : u))),
          )
          addInputs([record])
        } catch (error) {
          toast.error(`Could not add ${file.name}`, { description: (error as Error).message })
        } finally {
          setUploads((old) => old.filter((u) => u.name !== file.name))
        }
      }
    },
    [addInputs],
  )

  const browse = async () => {
    const native = await nativeApi()
    if (!native) return fileInput.current?.click()
    const result = await native.pick_media()
    addInputs(result.inputs)
    for (const error of result.errors) toast.error(error.path, { description: error.error })
  }

  const remove = useMutation({
    mutationFn: (id: string) => api.deleteInput(id),
    onSuccess: (_, id) => {
      setSelected((old) => old.filter((item) => item !== id))
      queryClient.invalidateQueries({ queryKey: ["inputs"] })
    },
  })

  const spec = TASKS.find((item) => item.id === task)!
  const usable = chosen.filter((input) => spec.accepts.includes(input.kind))
  const skipped = chosen.length - usable.length
  const ready = models.data?.tasks[task]?.ready ?? false

  return (
    <div
      className="flex min-h-full flex-col"
      onDragOver={(event) => {
        event.preventDefault()
        setDragging(true)
      }}
      onDragLeave={(event) => {
        if (event.currentTarget === event.target) setDragging(false)
      }}
      onDrop={(event) => {
        event.preventDefault()
        setDragging(false)
        uploadFiles([...event.dataTransfer.files])
      }}
    >
      <PageHeader
        title="New run"
        description="Pick images or videos, choose what to estimate, and run it on your GPU."
      />
      <input
        ref={fileInput}
        type="file"
        multiple
        accept="image/*,video/*,.mkv,.avi,.mov"
        className="hidden"
        onChange={(event) => {
          uploadFiles([...(event.target.files ?? [])])
          event.target.value = ""
        }}
      />
      <div className="grid flex-1 gap-6 p-8 xl:grid-cols-[minmax(0,1fr)_400px]">
        <section className="flex min-w-0 flex-col gap-6">
          <button
            onClick={browse}
            className={cn(
              "group relative flex flex-col items-center justify-center gap-3 overflow-hidden rounded-xl border-2 border-dashed px-6 py-10 text-center transition-all",
              dragging
                ? "border-primary bg-primary/10"
                : "border-border bg-card/40 hover:border-primary/60 hover:bg-card",
            )}
          >
            <div className="flex size-12 items-center justify-center rounded-full bg-primary/12 text-primary ring-8 ring-primary/5 transition-transform group-hover:scale-105">
              <Upload className="size-5" />
            </div>
            <div>
              <div className="font-medium">Drop images or videos here</div>
              <div className="mt-0.5 text-sm text-muted-foreground">
                or <span className="text-primary underline-offset-4 group-hover:underline">browse your files</span>{" "}
                · JPG, PNG, MP4, MOV, MKV, AVI…
              </div>
            </div>
          </button>

          {uploads.length > 0 && (
            <div className="flex flex-col gap-2">
              {uploads.map((upload) => (
                <div key={upload.name} className="flex items-center gap-3 rounded-lg border bg-card px-3 py-2 text-sm">
                  <Loader2 className="size-4 animate-spin text-primary" />
                  <span className="min-w-0 flex-1 truncate">{upload.name}</span>
                  <Progress value={upload.progress * 100} className="h-1.5 w-40" />
                </div>
              ))}
            </div>
          )}

          <div>
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-medium">
                Media library
                <span className="ml-2 font-normal text-muted-foreground">
                  {selected.length ? `${selected.length} selected` : "click to select"}
                </span>
              </h2>
              {selected.length > 0 && (
                <Button variant="ghost" size="sm" onClick={() => setSelected([])}>
                  Clear selection
                </Button>
              )}
            </div>
            {library.isLoading ? (
              <div className="h-40 animate-pulse rounded-xl bg-muted/50" />
            ) : inputs.length === 0 ? (
              <div className="rounded-xl border border-dashed px-6 py-12 text-center text-sm text-muted-foreground">
                No media yet. Files you add appear here so you can re-run them later.
              </div>
            ) : (
              <div className="grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-3">
                {inputs.map((input) => (
                  <MediaTile
                    key={input.id}
                    input={input}
                    selected={selected.includes(input.id)}
                    dimmed={selected.includes(input.id) && !spec.accepts.includes(input.kind)}
                    onToggle={() =>
                      setSelected((old) =>
                        old.includes(input.id) ? old.filter((id) => id !== input.id) : [...old, input.id],
                      )
                    }
                    onRemove={() => remove.mutate(input.id)}
                  />
                ))}
              </div>
            )}
          </div>
        </section>

        <aside className="flex flex-col gap-5 xl:sticky xl:top-6 xl:self-start">
          <div className="flex flex-col gap-2">
            <Label className="text-xs tracking-wide text-muted-foreground uppercase">What to estimate</Label>
            {TASKS.map((item) => (
              <TaskCard
                key={item.id}
                spec={item}
                active={task === item.id}
                ready={models.data?.tasks[item.id]?.ready}
                onSelect={() => setTask(item.id)}
              />
            ))}
          </div>
          <RunPanel
            task={task}
            inputs={usable}
            skipped={skipped}
            ready={ready}
            defaultDevice={settings.data?.device ?? "auto"}
            hasVipe={!!(settings.data?.vipe_executable || settings.data?.vipe_bundled)}
            onDone={() => setSelected([])}
          />
        </aside>
      </div>
    </div>
  )
}

function MediaTile(props: {
  input: InputRecord
  selected: boolean
  dimmed: boolean
  onToggle: () => void
  onRemove: () => void
}) {
  const { input } = props
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={props.onToggle}
      onKeyDown={(event) => (event.key === " " || event.key === "Enter") && props.onToggle()}
      className={cn(
        "group relative overflow-hidden rounded-lg border bg-card text-left transition-all outline-none focus-visible:ring-2 focus-visible:ring-ring",
        props.selected ? "border-primary ring-2 ring-primary/40" : "hover:border-foreground/20",
        props.dimmed && "opacity-45",
      )}
    >
      <div className="relative aspect-video bg-muted">
        <img src={urls.inputThumb(input.id)} alt="" className="size-full object-cover" loading="lazy" />
        <div className="absolute inset-x-0 bottom-0 h-12 bg-gradient-to-t from-black/60 to-transparent" />
        <div className="absolute bottom-1.5 left-2 flex items-center gap-1 text-[11px] font-medium text-white">
          {input.kind === "video" ? <Film className="size-3" /> : <ImageIcon className="size-3" />}
          {input.kind === "video" ? formatDuration(input.duration) : `${input.width}×${input.height}`}
        </div>
        <div
          className={cn(
            "absolute top-2 left-2 flex size-5 items-center justify-center rounded-full border-2 transition-all",
            props.selected
              ? "border-primary bg-primary text-primary-foreground"
              : "border-white/80 bg-black/20 opacity-0 group-hover:opacity-100",
          )}
        >
          {props.selected && <Check className="size-3" strokeWidth={3} />}
        </div>
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={(event) => {
                event.stopPropagation()
                props.onRemove()
              }}
              className="absolute top-1.5 right-1.5 flex size-6 items-center justify-center rounded-md bg-black/50 text-white opacity-0 transition-opacity group-hover:opacity-100 hover:bg-black/70"
            >
              {input.owned ? <Trash2 className="size-3.5" /> : <X className="size-3.5" />}
            </button>
          </TooltipTrigger>
          <TooltipContent>{input.owned ? "Delete uploaded copy" : "Remove from library"}</TooltipContent>
        </Tooltip>
      </div>
      <div className="px-2.5 py-2">
        <div className="truncate text-[13px] font-medium" title={input.path}>
          {input.name}
        </div>
        <div className="text-[11px] text-muted-foreground">
          {input.kind === "video"
            ? `${input.width}×${input.height} · ${input.fps?.toFixed(0)} fps · ${formatBytes(input.size)}`
            : formatBytes(input.size)}
        </div>
      </div>
    </div>
  )
}

function TaskCard(props: {
  spec: (typeof TASKS)[number]
  active: boolean
  ready?: boolean
  onSelect: () => void
}) {
  const Icon = props.spec.icon
  return (
    <button
      onClick={props.onSelect}
      className={cn(
        "flex items-start gap-3 rounded-xl border p-3.5 text-left transition-all",
        props.active
          ? "border-primary/70 bg-primary/[0.07] ring-1 ring-primary/40"
          : "bg-card hover:border-foreground/20",
      )}
    >
      <div
        className={cn(
          "flex size-9 shrink-0 items-center justify-center rounded-lg",
          props.active ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
        )}
      >
        <Icon className="size-[18px]" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{props.spec.title}</span>
          <span className="text-[11px] text-muted-foreground">
            {props.spec.accepts.length === 2 ? "image · video" : "video"}
          </span>
          {props.ready === false && (
            <Badge variant="outline" className="ml-auto border-amber-500/40 text-[10px] text-amber-600 dark:text-amber-400">
              models needed
            </Badge>
          )}
        </div>
        <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">{props.spec.blurb}</p>
      </div>
    </button>
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label className="text-[13px]">{label}</Label>
      {children}
      {hint && <p className="text-[11px] leading-relaxed text-muted-foreground">{hint}</p>}
    </div>
  )
}

function RunPanel(props: {
  task: TaskId
  inputs: InputRecord[]
  skipped: number
  ready: boolean
  defaultDevice: string
  hasVipe: boolean
  onDone: () => void
}) {
  const queryClient = useQueryClient()
  const [device, setDevice] = useState(props.defaultDevice)
  const [handView, setHandView] = useState<"exo" | "ego">("exo")
  const [hands, setHands] = useState<string[]>(["left", "right"])
  const [smooth, setSmooth] = useState(true)
  const [cameraMode, setCameraMode] = useState<"background" | "static" | "vipe">("static")
  const [postprocess, setPostprocess] = useState(true)
  const [anchorWorld, setAnchorWorld] = useState(true)
  const [range, setRange] = useState<[number, number] | null>(null)
  useEffect(() => {
    setDevice(props.defaultDevice)
  }, [props.defaultDevice])

  const single = props.inputs.length === 1 ? props.inputs[0] : null
  const motion = props.task === "body_motion"
  const total = single?.duration ?? 0
  useEffect(() => {
    setRange(single && motion ? [0, Math.min(total, 60)] : null)
  }, [single?.id, motion, total, single])

  const run = useMutation({
    mutationFn: async () => {
      const created = []
      for (const input of props.inputs) {
        const options: JobOptions = { device }
        let kind: JobKind
        if (motion) {
          kind = "motion-video"
          const start = range && single ? range[0] : 0
          const length = range && single ? range[1] - range[0] : Math.min((input.duration ?? 60) - start, 60)
          Object.assign(options, {
            camera_mode: cameraMode,
            postprocess,
            anchor_world: cameraMode === "static" && anchorWorld,
            start,
            duration: Math.max(0.5, Math.min(60, length)),
          })
        } else {
          kind = input.kind === "image" ? "pose-image" : "pose-video"
          if (props.task === "hand_pose") {
            Object.assign(options, { hand_view: handView, hands })
            if (kind === "pose-video") options.smooth_hand = handView === "ego" ? smooth : null
          }
        }
        created.push(
          ...(await api.createJobs({
            kind,
            task: props.task === "hand_pose" ? "hand" : "body",
            input_ids: [input.id],
            options,
          })),
        )
      }
      return created
    },
    onSuccess: (jobs) => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] })
      props.onDone()
      toast.success(jobs.length === 1 ? "Run queued" : `${jobs.length} runs queued`)
      navigate(jobs.length === 1 ? `/runs/${jobs[0].id}` : "/runs")
    },
    onError: (error) => toast.error("Could not start", { description: (error as Error).message }),
  })

  return (
    <div className="flex flex-col gap-4 rounded-xl border bg-card p-4">
      <Field label="Device" hint="Auto uses your NVIDIA GPU when available.">
        <Select value={device} onValueChange={setDevice}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="auto">Auto (recommended)</SelectItem>
            <SelectItem value="cuda">GPU (CUDA)</SelectItem>
            <SelectItem value="cpu">CPU (slow)</SelectItem>
          </SelectContent>
        </Select>
      </Field>

      {props.task === "hand_pose" && (
        <>
          <Field
            label="Camera viewpoint"
            hint={
              handView === "ego"
                ? "First-person: tracks the wearer's hands and stabilises crops."
                : "Third-person: every detected hand is processed independently."
            }
          >
            <ToggleGroup
              type="single"
              variant="outline"
              value={handView}
              onValueChange={(value) => value && setHandView(value as "exo" | "ego")}
              className="w-full"
            >
              <ToggleGroupItem value="exo" className="flex-1">Third-person</ToggleGroupItem>
              <ToggleGroupItem value="ego" className="flex-1">First-person</ToggleGroupItem>
            </ToggleGroup>
          </Field>
          <Field label="Hands">
            <ToggleGroup
              type="multiple"
              variant="outline"
              value={hands}
              onValueChange={(value) => value.length && setHands(value)}
              className="w-full"
            >
              <ToggleGroupItem value="left" className="flex-1">Left</ToggleGroupItem>
              <ToggleGroupItem value="right" className="flex-1">Right</ToggleGroupItem>
            </ToggleGroup>
          </Field>
          {handView === "ego" && (
            <SwitchRow label="Temporal smoothing" hint="Videos only" checked={smooth} onChange={setSmooth} />
          )}
        </>
      )}

      {motion && (
        <>
          <Field
            label="Is the camera moving?"
            hint={
              cameraMode === "static"
                ? "Tripod, phone on a stand, or any shot that doesn't pan or tilt. Best for in-place motion."
                : cameraMode === "vipe"
                  ? "Full camera tracking with ViPE: for handheld, panning or following shots."
                  : "Estimates camera rotation from background detail. Needs a textured background; plain walls can make the person drift."
            }
          >
            <ToggleGroup
              type="single"
              variant="outline"
              value={cameraMode === "static" ? "fixed" : "moving"}
              onValueChange={(value) => value && setCameraMode(value === "fixed" ? "static" : "background")}
              className="w-full"
            >
              <ToggleGroupItem value="fixed" className="flex-1">
                No, fixed
              </ToggleGroupItem>
              <ToggleGroupItem value="moving" className="flex-1">
                Yes, it pans / tilts
              </ToggleGroupItem>
            </ToggleGroup>
            {cameraMode !== "static" && (
              <Select value={cameraMode} onValueChange={(value) => setCameraMode(value as typeof cameraMode)}>
                <SelectTrigger className="mt-1 w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="background">Estimate rotation from background</SelectItem>
                  <SelectItem value="vipe" disabled={!props.hasVipe}>
                    Full camera tracking (ViPE){props.hasVipe ? "" : " - add with install.bat -Vipe"}
                  </SelectItem>
                </SelectContent>
              </Select>
            )}
          </Field>
          {cameraMode === "static" && (
            <SwitchRow
              label="Anchor to the camera"
              hint="Exact world motion for a fixed camera; no drift"
              checked={anchorWorld}
              onChange={setAnchorWorld}
            />
          )}
          <SwitchRow
            label="Contact & IK refinement"
            hint="Plants feet and removes sliding"
            checked={postprocess}
            onChange={setPostprocess}
          />
          {single && range && total > 0 && (
            <Field label="Clip" hint="Motion runs on up to 60 seconds at a time (resampled to 30 fps).">
              <Slider
                min={0}
                max={total}
                step={0.1}
                minStepsBetweenThumbs={5}
                value={range}
                onValueChange={(value) => {
                  let [start, end] = value as [number, number]
                  if (end - start > 60) {
                    if (start !== range[0]) end = start + 60
                    else start = end - 60
                  }
                  setRange([start, end])
                }}
                className="py-2"
              />
              <div className="flex justify-between text-[11px] text-muted-foreground tabular-nums">
                <span>{formatDuration(range[0])}</span>
                <span className="font-medium text-foreground">{formatDuration(range[1] - range[0])} selected</span>
                <span>{formatDuration(range[1])}</span>
              </div>
            </Field>
          )}
          {!single && props.inputs.some((input) => (input.duration ?? 0) > 60) && (
            <p className="flex gap-2 text-[11px] text-amber-600 dark:text-amber-400">
              <AlertTriangle className="size-3.5 shrink-0" />
              Videos longer than 60 s use their first 60 seconds. Select a single video to pick the range.
            </p>
          )}
        </>
      )}

      <div className="mt-1 border-t pt-4">
        {props.skipped > 0 && (
          <p className="mb-3 flex gap-2 text-xs text-muted-foreground">
            <AlertTriangle className="size-3.5 shrink-0 text-amber-500" />
            {props.skipped} image{props.skipped > 1 ? "s" : ""} skipped: Body Motion needs video.
          </p>
        )}
        {props.ready ? (
          <Button
            size="lg"
            className="w-full"
            disabled={!props.inputs.length || run.isPending}
            onClick={() => run.mutate()}
          >
            {run.isPending ? <Loader2 className="animate-spin" /> : <Play />}
            {props.inputs.length > 1
              ? `Run on ${props.inputs.length} files`
              : props.inputs.length
                ? "Run"
                : "Select media to run"}
          </Button>
        ) : (
          <Button size="lg" variant="secondary" className="w-full" onClick={() => navigate("/models")}>
            Download the models for this task
          </Button>
        )}
      </div>
    </div>
  )
}

function SwitchRow(props: { label: string; hint?: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3">
      <div>
        <div className="text-[13px] font-medium">{props.label}</div>
        {props.hint && <div className="text-[11px] text-muted-foreground">{props.hint}</div>}
      </div>
      <Switch checked={props.checked} onCheckedChange={props.onChange} />
    </label>
  )
}
