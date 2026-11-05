import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  ArrowLeft,
  Box,
  Camera,
  Clapperboard,
  Download,
  FileArchive,
  FileJson,
  Film,
  FolderOpen,
  Layers,
  Loader2,
  RotateCcw,
  Sparkles,
  Square,
  Upload,
  UserRound,
  X,
} from "lucide-react"
import { useEffect, useRef, useState, type ReactNode } from "react"
import { toast } from "sonner"

import { ActionList, ActionRow, OptionField, PanelSection } from "@/components/panel-section"
import { JobProgress, StatusBadge } from "@/components/status"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { Viewer, type ViewerHandle } from "@/components/viewer/viewer"
import {
  api,
  type ExportOptions,
  type FileEntry,
  type Job,
  type MmdModel,
  type PreviewVariant,
  type Preview,
  type RefineOptions,
  type Variant,
} from "@/lib/api"
import { formatBytes, formatDuration, jobElapsed, modeLabel, timeAgo } from "@/lib/format"
import { saveJobFile } from "@/lib/native"
import { navigate } from "@/lib/router"
import { readStored, writeStored } from "@/lib/storage"
import { cn } from "@/lib/utils"
import { RunMenu } from "@/pages/runs"

/** Poll the job and append its log incrementally. */
function useJob(jobId: string) {
  const [job, setJob] = useState<Job | null>(null)
  const [log, setLog] = useState("")
  const [missing, setMissing] = useState(false)
  const offset = useRef(0)
  useEffect(() => {
    let stopped = false
    let timer = 0
    const tick = async () => {
      try {
        const next = await api.job(jobId, offset.current)
        if (stopped) return
        offset.current = next.log_offset ?? offset.current
        if (next.log) setLog((old) => (old + next.log).slice(-400_000))
        setJob(next)
        const refining = next.refine?.status === "running" || next.refine?.status === "queued"
        const active = next.status === "running" || next.status === "queued" || refining
        timer = window.setTimeout(tick, active ? 900 : 5000)
      } catch {
        if (!stopped) setMissing(true)
      }
    }
    void tick()
    return () => {
      stopped = true
      window.clearTimeout(timer)
    }
  }, [jobId])
  return { job, log, missing }
}

export function RunDetailPage({ jobId }: { jobId: string }) {
  const { job, log, missing } = useJob(jobId)
  const viewer = useRef<ViewerHandle>(null)
  const done = job?.status === "done"
  const refined = job?.refine?.status === "done"
  // Open on the refined result whenever there is one; Raw stays one click away.
  const [variant, setVariant] = useState<Variant>("refined")
  const raw = useQuery({
    queryKey: ["preview", jobId, "raw"],
    queryFn: () => api.preview(jobId, "raw"),
    enabled: done,
    staleTime: Infinity,
  })
  // Keyed on the finish time so a re-refinement reloads the bundle.
  const refinedPreview = useQuery({
    queryKey: ["preview", jobId, "refined", job?.refine?.finished],
    queryFn: () => api.preview(jobId, "refined"),
    enabled: done && refined,
    staleTime: Infinity,
    // A new refinement keeps the previous one on screen until it has loaded.
    placeholderData: keepPreviousData,
  })
  // On opening, wait for the refined bundle rather than flashing the raw one first. Once the
  // viewer is up it stays mounted (keeping its view mode and camera): while a first
  // refinement loads, it keeps showing the raw result.
  const viewerShown = useRef(false)
  let shown: Variant = variant === "refined" && refined ? "refined" : "raw"
  let shownData = shown === "refined" ? refinedPreview.data : raw.data
  if (!shownData && viewerShown.current && raw.data) {
    shown = "raw"
    shownData = raw.data
  }
  if (shownData) viewerShown.current = true
  const preview = { data: shownData }
  // Fit to model: the same motion on a body with the model's limb proportions (World, 3D).
  const mmd = useMmdModel()
  const [body, setBody] = useState<"gem" | "model">("gem")
  const modelAvailable = !!mmd.model && shownData?.type === "motion"
  const modelOn = body === "model" && modelAvailable
  const modelPreview = useQuery({
    queryKey: ["model-preview", jobId, shown, mmd.model?.id, job?.refine?.finished],
    queryFn: async () => {
      const { variant } = await api.modelPreview(jobId, shown, mmd.model!.id)
      return { variant, data: await api.preview(jobId, variant) }
    },
    enabled: modelOn && !!shownData,
    staleTime: Infinity,
    retry: false,
  })
  useEffect(() => {
    if (!modelPreview.isError) return
    toast.error("Could not build the model preview", { description: (modelPreview.error as Error).message })
    setBody("gem")
  }, [modelPreview.isError, modelPreview.error])
  const viewerData = modelOn && modelPreview.data ? modelPreview.data.data : shownData
  const viewerVariant: PreviewVariant = modelOn && modelPreview.data ? modelPreview.data.variant : shown

  if (missing) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-sm text-muted-foreground">
        This run no longer exists.
        <Button variant="outline" onClick={() => navigate("/runs")}>
          Back to runs
        </Button>
      </div>
    )
  }
  if (!job) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="size-5 animate-spin text-muted-foreground" />
      </div>
    )
  }
  const label = modeLabel(job.kind, job.task)
  const active = job.status === "running" || job.status === "queued"

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex items-center gap-3 border-b px-6 py-3.5">
        <Button variant="ghost" size="icon" className="size-8" onClick={() => navigate("/runs")} aria-label="Back">
          <ArrowLeft />
        </Button>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2.5">
            <h1 className="truncate text-base font-semibold">{job.title}</h1>
            <StatusBadge status={job.status} />
          </div>
          <div className="text-xs text-muted-foreground">
            {label.title} · {job.kind.endsWith("image") ? "image" : "video"} · started {timeAgo(job.created)}
            {job.device ? ` · ${job.device.toUpperCase()}` : ""}
            {jobElapsed(job) != null ? ` · ${formatDuration(jobElapsed(job))}` : ""}
          </div>
        </div>
        {active && (
          <Button variant="outline" size="sm" onClick={() => api.cancel(job.id)}>
            <Square /> Cancel
          </Button>
        )}
        {(job.status === "failed" || job.status === "cancelled") && (
          <Button
            size="sm"
            onClick={async () => navigate(`/runs/${(await api.retry(job.id)).id}`)}
          >
            <RotateCcw /> Run again
          </Button>
        )}
        <Button variant="outline" size="sm" onClick={() => api.reveal(job.id)}>
          <FolderOpen /> Folder
        </Button>
        <RunMenu job={job} onDeleted={() => navigate("/runs")} />
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_340px] gap-4 p-4">
        <div className="flex min-h-0 min-w-0 flex-col gap-3">
          {done && raw.data && <CameraDriftBanner job={job} preview={raw.data} />}
          <div className="min-h-0 flex-1">
          {done && preview.data ? (
            <Viewer
              ref={viewer}
              jobId={job.id}
              preview={viewerData ?? preview.data}
              variant={viewerVariant}
              center={
                (refined || modelAvailable) && (
                  <div className="flex max-w-full flex-wrap items-center justify-center gap-2">
                    {refined && (
                      <ToggleGroup
                        type="single"
                        value={shown}
                        onValueChange={(value) => value && setVariant(value as Variant)}
                        aria-label="Result"
                        className="gap-0.5 rounded-lg bg-black/60 p-1 text-white shadow-lg ring-1 ring-primary/50 backdrop-blur"
                      >
                        {(["raw", "refined"] as const).map((value) => (
                          <ToggleGroupItem
                            key={value}
                            value={value}
                            className="h-8 gap-1.5 rounded-md px-4 text-sm font-medium text-white/70 capitalize hover:bg-white/10 hover:text-white data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:shadow-sm data-[state=on]:hover:bg-primary"
                          >
                            {value === "refined" && <Sparkles className="size-4" />}
                            {value}
                          </ToggleGroupItem>
                        ))}
                      </ToggleGroup>
                    )}
                    {modelAvailable && (
                      <ToggleGroup
                        type="single"
                        value={body}
                        onValueChange={(value) => value && setBody(value as "gem" | "model")}
                        aria-label="Body proportions"
                        className="gap-0.5 rounded-lg bg-black/60 p-1 text-white shadow-lg ring-1 ring-white/15 backdrop-blur"
                      >
                        <ToggleGroupItem
                          value="gem"
                          title="The dancer's proportions, as GEM estimated them"
                          className="h-8 rounded-md px-3 text-sm font-medium text-white/70 hover:bg-white/10 hover:text-white data-[state=on]:bg-white/90 data-[state=on]:text-black data-[state=on]:hover:bg-white/90"
                        >
                          GEM
                        </ToggleGroupItem>
                        <ToggleGroupItem
                          value="model"
                          title={`Preview with ${mmd.model!.name}'s limb proportions, as the fitted VMD moves it (Fit to model)`}
                          className="h-8 max-w-[11rem] gap-1.5 rounded-md px-3 text-sm font-medium text-white/70 hover:bg-white/10 hover:text-white data-[state=on]:bg-white/90 data-[state=on]:text-black data-[state=on]:hover:bg-white/90"
                        >
                          {modelOn && modelPreview.isFetching ? (
                            <Loader2 className="size-4 shrink-0 animate-spin" />
                          ) : (
                            <UserRound className="size-4 shrink-0" />
                          )}
                          <span className="truncate">{mmd.model!.name}</span>
                        </ToggleGroupItem>
                      </ToggleGroup>
                    )}
                  </div>
                )
              }
            />
          ) : (
            <StatusStage job={job} log={log} />
          )}
          </div>
        </div>
        <Tabs defaultValue={done ? "export" : "log"} key={String(done)} className="min-h-0">
          <TabsList className="w-full">
            <TabsTrigger value="export" disabled={!done}>
              Export
            </TabsTrigger>
            {job.kind !== "pose-image" && (
              <TabsTrigger value="refine" disabled={!done}>
                Refine
              </TabsTrigger>
            )}
            <TabsTrigger value="details">Details</TabsTrigger>
            <TabsTrigger value="log">Log</TabsTrigger>
          </TabsList>
          <TabsContent value="export" className="min-h-0 overflow-y-auto">
            {done && preview.data && (
              <ExportPanel job={job} preview={preview.data} viewer={viewer} variant={shown} mmd={mmd} />
            )}
          </TabsContent>
          <TabsContent value="refine" className="min-h-0 overflow-y-auto">
            {done &&
              (job.kind === "motion-video" ? (
                <MotionRefinePanel job={job} onRefined={() => setVariant("refined")} />
              ) : (
                <RefinePanel job={job} onRefined={() => setVariant("refined")} />
              ))}
          </TabsContent>
          <TabsContent value="details" className="min-h-0 overflow-y-auto">
            <Details job={job} preview={preview.data} />
          </TabsContent>
          <TabsContent value="log" className="min-h-0">
            <LogView text={log} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

/** Motion runs that estimated camera rotation: offer a fixed-camera re-run. */
function CameraDriftBanner({ job, preview }: { job: Job; preview: Preview }) {
  const dismissKey = `gem-studio:camera-banner-dismissed:${job.id}`
  const [dismissed, setDismissed] = useState(() => readStored(dismissKey, false))
  const camera = preview.camera
  const rotation = camera?.max_rotation_deg ?? 0
  // ViPE is chosen for a moving camera, so its rotation is expected rather than a warning sign.
  const vipe = job.options.camera_mode === "vipe" || camera?.mode === "vipe"
  if (preview.type !== "motion" || !camera || camera.mode === "static" || vipe || rotation < 2 || dismissed) {
    return null
  }
  return (
    <Alert className="relative border-amber-500/40 bg-amber-500/[0.06] py-2.5 pr-10">
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-1.5 right-1.5 size-7 text-muted-foreground"
        aria-label="Dismiss"
        title="Dismiss for this run"
        onClick={() => {
          writeStored(dismissKey, true)
          setDismissed(true)
        }}
      >
        <X />
      </Button>
      <AlertTitle className="text-[13px]">
        Camera movement was estimated: up to {rotation.toFixed(1)}° of rotation
      </AlertTitle>
      <AlertDescription className="flex items-center justify-between gap-4 text-xs">
        <span>
          If the camera did not actually move, this makes the person drift or rise in the World view. Re-run as a
          fixed camera to anchor the motion in place.
        </span>
        <Button
          size="sm"
          variant="outline"
          className="shrink-0"
          onClick={async () => {
            const [created] = await api.createJobs({
              kind: job.kind,
              task: job.task,
              input_ids: [job.input.id],
              options: { ...job.options, camera_mode: "static", anchor_world: true },
            })
            navigate(`/runs/${created.id}`)
          }}
        >
          Re-run as fixed camera
        </Button>
      </AlertDescription>
    </Alert>
  )
}

function StatusStage({ job, log }: { job: Job; log: string }) {
  const last = log.trim().split("\n").slice(-1)[0]
  return (
    <div className="relative flex h-full flex-col items-center justify-center overflow-hidden rounded-xl border bg-card">
      <img
        src={`/api/jobs/${job.id}/thumb`}
        alt=""
        className="absolute inset-0 size-full scale-110 object-cover opacity-20 blur-2xl"
      />
      <div className="relative w-full max-w-md px-8">
        {job.status === "failed" ? (
          <Alert variant="destructive" className="bg-card/90">
            <AlertTitle>Run failed</AlertTitle>
            <AlertDescription className="break-words">
              {job.error}
              <span className="mt-2 block text-xs text-muted-foreground">The Log tab has the full output.</span>
            </AlertDescription>
          </Alert>
        ) : job.status === "cancelled" ? (
          <div className="text-center text-sm text-muted-foreground">This run was cancelled.</div>
        ) : (
          <div className="flex flex-col gap-4 rounded-xl border bg-card/85 p-6 shadow-xl backdrop-blur">
            <div className="flex items-center gap-3">
              <img src={`/api/jobs/${job.id}/thumb`} alt="" className="size-14 rounded-lg object-cover" />
              <div className="min-w-0">
                <div className="truncate font-medium">{job.title}</div>
                <div className="text-xs text-muted-foreground">
                  {job.status === "queued" ? "Waiting for the GPU…" : `Elapsed ${formatDuration(jobElapsed(job))}`}
                </div>
              </div>
            </div>
            <JobProgress job={job} />
            {last && <div className="truncate font-mono text-[11px] text-muted-foreground">{last}</div>}
          </div>
        )}
      </div>
    </div>
  )
}

function LogView({ text }: { text: string }) {
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" })
  }, [text])
  return (
    <ScrollArea className="h-[calc(100vh-150px)] rounded-lg border bg-muted/30">
      <pre className="p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-muted-foreground">
        {text || "No output yet."}
      </pre>
      <div ref={end} />
    </ScrollArea>
  )
}

function Details({ job, preview }: { job: Job; preview?: Preview }) {
  const rows: [string, React.ReactNode][] = [
    ["Input", <span className="break-all">{job.input.path}</span>],
    ["Resolution", `${job.input.width} × ${job.input.height}`],
  ]
  if (job.input.kind === "video") {
    rows.push(["Source", `${formatDuration(job.input.duration)} · ${job.input.fps} fps · ${job.input.frame_count} frames`])
  }
  rows.push(["Device", job.device?.toUpperCase() ?? "-"])
  if (job.inference_seconds) rows.push(["Inference", formatDuration(job.inference_seconds)])
  if (preview) {
    rows.push(["Frames", String(preview.frame_count)])
    rows.push(["Result", `${preview.variant}${preview.fps ? ` · ${Number(preview.fps.toFixed(2))} fps` : ""}`])
    const people = new Set(preview.frames.flatMap((frame) => frame.instances.map((item) => item.track)))
    rows.push([job.task === "hand" ? "Hands" : "Tracks", String(people.size)])
  }
  for (const [key, value] of Object.entries(job.options)) {
    if (value == null || key === "vipe_executable") continue
    rows.push([key.replace(/_/g, " "), Array.isArray(value) ? value.join(", ") : String(value)])
  }
  return (
    <dl className="grid grid-cols-[110px_1fr] gap-x-3 gap-y-2.5 rounded-lg border bg-card p-4 text-xs">
      {rows.map(([key, value]) => (
        <div key={key} className="contents">
          <dt className="text-muted-foreground capitalize">{key}</dt>
          <dd className="min-w-0">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

// -- exports -----------------------------------------------------------------

async function deliver(jobId: string, entry: FileEntry) {
  const saved = await saveJobFile(jobId, entry.path)
  if (saved && saved !== "browser") toast.success("Saved", { description: saved })
}

function ExportPanel({
  job,
  preview,
  viewer,
  variant,
  mmd,
}: {
  job: Job
  preview: Preview
  viewer: React.RefObject<ViewerHandle | null>
  variant: Variant
  mmd: MmdModelChoice
}) {
  const queryClient = useQueryClient()
  const files = useQuery({ queryKey: ["files", job.id], queryFn: () => api.files(job.id) })
  const system = useQuery({ queryKey: ["system"], queryFn: api.system })
  const [busy, setBusy] = useState<string | null>(null)
  const [renderProgress, setRenderProgress] = useState<number | null>(null)
  const [space, setSpace] = useState<"world" | "camera">(preview.views.includes("world") ? "world" : "camera")
  const [step, setStep] = useState("1")
  const isMotion = preview.type === "motion"
  // Skeletons come from GEM's SOMA rig (Body Motion); Pose videos export mesh shapes.
  const [format, setFormat] = useState<"skeleton" | "shapes">(isMotion ? "skeleton" : "shapes")
  const [naming, setNaming] = useState<"soma" | "mmd">("soma")
  const [armAngle, setArmAngle] = useState(35)
  const [mmdScale, setMmdScale] = useState("12.5")
  const [faceForward, setFaceForward] = useState(true)
  const fitModel = mmd.model
  // A moving camera (ViPE or background estimate) is worth exporting by default.
  const movingCamera = !!preview.camera?.mode && preview.camera.mode !== "static"
  const [includeCamera, setIncludeCamera] = useState(movingCamera)
  const abort = useRef<AbortController | null>(null)
  const isVideo = preview.mode === "video"
  const hasWorld = preview.views.includes("world")
  const stem = job.title.replace(/\.[^.]+$/, "") + (variant === "refined" ? "_refined" : "")
  const blender = system.data?.blender
  const exportOptions: ExportOptions = {
    variant,
    space,
    step: Number(step),
    naming,
    mode: format,
    camera: includeCamera,
  }

  // Morph-target size: one float32 position delta per vertex per sampled frame and track.
  const prefix = space === "world" && hasWorld ? "world_" : "camera_"
  const tracks = new Map<string, number>()
  for (const frame of preview.frames) {
    for (const item of [...frame.instances, ...(frame.world ?? [])]) {
      if (item.stream.startsWith(prefix)) tracks.set(`${item.stream}|${item.track}`, preview.streams[item.stream].vertex_count)
    }
  }
  const sampled = Math.ceil(preview.frame_count / Number(step))
  const estimate =
    format === "skeleton"
      ? 2_000_000 + sampled * 77 * 16 // bind mesh + one quaternion per joint per frame
      : [...tracks.values()].reduce((total, vertices) => total + sampled * vertices * 12, 0)
  const fps = preview.fps ?? 30

  const run = async (id: string, work: () => Promise<FileEntry | void>) => {
    setBusy(id)
    try {
      const entry = await work()
      await queryClient.invalidateQueries({ queryKey: ["files", job.id] })
      if (entry) await deliver(job.id, entry)
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        toast.error("Export failed", { description: (error as Error).message })
      }
    } finally {
      setBusy(null)
    }
  }

  const frameSuffix = () => (isVideo ? `_f${String(viewer.current!.frame()).padStart(5, "0")}` : "")
  const clientExport = (id: string, name: () => string, make: () => Promise<Blob>) =>
    run(id, async () => api.uploadExport(job.id, name(), await make()))
  const serverExport = (kind: string, extra: Partial<ExportOptions> = {}) =>
    run(kind, () => api.makeExport(job.id, kind, { ...exportOptions, ...extra }))
  const predictionFiles = (files.data ?? []).filter((file) =>
    variant === "refined" ? file.name.startsWith("predictions_refined.") : /^predictions\.(pt|json)$/.test(file.name),
  )

  const exported = (files.data ?? []).filter((file) => file.path.startsWith("exports/")).reverse()
  const ordinal = (value: number) => (value === 2 ? "nd" : value === 3 ? "rd" : "th")

  return (
    <div className="flex flex-col gap-3 pb-4">
      {variant === "refined" && (
        <div className="flex items-center gap-2 rounded-lg border border-primary/30 bg-primary/[0.06] px-3 py-2 text-xs">
          <Sparkles className="size-3.5 text-primary" /> Exporting the refined result (see the Refine tab)
        </div>
      )}
      {isVideo && (
        <PanelSection
          id="export-animation"
          title="Animation"
          summary={[
            format === "skeleton" ? "Skeleton" : "Shape keys",
            format === "skeleton" && (naming === "mmd" ? "MMD names" : "Standard names"),
            hasWorld && (space === "world" ? "World" : "Camera"),
            includeCamera && "with camera",
            step !== "1" && `every ${step}${ordinal(Number(step))}`,
          ]
            .filter(Boolean)
            .join(" · ")}
        >
          <OptionField
            label="Format"
            hint={
              format === "skeleton"
                ? "Animated armature + skinned body. Small, retargetable; skinning differs from GEM's surface by ~3 mm."
                : isMotion
                  ? "Every frame's exact mesh (keeps pose correctives). Large; no bones."
                  : "Every frame's exact mesh. Skeletons are available for Body Motion runs."
            }
          >
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={format}
              onValueChange={(value) => value && setFormat(value as "skeleton" | "shapes")}
              className="w-full"
            >
              <ToggleGroupItem value="skeleton" className="flex-1" disabled={!isMotion}>
                Skeleton (skinned)
              </ToggleGroupItem>
              <ToggleGroupItem value="shapes" className="flex-1">
                Shape keys
              </ToggleGroupItem>
            </ToggleGroup>
          </OptionField>
          {format === "skeleton" && (
            <OptionField
              label="Bone names"
              hint={
                naming === "mmd"
                  ? "センター, 上半身, 左腕, 左ひざ… (MMD standard bones)"
                  : "Hips, Spine1, LeftArm, LeftShin… (SOMA joints)"
              }
            >
              <ToggleGroup
                type="single"
                variant="outline"
                size="sm"
                value={naming}
                onValueChange={(value) => value && setNaming(value as "soma" | "mmd")}
                className="w-full"
              >
                <ToggleGroupItem value="soma" className="flex-1">
                  Standard
                </ToggleGroupItem>
                <ToggleGroupItem value="mmd" className="flex-1">
                  MMD
                </ToggleGroupItem>
              </ToggleGroup>
            </OptionField>
          )}
          {hasWorld && (
            <OptionField label="Space">
              <ToggleGroup
                type="single"
                variant="outline"
                size="sm"
                value={space}
                onValueChange={(value) => value && setSpace(value as "world" | "camera")}
                className="w-full"
              >
                <ToggleGroupItem value="world" className="flex-1">
                  World (moves)
                </ToggleGroupItem>
                <ToggleGroupItem value="camera" className="flex-1">
                  Camera
                </ToggleGroupItem>
              </ToggleGroup>
            </OptionField>
          )}
          <SwitchField
            label="Include camera"
            hint={
              space === "world" && hasWorld
                ? "The capture camera, keyed with the body (position, rotation, field of view): look through it to see the video's framing."
                : "A camera at the origin with the video's field of view."
            }
            checked={includeCamera}
            onChange={setIncludeCamera}
          />
          <OptionField
            label="Frame rate"
            hint={`${sampled} keyframes · ${
              format === "skeleton" ? "77 joints" : `${tracks.size} track${tracks.size === 1 ? "" : "s"}`
            } · ≈ ${formatBytes(estimate)}`}
          >
            <Select value={step} onValueChange={setStep}>
              <SelectTrigger size="sm" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {[1, 2, 3, 4].map((value) => (
                  <SelectItem key={value} value={String(value)}>
                    {value === 1 ? "Every frame" : `Every ${value}${ordinal(value)} frame`} ·{" "}
                    {Number((fps / value).toFixed(1))} fps
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </OptionField>
          <ActionList>
            <ActionRow
              icon={Film}
              title="glTF animation (.glb)"
              hint={
                format === "skeleton"
                  ? "Skinned skeleton · Blender, Unity, Unreal, three.js · no Blender needed"
                  : "Shape-key animation · Blender, Unity, Unreal, three.js"
              }
              busy={busy === "gltf_animation" || busy === "gltf_skeleton"}
              onClick={() => serverExport(format === "skeleton" ? "gltf_skeleton" : "gltf_animation")}
            />
            <Tooltip>
              <TooltipTrigger asChild>
                <div>
                  <ActionRow
                    icon={Film}
                    title="FBX animation"
                    hint={
                      !blender
                        ? "Needs Blender (set path in Settings)"
                        : format === "skeleton"
                          ? "Armature + skin · Maya, Max, Unity, Unreal"
                          : "Blend-shape animation · Maya, Max, Unity, Unreal"
                    }
                    busy={busy === "fbx"}
                    disabled={!blender}
                    onClick={() => serverExport("fbx")}
                  />
                </div>
              </TooltipTrigger>
              <TooltipContent className="max-w-64">
                {blender
                  ? `Converted by Blender: ${blender}`
                  : "FBX is written by a local Blender install (free). Install Blender, or set blender.exe in Settings."}
              </TooltipContent>
            </Tooltip>
            <ActionRow
              icon={Layers}
              title="Point cache (OBJ + PC2)"
              hint="Every frame, all tracks · Mesh Cache modifier"
              busy={busy === "point_cache"}
              onClick={() => serverExport("point_cache")}
            />
            <ActionRow
              icon={FileArchive}
              title="OBJ sequence (zip)"
              hint="One OBJ per frame and track"
              busy={busy === "obj_sequence"}
              onClick={() => serverExport("obj_sequence")}
            />
          </ActionList>
        </PanelSection>
      )}

      {isVideo && isMotion && (
        <PanelSection
          id="export-mmd"
          title="MMD (MikuMikuDance)"
          summary={
            fitModel
              ? `Fitted to ${fitModel.name}`
              : `${armAngle}° arms · ${mmdScale === "12.5" ? "standard scale" : `scale ${mmdScale}`}`
          }
        >
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            VMD motion for any MMD model, matched by standard bone names (センター, 上半身, 左腕, 左足…, fingers). Leg IK is
            switched off so the legs follow the capture exactly. The skeleton glTF/FBX above can also use MMD bone
            names.
          </p>
          <MmdModelField
            model={fitModel}
            loading={mmd.loading}
            onChange={mmd.choose}
          />
          {fitModel ? (
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              Arm angle ({fitModel.arm_angle_deg}°) and scale come from the model; scale matches its legs (
              {fitModel.leg_length} units). Each frame the model&apos;s lowest point is put where the dancer&apos;s
              was, so hands, feet or knees meet the floor despite different proportions.
            </p>
          ) : (
            <>
              <RangeField
                label="Model arm angle (A-pose)"
                value={armAngle}
                display={`${armAngle}°`}
                min={0}
                max={50}
                step={1}
                left="T-pose (0°)"
                right="Most MMD models: 30–40°"
                onChange={setArmAngle}
              />
              <OptionField label="Scale">
                <Select value={mmdScale} onValueChange={setMmdScale}>
                  <SelectTrigger size="sm" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="12.5">Standard · 1 unit = 8 cm</SelectItem>
                    <SelectItem value="10">Smaller model · 1 unit = 10 cm</SelectItem>
                    <SelectItem value="15">Larger model · 1 unit ≈ 6.7 cm</SelectItem>
                  </SelectContent>
                </Select>
              </OptionField>
            </>
          )}
          <SwitchField
            label="Face the camera at the start"
            checked={faceForward}
            onChange={setFaceForward}
          />
          <ActionList>
            <ActionRow
              icon={Film}
              title="VMD motion (.vmd)"
              hint={`${fitModel ? `Fitted to ${fitModel.name} · ` : ""}MikuMikuDance, MMM, mmd_tools · 30 fps`}
              busy={busy === "vmd"}
              onClick={() =>
                serverExport(
                  "vmd",
                  fitModel
                    ? { model: fitModel.id, arm_angle_deg: null, scale: "auto", face_forward: faceForward }
                    : { arm_angle_deg: armAngle, scale: Number(mmdScale), face_forward: faceForward },
                )
              }
            />
          </ActionList>
        </PanelSection>
      )}

      <PanelSection id="export-frame" title="Current frame" summary="GLB · OBJ · PNG">
        <ActionList>
          <ActionRow
            icon={Box}
            title="Mesh · GLB"
            hint="Blender, Unity, Unreal, web"
            busy={busy === "glb"}
            onClick={() =>
              clientExport("glb", () => `${stem}${frameSuffix()}.glb`, () => viewer.current!.exportFrame("glb"))
            }
          />
          <ActionRow
            icon={Box}
            title="Mesh · OBJ"
            busy={busy === "obj"}
            onClick={() =>
              clientExport("obj", () => `${stem}${frameSuffix()}.obj`, () => viewer.current!.exportFrame("obj"))
            }
          />
          <ActionRow
            icon={Camera}
            title="Screenshot (PNG)"
            hint="Current view at full resolution"
            busy={busy === "png"}
            onClick={() =>
              clientExport("png", () => `${stem}${frameSuffix()}_${viewer.current!.view()}.png`, () =>
                viewer.current!.screenshot(),
              )
            }
          />
        </ActionList>
      </PanelSection>

      {isVideo && (
        <PanelSection id="export-video" title="Video" summary="MP4 render">
          <ActionList>
            <ActionRow
              icon={Clapperboard}
              title="Render video (MP4)"
              hint="Exactly what the viewer shows, frame by frame"
              onClick={() => undefined}
              control={
                renderProgress == null ? (
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={busy != null}
                    onClick={() => {
                      const controller = new AbortController()
                      abort.current = controller
                      setRenderProgress(0)
                      const view = viewer.current!.view()
                      void run("video", async () => {
                        const name = `${stem}_${view}.mp4`
                        await viewer.current!.renderVideo(name, setRenderProgress, controller.signal)
                        const list = await api.files(job.id)
                        return list.find((file) => file.name === name)
                      }).finally(() => setRenderProgress(null))
                    }}
                  >
                    Render
                  </Button>
                ) : (
                  <Button size="sm" variant="outline" onClick={() => abort.current?.abort()}>
                    Stop
                  </Button>
                )
              }
            >
              {renderProgress != null && (
                <div className="flex items-center gap-2">
                  <Progress value={renderProgress * 100} className="h-1.5" />
                  <span className="w-9 text-right text-[11px] text-muted-foreground tabular-nums">
                    {Math.round(renderProgress * 100)}%
                  </span>
                </div>
              )}
            </ActionRow>
          </ActionList>
        </PanelSection>
      )}

      <PanelSection id="export-data" title="Data" summary="Predictions · NumPy" defaultOpen={false}>
        <ActionList>
          {predictionFiles.map((file) => (
            <ActionRow
              key={file.name}
              icon={file.name.endsWith(".json") ? FileJson : Layers}
              title={file.name.endsWith(".json") ? "Parameters & keypoints (JSON)" : "Full GEM result (.pt)"}
              hint={formatBytes(file.size)}
              action="Save"
              onClick={() => deliver(job.id, file)}
            />
          ))}
          <ActionRow
            icon={FileArchive}
            title="NumPy arrays (.npz)"
            hint="Vertices, faces, frame mapping"
            busy={busy === "npz"}
            onClick={() => serverExport("npz")}
          />
        </ActionList>
      </PanelSection>

      {exported.length > 0 && (
        <PanelSection
          id="export-files"
          title="Exported files"
          summary={`${exported.length} file${exported.length === 1 ? "" : "s"}`}
          defaultOpen={false}
        >
          <div className="-mx-2 -my-2 flex flex-col">
            {exported.map((file) => (
              <button
                key={file.path}
                onClick={() => deliver(job.id, file)}
                className="group flex items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs hover:bg-accent"
              >
                <Download className="size-3.5 text-muted-foreground group-hover:text-foreground" />
                <span className="min-w-0 flex-1 truncate">{file.name}</span>
                <span className="text-muted-foreground tabular-nums">{formatBytes(file.size)}</span>
              </button>
            ))}
          </div>
        </PanelSection>
      )}
    </div>
  )
}

const DETAIL_LABELS: Record<RefineOptions["detail"], string> = {
  smooth: "Smoother",
  balanced: "Balanced",
  sharp: "Sharper",
}

/** Shared: toast and switch to Refined when a refinement finishes. */
function useRefineCompletion(job: Job, onRefined: () => void) {
  const queryClient = useQueryClient()
  const current = job.refine
  const previous = useRef(current?.status)
  useEffect(() => {
    if (previous.current && previous.current !== "done" && current?.status === "done") {
      toast.success("Refined result ready")
      queryClient.invalidateQueries({ queryKey: ["files", job.id] })
      onRefined()
    }
    if (previous.current !== "failed" && current?.status === "failed") {
      toast.error("Refinement failed", { description: current.error ?? undefined })
    }
    previous.current = current?.status
  }, [current?.status, current?.error, job.id, onRefined, queryClient])
}

const MOTION_DEFAULTS: RefineOptions = {
  world: "anchor",
  strength: 0,
  detail: "balanced",
  lock_shape: true,
  plant_feet: true,
  contact_threshold: 0.5,
  floor_offset_cm: 0,
  level: "auto",
  tilt_deg: 0,
  roll_deg: 0,
  depth_smoothing: 4,
  lock_feet: true,
  plant_hands: true,
}

const signed = (value: number, unit: string) => `${value > 0 ? "+" : ""}${value.toFixed(1)}${unit}`

function MotionRefinePanel({ job, onRefined }: { job: Job; onRefined: () => void }) {
  useRefineCompletion(job, onRefined)
  const current = job.refine
  const active = current?.status === "running" || current?.status === "queued"
  const fixedCamera = job.options.camera_mode === "static"
  // A moving camera keeps GEM's world, which is already gravity-aligned: no auto-level
  // (the drifting floor would tilt it), and the floor follows the feet instead.
  const cameraDefaults: Partial<RefineOptions> = fixedCamera
    ? { world: "anchor" }
    : { world: "gem", level: "off", follow_floor: true }
  const [options, setOptions] = useState<RefineOptions>(() => ({
    ...MOTION_DEFAULTS,
    ...cameraDefaults,
    ...(current?.options ?? {}),
  }))
  const set = (patch: Partial<RefineOptions>) => setOptions((old) => ({ ...old, ...patch }))
  const report = current?.status === "done" ? current.report : null
  const plant = report?.plant_feet
  const sensitivity = 1 - (options.contact_threshold ?? 0.5)
  const tilt = options.tilt_deg ?? 0
  const roll = options.roll_deg ?? 0
  const offset = options.floor_offset_cm ?? 0

  return (
    <div className="flex flex-col gap-3 pb-4">
      <PanelSection
        id="refine-world"
        title="World trajectory"
        summary={options.world === "anchor" ? "Anchored to camera" : "GEM world"}
      >
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={options.world}
          onValueChange={(world) => world && set({ world: world as RefineOptions["world"] })}
          className="w-full"
        >
          <ToggleGroupItem value="anchor" className="flex-1">
            Anchored to camera
          </ToggleGroupItem>
          <ToggleGroupItem value="gem" className="flex-1">
            GEM world
          </ToggleGroupItem>
        </ToggleGroup>
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          {options.world === "anchor"
            ? "Exact for a camera that did not move: the camera-space motion under GEM's gravity estimate. No drift."
            : "GEM's integrated world motion, for a camera that moved. Its height can drift; Follow the floor (Floor & contacts) removes that."}
          {options.world === "anchor" && !fixedCamera && (
            <span className="mt-1 block text-amber-600 dark:text-amber-400">
              This run assumed a moving camera; anchoring is only correct if it was actually still.
            </span>
          )}
        </p>
      </PanelSection>

      <PanelSection
        id="refine-stability"
        title="Depth stability"
        summary={options.depth_smoothing ? `${options.depth_smoothing.toFixed(1)} frames` : "Off"}
      >
        <RangeField
          label="Depth stabilization"
          value={options.depth_smoothing ?? 0}
          display={options.depth_smoothing ? `${options.depth_smoothing.toFixed(1)} frames` : "Off"}
          min={0}
          max={10}
          step={0.5}
          left="Off"
          right="Strong"
          onChange={(depth_smoothing) => set({ depth_smoothing })}
        />
        <p className="-mt-2 text-[11px] leading-relaxed text-muted-foreground">
          A single camera judges distance far less precisely than position, so the body wobbles toward and away from
          the camera. This smooths only that distance, along each frame&apos;s line of sight, so the overlay still
          lines up.
        </p>
      </PanelSection>

      <PanelSection
        id="refine-level"
        title="Level & tilt"
        summary={[
          options.level === "off" ? "Auto-level off" : "Auto-level",
          tilt && `lean ${signed(tilt, "°")}`,
          roll && `sideways ${signed(roll, "°")}`,
        ]
          .filter(Boolean)
          .join(" · ")}
      >
        <SwitchField
          label="Auto-level on the floor"
          hint="Fixes a camera tilted up or down: the planted feet define the floor"
          checked={options.level !== "off"}
          onChange={(on) => set({ level: on ? "auto" : "off" })}
        />
        {!fixedCamera && options.world === "gem" && options.level !== "off" && (
          <p className="-mt-2 text-[11px] leading-relaxed text-amber-600 dark:text-amber-400">
            With a moving camera GEM&apos;s world is already level; drift in the trajectory can make the floor
            look tilted and auto-level then tilts the world. Prefer Follow the floor.
          </p>
        )}
        <RangeField
          label="Lean toward / away from camera"
          value={tilt}
          display={signed(tilt, "°")}
          min={-15}
          max={15}
          step={0.5}
          left="Away (back)"
          right="Toward (forward)"
          onChange={(tilt_deg) => set({ tilt_deg })}
        />
        <RangeField
          label="Sideways lean"
          value={roll}
          display={signed(roll, "°")}
          min={-15}
          max={15}
          step={0.5}
          left="Camera left"
          right="Camera right"
          onChange={(roll_deg) => set({ roll_deg })}
        />
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Manual lean is added on top of auto-level. Check it from the side in the World view.
        </p>
      </PanelSection>

      <PanelSection
        id="refine-feet"
        title="Floor & contacts"
        summary={[
          options.plant_feet ? `Planted · ${Math.round(sensitivity * 100)}%` : "Not planted",
          options.plant_hands && "hands too",
          options.lock_feet && "no sliding",
          options.world === "gem" && options.follow_floor && "floor follows",
          offset && `body ${signed(offset, " cm")}`,
        ]
          .filter(Boolean)
          .join(" · ")}
      >
        <SwitchField
          label="Plant feet flat on the floor"
          hint="Hip height + leg IK on predicted foot contacts"
          checked={!!options.plant_feet}
          onChange={(plant_feet) => set({ plant_feet })}
        />
        <SwitchField
          label="Plant hands on the floor"
          hint="GEM's hand contacts: a still hand near the floor rests on it (arm IK) and doesn't slide"
          checked={!!options.plant_hands}
          onChange={(plant_hands) => set({ plant_hands })}
        />
        <SwitchField
          label="Stop contact sliding"
          hint="Keeps planted feet and hands in place by moving the body; turn off for gliding footwork"
          checked={!!options.lock_feet}
          onChange={(lock_feet) => set({ lock_feet })}
        />
        {options.world === "gem" && (
          <SwitchField
            label="Follow the floor"
            hint="Removes slow up/down drift: the floor follows the planted feet (smoothed over about a second). For moving cameras."
            checked={!!options.follow_floor}
            onChange={(follow_floor) => set({ follow_floor })}
          />
        )}
        <RangeField
          label="Contact sensitivity"
          value={sensitivity}
          display={`${Math.round(sensitivity * 100)}%`}
          min={0.2}
          max={0.8}
          step={0.05}
          left="Fewer frames planted"
          right="More frames planted"
          disabled={!options.plant_feet && !options.plant_hands}
          onChange={(value) => set({ contact_threshold: Number((1 - value).toFixed(2)) })}
        />
        <RangeField
          label="Body height"
          value={offset}
          display={offset ? signed(offset, " cm") : "On the floor"}
          min={-5}
          max={5}
          step={0.5}
          left="Sink into floor"
          right="Lift off floor"
          onChange={(floor_offset_cm) => set({ floor_offset_cm })}
        />
        <p className="-mt-2 text-[11px] leading-relaxed text-muted-foreground">
          At 0 the soles rest exactly on the floor. Positive values lift the whole body above it (for example
          to leave room for shoe soles), negative values sink it in.
        </p>
      </PanelSection>

      <PanelSection
        id="refine-smoothing"
        title="Smoothing"
        summary={options.strength ? `${options.strength.toFixed(1)} frames · ${DETAIL_LABELS[options.detail]}` : "Off"}
      >
        <RangeField
          label="Strength"
          value={options.strength}
          display={options.strength ? `${options.strength.toFixed(1)} frames` : "Off"}
          min={0}
          max={8}
          step={0.5}
          left="Off"
          right="Strong"
          onChange={(strength) => set({ strength })}
        />
        <div className={cn(!options.strength && "pointer-events-none opacity-50")}>
          <DetailField value={options.detail} onChange={(detail) => set({ detail })} />
        </div>
        <p className="text-[11px] text-muted-foreground">
          Body Motion is already temporally modelled, so smoothing starts off. Add a little for noisy clips.
        </p>
      </PanelSection>

      <ApplyBar
        job={job}
        active={active}
        label={current?.status === "done" ? "Apply again" : "Apply"}
        onApply={() => api.refine(job.id, options)}
        onDefaults={() => setOptions({ ...MOTION_DEFAULTS, ...cameraDefaults })}
      >
        {report && (
          <>
            {report.mode === "anchored_to_fixed_camera" ? "Anchored to the camera" : "GEM world"}
            {report.level?.applied_deg != null &&
              ` · levelled ${report.level.applied_deg}° (${Math.abs(report.level.toward_camera_deg ?? 0).toFixed(1)}° ${
                (report.level.toward_camera_deg ?? 0) < 0 ? "back" : "forward"
              }, ${Math.abs(report.level.sideways_deg ?? 0).toFixed(1)}° sideways) from the ${
                report.level.method === "floor_plane" ? "floor plane" : "body balance"
              }`}
            {report.floor === "foot_contact_follow"
              ? " · floor follows the feet"
              : report.floor === "foot_contact_median"
                ? " · floor from foot contacts"
                : " · floor from lowest points"}
            {report.depth && ` · ${report.depth.depth_jitter_removed_cm} cm of depth jitter removed`}
            {report.lock_feet && " · contacts locked"}
            {plant && (
              <>
                {" "}
                · planted-foot height error {plant.planted_height_error_cm_before} →{" "}
                <span className="font-medium text-foreground">{plant.planted_height_error_cm_after} cm</span> · feet
                planted in {Math.round(plant.planted_frame_fraction * 100)}% of frames
                {plant.hands_planted_frame_fraction != null &&
                  `, hands in ${Math.round(plant.hands_planted_frame_fraction * 100)}%`}
              </>
            )}
          </>
        )}
      </ApplyBar>
    </div>
  )
}

/** Apply (+ optional Defaults) with progress and a "last result" note; shared by both Refine panels. */
function ApplyBar(props: {
  job: Job
  active: boolean
  label: string
  onApply: () => Promise<unknown>
  onDefaults?: () => void
  children?: ReactNode
}) {
  const current = props.job.refine
  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-2">
        <Button
          className="flex-1"
          disabled={props.active}
          onClick={async () => {
            try {
              await props.onApply()
            } catch (error) {
              toast.error("Could not start", { description: (error as Error).message })
            }
          }}
        >
          {props.active ? <Loader2 className="animate-spin" /> : <Sparkles />}
          {props.active ? "Refining…" : props.label}
        </Button>
        {props.onDefaults && (
          <Button variant="outline" disabled={props.active} onClick={props.onDefaults}>
            Defaults
          </Button>
        )}
      </div>
      {props.active && current && (
        <JobProgress
          job={{ ...props.job, stage: current.stage, detail: current.detail ?? null, progress: current.progress }}
        />
      )}
      {props.children && !props.active && (
        <div className="rounded-lg border bg-card p-3 text-[11px] leading-relaxed text-muted-foreground">
          <div className="mb-1 font-medium text-foreground">Last result</div>
          {props.children}
        </div>
      )}
    </div>
  )
}

const MMD_MODEL_KEY = "gem-studio:mmd-model"

interface MmdModelChoice {
  model: MmdModel | null
  loading: boolean
  choose: (model: MmdModel | null) => void
}

/** The MMD model chosen under Fit to model (remembered across runs and sessions). */
function useMmdModel(): MmdModelChoice {
  const queryClient = useQueryClient()
  const [id, setId] = useState<string | null>(() => readStored(MMD_MODEL_KEY, null))
  const query = useQuery({
    queryKey: ["mmd-model", id],
    queryFn: () => api.mmdModel(id!),
    enabled: !!id,
    retry: false,
    staleTime: Infinity,
  })
  const choose = (model: MmdModel | null) => {
    setId(model?.id ?? null)
    writeStored(MMD_MODEL_KEY, model?.id ?? null)
    if (model) queryClient.setQueryData(["mmd-model", model.id], model)
  }
  useEffect(() => {
    // The stored profile is gone (data folder cleared): forget it.
    if (query.isError) choose(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query.isError])
  return { model: id && query.data ? query.data : null, loading: !!id && query.isLoading, choose }
}

/** Optional MMD model (.pmx) the VMD export is fitted to. */
function MmdModelField(props: {
  model: MmdModel | null
  loading: boolean
  onChange: (model: MmdModel | null) => void
}) {
  const input = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const upload = async (file: File) => {
    setUploading(true)
    try {
      props.onChange(await api.uploadMmdModel(file))
    } catch (error) {
      toast.error("Could not read the model", { description: (error as Error).message })
    } finally {
      setUploading(false)
      if (input.current) input.current.value = ""
    }
  }
  const { model } = props
  return (
    <OptionField label="Fit to model">
      <input
        ref={input}
        type="file"
        accept=".pmx"
        className="hidden"
        onChange={(event) => event.target.files?.[0] && upload(event.target.files[0])}
      />
      {model ? (
        <div className="flex items-center gap-2 rounded-lg border p-2">
          <div className="min-w-0 flex-1 text-[11px] leading-snug">
            <div className="truncate font-medium text-foreground" title={model.name}>
              {model.name}
            </div>
            <div className="text-muted-foreground tabular-nums">
              {model.height} units tall · legs {model.leg_length} · arms {model.arm_length} ·{" "}
              {model.arm_angle_deg}° A-pose
            </div>
          </div>
          <Button variant="outline" size="sm" disabled={uploading} onClick={() => input.current?.click()}>
            {uploading ? <Loader2 className="animate-spin" /> : <Upload />} Change
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="size-8"
            aria-label="Don't fit to a model"
            onClick={() => props.onChange(null)}
          >
            <X />
          </Button>
        </div>
      ) : (
        <div className="flex flex-col gap-1.5">
          <Button
            variant="outline"
            size="sm"
            className="w-full"
            disabled={uploading || props.loading}
            onClick={() => input.current?.click()}
          >
            {uploading || props.loading ? <Loader2 className="animate-spin" /> : <Upload />} Choose model (.pmx)…
          </Button>
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            Optional. Models have other proportions than the dancer (shorter arms, longer legs), so the motion alone can
            leave them floating or sinking. Studio reads only the bones and body mesh; the file isn&apos;t kept.
          </p>
        </div>
      )}
    </OptionField>
  )
}

function SwitchField(props: { label: string; hint?: string; checked: boolean; onChange: (value: boolean) => void }) {
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

function DetailField(props: { value: RefineOptions["detail"]; onChange: (value: RefineOptions["detail"]) => void }) {
  return (
    <OptionField label="Fast motion" hint="Sharper keeps quick gestures crisp; Smoother also calms fast movement.">
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        value={props.value}
        onValueChange={(detail) => detail && props.onChange(detail as RefineOptions["detail"])}
        className="w-full"
      >
        {(Object.keys(DETAIL_LABELS) as RefineOptions["detail"][]).map((value) => (
          <ToggleGroupItem key={value} value={value} className="flex-1">
            {DETAIL_LABELS[value]}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </OptionField>
  )
}

function RangeField(props: {
  label: string
  value: number
  display: string
  min: number
  max: number
  step: number
  left: string
  right: string
  disabled?: boolean
  onChange: (value: number) => void
}) {
  return (
    <div className={cn("flex flex-col gap-2", props.disabled && "pointer-events-none opacity-50")}>
      <div className="flex items-center justify-between">
        <Label className="text-[13px]">{props.label}</Label>
        <span className="text-xs text-muted-foreground tabular-nums">{props.display}</span>
      </div>
      <Slider
        min={props.min}
        max={props.max}
        step={props.step}
        value={[props.value]}
        onValueChange={([value]) => props.onChange(value)}
      />
      <div className="flex justify-between text-[11px] text-muted-foreground">
        <span>{props.left}</span>
        <span>{props.right}</span>
      </div>
    </div>
  )
}

function RefinePanel({ job, onRefined }: { job: Job; onRefined: () => void }) {
  const current = job.refine
  const [options, setOptions] = useState<RefineOptions>(
    current?.options?.strength != null ? current.options : { strength: 2, detail: "balanced", lock_shape: true },
  )
  const active = current?.status === "running" || current?.status === "queued"
  useRefineCompletion(job, onRefined)
  const subject = job.task === "hand" ? "hand" : "body"

  return (
    <div className="flex flex-col gap-3 pb-4">
      <PanelSection
        id="refine-pose-smoothing"
        title="Temporal smoothing"
        summary={`${options.strength.toFixed(1)} frames · ${DETAIL_LABELS[options.detail]}${options.lock_shape ? " · shape locked" : ""}`}
      >
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Runs GEM&apos;s motion-adaptive filter over the predicted rotations, position and {subject} shape, then
          rebuilds the meshes. It removes frame-to-frame jitter and backs off during fast movement. The raw result is
          kept; switch between them in the viewer.
        </p>
        <RangeField
          label="Strength"
          value={options.strength}
          display={`${options.strength.toFixed(1)} frame${options.strength === 1 ? "" : "s"}`}
          min={0.5}
          max={8}
          step={0.5}
          left="Light"
          right="Strong (GEM default: 2)"
          onChange={(strength) => setOptions({ ...options, strength })}
        />
        <DetailField value={options.detail} onChange={(detail) => setOptions({ ...options, detail })} />
        <SwitchField
          label={`Lock ${subject} shape`}
          hint="Same proportions for the whole clip"
          checked={options.lock_shape}
          onChange={(lock_shape) => setOptions({ ...options, lock_shape })}
        />
      </PanelSection>

      <ApplyBar
        job={job}
        active={active}
        label={current?.status === "done" ? "Apply again" : "Apply smoothing"}
        onApply={() => api.refine(job.id, options)}
        onDefaults={() => setOptions({ strength: 2, detail: "balanced", lock_shape: true })}
      >
        {current?.status === "done" && current.options.strength != null && (
          <>
            Strength {current.options.strength}, {DETAIL_LABELS[current.options.detail].toLowerCase()}
            {current.options.lock_shape ? ", shape locked" : ""}.
          </>
        )}
      </ApplyBar>
    </div>
  )
}
