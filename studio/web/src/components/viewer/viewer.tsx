import {
  Box,
  ChevronLeft,
  ChevronRight,
  Crosshair,
  Eye,
  EyeOff,
  Globe2,
  ImageIcon,
  ImageOff,
  Loader2,
  Pause,
  Play,
  Repeat,
  ScanEye,
  SlidersHorizontal,
  Spline,
  Video,
  VideoOff,
  Volume1,
  Volume2,
  VolumeX,
} from "lucide-react"
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react"

import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { api, urls, type Preview, type PreviewFrame, type Variant } from "@/lib/api"
import { formatTimecode } from "@/lib/format"
import { MUTED_KEY, VOLUME_KEY, readStored, writeStored } from "@/lib/storage"
import { cn } from "@/lib/utils"
import { ViewerEngine, type ViewMode } from "./engine"
import { MeshStore } from "./mesh-store"

const HAND_CHAINS = [
  [0, 1, 2, 3, 4],
  [0, 5, 6, 7, 8, 9],
  [0, 10, 11, 12, 13, 14],
  [0, 15, 16, 17, 18, 19],
  [0, 20, 21, 22, 23, 24],
]
const KP_COLORS: Record<string, string> = { left: "#50e178", right: "#50afff", body: "#ffc857" }
// Corner preview of the source video in 3D/World: share of the Overlay size, inset in px.
const PIP_SCALE = 0.22
const PIP_MARGIN = 12

const MESH_COLORS = ["#9be15d", "#7cc4ff", "#f0a6ff", "#ffd27a", "#e8e8e8"]

function drawKeypoints(ctx: CanvasRenderingContext2D, record: PreviewFrame | undefined, sx: number, sy: number) {
  if (!record) return
  const radius = Math.max(2, Math.min(4.5, 2.2 * Math.sqrt(sx * sy) + 1))
  for (const instance of record.instances) {
    const points = instance.kp2d
    if (!points) continue
    const color = KP_COLORS[instance.side ?? "body"]
    const valid = (index: number) => points[index] && points[index][0] != null && points[index][1] != null
    const at = (index: number) => [(points[index][0]! + 0.5) * sx, (points[index][1]! + 0.5) * sy]
    ctx.lineWidth = Math.max(1.5, radius * 0.7)
    ctx.strokeStyle = color
    ctx.lineCap = "round"
    if (instance.topology === "left" || instance.topology === "right") {
      for (const chain of HAND_CHAINS) {
        ctx.beginPath()
        let open = false
        for (const index of chain) {
          if (!valid(index)) {
            open = false
            continue
          }
          const [x, y] = at(index)
          if (open) ctx.lineTo(x, y)
          else ctx.moveTo(x, y)
          open = true
        }
        ctx.stroke()
      }
    }
    for (let index = 0; index < points.length; index++) {
      if (!valid(index)) continue
      const [x, y] = at(index)
      ctx.beginPath()
      ctx.arc(x, y, radius, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()
      ctx.lineWidth = 1
      ctx.strokeStyle = "rgba(0,0,0,0.55)"
      ctx.stroke()
    }
  }
}

export interface ViewerHandle {
  screenshot(): Promise<Blob>
  exportFrame(format: "glb" | "obj"): Promise<Blob>
  renderVideo(name: string, onProgress: (fraction: number) => void, signal: AbortSignal): Promise<void>
  frame(): number
  view(): ViewMode
}

export interface ViewerProps {
  jobId: string
  preview: Preview
  variant?: Variant
  /** Rendered centred at the top of the viewer (the Raw/Refined switch). */
  center?: React.ReactNode
}

export const Viewer = forwardRef<ViewerHandle, ViewerProps>(function Viewer(
  { jobId, preview, variant = "raw", center },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const keypointRef = useRef<HTMLCanvasElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const engineRef = useRef<ViewerEngine | null>(null)
  // The whole viewer area; Overlay uses the media-shaped box inside it, 3D and World all of it.
  const [box, setBox] = useState({ width: 0, height: 0 })
  const [ready, setReady] = useState(false)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [loop, setLoop] = useState(true)
  const [rate, setRate] = useState("1")
  const [volume, setVolume] = useState(() => readStored(VOLUME_KEY, 0.8))
  const [muted, setMuted] = useState(() => readStored(MUTED_KEY, false))
  const [view, setView] = useState<ViewMode>("overlay")
  const [showMesh, setShowMesh] = useState(true)
  const [showMedia, setShowMedia] = useState(true)
  const [showKeypoints, setShowKeypoints] = useState(false)
  const [opacity, setOpacity] = useState(0.85)
  const [color, setColor] = useState(MESH_COLORS[0])
  const [follow, setFollow] = useState(true)
  const [exporting, setExporting] = useState(false)

  const isVideo = preview.mode === "video"
  const fps = preview.fps ?? 30
  const frameCount = preview.frame_count
  const [imageWidth, imageHeight] = preview.image_size
  // Overlay: the media-shaped box centred in the viewer. 3D/World: the whole viewer.
  const fitted = useMemo(() => {
    const aspect = imageWidth / imageHeight
    return box.width / Math.max(box.height, 1) > aspect
      ? { width: Math.floor(box.height * aspect), height: Math.floor(box.height) }
      : { width: Math.floor(box.width), height: Math.floor(box.width / aspect) }
  }, [box.width, box.height, imageWidth, imageHeight])
  const stage = useMemo(
    () => (view === "overlay" ? fitted : { width: Math.floor(box.width), height: Math.floor(box.height) }),
    [view, fitted, box.width, box.height],
  )
  const hasKeypoints = preview.type === "pose"

  // One engine per viewer; switching variant only swaps its vertex source.
  useEffect(() => {
    const store = new MeshStore(jobId, preview, variant)
    let cancelled = false
    store.loadFaces().then(() => {
      if (cancelled || !canvasRef.current) return
      if (engineRef.current) engineRef.current.setSource(preview, store)
      else {
        engineRef.current = new ViewerEngine(canvasRef.current, preview, store)
        engineRef.current.onFollowChange = setFollow
        setReady(true)
      }
    })
    return () => {
      cancelled = true
    }
  }, [jobId, preview, variant])
  useEffect(
    () => () => {
      engineRef.current?.dispose()
      engineRef.current = null
    },
    [],
  )

  // Track the viewer area; the stage is derived from it per view.
  useLayoutEffect(() => {
    const container = containerRef.current
    if (!container) return
    const observer = new ResizeObserver(() => {
      const { width, height } = container.getBoundingClientRect()
      setBox({ width, height })
    })
    observer.observe(container)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    engineRef.current?.setSize(stage.width, stage.height)
  }, [stage.width, stage.height, ready])
  useEffect(() => {
    engineRef.current?.setMode(view)
  }, [view, ready])
  useEffect(() => {
    engineRef.current?.setFrame(frame)
  }, [frame, ready])
  useEffect(() => {
    engineRef.current?.setStyle({ showMesh, opacity, color })
  }, [showMesh, opacity, color, ready])
  useEffect(() => {
    if (engineRef.current) engineRef.current.follow = follow
  }, [follow, ready])

  // Keypoint layer
  useEffect(() => {
    const canvas = keypointRef.current
    if (!canvas) return
    const ratio = Math.min(window.devicePixelRatio, 2)
    canvas.width = stage.width * ratio
    canvas.height = stage.height * ratio
    const ctx = canvas.getContext("2d")!
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (view === "overlay" && showKeypoints) {
      drawKeypoints(
        ctx,
        preview.frames[frame],
        (stage.width * ratio) / imageWidth,
        (stage.height * ratio) / imageHeight,
      )
    }
  }, [frame, stage, view, showKeypoints, preview, imageWidth, imageHeight])

  // The video is the clock: every presented frame updates the mesh frame.
  useEffect(() => {
    const video = videoRef.current
    if (!video || !isVideo) return
    let handle = 0
    const onFrame = (_: number, metadata: VideoFrameCallbackMetadata) => {
      setFrame(Math.max(0, Math.min(frameCount - 1, Math.round(metadata.mediaTime * fps))))
      handle = video.requestVideoFrameCallback(onFrame)
    }
    handle = video.requestVideoFrameCallback(onFrame)
    const onPlay = () => setPlaying(true)
    const onPause = () => setPlaying(false)
    video.addEventListener("play", onPlay)
    video.addEventListener("pause", onPause)
    return () => {
      video.cancelVideoFrameCallback(handle)
      video.removeEventListener("play", onPlay)
      video.removeEventListener("pause", onPause)
    }
  }, [isVideo, fps, frameCount])

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = Number(rate)
  }, [rate])
  useEffect(() => {
    if (videoRef.current) videoRef.current.loop = loop
  }, [loop])

  useEffect(() => {
    const video = videoRef.current
    if (video) {
      video.volume = volume
      video.muted = muted || !preview.audio
    }
    writeStored(VOLUME_KEY, volume)
    writeStored(MUTED_KEY, muted)
  }, [volume, muted, preview.audio])

  const seek = useCallback(
    (target: number) => {
      const clamped = Math.max(0, Math.min(frameCount - 1, target))
      setFrame(clamped)
      const video = videoRef.current
      if (video) video.currentTime = (clamped + 0.5) / fps
    },
    [frameCount, fps],
  )

  const togglePlay = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) void video.play()
    else video.pause()
  }, [])

  // Keyboard: space, arrows.
  useEffect(() => {
    if (!isVideo) return
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable) return
      if (event.code === "Space") {
        event.preventDefault()
        togglePlay()
      } else if (event.code === "ArrowRight" || event.code === "ArrowLeft") {
        event.preventDefault()
        videoRef.current?.pause()
        const step = (event.shiftKey ? 10 : 1) * (event.code === "ArrowRight" ? 1 : -1)
        seek(frame + step)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [isVideo, togglePlay, seek, frame])

  // -- exports -------------------------------------------------------------
  const composite = useCallback(
    (target: HTMLCanvasElement, record: PreviewFrame | undefined) => {
      const engine = engineRef.current!
      const ctx = target.getContext("2d")!
      if (view === "overlay" && showMedia) {
        const media = (videoRef.current ?? imageRef.current) as CanvasImageSource
        ctx.drawImage(media, 0, 0, target.width, target.height)
      } else {
        ctx.fillStyle = getComputedStyle(containerRef.current!).backgroundColor
        ctx.fillRect(0, 0, target.width, target.height)
      }
      ctx.drawImage(engine.renderer.domElement, 0, 0, target.width, target.height)
      if (view === "overlay" && showKeypoints) {
        drawKeypoints(ctx, record, target.width / imageWidth, target.height / imageHeight)
      }
    },
    [view, showMedia, showKeypoints, imageWidth, imageHeight],
  )

  useImperativeHandle(
    ref,
    () => ({
      frame: () => frame,
      view: () => view,
      async screenshot() {
        const canvas = document.createElement("canvas")
        const scale = view === "overlay" ? 1 : Math.min(window.devicePixelRatio, 2)
        canvas.width = view === "overlay" ? imageWidth : Math.round(stage.width * scale)
        canvas.height = view === "overlay" ? imageHeight : Math.round(stage.height * scale)
        await engineRef.current!.renderFrameNow(frame)
        composite(canvas, preview.frames[frame])
        return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob!), "image/png"))
      },
      exportFrame: (format) => engineRef.current!.exportFrame(format),
      async renderVideo(name, onProgress, signal) {
        const video = videoRef.current
        const engine = engineRef.current
        if (!video || !engine) throw new Error("Video export needs a video result")
        video.pause()
        setExporting(true)
        // Even dimensions, capped at 1920 px wide.
        const scale = Math.min(1, 1920 / imageWidth)
        const width = 2 * Math.round((imageWidth * scale) / 2)
        const height = 2 * Math.round((imageHeight * scale) / 2)
        const canvas = document.createElement("canvas")
        canvas.width = width
        canvas.height = height
        const ratio = engine.renderer.getPixelRatio()
        engine.renderer.setPixelRatio(1)
        engine.setSize(width, height)
        await api.renderStart(jobId, name, fps)
        try {
          for (let index = 0; index < frameCount; index++) {
            if (signal.aborted) throw new DOMException("Export cancelled", "AbortError")
            await new Promise<void>((resolve) => {
              const done = () => resolve()
              video.addEventListener("seeked", done, { once: true })
              video.currentTime = (index + 0.5) / fps
            })
            await engine.renderFrameNow(index)
            composite(canvas, preview.frames[index])
            const blob = await new Promise<Blob>((resolve) =>
              canvas.toBlob((value) => resolve(value!), "image/jpeg", 0.93),
            )
            await api.renderFrame(jobId, blob)
            onProgress((index + 1) / frameCount)
          }
          await api.renderFinish(jobId)
        } catch (error) {
          await api.renderAbort(jobId).catch(() => undefined)
          throw error
        } finally {
          engine.renderer.setPixelRatio(ratio)
          engine.setSize(stage.width, stage.height)
          engine.setFrame(frame)
          setExporting(false)
        }
      },
    }),
    [frame, view, composite, preview, imageWidth, imageHeight, stage, jobId, fps, frameCount],
  )

  const views: { value: ViewMode; label: string; icon: typeof Box }[] = [
    { value: "overlay", label: "Overlay", icon: ScanEye },
    { value: "camera3d", label: "3D", icon: Box },
    ...(preview.views.includes("world") ? [{ value: "world" as const, label: "World", icon: Globe2 }] : []),
  ]
  const time = frame / fps

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-xl border bg-viewer">
      <div ref={containerRef} className="@container relative flex min-h-0 flex-1 items-center justify-center bg-viewer">
        {isVideo && (
          <video
            ref={videoRef}
            src={urls.previewMedia(jobId)}
            // Unmuted by the volume effect when the preview carries sound.
            muted
            playsInline
            preload="auto"
            onEnded={() => setPlaying(false)}
            // Sized like the Overlay stage; the corner preview is the same element translated
            // and scaled (one set of animated properties), so shrinking and growing back are
            // exact reverses. Radius and ring are sized up to come out at ~6 px / 1 px.
            style={{
              left: (box.width - fitted.width) / 2,
              top: (box.height - fitted.height) / 2,
              width: fitted.width,
              height: fitted.height,
              ...(view !== "overlay" && {
                translate: `${PIP_MARGIN - (box.width - fitted.width) / 2}px ${
                  box.height - PIP_MARGIN - fitted.height * PIP_SCALE - (box.height - fitted.height) / 2
                }px`,
                scale: String(PIP_SCALE),
              }),
            }}
            className={cn(
              // Only the shrink/grow animates; size and position follow the viewer instantly.
              "absolute origin-top-left bg-black object-fill transition-[translate,scale,opacity,border-radius,box-shadow] duration-300 ease-out",
              view !== "overlay" && "z-20 rounded-[27px] opacity-90 shadow-2xl ring-[4px] ring-white/15",
              // Hidden, not removed: the video keeps driving playback.
              !showMedia && "invisible",
            )}
          />
        )}
        <div className="relative" style={{ width: stage.width, height: stage.height }}>
          {!isVideo && (
            <img
              ref={imageRef}
              src={urls.previewMedia(jobId)}
              alt=""
              className={cn("absolute inset-0 size-full object-fill", (view !== "overlay" || !showMedia) && "invisible")}
            />
          )}
          <canvas ref={canvasRef} className="absolute inset-0 z-10 size-full" />
          <canvas ref={keypointRef} className="pointer-events-none absolute inset-0 z-10 size-full" />
        </div>

        {!ready && (
          <div className="absolute inset-0 flex items-center justify-center">
            <Loader2 className="size-6 animate-spin text-white/60" />
          </div>
        )}
        {exporting && (
          <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/35 text-sm text-white backdrop-blur-[2px]">
            <Loader2 className="mr-2 size-4 animate-spin" /> Rendering video…
          </div>
        )}

        {/* Floating toolbar: views left, result centred, display options right. A narrow
            viewer moves the centre item onto a second row. */}
        <div className="pointer-events-none absolute inset-x-3 top-3 z-20 grid grid-cols-[auto_1fr] items-start gap-2 @xl:grid-cols-[1fr_auto_1fr] [&>*]:pointer-events-auto">
          <ToggleGroup
            type="single"
            value={view}
            onValueChange={(value) => value && setView(value as ViewMode)}
            className="col-start-1 row-start-1 justify-self-start rounded-lg bg-black/55 p-0.5 text-white backdrop-blur"
          >
            {views.map((item) => (
              <ToggleGroupItem
                key={item.value}
                value={item.value}
                className="h-7 gap-1.5 rounded-md px-2.5 text-xs text-white/75 hover:bg-white/10 hover:text-white data-[state=on]:bg-white/20 data-[state=on]:text-white"
              >
                <item.icon className="size-3.5" />
                {item.label}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <div className="col-span-2 justify-self-center empty:hidden @xl:col-span-1 @xl:col-start-2 @xl:row-start-1">{center}</div>
          <div className="col-start-2 row-start-1 flex items-center gap-1 justify-self-end rounded-lg @xl:col-start-3 bg-black/55 p-0.5 text-white backdrop-blur">
            <IconToggle label={showMesh ? "Hide mesh" : "Show mesh"} active={showMesh} onClick={() => setShowMesh(!showMesh)}>
              {showMesh ? <Eye /> : <EyeOff />}
            </IconToggle>
            <IconToggle
              label={`${showMedia ? "Hide" : "Show"} source ${isVideo ? "video" : "image"}`}
              active={showMedia}
              onClick={() => setShowMedia(!showMedia)}
              disabled={!isVideo && view !== "overlay"}
            >
              {isVideo ? showMedia ? <Video /> : <VideoOff /> : showMedia ? <ImageIcon /> : <ImageOff />}
            </IconToggle>
            {hasKeypoints && (
              <IconToggle
                label="2D keypoints"
                active={showKeypoints}
                onClick={() => setShowKeypoints(!showKeypoints)}
                disabled={view !== "overlay"}
              >
                <Spline />
              </IconToggle>
            )}
            {view !== "overlay" && (
              <IconToggle
                label={
                  view === "camera3d"
                    ? follow
                      ? "Looking through the capture camera"
                      : "Back to the camera's view"
                    : follow
                      ? "Following the subject"
                      : "Re-centre camera"
                }
                active={follow}
                onClick={() => {
                  setFollow(true)
                  engineRef.current?.resetCamera()
                }}
              >
                <Crosshair />
              </IconToggle>
            )}
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="ghost" size="icon" className="size-7 text-white/80 hover:bg-white/10 hover:text-white">
                  <SlidersHorizontal />
                </Button>
              </PopoverTrigger>
              <PopoverContent align="end" className="w-64">
                <div className="flex flex-col gap-4">
                  <div className="flex flex-col gap-2">
                    <Label className="text-xs">Overlay opacity · {Math.round(opacity * 100)}%</Label>
                    <Slider min={0.15} max={1} step={0.01} value={[opacity]} onValueChange={([value]) => setOpacity(value)} />
                  </div>
                  <div className="flex flex-col gap-2">
                    <Label className="text-xs">Body colour</Label>
                    <div className="flex gap-2">
                      {MESH_COLORS.map((value) => (
                        <button
                          key={value}
                          aria-label={value}
                          onClick={() => setColor(value)}
                          className={cn(
                            "size-6 rounded-full ring-offset-2 ring-offset-popover transition",
                            color === value && "ring-2 ring-ring",
                          )}
                          style={{ background: value }}
                        />
                      ))}
                    </div>
                  </div>
                  {hasKeypoints && (
                    <label className="flex items-center justify-between text-xs">
                      2D keypoints
                      <Switch checked={showKeypoints} onCheckedChange={setShowKeypoints} />
                    </label>
                  )}
                </div>
              </PopoverContent>
            </Popover>
          </div>
        </div>
      </div>

      {isVideo && (
        <div className="flex items-center gap-3 border-t border-white/5 bg-card px-3 py-2.5">
          <Button size="icon" variant="ghost" className="size-8" onClick={() => seek(frame - 1)} aria-label="Previous frame">
            <ChevronLeft />
          </Button>
          <Button size="icon" className="size-9 rounded-full" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>
            {playing ? <Pause className="fill-current" /> : <Play className="ml-0.5 fill-current" />}
          </Button>
          <Button size="icon" variant="ghost" className="size-8" onClick={() => seek(frame + 1)} aria-label="Next frame">
            <ChevronRight />
          </Button>
          <div className="w-28 shrink-0 text-xs tabular-nums">
            <div>{formatTimecode(time)}</div>
            <div className="text-muted-foreground">
              frame {frame + 1}/{frameCount}
            </div>
          </div>
          <Slider
            className="flex-1"
            min={0}
            max={frameCount - 1}
            step={1}
            value={[frame]}
            onValueChange={([value]) => {
              videoRef.current?.pause()
              seek(value)
            }}
          />
          {preview.audio && (
            <div className="flex shrink-0 items-center gap-1.5">
              <IconToggle
                label={muted ? "Unmute" : "Mute"}
                active={!muted && volume > 0}
                onClick={() => {
                  if (muted || volume === 0) {
                    setMuted(false)
                    if (volume === 0) setVolume(0.5)
                  } else setMuted(true)
                }}
                plain
              >
                {muted || volume === 0 ? <VolumeX /> : volume < 0.5 ? <Volume1 /> : <Volume2 />}
              </IconToggle>
              <Slider
                aria-label="Volume"
                className="w-20"
                min={0}
                max={1}
                step={0.01}
                value={[muted ? 0 : volume]}
                onValueChange={([value]) => {
                  setVolume(value)
                  setMuted(value === 0)
                }}
              />
            </div>
          )}
          <IconToggle label="Loop" active={loop} onClick={() => setLoop(!loop)} plain>
            <Repeat />
          </IconToggle>
          <Select value={rate} onValueChange={setRate}>
            <SelectTrigger size="sm" className="w-20">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {["0.25", "0.5", "1", "1.5", "2"].map((value) => (
                <SelectItem key={value} value={value}>
                  {value}×
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
    </div>
  )
})

function IconToggle(props: {
  label: string
  active: boolean
  onClick: () => void
  disabled?: boolean
  plain?: boolean
  children: React.ReactNode
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          disabled={props.disabled}
          onClick={props.onClick}
          aria-pressed={props.active}
          className={cn(
            "size-7",
            props.plain
              ? props.active
                ? "text-primary"
                : "text-muted-foreground"
              : cn("text-white/70 hover:bg-white/10 hover:text-white", props.active && "bg-white/15 text-white"),
          )}
        >
          {props.children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{props.label}</TooltipContent>
    </Tooltip>
  )
}
