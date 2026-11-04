import { useQuery } from "@tanstack/react-query"
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Crosshair,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  UserRound,
  Volume1,
  Volume2,
  VolumeX,
} from "lucide-react"
import { useEffect, useRef, useState } from "react"

import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { api, urls, type InputRecord, type Subject } from "@/lib/api"
import { formatTimecode } from "@/lib/format"
import { MUTED_KEY, VOLUME_KEY, readStored, writeStored } from "@/lib/storage"
import { cn } from "@/lib/utils"

/** Inclusive source-frame indices [first, last]. */
export type ClipRange = [number, number]
type Edge = "start" | "end"

/** Body Motion runs on at most 60 seconds at a time. */
const MAX_SECONDS = 60
const FILMSTRIP = 10

export function clipLimits(input: InputRecord) {
  const fps = input.fps || 30
  const count = Math.max(1, input.frame_count || Math.round((input.duration ?? 0) * fps) || 1)
  // The cut spans the clip's frames plus 0.4 of a frame (see clipToSeconds); keep it within the limit.
  return { fps, count, maxLength: Math.max(1, Math.floor(MAX_SECONDS * fps - 0.4)) }
}

export function defaultClip(input: InputRecord): ClipRange {
  const { count, maxLength } = clipLimits(input)
  return [0, Math.min(count, maxLength) - 1]
}

/**
 * Start and duration (seconds) for GEM so that its clip runs from frame ``first`` to
 * frame ``last`` inclusive. ffmpeg's accurate seek keeps frames from the start time on,
 * so the start sits half a frame early. GEM's 30 fps conversion drops a final frame
 * that is only half covered, so the end covers most of the last frame, yet stays short
 * of the next one and of the video's end (verified against normalize_video).
 */
export function clipToSeconds(input: InputRecord, [first, last]: ClipRange) {
  const { fps } = clipLimits(input)
  const start = first === 0 ? 0 : (first - 0.5) / fps
  const end = (last + 0.9) / fps
  const round = (value: number) => Math.round(value * 1e6) / 1e6
  return { start: round(start), duration: round(end - start) }
}

/** Whether a picked subject lies inside the clip (only then does the run use it). */
export function subjectInClip(subject: Subject | null, [first, last]: ClipRange) {
  return !!subject && subject.frame >= first && subject.frame <= last
}

/** The picked person, cropped from their frame. */
export function SubjectThumb(props: { input: InputRecord; subject: Subject; size: number; className?: string }) {
  const { input, subject, size } = props
  const [x1, y1, x2, y2] = subject.box
  // Fit the box (with a little margin) into the square; the frame is drawn at that scale.
  const scale = (size / Math.max(x2 - x1, y2 - y1, 1)) * 0.9
  return (
    <div
      className={cn("shrink-0 rounded bg-black bg-no-repeat", props.className)}
      style={{
        width: size,
        height: size,
        backgroundImage: `url(${urls.inputFrame(input.id, subject.frame, 360)})`,
        backgroundSize: `${input.width * scale}px ${input.height * scale}px`,
        backgroundPosition: `${size / 2 - ((x1 + x2) / 2) * scale}px ${size / 2 - ((y1 + y2) / 2) * scale}px`,
      }}
    />
  )
}

/** Floating card for picking the exact first and last frame, sized to the video. */
export function ClipDialog(props: {
  input: InputRecord
  value: ClipRange
  onChange: (value: ClipRange) => void
  subject: Subject | null
  onSubjectChange: (value: Subject | null) => void
  /** Open straight into picking the subject on the current frame. */
  startPicking?: boolean
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  return (
    <Dialog open={props.open} onOpenChange={props.onOpenChange}>
      <DialogContent className="max-h-[calc(100vh-2rem)] w-auto max-w-[calc(100vw-2rem)] gap-4 overflow-y-auto p-5 sm:max-w-[calc(100vw-2rem)]">
        <div className="pr-8">
          <DialogTitle className="text-base">Clip</DialogTitle>
          <DialogDescription className="text-xs">
            {props.input.name} · choose the exact first and last frame (up to {MAX_SECONDS} s) and who to follow.
          </DialogDescription>
        </div>
        {/* Mounted only while the card is shown (Radix), so each opening starts fresh. */}
        <ClipEditor {...props} portrait={props.input.height > props.input.width} />
      </DialogContent>
    </Dialog>
  )
}

function ClipEditor(props: {
  input: InputRecord
  value: ClipRange
  onChange: (value: ClipRange) => void
  subject: Subject | null
  onSubjectChange: (value: Subject | null) => void
  startPicking?: boolean
  onOpenChange: (open: boolean) => void
  portrait: boolean
}) {
  const { input, value, onChange, subject, portrait } = props
  const { fps, count, maxLength } = clipLimits(input)
  const [first, last] = value
  const videoRef = useRef<HTMLVideoElement>(null)
  const [playhead, setPlayhead] = useState(first)
  const playheadRef = useRef(playhead)
  playheadRef.current = playhead
  const [playing, setPlaying] = useState(false)
  // The browser cannot decode every format (e.g. HEVC); fall back to frames cut by the server.
  const [fallback, setFallback] = useState(false)
  // Picking the subject: the people on the shown frame become clickable boxes.
  const [picking, setPicking] = useState(!!props.startPicking)
  const [volume, setVolume] = useState(() => readStored(VOLUME_KEY, 0.8))
  const [muted, setMuted] = useState(() => readStored(MUTED_KEY, false))

  useEffect(() => {
    const video = videoRef.current
    if (video) {
      video.volume = volume
      video.muted = muted
    }
    writeStored(VOLUME_KEY, volume)
    writeStored(MUTED_KEY, muted)
  }, [volume, muted, fallback])

  // While paused, show the playhead frame by seeking to its middle (exactly that frame).
  useEffect(() => {
    const video = videoRef.current
    if (!video || fallback || playing) return
    video.pause()
    video.currentTime = (playhead + 0.5) / fps
  }, [playhead, fps, fallback, playing])

  // Play the selection in a loop. requestVideoFrameCallback reports each presented frame's
  // exact timestamp; animation frames (reading currentTime) are the fallback.
  useEffect(() => {
    const video = videoRef.current
    if (!playing || !video) return
    const show = (frame: number) => {
      if (frame > last || video.ended) {
        video.currentTime = (first + 0.5) / fps
        if (video.paused) void video.play().catch(() => setPlaying(false))
      } else {
        setPlayhead(Math.max(first, frame))
      }
    }
    let handle = 0
    const exact = "requestVideoFrameCallback" in video
    const onFrame = (_now: number, meta: VideoFrameCallbackMetadata) => {
      show(Math.round(meta.mediaTime * fps))
      handle = video.requestVideoFrameCallback(onFrame)
    }
    const onTick = () => {
      show(Math.floor(video.currentTime * fps))
      handle = requestAnimationFrame(onTick)
    }
    // Resume from the playhead when it is inside the clip, else from its start.
    const from = playheadRef.current
    video.currentTime = ((from >= first && from < last ? from : first) + 0.5) / fps
    void video.play().catch(() => setPlaying(false))
    handle = exact ? video.requestVideoFrameCallback(onFrame) : requestAnimationFrame(onTick)
    // Backup for throttled background pages, where frame callbacks slow down.
    const onTime = () => show(Math.floor(video.currentTime * fps))
    video.addEventListener("timeupdate", onTime)
    return () => {
      if (exact) video.cancelVideoFrameCallback(handle)
      else cancelAnimationFrame(handle)
      video.removeEventListener("timeupdate", onTime)
      video.pause()
    }
  }, [playing, first, last, fps])

  // Detect people once scrubbing settles on a frame (each lookup decodes a full frame).
  const [lookupFrame, setLookupFrame] = useState(playhead)
  useEffect(() => {
    if (!picking || playing) return
    const timer = setTimeout(() => setLookupFrame(playhead), 150)
    return () => clearTimeout(timer)
  }, [picking, playing, playhead])
  const people = useQuery({
    queryKey: ["people", input.id, lookupFrame],
    queryFn: () => api.inputPeople(input.id, lookupFrame),
    enabled: picking && !playing && lookupFrame === playhead,
    staleTime: Infinity,
    retry: false,
  })
  const shownPeople = picking && !playing && people.data?.frame === playhead ? people.data : null

  const togglePlaying = () => {
    if (fallback) return
    if (!playing) setPicking(false)
    setPlaying(!playing)
  }
  const togglePicking = () => {
    setPlaying(false)
    setPicking(!picking)
  }

  const seek = (frame: number) => {
    setPlaying(false)
    setPlayhead(Math.round(Math.min(Math.max(frame, 0), count - 1)))
  }

  const setEdge = (edge: Edge, frame: number) => {
    let [start, end] = value
    frame = Math.round(Math.min(Math.max(frame, 0), count - 1))
    if (edge === "start") {
      start = frame
      end = Math.max(end, start)
      if (end - start + 1 > maxLength) end = start + maxLength - 1
    } else {
      end = frame
      start = Math.min(start, end)
      if (end - start + 1 > maxLength) start = end - maxLength + 1
    }
    onChange([start, end])
    seek(frame)
  }

  const onKeyDown = (event: React.KeyboardEvent) => {
    // Inputs, buttons and sliders handle their own keys (Space on a button already clicks it).
    if ((event.target as HTMLElement).closest("input, button, [role=slider]")) return
    const key = event.key.toLowerCase()
    if (key === " ") {
      event.preventDefault()
      togglePlaying()
    } else if (key === "p") togglePicking()
    else if (key === "arrowleft" || key === "arrowright") {
      event.preventDefault()
      seek(playhead + (key === "arrowleft" ? -1 : 1) * (event.shiftKey ? 10 : 1))
    } else if (key === "i") setEdge("start", playhead)
    else if (key === "o") setEdge("end", playhead)
  }

  const frames = last - first + 1
  // The box always has the video's shape (subject boxes are placed in percent of it):
  // a wide video is sized by width, capped so its height leaves room for the controls below
  // (about 610 px) and stays within half the window.
  const maxHeight = "max(240px, min(50vh, 100vh - 610px))"
  const previewSize = portrait
    ? { height: "min(66vh, 720px)" }
    : { width: `min(820px, calc(100vw - 4.5rem), calc(${maxHeight} * ${input.width / input.height}))` }

  const preview = (
    <div
      style={{ ...previewSize, aspectRatio: `${input.width} / ${input.height}` }}
      className={cn(
        "relative flex max-w-full items-center justify-center overflow-hidden rounded-lg bg-black",
        !picking && "cursor-pointer",
      )}
      onClick={() => !picking && togglePlaying()}
    >
      {fallback ? (
        <img
          src={urls.inputFrame(input.id, playhead)}
          alt={`Frame ${playhead + 1}`}
          className="size-full object-contain"
        />
      ) : (
        <video
          ref={videoRef}
          src={urls.inputMedia(input.id)}
          playsInline
          preload="auto"
          onError={() => setFallback(true)}
          className="size-full object-contain"
        />
      )}
      <div className="pointer-events-none absolute top-2 left-2 rounded-md bg-black/65 px-2 py-1 text-xs text-white tabular-nums backdrop-blur">
        frame {playhead + 1}
        {playhead === first && <span className="ml-1 text-primary">· start</span>}
        {playhead === last && <span className="ml-1 text-primary">· end</span>}
        {(playhead < first || playhead > last) && <span className="ml-1 text-white/60">· outside the clip</span>}
      </div>
      {/* The media fills the box exactly (same aspect ratio), so boxes are placed in percent. */}
      {shownPeople?.people.map((person, index) => (
        <button
          key={index}
          type="button"
          aria-label={`Follow person ${index + 1}`}
          title="Follow this person"
          onClick={(event) => {
            event.stopPropagation()
            props.onSubjectChange({ frame: shownPeople.frame, box: person.box })
            setPicking(false)
          }}
          style={boxStyle(person.box, shownPeople.width, shownPeople.height)}
          className="group absolute cursor-pointer rounded-sm border-2 border-white/85 shadow-[0_0_0_1px_rgba(0,0,0,0.5)] transition-colors hover:border-primary hover:bg-primary/20"
        >
          <span className="absolute -top-px -left-px rounded-br-sm bg-white/85 px-1 text-[10px] font-semibold text-black group-hover:bg-primary group-hover:text-primary-foreground">
            {index + 1}
          </span>
        </button>
      ))}
      {!picking && subject && playhead === subject.frame && (
        <div
          style={boxStyle(subject.box, input.width, input.height)}
          className="pointer-events-none absolute rounded-sm border-2 border-primary"
        >
          <span className="absolute -top-px -left-px rounded-br-sm bg-primary px-1 text-[10px] font-semibold text-primary-foreground">
            Subject
          </span>
        </div>
      )}
      {picking && (
        <div className="pointer-events-none absolute inset-x-0 bottom-2 flex justify-center px-2">
          <div className="flex items-center gap-1.5 rounded-md bg-black/70 px-2.5 py-1 text-xs text-white backdrop-blur">
            {people.isError ? (
              <span className="text-amber-300">{(people.error as Error).message}</span>
            ) : !shownPeople ? (
              <>
                <Loader2 className="size-3.5 animate-spin" /> Finding people…
              </>
            ) : shownPeople.people.length ? (
              <>Click the person to follow · step to another frame if they are hidden</>
            ) : (
              <>No one found on this frame · step to another</>
            )}
          </div>
        </div>
      )}
    </div>
  )

  const transport = (
    <div className="flex flex-col gap-2.5">
      <Timeline
        input={input}
        count={count}
        first={first}
        last={last}
        playhead={playhead}
        onEdge={setEdge}
        onSeek={seek}
      />
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="icon"
          className="size-8 rounded-full"
          disabled={fallback}
          onClick={togglePlaying}
          aria-label={playing ? "Pause" : "Play selection"}
          title={playing ? "Pause (Space)" : "Play selection (Space)"}
        >
          {playing ? <Pause className="fill-current" /> : <Play className="ml-0.5 fill-current" />}
        </Button>
        <div className="text-xs tabular-nums">
          {formatTimecode(playhead / fps)}{" "}
          <span className="text-muted-foreground">/ {formatTimecode(count / fps)}</span>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          <Button
            variant="outline"
            size="sm"
            className="h-7 px-2 text-xs"
            onClick={() => setEdge("start", playhead)}
            title="Set start to this frame (I)"
          >
            Set start
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-7 px-2 text-xs"
            onClick={() => setEdge("end", playhead)}
            title="Set end to this frame (O)"
          >
            Set end
          </Button>
          {!fallback && (
            <div className="ml-1 flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                className="size-7"
                aria-label={muted ? "Unmute" : "Mute"}
                title={muted ? "Unmute" : "Mute"}
                onClick={() => {
                  if (muted || volume === 0) {
                    setMuted(false)
                    if (volume === 0) setVolume(0.5)
                  } else setMuted(true)
                }}
              >
                {muted || volume === 0 ? <VolumeX /> : volume < 0.5 ? <Volume1 /> : <Volume2 />}
              </Button>
              <Slider
                aria-label="Volume"
                className="w-16"
                min={0}
                max={1}
                step={0.01}
                value={[muted ? 0 : volume]}
                onValueChange={([next]) => {
                  setVolume(next)
                  setMuted(next === 0)
                }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )

  const controls = (
    <div className={cn("flex flex-col gap-3", portrait && "h-full")}>
      <div className={cn("grid gap-3", !portrait && "sm:grid-cols-2")}>
        <EdgeControl
          label="Start"
          frame={first}
          fps={fps}
          count={count}
          current={!playing && playhead === first}
          onShow={() => seek(first)}
          onChange={(frame) => setEdge("start", frame)}
        />
        <EdgeControl
          label="End"
          frame={last}
          fps={fps}
          count={count}
          current={!playing && playhead === last}
          onShow={() => seek(last)}
          onChange={(frame) => setEdge("end", frame)}
        />
      </div>
      <div className="text-xs tabular-nums">
        <span className="font-medium">{frames} frames</span>
        <span className="text-muted-foreground">
          {" "}
          · {(frames / fps).toFixed(2)} s of {(count / fps).toFixed(2)} s
          {Math.abs(fps - 30) > 0.5 ? ` · ${Number(fps.toFixed(2))} fps, resampled to 30 fps` : ""}
        </span>
      </div>
      <SubjectControl
        input={input}
        subject={subject}
        inClip={subjectInClip(subject, value)}
        picking={picking}
        onPick={togglePicking}
        onShow={() => subject && seek(subject.frame)}
        onClear={() => {
          props.onSubjectChange(null)
          setPicking(false)
        }}
        wide={!portrait}
      />
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        Space plays · ←/→ step a frame (Shift: 10) · I / O set start / end at the playhead · P picks the subject. Drag
        the green handles to trim, anywhere else on the strip to scrub.
      </p>
      <div className={cn("flex gap-2", portrait ? "mt-auto" : "justify-end")}>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            onChange(defaultClip(input))
            seek(0)
          }}
        >
          <RotateCcw /> Reset
        </Button>
        <Button size="sm" className={cn(portrait && "flex-1")} onClick={() => props.onOpenChange(false)}>
          Done
        </Button>
      </div>
    </div>
  )

  return (
    <div onKeyDown={onKeyDown} className={portrait ? "flex gap-5" : "flex flex-col gap-4"}>
      <div className="flex min-w-0 flex-col gap-3" style={portrait ? undefined : { width: previewSize.width }}>
        {preview}
        {transport}
      </div>
      {portrait ? <div className="flex w-[300px] shrink-0 flex-col">{controls}</div> : controls}
    </div>
  )
}

/** Absolute position of a source-pixel box over media that fills its container. */
function boxStyle([x1, y1, x2, y2]: [number, number, number, number], width: number, height: number) {
  return {
    left: `${(x1 / width) * 100}%`,
    top: `${(y1 / height) * 100}%`,
    width: `${((x2 - x1) / width) * 100}%`,
    height: `${((y2 - y1) / height) * 100}%`,
  }
}

/** Who the run follows: the largest person (automatic) or one picked on a frame. */
function SubjectControl(props: {
  input: InputRecord
  subject: Subject | null
  inClip: boolean
  picking: boolean
  onPick: () => void
  onShow: () => void
  onClear: () => void
  /** Wide card (video above the controls): the button sits on the same row. */
  wide: boolean
}) {
  const { subject } = props
  const pick = (
    <Button
      variant={props.picking ? "default" : "outline"}
      size="sm"
      className="h-7 shrink-0 text-xs"
      onClick={props.onPick}
    >
      <Crosshair /> {props.picking ? "Cancel picking" : subject ? "Pick again on this frame" : "Pick on this frame"}
    </Button>
  )
  return (
    <div className="flex flex-col gap-2 rounded-lg border p-2.5">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium">Subject</span>
        {subject && (
          <button
            type="button"
            className="text-[11px] text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
            onClick={props.onClear}
          >
            Use automatic
          </button>
        )}
      </div>
      <div className="flex items-center gap-2.5">
        {subject ? (
          <button type="button" onClick={props.onShow} title="Show the frame it was picked on">
            <SubjectThumb input={props.input} subject={subject} size={40} className="ring-2 ring-primary/60" />
          </button>
        ) : (
          <div className="grid size-10 shrink-0 place-items-center rounded bg-muted text-muted-foreground">
            <UserRound className="size-5" />
          </div>
        )}
        <div className="min-w-0 flex-1 text-[11px] leading-snug">
          {!subject ? (
            <span className="text-muted-foreground">Automatic: the largest person over the clip.</span>
          ) : props.inClip ? (
            <span>
              Picked on frame {subject.frame + 1}. The run follows this person{" "}
              <span className="text-muted-foreground">(where they cross someone, it can still switch).</span>
            </span>
          ) : (
            <span className="text-amber-600 dark:text-amber-400">
              Picked on frame {subject.frame + 1}, outside the clip, so the run uses the largest person. Pick again
              inside the clip.
            </span>
          )}
        </div>
        {props.wide && pick}
      </div>
      {!props.wide && pick}
    </div>
  )
}

/**
 * Player-style timeline: a filmstrip of the whole video, the clip framed by trim handles,
 * and a playhead. Drag a handle to trim, anywhere else to scrub.
 */
function Timeline(props: {
  input: InputRecord
  count: number
  first: number
  last: number
  playhead: number
  onEdge: (edge: Edge, frame: number) => void
  onSeek: (frame: number) => void
}) {
  const { count, first, last, playhead } = props
  const track = useRef<HTMLDivElement>(null)
  const drag = useRef<Edge | "playhead" | null>(null)
  const pct = (frame: number) => `${(frame / count) * 100}%`

  // Pointer position on the track in frame cells (0 .. count).
  const cellAt = (clientX: number) => {
    const box = track.current!.getBoundingClientRect()
    return (Math.min(Math.max(clientX - box.left, 0), box.width) / box.width) * count
  }
  const apply = (clientX: number) => {
    const cell = cellAt(clientX)
    if (drag.current === "start") props.onEdge("start", Math.round(cell))
    else if (drag.current === "end") props.onEdge("end", Math.round(cell) - 1)
    else props.onSeek(Math.floor(cell))
  }

  const thumbs = Array.from({ length: FILMSTRIP }, (_, index) =>
    Math.min(count - 1, Math.round(((index + 0.5) / FILMSTRIP) * count - 0.5)),
  )

  return (
    <div
      ref={track}
      className="relative mx-2.5 h-12 cursor-pointer touch-none rounded-md bg-muted select-none"
      onPointerDown={(event) => {
        const box = track.current!.getBoundingClientRect()
        const x = event.clientX - box.left
        const toStart = Math.abs(x - (first / count) * box.width)
        const toEnd = Math.abs(x - ((last + 1) / count) * box.width)
        // Within reach of a handle: trim (the nearer handle wins for very short clips); else scrub.
        drag.current = Math.min(toStart, toEnd) <= 12 ? (toStart <= toEnd ? "start" : "end") : "playhead"
        event.currentTarget.setPointerCapture(event.pointerId)
        apply(event.clientX)
      }}
      onPointerMove={(event) => drag.current && apply(event.clientX)}
      onPointerUp={() => (drag.current = null)}
      onPointerCancel={() => (drag.current = null)}
    >
      <div className="absolute inset-0 flex overflow-hidden rounded-md">
        {thumbs.map((frame, index) => (
          <img
            key={index}
            src={urls.inputFrame(props.input.id, frame, 96)}
            alt=""
            draggable={false}
            className="h-full min-w-0 flex-1 object-cover opacity-75"
          />
        ))}
      </div>
      {/* Outside the clip is dimmed. */}
      <div className="absolute inset-y-0 left-0 rounded-l-md bg-black/65" style={{ width: pct(first) }} />
      <div className="absolute inset-y-0 right-0 rounded-r-md bg-black/65" style={{ left: pct(last + 1) }} />
      <div
        className="pointer-events-none absolute inset-y-0 border-y-2 border-primary"
        style={{ left: pct(first), width: `calc(${pct(last + 1)} - ${pct(first)})` }}
      />
      <TrimHandle
        edge="start"
        frame={first}
        count={count}
        left={pct(first)}
        onStep={(delta) => props.onEdge("start", first + delta)}
      />
      <TrimHandle
        edge="end"
        frame={last}
        count={count}
        left={pct(last + 1)}
        onStep={(delta) => props.onEdge("end", last + delta)}
      />
      <div
        className="pointer-events-none absolute -top-1.5 -bottom-1.5 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_3px_rgba(0,0,0,0.8)]"
        style={{ left: `${((playhead + 0.5) / count) * 100}%` }}
      >
        <div className="absolute -top-1 left-1/2 size-2.5 -translate-x-1/2 rounded-full bg-white shadow" />
      </div>
    </div>
  )
}

function TrimHandle(props: {
  edge: Edge
  frame: number
  count: number
  left: string
  onStep: (delta: number) => void
}) {
  return (
    <div
      role="slider"
      tabIndex={0}
      aria-label={props.edge === "start" ? "Clip start" : "Clip end"}
      aria-valuemin={1}
      aria-valuemax={props.count}
      aria-valuenow={props.frame + 1}
      onKeyDown={(event) => {
        if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
          event.preventDefault()
          props.onStep((event.key === "ArrowLeft" ? -1 : 1) * (event.shiftKey ? 10 : 1))
        }
      }}
      className={cn(
        "absolute inset-y-0 flex w-2.5 cursor-ew-resize items-center justify-center bg-primary outline-none focus-visible:ring-2 focus-visible:ring-ring",
        props.edge === "start" ? "-translate-x-full rounded-l-md" : "rounded-r-md",
      )}
      style={{ left: props.left }}
    >
      <div className="h-4 w-0.5 rounded-full bg-primary-foreground/70" />
    </div>
  )
}

function EdgeControl(props: {
  label: string
  frame: number
  fps: number
  count: number
  current: boolean
  onShow: () => void
  onChange: (frame: number) => void
}) {
  const [draft, setDraft] = useState(String(props.frame + 1))
  useEffect(() => setDraft(String(props.frame + 1)), [props.frame])
  const commit = () => {
    const value = Number.parseInt(draft, 10)
    if (Number.isFinite(value)) props.onChange(value - 1)
    else setDraft(String(props.frame + 1))
  }
  const step = (delta: number) => props.onChange(props.frame + delta)

  return (
    <div
      onClick={props.onShow}
      className={cn(
        "flex cursor-pointer flex-col gap-1.5 rounded-lg border p-2.5 transition-colors",
        props.current ? "border-primary/60 bg-primary/5" : "hover:bg-muted/40",
      )}
    >
      <div className="flex items-center justify-between text-xs">
        <span className={cn("font-medium", props.current && "text-primary")}>{props.label}</span>
        <span className="text-muted-foreground tabular-nums">{formatTimecode(props.frame / props.fps)}</span>
      </div>
      <div className="flex items-center gap-1">
        <StepButton label="Back 10 frames" onClick={() => step(-10)}>
          <ChevronsLeft />
        </StepButton>
        <StepButton label="Back 1 frame" onClick={() => step(-1)}>
          <ChevronLeft />
        </StepButton>
        <Input
          aria-label={`${props.label} frame`}
          inputMode="numeric"
          value={draft}
          onFocus={props.onShow}
          onChange={(event) => setDraft(event.target.value.replace(/[^\d]/g, ""))}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") commit()
            if (event.key === "ArrowUp" || event.key === "ArrowDown") {
              event.preventDefault()
              step((event.key === "ArrowUp" ? 1 : -1) * (event.shiftKey ? 10 : 1))
            }
          }}
          className="h-8 min-w-0 flex-1 text-center tabular-nums"
        />
        <StepButton label="Forward 1 frame" onClick={() => step(1)}>
          <ChevronRight />
        </StepButton>
        <StepButton label="Forward 10 frames" onClick={() => step(10)}>
          <ChevronsRight />
        </StepButton>
      </div>
      <div className="text-[11px] text-muted-foreground tabular-nums">
        frame {props.frame + 1} of {props.count}
      </div>
    </div>
  )
}

function StepButton(props: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <Button
      variant="outline"
      size="icon"
      aria-label={props.label}
      title={props.label}
      className="size-8 shrink-0"
      onClick={(event) => {
        event.stopPropagation()
        props.onClick()
      }}
    >
      {props.children}
    </Button>
  )
}
