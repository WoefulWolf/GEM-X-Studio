import type { Job, JobKind, Task } from "./api"

export function formatBytes(bytes: number) {
  if (!bytes) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB"]
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / 1024 ** index
  return `${value.toFixed(value >= 100 || index === 0 ? 0 : 1)} ${units[index]}`
}

export function formatDuration(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds)) return "-"
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  if (minutes < 60) return `${minutes}m ${rest.toString().padStart(2, "0")}s`
  return `${Math.floor(minutes / 60)}h ${(minutes % 60).toString().padStart(2, "0")}m`
}

export function formatTimecode(seconds: number) {
  const minutes = Math.floor(seconds / 60)
  return `${minutes}:${(seconds % 60).toFixed(2).padStart(5, "0")}`
}

export function timeAgo(epochSeconds: number) {
  const delta = Date.now() / 1000 - epochSeconds
  if (delta < 45) return "just now"
  if (delta < 3600) return `${Math.round(delta / 60)} min ago`
  if (delta < 86400) return `${Math.round(delta / 3600)} h ago`
  return new Date(epochSeconds * 1000).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  })
}

export const MODE_LABELS: Record<string, { title: string; short: string }> = {
  "pose-image:body": { title: "Body Pose", short: "Body · image" },
  "pose-video:body": { title: "Body Pose", short: "Body · video" },
  "pose-image:hand": { title: "Hand Pose", short: "Hand · image" },
  "pose-video:hand": { title: "Hand Pose", short: "Hand · video" },
  "motion-video:body": { title: "Body Motion", short: "Motion · video" },
}

export function modeLabel(kind: JobKind, task: Task) {
  return MODE_LABELS[`${kind}:${task}`] ?? { title: kind, short: kind }
}

export function jobElapsed(job: Job) {
  if (!job.started) return null
  return (job.finished ?? Date.now() / 1000) - job.started
}
