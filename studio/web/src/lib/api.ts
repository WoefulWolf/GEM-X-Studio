// Typed client for the local Studio server.

export type MediaKind = "image" | "video"
export type JobKind = "pose-image" | "pose-video" | "motion-video"
export type Task = "body" | "hand"
export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled"

export interface InputRecord {
  id: string
  name: string
  path: string
  kind: MediaKind
  size: number
  owned: boolean
  added: number
  width: number
  height: number
  fps?: number
  frame_count?: number
  duration?: number | null
}

export interface JobOptions {
  device?: string
  detector_device?: string
  hand_view?: "exo" | "ego"
  hands?: ("left" | "right")[]
  smooth_hand?: boolean | null
  camera_mode?: "background" | "static" | "vipe"
  start?: number
  duration?: number | null
  postprocess?: boolean
  anchor_world?: boolean
  /** Body Motion: track this person instead of the largest one. */
  subject?: Subject | null
}

/** A person picked on one source frame: frame index and box (x1, y1, x2, y2) in source pixels. */
export interface Subject {
  frame: number
  box: [number, number, number, number]
}

/** People the detector finds on one source frame, largest first. */
export interface FramePeople {
  frame: number
  width: number
  height: number
  people: { box: [number, number, number, number]; score: number }[]
}

export interface Job {
  id: string
  title: string
  kind: JobKind
  task: Task
  input: InputRecord
  options: JobOptions
  status: JobStatus
  stage: string | null
  detail: string | null
  progress: [number, number] | null
  error: string | null
  device: string | null
  created: number
  started: number | null
  finished: number | null
  inference_seconds: number | null
  refine?: RefineState | null
  log?: string
  log_offset?: number
}

export interface RefineOptions {
  strength: number
  detail: "smooth" | "balanced" | "sharp"
  lock_shape: boolean
  /** Motion: "anchor" rebuilds world motion from camera space (fixed camera); "gem" keeps GEM's. */
  world?: "anchor" | "gem"
  plant_feet?: boolean
  /** GEM foot-contact probability that counts as planted (0-1). */
  contact_threshold?: number
  floor_offset_cm?: number
  /** Motion: level the world on the planted-feet floor plane ("auto") or not. */
  level?: "auto" | "off"
  /** Motion: manual extra lean toward (+) / away from (-) the camera, and sideways (+ right). */
  tilt_deg?: number
  roll_deg?: number
  /** Motion: smoothing width (30 fps frames) of the distance to the camera only; 0 = off. */
  depth_smoothing?: number
  /** Motion: cancel horizontal sliding of planted feet. */
  lock_feet?: boolean
  /** Motion, GEM world: the floor follows the planted feet, removing slow vertical drift. */
  follow_floor?: boolean
}

export interface RefineReport {
  mode?: "anchored_to_fixed_camera" | "gem_world"
  floor?: string
  level?: {
    method: "floor_plane" | "balance" | "insufficient_contact"
    measured_tilt_deg?: number
    toward_camera_deg?: number
    sideways_deg?: number
    applied_deg?: number
  }
  depth?: { sigma_frames: number; depth_jitter_removed_cm: number }
  lock_feet?: { planted_slide_mm_per_frame_before: number; max_root_shift_cm: number }
  plant_feet?: {
    planted_height_error_cm_before: number
    planted_height_error_cm_after: number
    planted_frame_fraction: number
  }
}

export interface RefineState {
  status: JobStatus
  options: RefineOptions
  stage: string | null
  detail?: string | null
  progress: [number, number] | null
  error: string | null
  finished?: number | null
  report?: RefineReport | null
}

export type Variant = "raw" | "refined"

export interface ExportOptions {
  variant?: Variant
  space?: "auto" | "world" | "camera"
  step?: number
  /** Skeletal exports: joint names. */
  naming?: "soma" | "mmd"
  /** FBX: skinned skeleton or shape-key animation. */
  mode?: "skeleton" | "shapes"
  /** VMD: model rest arm angle (degrees), metres-to-MMD-units scale, face forward. */
  arm_angle_deg?: number
  scale?: number
  face_forward?: boolean
  /** glTF/FBX animation: include the capture camera (animated in world space). */
  camera?: boolean
}

export interface ModelComponent {
  id: string
  name: string
  description: string
  files: string[]
  missing: string[]
  ready: boolean
  source: string
  approx_mb: number
  license_url: string | null
}

export interface DownloadState {
  component: string
  status: "queued" | "downloading" | "done" | "error"
  file: string | null
  downloaded: number
  total: number
  error: string | null
}

export type TaskId = "body_pose" | "hand_pose" | "body_motion"

export interface ModelsResponse {
  components: ModelComponent[]
  tasks: Record<TaskId, { name: string; components: string[]; ready: boolean }>
  downloads: Record<string, DownloadState>
}

export interface SystemInfo {
  python: string
  platform: string
  machine: string
  repo_root: string
  data_dir: string
  /** Portable install: library caches live inside the install folder. */
  portable?: boolean
  cache_dir?: string | null
  weights_dir: string
  soma_dir: string
  ffmpeg: string | null
  native: boolean
  pywebview: boolean
  blender: string | null
  runtime: null | {
    torch?: string
    cuda_available?: boolean
    cuda_version?: string | null
    gpus?: { name: string; memory_gb: number }[]
    mps_available?: boolean
    torch_error?: string
    gem?: boolean
    gem_error?: string
    onnxruntime?: string | null
    onnx_providers?: string[]
    open3d?: boolean
    probe_error?: string
  }
}

export interface Settings {
  device: string
  vipe_executable: string
  /** ViPE installed by install.bat -Vipe (.venv-vipe); used when vipe_executable is empty. */
  vipe_bundled?: string | null
  blender_executable: string
  theme: string
}

export interface FileEntry {
  name: string
  path: string
  size: number
  mtime: number
}

export interface PreviewInstance {
  stream: string
  slot: number
  topology: string
  track: string
  label: string
  side?: string | null
  bbox?: number[]
  kp2d?: (number | null)[][]
}

export interface PreviewFrame {
  K?: number[][]
  instances: PreviewInstance[]
  world?: PreviewInstance[]
}

export interface Preview {
  type: "pose" | "motion"
  task: Task
  mode: "image" | "video"
  fps: number | null
  frame_count: number
  image_size: [number, number]
  variant: string
  topologies: string[]
  streams: Record<string, { slots: number; vertex_count: number }>
  frames: PreviewFrame[]
  trajectory?: number[][]
  camera?: { mode?: string | null; max_rotation_deg?: number }
  ground?: number | null
  /** Motion: camera-to-display rotation for the 3D view (gravity up, heading removed). */
  camera_frame?: number[][]
  /**
   * Motion: capture-camera pose per frame. ``rotation`` (row-major 3x3) maps OpenCV camera
   * coordinates to the Y-up world, ``position`` is the camera centre. The display copies are
   * lightly smoothed for drawing the frustum and path.
   */
  camera_path?: {
    rotation: number[][]
    position: number[][]
    display_rotation: number[][]
    display_position: number[][]
  }
  /** Motion: per-frame horizontal facing (x, z) in world space. */
  facing?: number[][]
  views: string[]
  media: string
  /** The video carries the source's sound. */
  audio?: boolean
}

function meta(name: string) {
  return document.querySelector<HTMLMetaElement>(`meta[name="${name}"]`)?.content ?? ""
}

export const token = meta("studio-token")
export const isNativeShell = meta("studio-native") === "1"

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message)
  }
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (token) headers.set("X-Studio-Token", token)
  if (init.body && typeof init.body === "string") headers.set("Content-Type", "application/json")
  const response = await fetch(path, { ...init, headers })
  if (!response.ok) {
    let message = response.statusText
    try {
      const body = await response.json()
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, message)
  }
  const type = response.headers.get("content-type") ?? ""
  return (type.includes("json") ? response.json() : response.arrayBuffer()) as Promise<T>
}

const json = (method: string, body?: unknown): RequestInit => ({
  method,
  body: body === undefined ? undefined : JSON.stringify(body),
})

export const api = {
  system: () => request<SystemInfo>("/api/system"),
  settings: () => request<Settings>("/api/settings"),
  saveSettings: (values: Partial<Settings>) => request<Settings>("/api/settings", json("PUT", values)),
  models: () => request<ModelsResponse>("/api/models"),
  download: (body: { tasks?: TaskId[]; components?: string[] }) =>
    request("/api/models/download", json("POST", body)),
  inputs: () => request<InputRecord[]>("/api/inputs"),
  deleteInput: (id: string) => request(`/api/inputs/${id}`, json("DELETE")),
  registerPaths: (paths: string[]) =>
    request<{ inputs: InputRecord[]; errors: { path: string; error: string }[] }>(
      "/api/inputs/register",
      json("POST", { paths }),
    ),
  jobs: () => request<Job[]>("/api/jobs"),
  job: (id: string, logFrom = 0) => request<Job>(`/api/jobs/${id}?log_from=${logFrom}`),
  inputPeople: (id: string, index: number) => request<FramePeople>(`/api/inputs/${id}/people?index=${index}`),
  createJobs: (body: { kind: JobKind; task: Task; input_ids: string[]; options: JobOptions }) =>
    request<Job[]>("/api/jobs", json("POST", body)),
  cancel: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, json("POST")),
  retry: (id: string) => request<Job>(`/api/jobs/${id}/retry`, json("POST")),
  deleteJob: (id: string) => request(`/api/jobs/${id}`, json("DELETE")),
  reveal: (id: string) => request(`/api/jobs/${id}/reveal`, json("POST")),
  preview: (id: string, variant: Variant = "raw") =>
    request<Preview>(`/api/jobs/${id}/preview?variant=${variant}`),
  refine: (id: string, options: Partial<RefineOptions>) =>
    request<Job>(`/api/jobs/${id}/refine`, json("POST", options)),
  files: (id: string) => request<FileEntry[]>(`/api/jobs/${id}/files`),
  makeExport: (id: string, kind: string, options: ExportOptions = {}) =>
    request<FileEntry>(`/api/jobs/${id}/exports/${kind}`, json("POST", options)),
  uploadExport: (id: string, name: string, blob: Blob) =>
    request<FileEntry>(`/api/jobs/${id}/exports-upload?name=${encodeURIComponent(name)}`, {
      method: "POST",
      body: blob,
    }),
  renderStart: (id: string, name: string, fps: number) =>
    request<{ name: string }>(`/api/jobs/${id}/render/start`, json("POST", { name, fps })),
  renderFrame: (id: string, frame: Blob) =>
    request(`/api/jobs/${id}/render/frame`, { method: "POST", body: frame }),
  renderFinish: (id: string) => request<FileEntry>(`/api/jobs/${id}/render/finish`, json("POST")),
  renderAbort: (id: string) => request(`/api/jobs/${id}/render/abort`, json("POST")),
}

export function uploadInput(file: File, onProgress?: (fraction: number) => void) {
  // XHR rather than fetch: upload progress events.
  return new Promise<InputRecord>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open("POST", `/api/inputs/upload?name=${encodeURIComponent(file.name)}`)
    if (token) xhr.setRequestHeader("X-Studio-Token", token)
    xhr.upload.onprogress = (event) => event.lengthComputable && onProgress?.(event.loaded / event.total)
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText))
      else {
        let message = xhr.statusText
        try {
          message = JSON.parse(xhr.responseText).detail
        } catch {
          /* not JSON */
        }
        reject(new ApiError(xhr.status, message))
      }
    }
    xhr.onerror = () => reject(new ApiError(0, "Upload failed"))
    xhr.send(file)
  })
}

export const urls = {
  inputThumb: (id: string) => `/api/inputs/${id}/thumb`,
  inputMedia: (id: string) => `/api/inputs/${id}/media`,
  /** One exact source frame (JPEG), cut the way GEM cuts clips. */
  inputFrame: (id: string, index: number, height = 720) => `/api/inputs/${id}/frame?index=${index}&height=${height}`,
  jobThumb: (id: string) => `/api/jobs/${id}/thumb`,
  previewMedia: (id: string) => `/api/jobs/${id}/preview/media`,
  mesh: (id: string, stream: string, start: number, count: number, variant: Variant = "raw") =>
    `/api/jobs/${id}/mesh/${stream}?start=${start}&count=${count}&variant=${variant}`,
  faces: (id: string, topology: string, variant: Variant = "raw") =>
    `/api/jobs/${id}/faces/${topology}?variant=${variant}`,
  file: (id: string, path: string) => `/api/jobs/${id}/files/${path}`,
}
