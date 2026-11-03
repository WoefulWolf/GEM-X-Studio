// Bridge to the pywebview host (GEM-X Studio desktop window). In a regular
// browser every helper falls back to standard web behaviour.
import { urls, type InputRecord } from "./api"

interface NativeApi {
  pick_media(): Promise<{ inputs: InputRecord[]; errors: { path: string; error: string }[] }>
  save_file(jobId: string, relative: string): Promise<string | null>
  open_external(url: string): Promise<void>
}

declare global {
  interface Window {
    pywebview?: { api: NativeApi }
  }
}

let ready: Promise<NativeApi | null> | null = null

/** Resolves once pywebview has injected its API (or quickly to null in a browser). */
export function nativeApi(): Promise<NativeApi | null> {
  ready ??= new Promise((resolve) => {
    if (window.pywebview?.api) return resolve(window.pywebview.api)
    const timer = window.setTimeout(() => resolve(null), 1500)
    window.addEventListener("pywebviewready", () => {
      window.clearTimeout(timer)
      resolve(window.pywebview?.api ?? null)
    })
  })
  return ready
}

/** Save a job output: native "Save as" dialog in the app, a normal download in a browser. */
export async function saveJobFile(jobId: string, relative: string): Promise<string | null> {
  const native = await nativeApi()
  if (native) return native.save_file(jobId, relative)
  const anchor = document.createElement("a")
  anchor.href = urls.file(jobId, relative)
  anchor.download = relative.split("/").pop() ?? "download"
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  return "browser"
}

export async function openExternal(url: string) {
  const native = await nativeApi()
  if (native) await native.open_external(url)
  else window.open(url, "_blank", "noopener")
}
