/** Per-viewer preferences in localStorage, which can be unavailable (private windows, blocked site data). */
export function readStored<T>(key: string, fallback: T): T {
  try {
    const value = localStorage.getItem(key)
    return value == null ? fallback : (JSON.parse(value) as T)
  } catch {
    return fallback
  }
}

export function writeStored(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Not persisted; the setting still applies for this session.
  }
}

/** Playback volume shared by the result viewer and the clip editor. */
export const VOLUME_KEY = "gem-studio:volume"
export const MUTED_KEY = "gem-studio:muted"
