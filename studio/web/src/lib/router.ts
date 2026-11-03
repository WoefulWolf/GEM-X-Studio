import { useSyncExternalStore } from "react"

// Minimal hash router: #/new, #/runs, #/runs/<id>, #/models, #/settings.
function subscribe(callback: () => void) {
  window.addEventListener("hashchange", callback)
  return () => window.removeEventListener("hashchange", callback)
}

const snapshot = () => window.location.hash.replace(/^#/, "") || "/new"

export function useRoute() {
  const path = useSyncExternalStore(subscribe, snapshot)
  return path.split("/").filter(Boolean)
}

export function navigate(path: string) {
  window.location.hash = path
}
