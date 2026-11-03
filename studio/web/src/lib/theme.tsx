import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react"

export type ThemeChoice = "light" | "dark" | "system"

type ThemeContext = {
  theme: ThemeChoice
  resolved: "light" | "dark"
  setTheme: (theme: ThemeChoice) => void
}

const Context = createContext<ThemeContext | null>(null)
const KEY = "gem-studio-theme"

function readStored(): ThemeChoice {
  try {
    const value = localStorage.getItem(KEY)
    if (value === "light" || value === "dark" || value === "system") return value
  } catch {
    /* storage unavailable */
  }
  return "dark"
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<ThemeChoice>(readStored)
  const [systemDark, setSystemDark] = useState(
    () => window.matchMedia("(prefers-color-scheme: dark)").matches,
  )
  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)")
    const update = () => setSystemDark(query.matches)
    query.addEventListener("change", update)
    return () => query.removeEventListener("change", update)
  }, [])
  const resolved = theme === "system" ? (systemDark ? "dark" : "light") : theme
  useEffect(() => {
    document.documentElement.classList.toggle("dark", resolved === "dark")
    document.documentElement.style.colorScheme = resolved
  }, [resolved])
  const value = useMemo(
    () => ({
      theme,
      resolved,
      setTheme: (next: ThemeChoice) => {
        setThemeState(next)
        try {
          localStorage.setItem(KEY, next)
        } catch {
          /* storage unavailable */
        }
      },
    }),
    [theme, resolved],
  )
  return <Context.Provider value={value}>{children}</Context.Provider>
}

export function useTheme() {
  const context = useContext(Context)
  if (!context) throw new Error("useTheme must be used inside ThemeProvider")
  return context
}
