import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { CheckCircle2, CircleAlert } from "lucide-react"
import { useEffect, useState } from "react"
import { toast } from "sonner"

import { PageHeader } from "@/components/logo"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { api } from "@/lib/api"
import { useTheme, type ThemeChoice } from "@/lib/theme"

export function SettingsPage() {
  const queryClient = useQueryClient()
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settings })
  const system = useQuery({
    queryKey: ["system"],
    queryFn: api.system,
    refetchInterval: (query) => (query.state.data?.runtime ? false : 1500),
  })
  const { theme, setTheme } = useTheme()
  const [vipe, setVipe] = useState("")
  const [blender, setBlender] = useState("")
  useEffect(() => {
    setVipe(settings.data?.vipe_executable ?? "")
  }, [settings.data?.vipe_executable])
  useEffect(() => {
    setBlender(settings.data?.blender_executable ?? "")
  }, [settings.data?.blender_executable])
  const save = useMutation({
    mutationFn: api.saveSettings,
    onSuccess: (data) => {
      queryClient.setQueryData(["settings"], data)
      queryClient.invalidateQueries({ queryKey: ["system"] })
      toast.success("Settings saved")
    },
  })
  const runtime = system.data?.runtime

  return (
    <div>
      <PageHeader title="Settings" description="Defaults for new runs and a quick health check of this machine." />
      <div className="grid max-w-5xl gap-6 p-8 lg:grid-cols-2">
        <Card title="Preferences">
          <Row label="Appearance">
            <Select value={theme} onValueChange={(value) => setTheme(value as ThemeChoice)}>
              <SelectTrigger className="w-44">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="dark">Dark</SelectItem>
                <SelectItem value="light">Light</SelectItem>
                <SelectItem value="system">Match Windows</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <Row label="Default device">
            <Select
              value={settings.data?.device ?? "auto"}
              onValueChange={(device) => save.mutate({ device })}
            >
              <SelectTrigger className="w-44">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto</SelectItem>
                <SelectItem value="cuda">GPU (CUDA)</SelectItem>
                <SelectItem value="cpu">CPU</SelectItem>
              </SelectContent>
            </Select>
          </Row>
          <div className="flex flex-col gap-2">
            <Label className="text-[13px]">ViPE executable (optional)</Label>
            <div className="flex gap-2">
              <Input
                value={vipe}
                placeholder="C:\path\to\vipe\.venv\Scripts\vipe.exe"
                onChange={(event) => setVipe(event.target.value)}
                className="font-mono text-xs"
              />
              <Button
                variant="secondary"
                disabled={vipe === (settings.data?.vipe_executable ?? "")}
                onClick={() => save.mutate({ vipe_executable: vipe.trim() })}
              >
                Save
              </Button>
            </div>
            <p className="text-[11px] text-muted-foreground">
              Enables the "Moving camera (ViPE)" option for Body Motion. ViPE is installed separately; see
              docs/VIPE_CAMERA.md.
            </p>
          </div>
          <div className="flex flex-col gap-2">
            <Label className="text-[13px]">Blender executable (for FBX export)</Label>
            <div className="flex gap-2">
              <Input
                value={blender}
                placeholder={system.data?.blender ?? "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"}
                onChange={(event) => setBlender(event.target.value)}
                className="font-mono text-xs"
              />
              <Button
                variant="secondary"
                disabled={blender === (settings.data?.blender_executable ?? "")}
                onClick={() => save.mutate({ blender_executable: blender.trim() })}
              >
                Save
              </Button>
            </div>
            <p className="text-[11px] text-muted-foreground">
              {system.data?.blender
                ? `Using ${system.data.blender}. Leave empty to auto-detect.`
                : "Not found. Install Blender (free, blender.org) or point to blender.exe; it converts animations to FBX."}
            </p>
          </div>
        </Card>

        <Card title="System">
          {!runtime ? (
            <div className="text-sm text-muted-foreground">Checking the Python runtime…</div>
          ) : (
            <dl className="grid grid-cols-[130px_1fr] gap-x-3 gap-y-2 text-xs">
              <Check ok={!!runtime.torch} label="PyTorch" value={runtime.torch ?? runtime.torch_error ?? "missing"} />
              <Check
                ok={!!runtime.cuda_available}
                warn
                label="CUDA"
                value={runtime.cuda_available ? `available · CUDA ${runtime.cuda_version}` : "not available - CPU only"}
              />
              {(runtime.gpus ?? []).map((gpu, index) => (
                <Check key={index} ok label={`GPU ${index}`} value={`${gpu.name} · ${gpu.memory_gb} GB`} />
              ))}
              <Check ok={!!runtime.gem} label="GEM-X package" value={runtime.gem ? "installed" : (runtime.gem_error ?? "")} />
              <Check
                ok={!!runtime.onnxruntime}
                label="ONNX Runtime"
                value={runtime.onnxruntime ? `${runtime.onnxruntime} · ${(runtime.onnx_providers ?? []).join(", ")}` : "missing"}
              />
              <Check ok={!!system.data?.ffmpeg} label="FFmpeg" value={system.data?.ffmpeg ?? "missing"} />
              <Check
                ok={!!runtime.onnx_providers?.includes("CUDAExecutionProvider")}
                warn
                label="GPU detectors"
                value={
                  runtime.onnx_providers?.includes("CUDAExecutionProvider")
                    ? "person/hand detectors run on the GPU"
                    : "detectors run on the CPU (re-run install.bat to enable GPU)"
                }
              />
              <Check ok={!!system.data?.blender} warn label="Blender (FBX)" value={system.data?.blender ?? "not found"} />
              <Check
                ok={!!system.data?.native}
                warn
                label="App window"
                value={system.data?.native ? "native (WebView2)" : "browser mode"}
              />
            </dl>
          )}
        </Card>

        <Card title="Locations" className="lg:col-span-2">
          <dl className="grid grid-cols-[130px_1fr] gap-x-3 gap-y-2 font-mono text-xs">
            {[
              ["Repository", system.data?.repo_root],
              ["Models", system.data?.weights_dir],
              ["SOMA assets", system.data?.soma_dir],
              ["Runs & media", system.data?.data_dir],
              ["Caches", system.data?.portable ? `${system.data.cache_dir} (portable)` : "user profile (shared)"],
              ["Python", `${system.data?.python ?? ""} · ${system.data?.platform ?? ""}`],
            ].map(([label, value]) => (
              <div key={label} className="contents">
                <dt className="font-sans text-muted-foreground">{label}</dt>
                <dd className="break-all">{value}</dd>
              </div>
            ))}
          </dl>
        </Card>
      </div>
    </div>
  )
}

function Card({ title, children, className }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <section className={`flex flex-col gap-5 rounded-xl border bg-card p-5 ${className ?? ""}`}>
      <h2 className="text-sm font-medium">{title}</h2>
      {children}
    </section>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <Label className="text-[13px]">{label}</Label>
      {children}
    </div>
  )
}

function Check({ ok, warn, label, value }: { ok: boolean; warn?: boolean; label: string; value: string }) {
  return (
    <div className="contents">
      <dt className="flex items-center gap-1.5 text-muted-foreground">
        {ok ? (
          <CheckCircle2 className="size-3.5 text-emerald-500" />
        ) : (
          <CircleAlert className={`size-3.5 ${warn ? "text-amber-500" : "text-destructive"}`} />
        )}
        {label}
      </dt>
      <dd className="min-w-0 break-words">{value}</dd>
    </div>
  )
}
