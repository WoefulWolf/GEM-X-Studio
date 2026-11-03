# GEM-X Studio

[Overview](../README.md) · [Installation](INSTALL.md) · [Inference guide](DEMO.md)

GEM-X Studio is a local desktop app for GEM-X inference. You pick images or
videos, choose **Body Pose**, **Hand Pose** or **Body Motion**, run on your GPU,
inspect the result in an interactive 3D viewer, and export meshes, data or
rendered video. Everything runs on your machine.

<p align="center">
  <em>Media library and task picker → live progress → overlay / 3D / world viewer → exports</em>
</p>

## Install on Windows (one click)

Requirements: 64-bit Windows 10/11, an internet connection, and ideally an
NVIDIA GPU with a recent driver. Python, Node.js and FFmpeg do **not** need to be
installed beforehand.

1. Download or clone this repository.
2. Double-click **`install.bat`** in the repository root.
3. Choose which models to download when asked (or skip and use the Models page
   later), then start **GEM-X Studio** from the Desktop or Start Menu shortcut.

The installer keeps everything inside the repository folder:

| Folder | Contents |
| --- | --- |
| `.tools\uv`, `.tools\python` | [uv](https://github.com/astral-sh/uv) and a private Python 3.12 |
| `.tools\node` | Portable Node.js, only downloaded if Node 20+ is missing (used to build the UI) |
| `.venv` | PyTorch, GEM-X (`.[video,hand]`) and Studio |
| `weights`, `assets\soma` | Model checkpoints and SOMA body-model assets |
| `studio_data` | Your media library, runs, exports and settings |

It detects your NVIDIA driver and installs PyTorch **2.10**, the series GEM is
validated with, in the matching CUDA build (13.0, 12.8 or 12.6), or the CPU build
when no GPU is found. Newer PyTorch builds are deliberately not used: they have
triggered CUDA faults on some GPUs. With the CUDA 13.0 build it also installs
`onnxruntime-gpu`, so GEM's person and hand detectors run on the GPU instead of
loading every CPU core (they default to CPU otherwise). Re-running
`install.bat` is safe; finished steps are skipped, so it doubles as an updater
after `git pull`.

Installer options (pass after `install.bat`):

```bat
install.bat -Models body_pose,body_motion -Yes   :: unattended, chosen models
install.bat -Torch cpu                           :: force a PyTorch build (auto, cpu, cu126, cu128, cu130)
install.bat -NoShortcut                          :: skip Desktop / Start Menu shortcuts
install.bat -Portable                            :: keep every cache inside the folder (see below)
install.bat -Portable -KeepCache                 :: ...but keep the download caches for faster reinstalls
install.bat -Vipe                                :: add ViPE camera tracking for moving cameras (see below)
install.bat -Help                                :: all options
```

### Portable install

By default the installer reuses the download caches in your user profile
(`%LOCALAPPDATA%\uv\cache`, `%LOCALAPPDATA%\npm-cache`), which saves time and
space when other Python or Node projects share them. While Studio runs, a few
libraries also keep small caches in the profile: the Hugging Face transfer cache,
NVIDIA Warp kernels compiled by SOMA, and the CUDA compute cache. The app window's
browser profile lives in `%LOCALAPPDATA%\GEM-X Studio`.

Answer **yes** to *Portable install*, or pass `-Portable`, to keep all of that
inside the folder:

| Location | Contents |
| --- | --- |
| `.tools\cache\uv`, `npm`, `huggingface` | Install-time downloads; emptied when the installer finishes, unless you pass `-KeepCache` |
| `.tools\cache\warp`, `nvidia`, `torch`, ... | Caches written while Studio runs (a few MB) |
| `.tools\webview` | The app window's browser profile |

The choice is stored in `.tools\portable`: Studio reads it at startup, and later
`install.bat` runs keep the setting. `install.bat -Shared` switches back to the
shared caches. Settings → Locations shows which mode is active. The only
files a portable install places elsewhere are the shortcuts; add `-NoShortcut` to
skip them.

### ViPE (moving cameras)

[ViPE](https://github.com/nv-tlabs/vipe) estimates a moving camera (handheld,
panning, following shots) for Body Motion. Answer **yes** to *Install ViPE?*, or
run `install.bat -Vipe`, to add it in its own environment, `.venv-vipe` (about
4 GB; ViPE needs `transformers` 4 while Studio uses 5). Studio finds it
automatically: *Full camera tracking (ViPE)* becomes available on New run, and
Settings shows which ViPE is in use. Fixed-camera clips don't need it.

- It needs an NVIDIA driver with CUDA 13 (driver 580 or newer).
- The first run downloads about 3 GB of ViPE models (GroundingDINO, SAM, MoGe-2,
  GeoCalib, DROID-SLAM), which takes several minutes; later runs start right away.
  With a portable install they are kept in `.tools\cache`.
- ViPE is published as source code that compiles CUDA extensions while it
  installs, and one of its libraries (gtsam) has no Windows build. The installer
  therefore uses a prebuilt Windows package, made with
  `scripts\windows\vipe\build_wheel.ps1` (ViPE plus `windows.patch`, compiled
  against Studio's PyTorch 2.10 / CUDA 13.0). Without gtsam, Studio runs ViPE's
  `pose_only` pipeline instead of GEM's default `pose_only_long`; GPU memory
  grows with clip length, so very long clips need more VRAM.

To remove the environment and shortcuts while keeping models and runs, run
`powershell -ExecutionPolicy Bypass -File scripts\windows\uninstall.ps1`
(add `-All` to delete models and runs too).

## Launching

- **Desktop / Start Menu shortcut** or **`studio.bat`**: opens Studio in its own
  app window.
- **`studio.bat --browser`**: serves Studio and opens it in your default browser.
- On Linux/macOS, inside an environment with GEM installed:
  `pip install -e "studio[native]"`, build the UI once (below), then
  `python -m gem_studio` (or `python -m gem_studio --browser`).

### Why the app window is a native webview

The app window is a thin [pywebview](https://pywebview.flowrl.com/) shell
around the operating system's own web engine (Microsoft Edge WebView2 on
Windows, WKWebView on macOS). It adds about a megabyte and no extra runtime.
Tauri and Wails also use the system webview, but they would need a Rust or Go
toolchain (or prebuilt binaries) on every machine, while still having to launch
the Python/PyTorch process that does the actual work. pywebview lives inside
that same Python environment. If WebView2 is missing, Studio falls back to your
browser automatically.

## Using Studio

**New run.** Drop files on the window or click *browse*. In the app window this
opens the native file picker and references files in place; in a browser, files
are uploaded into `studio_data\inputs`. Every file stays in the **media library**
for re-runs. Select one or more items, pick a task and press **Run**. Multiple
files are queued as separate runs.

| Task | Inputs | Options |
| --- | --- | --- |
| Body Pose | images, videos (framewise) | device |
| Hand Pose | images, videos (framewise) | third-person (exo) or first-person (ego) view, left/right hands, temporal smoothing for ego video |
| Body Motion | videos, up to 60 s per run | is the camera moving (fixed by default; moving uses background rotation estimates or ViPE); anchor to the camera; contact & IK refinement; clip range |

**Clip (Body Motion).** With one video selected, the run panel shows the frames that
will be analysed; **Change start/end** opens a card sized to the video for picking the
exact first and last frame. Drag the green trim handles on the filmstrip, scrub the
white playhead and press **Set start** / **Set end** (I / O), or type frame numbers and
step one or ten frames. **Play** loops the selection with sound. The preview shows
exactly the frame GEM will use: the start and duration Studio sends make GEM's cut
begin and end on the chosen frames (sources that are not 30 fps are then resampled).

**Runs.** Runs execute one at a time on the GPU, each in its own process, so a
crash or cancel never takes the app down and GPU memory is released after each
run. Progress, the current stage and the full log are shown live. Runs persist
across restarts.

**Viewer.**

- *Overlay*: the mesh drawn over the source frames with each frame's camera
  intrinsics, so it lines up exactly. Toggle the mesh, the source video (also the
  small picture-in-picture in 3D and World), 2D keypoints, opacity and colour.
- *3D*: camera-centred. The camera sits at the origin with its real tilt and
  field of view, shown as a frustum with a line down to the floor (its height),
  and the body appears exactly where it was relative to the lens. For Body
  Motion the scene is gravity-aligned (levelled when refined), so a camera that
  filmed upward shows as tilted while the body stands upright on a flat floor. The
  view opens looking through the capture camera (same position, direction and
  field of view as the video); orbit freely, and the crosshair button returns to
  the camera's view.
- *World* (Body Motion): world-space motion on a ground grid with the root
  trajectory, opening on a three-quarter view from the subject's front. The view follows the subject until you orbit; the crosshair
  re-centres and resumes following.
- Space plays/pauses, ←/→ step one frame (Shift for 10), with speed and loop
  controls. Videos with sound play it, cut to the analysed clip; the volume
  control next to loop (click the speaker to mute) is remembered. Runs made
  before sound support get it the first time they are opened, as long as the
  original video is still where it was.

**Refine (Pose videos).** Framewise Body and Hand Pose can jitter from frame to
frame. The *Refine* tab runs GEM's motion-adaptive smoother
(`gem.pose.smoothing`) over the predicted rotations, camera translation and
shape, then decodes the meshes again through SOMA; vertices are never averaged.
It typically takes a few seconds.

- *Strength*: the widest smoothing window, in 30 fps frames (GEM's default is 2).
- *Fast motion*: Sharper keeps quick gestures crisp, Smoother also calms fast
  movement (it scales the speeds at which smoothing backs off).
- *Lock shape*: keeps body or hand proportions constant across the clip.

The raw result is kept. The green **Raw / Refined** switch at the top of the viewer
compares them (runs open on Refined once one exists), and exports follow the selected one (`predictions_refined.pt` holds the refined
parameters and geometry).

**Refine (Body Motion).** GEM builds world motion by integrating the predicted
root velocity. On long in-place movement (dance, exercise), small errors can
accumulate into drift or a slow climb, and Contact/IK can add height on sliding
footwork. Motion's Refine tab offers:

- *World trajectory*: **Anchored to camera** (default when the run used a fixed
  camera) places the camera-space motion in the world under one rotation, GEM's
  own gravity estimate averaged over the clip. It is exact for a still camera and
  has no drift. **GEM world** keeps GEM's integrated motion, for moving cameras.
- *Stability*: a single camera judges distance far less precisely than
  position, so the body wobbles toward and away from the camera, most visible
  from the side. *Depth stabilization* (default 4 frames) smooths only the
  distance to the camera, moving each frame along its own line of sight so the
  overlay still lines up. *Stop foot sliding* (on by default) keeps planted feet
  in place by shifting the body; the correction is eased in and short-term
  only, so it never pulls the motion away from its anchored position. Turn it
  off for deliberately gliding footwork. On the test dance clip, depth jitter
  fell from 3.4 cm to 1.2 cm (the same as sideways), root acceleration along
  depth by about 12×, and planted-foot sliding from 7.3 to 2.9 mm per frame.
- *Auto-level on the floor* (on by default): GEM can misjudge the camera's
  pitch, for example a low phone tilted upward, which leaves the world tilted
  so the body appears to lean back from the side and feet float as they step
  forward or back. Studio fits a plane through the planted feet over the whole
  clip, which is the floor and therefore gravity, and rotates the world (pitch
  and roll only) to make it level. If the feet cover too little ground for a
  plane, it uses balance instead: the centre of mass over the planted feet. On
  the test dance clip it measured 6.9° (6.6° back, 1.7° sideways); afterwards
  the floor is level to 0.4° and the centre of mass sits over the feet. *Lean
  toward/away from camera* and *Sideways lean* add a manual correction on top;
  check it from the side in the World view.
- *Plant feet flat on the floor* (on by default): adjusts hip height and runs
  GEM's CCD leg IK so that during predicted foot contact the ankle and ball of
  the foot sit at their anatomical flat-foot heights (from SOMA's rest pose, the
  same for both feet). *Contact sensitivity* sets how many frames count as
  planted.
- *Body height*: 0 rests the soles exactly on the floor; positive values lift the body
  above it (e.g. to leave room for shoe soles), negative values sink it in.
- *Smoothing*: the same adaptive filter as for Pose, off by default since Motion
  is already temporal.

The floor (y = 0) is set from the body's lowest point during GEM-predicted foot
contact, and the results line reports the planted-foot height error before and
after. New Body Motion runs ask whether the camera is moving; with **No, fixed**
(the default) the anchored world is produced automatically, and GEM's world is
kept under *Raw*. A run that estimated camera rotation shows the amount, with a
one-click *Re-run as fixed camera*.

**Export.**

| Export | Contents |
| --- | --- |
| glTF animation (`.glb`) | **Skeleton** (Body Motion, default): a skinned body and an animated 77-joint SOMA armature, written natively without Blender. **Shape keys**: every frame's exact mesh as morph targets, one node per track. Options: World or Camera space, every Nth frame |
| FBX animation | The same skeleton (armature + skin) or shape-key animation, converted by a local Blender install |
| VMD motion (MMD) | MikuMikuDance motion for any MMD model, matched by standard bone names (Body Motion) |
| Point cache | Rest OBJ + `.pc2` per track (Blender *Mesh Cache* modifier, Maya, 3ds Max) |
| OBJ sequence | One OBJ per frame and track, zipped |
| Current frame GLB / OBJ | The meshes on screen, Y-up metres |
| Screenshot | PNG of the current view at source resolution |
| Rendered video | H.264 MP4 of exactly what the viewer shows, rendered frame by frame, with the clip's sound |
| `predictions.pt` / `.json` | The complete GEM result (identical to `gem-infer --output`) and a readable JSON of parameters, joints, 2D keypoints and boxes |
| NumPy `.npz` | All vertices, faces and frame/slot mapping |

**Skeleton exports** use SOMA's own rig and skinning weights. Bones rest in a
T-pose with world-aligned axes, and each frame stores every joint's rotation (plus
the hips' translation), so the armature retargets cleanly in Blender, Unity,
Unreal or Maya. Linear skinning omits SOMA's pose correctives, so the skinned
surface differs from GEM's by about 3 mm on average. Use **Shape keys** when the
exact surface matters; they are much larger (about 0.2 MB per frame versus about
2 MB for the whole skeleton export). **Bone names** can be *Standard* (SOMA:
Hips, Spine1, LeftArm, LeftShin, …) or *MMD* (センター, 上半身, 左腕, 左ひざ, …).
Skeletons are built from Body Motion results; Pose videos export shape keys.

**MMD (VMD).** The VMD exporter retargets the motion onto MMD's standard bones:
センター (translation), 下半身, 上半身, 上半身2, 首, 頭, shoulders, arms, elbows,
wrists, all finger joints, legs, knees, ankles and 足先EX. Rotations are mapped
through world-space deltas, so SOMA's extra spine and neck joints fold into the
MMD bones. Played back on an MMD-style skeleton, the ankles match GEM exactly and
the head and wrists within about 1.5 cm. Options:

- *Model arm angle*: MMD models rest in an A-pose, typically 30–40° (default
  35°). Use 0° for T-pose models.
- *Scale*: metres to MMD units (default 1 unit = 8 cm).
- *Face the camera at the start*: turns the motion so the dancer faces the MMD
  camera.

The motion is keyed at 30 fps. Leg IK (左足ＩＫ, 右足ＩＫ, つま先ＩＫ) is switched off
in the file so the legs follow the capture exactly.

Exports are written to the run's `exports` folder (**Folder** opens it). In the
app window, *Save* opens a native Save As dialog.

Rendered video uses the viewer itself, so it works on Windows without Open3D.
The Open3D renderer (`gem-render`) remains available from the command line on
its [supported platforms](INSTALL.md#rendering-dependencies).

## Models

The **Models** page shows what each task needs and downloads it with one click,
with progress, from the sources in [Pretrained models](PRETRAINED_MODELS.md):
GEM-X Body Pose and Motion, EgoHand, the RT-DETR hand detector, SOMA v0.3.3
assets and the YOLOX person detector. Each model has its own license on its
Hugging Face page. The installer offers the same downloads, which can also be
run directly:

```bat
.venv\Scripts\python -m gem_studio.models download body_pose hand_pose body_motion
.venv\Scripts\python -m gem_studio.models status
```

## Settings

- **Default device**: Auto (CUDA when available), GPU or CPU.
- **ViPE executable**: the ViPE used for *Full camera tracking*. Leave it empty
  to use the one `install.bat -Vipe` set up, or point to your own install
  (see [ViPE cameras](VIPE_CAMERA.md)).
- **Blender executable**: used for FBX export; auto-detected when empty.
- The System panel reports PyTorch/CUDA, GPUs, whether detectors run on the GPU,
  ONNX Runtime, FFmpeg, Blender and paths.

Environment variables `GEM_WEIGHTS_DIR`, `GEM_SOMA_ASSETS` and `GEM_STUDIO_DATA`
override the model, asset and data folders.

## Troubleshooting

- **Runs in the background**: Studio opts out of Windows' efficiency mode
  (EcoQoS), so a run keeps full speed while you work in other windows.
- **Screen goes black for a few seconds and the run fails with "illegal memory
  access"**: Windows reset the graphics driver (Event Viewer → System shows
  `nvlddmkm` event 153, "Reset TDR occurred"). This is not a VRAM shortage. If it
  happens at random and a retry works, check GPU overclocks or undervolts (for
  example an MSI Afterburner voltage/frequency curve): sustained compute is less
  forgiving than games. Test with the stock settings.

## Development

```bash
# terminal 1: API server with token checks disabled for the Vite proxy
python -m gem_studio --dev --no-open --port 8765
# terminal 2: hot-reloading UI on http://localhost:5173
cd studio/web && npm install && npm run dev
```

`npm run build` writes the production UI to `studio/gem_studio/static`, which
the server serves. The UI uses React, Tailwind CSS v4,
[shadcn/ui](https://ui.shadcn.com) components (`npx shadcn@latest add <name>`)
and three.js.

Layout:

| Path | Role |
| --- | --- |
| `studio/gem_studio/app.py` | FastAPI server: inputs, jobs, previews, exports, models |
| `studio/gem_studio/worker.py` | Per-run subprocess that calls `gem.inference.cli`, adds progress reporting and builds the preview |
| `studio/gem_studio/preview.py` | Converts predictions into the viewer bundle (fp16 vertex streams, keypoints, intrinsics, browser-playable media) |
| `studio/gem_studio/exports.py` | Shape-key glTF, Blender FBX, OBJ sequence, point cache and NPZ exporters |
| `studio/gem_studio/skeleton.py` | Per-frame SOMA skeleton and skinning data (cached next to the preview) |
| `studio/gem_studio/rigging.py` | Skinned glTF, MMD bone mapping and the VMD writer |
| `studio/gem_studio/refine.py` | Pose smoothing, and Motion anchoring, floor placement, foot planting and smoothing |
| `studio/gem_studio/models.py` | Model catalog and downloader |
| `studio/web/src/components/viewer` | three.js viewer (`engine.ts`) and streaming mesh cache |
| `scripts/windows/install.ps1` | Windows installer |

The server listens on `127.0.0.1` only. State-changing requests require a
per-launch token embedded in the page, and every request must use a loopback
`Host` header, so other websites cannot drive the local API.
