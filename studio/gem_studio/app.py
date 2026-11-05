"""FastAPI server for GEM-X Studio. Binds to localhost only.

Mutating requests require the per-launch token embedded in index.html, and
every request must name a loopback Host; together these stop other websites
(CSRF, DNS rebinding) from driving the local API.
"""

import asyncio
import json
import mimetypes
import os
import secrets
import struct
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import camera_path, inputs, mmd_model, models, subjects, system
from .exports import EXPORTERS, find_blender, load_faces, load_slots
from .jobs import JobManager, job_files
from .paths import (
    SETTINGS_FILE,
    STATIC_DIR,
    bundled_vipe,
    ensure_dirs,
    ensure_ffmpeg_on_path,
    python_executable,
    vipe_pipeline,
)
from .preview import ensure_preview_audio

DEFAULT_SETTINGS = {
    "device": "auto",
    "vipe_executable": "",
    "blender_executable": "",
    "theme": "system",
}
VARIANTS = {
    "raw": "preview",
    "refined": "refined",
    # Body Motion with an MMD model's limb proportions (Fit to model), a skinned mesh.
    "model_raw": "model_raw",
    "model_refined": "model_refined",
}
KINDS = {
    ("pose-image", "body"): "body_pose",
    ("pose-video", "body"): "body_pose",
    ("pose-image", "hand"): "hand_pose",
    ("pose-video", "hand"): "hand_pose",
    ("motion-video", "body"): "body_motion",
}


def load_settings():
    try:
        return {**DEFAULT_SETTINGS, **json.loads(SETTINGS_FILE.read_text())}
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_SETTINGS)


def _ranged(path, request, media_type=None):
    """Byte-range file responses so <video> can seek in large files."""
    path = Path(path)
    if not path.is_file():
        raise HTTPException(404)
    size = path.stat().st_size
    media_type = media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    header = request.headers.get("range")
    if not header or not header.startswith("bytes="):
        return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})
    start_text, _, end_text = header[6:].split(",")[0].partition("-")
    if start_text:
        start = int(start_text)
        end = min(int(end_text), size - 1) if end_text else size - 1
    else:
        start, end = max(size - int(end_text), 0), size - 1
    if start >= size or start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def chunks():
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = handle.read(min(1 << 20, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        chunks(),
        status_code=206,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
        },
    )


class RenderSession:
    """Encode viewer frames (JPEG, posted one by one) into an H.264 MP4.

    ``audio``: the preview video whose sound is laid under the frames (same timeline).
    """

    def __init__(self, target, fps, audio=None):
        self.target = Path(target)
        self.fps = fps
        self.audio = audio
        self.partial = self.target.with_name(self.target.name + ".part.mp4")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", f"{fps:.6f}", "-i", "pipe:0",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "medium",
                "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.partial),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )  # fmt: skip

    def write(self, data):
        self.process.stdin.write(data)
        self.frames += 1

    def finish(self):
        self.process.stdin.close()
        error = self.process.stderr.read().decode(errors="replace")
        if self.process.wait() != 0:
            self.partial.unlink(missing_ok=True)
            raise RuntimeError(error or "ffmpeg failed")
        if self.audio is not None and self.frames:
            with_sound = self.target.with_name(self.target.name + ".audio.part.mp4")
            muxed = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-i", str(self.partial), "-i", str(self.audio),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-t", f"{self.frames / self.fps:.6f}", "-movflags", "+faststart", str(with_sound),
                ],
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )  # fmt: skip
            if muxed.returncode == 0:
                os.replace(with_sound, self.partial)
            else:
                with_sound.unlink(missing_ok=True)  # keep the silent render rather than fail
        os.replace(self.partial, self.target)
        return self.target

    def abort(self):
        self.process.kill()
        self.process.wait()
        self.partial.unlink(missing_ok=True)


def create_app(*, port, native=False, dev=False, token=None):
    ensure_dirs()
    ensure_ffmpeg_on_path()
    system.start_probe()
    token = token or secrets.token_urlsafe(24)
    jobs = JobManager()
    downloads = models.DownloadManager()
    renders = {}
    render_lock = threading.Lock()
    app = FastAPI(title="GEM-X Studio", docs_url="/api/docs" if dev else None, redoc_url=None)
    app.state.token = token
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", "127.0.0.1", "localhost"}

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "")
        if not dev and host not in allowed_hosts:
            return JSONResponse({"detail": "Forbidden host"}, status_code=403)
        if (
            not dev
            and request.url.path.startswith("/api/")
            and request.method not in ("GET", "HEAD", "OPTIONS")
            and not secrets.compare_digest(request.headers.get("x-studio-token", ""), token)
        ):
            return JSONResponse({"detail": "Missing or invalid Studio token"}, status_code=403)
        return await call_next(request)

    def job_dir(job_id):
        try:
            return jobs.directory(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found") from None

    # -- system / settings -------------------------------------------------
    @app.get("/api/system")
    def get_system():
        return {
            **system.info(native=native),
            "blender": find_blender(load_settings().get("blender_executable")),
        }

    @app.get("/api/settings")
    def get_settings():
        return {**load_settings(), "vipe_bundled": bundled_vipe()}

    @app.put("/api/settings")
    def put_settings(values: dict = Body(...)):
        settings = {**load_settings(), **{k: v for k, v in values.items() if k in DEFAULT_SETTINGS}}
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2))
        return {**settings, "vipe_bundled": bundled_vipe()}

    # -- models ------------------------------------------------------------
    @app.get("/api/models")
    def get_models():
        return {**models.catalog(), "downloads": downloads.status()}

    @app.post("/api/models/download")
    def download_models(payload: dict = Body(...)):
        ids = list(payload.get("components") or [])
        for task in payload.get("tasks") or []:
            ids += models.TASKS[task]["components"]
        try:
            downloads.enqueue(list(dict.fromkeys(ids)))
        except KeyError as error:
            raise HTTPException(400, f"Unknown component {error}") from None
        return downloads.status()

    # -- inputs ------------------------------------------------------------
    @app.get("/api/inputs")
    def list_inputs():
        return inputs.list_inputs()

    @app.post("/api/inputs/upload")
    async def upload_input(request: Request, name: str):
        try:
            return await inputs.upload(name, request.stream())
        except ValueError as error:
            raise HTTPException(400, str(error)) from None

    @app.post("/api/inputs/register")
    def register_inputs(payload: dict = Body(...)):
        records, errors = [], []
        for path in payload.get("paths", []):
            try:
                records.append(inputs.register(path))
            except Exception as error:
                errors.append({"path": path, "error": str(error)})
        return {"inputs": records, "errors": errors}

    def input_or_404(input_id):
        try:
            return inputs.get(input_id)
        except KeyError:
            raise HTTPException(404, "Input not found") from None

    @app.get("/api/inputs/{input_id}/thumb")
    def input_thumb(input_id: str):
        input_or_404(input_id)
        return FileResponse(inputs.thumb_path(input_id), media_type="image/jpeg")

    @app.get("/api/inputs/{input_id}/media")
    def input_media(input_id: str, request: Request):
        return _ranged(input_or_404(input_id)["path"], request)

    @app.get("/api/inputs/{input_id}/frame")
    def input_frame(input_id: str, index: int = 0, height: int = 720):
        record = input_or_404(input_id)
        if record["kind"] != "video":
            raise HTTPException(400, "Not a video")
        try:
            data = inputs.frame_jpeg(record, max(0, index), max_height=min(max(height, 48), 1080))
        except ValueError as error:
            raise HTTPException(404, str(error)) from None
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/inputs/{input_id}/people")
    def input_people(input_id: str, index: int = 0):
        """People on one source frame, for picking a Body Motion run's subject."""
        record = input_or_404(input_id)
        if record["kind"] != "video":
            raise HTTPException(400, "Not a video")
        try:
            return subjects.detect_people(record, max(0, index))
        except FileNotFoundError as error:
            raise HTTPException(409, str(error)) from None
        except ValueError as error:
            raise HTTPException(404, str(error)) from None

    @app.delete("/api/inputs/{input_id}")
    def delete_input(input_id: str):
        input_or_404(input_id)
        inputs.delete(input_id)
        return {"ok": True}

    # -- jobs --------------------------------------------------------------
    @app.get("/api/jobs")
    def list_jobs():
        return jobs.list()

    @app.post("/api/jobs")
    def create_jobs(payload: dict = Body(...)):
        kind, task = payload.get("kind"), payload.get("task")
        if (kind, task) not in KINDS:
            raise HTTPException(400, f"Unsupported combination {kind} / {task}")
        requirement = models.TASKS[KINDS[(kind, task)]]
        missing = [
            models.COMPONENTS_BY_ID[c].name
            for c in requirement["components"]
            if models.COMPONENTS_BY_ID[c].missing()
        ]
        if missing:
            raise HTTPException(409, "Missing models: " + ", ".join(missing))
        options = dict(payload.get("options") or {})
        if options.get("subject"):
            subject = options["subject"]
            try:
                box = [float(value) for value in subject["box"]]
                frame = int(subject["frame"])
            except (KeyError, TypeError, ValueError):
                raise HTTPException(400, "Subject needs a frame and a box") from None
            if kind != "motion-video" or len(box) != 4 or frame < 0:
                raise HTTPException(400, "A subject is a source frame and box, for Body Motion")
            options["subject"] = {"frame": frame, "box": box}
        settings = load_settings()
        options.setdefault("device", settings["device"])
        vipe = settings.get("vipe_executable") or bundled_vipe()
        if vipe:
            options.setdefault("vipe_executable", vipe)
            if vipe_pipeline(vipe):
                options.setdefault("vipe_pipeline", vipe_pipeline(vipe))
        created = []
        for input_id in payload.get("input_ids", []):
            record = input_or_404(input_id)
            expected = "image" if kind == "pose-image" else "video"
            if record["kind"] != expected:
                raise HTTPException(400, f"{record['name']} is not a {expected}")
            created.append(jobs.create(kind=kind, task=task, input_record=record, options=options))
        return created

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, log_from: int = 0):
        job_dir(job_id)
        text, offset = jobs.log(job_id, log_from)
        return {**jobs.get(job_id), "log": text, "log_offset": offset}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job_dir(job_id)
        jobs.cancel(job_id)
        return jobs.get(job_id)

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str):
        job_dir(job_id)
        return jobs.retry(job_id)

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        job_dir(job_id)
        jobs.delete(job_id)
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/thumb")
    def job_thumb(job_id: str):
        job_dir(job_id)
        try:
            return FileResponse(inputs.thumb_path(jobs.get(job_id)["input"]["id"]))
        except KeyError:
            raise HTTPException(404) from None

    @app.post("/api/jobs/{job_id}/reveal")
    def reveal_job(job_id: str):
        directory = job_dir(job_id)
        if sys.platform == "win32":
            os.startfile(directory)  # noqa: S606 - local folder for the local user
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(directory)])
        return {"ok": True}

    # -- preview -----------------------------------------------------------
    def variant_dir(job_id, variant):
        if variant not in VARIANTS:
            raise HTTPException(400, "Unknown variant")
        return job_dir(job_id) / VARIANTS[variant]

    @app.get("/api/jobs/{job_id}/preview")
    def preview(job_id: str, variant: str = "raw"):
        path = variant_dir(job_id, variant) / "preview.json"
        if not path.is_file():
            raise HTTPException(404, "Preview not ready")
        ensure_preview_audio(jobs.get(job_id), job_dir(job_id))
        camera_path.ensure(path.parent)  # Motion runs built before camera paths existed
        return FileResponse(path, media_type="application/json")

    @app.get("/api/jobs/{job_id}/preview/media")
    def preview_media(job_id: str, request: Request):
        directory = job_dir(job_id) / "preview"
        for name in ("source.mp4", "source.jpg"):
            if (directory / name).is_file():
                return _ranged(directory / name, request)
        raise HTTPException(404)

    @app.get("/api/jobs/{job_id}/mesh/{stream}")
    def mesh(job_id: str, stream: str, start: int = 0, count: int = 32, variant: str = "raw"):
        try:
            data = load_slots(variant_dir(job_id, variant), stream, start, max(1, min(count, 256)))
        except (OSError, ValueError):
            raise HTTPException(404, "Unknown stream") from None
        return Response(
            data.tobytes(),
            media_type="application/octet-stream",
            headers={"X-Slots": str(len(data)), "Cache-Control": "no-cache"},
        )

    @app.post("/api/jobs/{job_id}/model-preview")
    async def make_model_preview(job_id: str, options: dict = Body(...)):
        """Build (or reuse) the preview with an MMD model's limb proportions."""
        from . import model_preview

        variant, model_id = options.get("variant", "refined"), str(options.get("model") or "")
        if variant not in ("raw", "refined"):
            raise HTTPException(400, "Unknown variant")
        try:
            mmd_model.model_info(model_id)
        except KeyError:
            raise HTTPException(
                404, "That MMD model is no longer stored; choose it again"
            ) from None
        if not (variant_dir(job_id, variant) / "preview.json").is_file():
            raise HTTPException(409, "Run has no results for this variant yet")
        directory = job_dir(job_id)
        if not model_preview.is_current(directory, variant, model_id):

            def work():
                return subprocess.run(
                    [python_executable(), "-m", "gem_studio.model_preview", str(directory),
                     variant, model_id],
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=900,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )  # fmt: skip

            result = await asyncio.to_thread(work)
            if result.returncode != 0 or not model_preview.is_current(
                directory, variant, model_id
            ):
                tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-6:])
                raise HTTPException(500, f"Could not build the model preview:\n{tail}")
        return {"variant": f"model_{variant}"}

    @app.get("/api/jobs/{job_id}/model-skin/{name}")
    def model_skin(job_id: str, name: str, variant: str = "model_refined"):
        from .model_preview import SKIN_FILES

        if name not in SKIN_FILES or not variant.startswith("model_"):
            raise HTTPException(404, "Unknown skin data")
        path = variant_dir(job_id, variant) / f"skin_{name}.bin"
        if not path.is_file():
            raise HTTPException(404, "Model preview not built")
        return FileResponse(path, media_type="application/octet-stream")

    @app.get("/api/jobs/{job_id}/faces/{topology}")
    def faces(job_id: str, topology: str, variant: str = "raw"):
        try:
            data = load_faces(variant_dir(job_id, variant), topology).astype("<i4")
        except (OSError, ValueError):
            raise HTTPException(404, "Unknown topology") from None
        return Response(data.tobytes(), media_type="application/octet-stream")

    @app.post("/api/jobs/{job_id}/refine")
    def refine_job(job_id: str, options: dict = Body(...)):
        job_dir(job_id)
        allowed = {
            "strength",
            "detail",
            "lock_shape",
            "world",
            "plant_feet",
            "contact_threshold",
            "floor_offset_cm",
            "level",
            "tilt_deg",
            "roll_deg",
            "depth_smoothing",
            "lock_feet",
            "follow_floor",
            "plant_hands",
            "contact_reach_cm",
        }
        try:
            return jobs.refine(job_id, {k: v for k, v in options.items() if k in allowed})
        except ValueError as error:
            raise HTTPException(409, str(error)) from None

    # -- exports -----------------------------------------------------------
    @app.get("/api/jobs/{job_id}/files")
    def files(job_id: str):
        job_dir(job_id)
        return job_files(job_id)

    @app.get("/api/jobs/{job_id}/files/{relative:path}")
    def download_file(job_id: str, relative: str, request: Request, inline: bool = False):
        directory = job_dir(job_id).resolve()
        path = (directory / relative).resolve()
        if directory not in path.parents or not path.is_file():
            raise HTTPException(404)
        if inline:
            return _ranged(path, request)
        return FileResponse(path, filename=path.name)

    def ensure_skeleton(job_id, variant, space):
        """Build the cached SOMA skeleton in a subprocess (needs torch; CPU only)."""
        from .skeleton import skeleton_path

        path = skeleton_path(job_dir(job_id), variant, space)
        result = subprocess.run(
            [
                python_executable(),
                "-m",
                "gem_studio.skeleton",
                str(job_dir(job_id)),
                variant,
                space,
            ],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0 or not path.is_file():
            tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-6:])
            raise RuntimeError(f"Could not build the skeleton:\n{tail}")
        return path

    @app.post("/api/jobs/{job_id}/exports/{kind}")
    async def make_export(job_id: str, kind: str, options: dict = Body(default={})):
        from . import rigging

        if kind not in EXPORTERS and kind not in ("gltf_skeleton", "vmd"):
            raise HTTPException(400, "Unknown export")
        variant = options.get("variant", "raw")
        source = variant_dir(job_id, variant)
        if not (source / "preview.json").is_file():
            raise HTTPException(409, "Run has no results for this variant yet")
        info = json.loads((source / "preview.json").read_text())
        space = options.get("space", "auto")
        if space == "auto":
            space = "world" if "world" in info.get("views", []) else "camera"
        step = max(1, int(options.get("step", 1)))
        naming = "mmd" if options.get("naming") == "mmd" else "soma"
        camera = bool(options.get("camera")) and kind in ("gltf_animation", "gltf_skeleton", "fbx")
        if camera:
            camera_path.ensure(source)
            info = json.loads((source / "preview.json").read_text())
        skeletal = kind in ("gltf_skeleton", "vmd") or (
            kind == "fbx" and options.get("mode", "skeleton") == "skeleton"
        )
        if skeletal and info.get("type") != "motion":
            if kind == "fbx" and "mode" not in options:
                skeletal = False  # Pose runs: shape-key FBX
            else:
                raise HTTPException(409, "Skeleton exports are available for Body Motion runs")
        suffix = []
        fit_model, arm_angle, vmd_scale = None, options.get("arm_angle_deg", 35), None
        if kind == "vmd":
            name, space = "motion.vmd", "world"
            vmd_scale = options.get("scale", 12.5)
            if options.get("model"):
                try:
                    fit_model = mmd_model.load_profile(str(options["model"]))
                except KeyError:
                    raise HTTPException(
                        404, "That MMD model is no longer stored; choose it again"
                    ) from None
                suffix.append("fit")
            elif vmd_scale == "auto" or arm_angle is None:
                raise HTTPException(400, "Automatic scale and arm angle need a model to fit to")
        elif kind == "gltf_skeleton":
            name = "animation_skeleton.glb"
        else:
            name = EXPORTERS[kind][0]
            if kind == "fbx" and skeletal:
                name = "animation_skeleton.fbx"
        if kind in ("gltf_animation", "gltf_skeleton", "fbx"):
            suffix += [space] + ([f"every{step}"] if step > 1 else [])
        if skeletal and kind != "vmd" and naming == "mmd":
            suffix.append("mmd")
        if camera:
            suffix.append("camera")
        if variant != "raw":
            suffix.append(variant)
        if suffix:
            stem, dot, extension = name.rpartition(".")
            name = f"{stem}_{'_'.join(suffix)}{dot}{extension}"
        target = job_dir(job_id) / "exports" / name
        target.parent.mkdir(exist_ok=True)
        partial = target.with_name(f"{target.stem}.part{target.suffix}")
        blender = find_blender(load_settings().get("blender_executable"))
        if kind == "fbx" and not blender:
            raise HTTPException(
                409, "FBX export needs Blender; install it or set its path in Settings"
            )

        def work():
            skeleton = ensure_skeleton(job_id, variant, space) if skeletal else None
            if kind == "vmd":
                report = rigging.export_vmd(
                    skeleton,
                    partial,
                    arm_angle_deg=None if arm_angle is None else float(arm_angle),
                    scale="auto" if vmd_scale == "auto" else float(vmd_scale),
                    face_forward=bool(options.get("face_forward", True)),
                    model=fit_model,
                )
                if report:
                    print(f"VMD fitted to the model: {report}", flush=True)
                return partial
            if kind == "gltf_skeleton":
                return rigging.export_skinned_gltf(
                    skeleton, source, partial, step=step, naming=naming, camera=camera, space=space
                )
            if kind == "fbx":
                make_glb = None
                if skeletal:

                    def make_glb(path):  # noqa: E306
                        rigging.export_skinned_gltf(
                            skeleton,
                            source,
                            path,
                            step=step,
                            naming=naming,
                            camera=camera,
                            space=space,
                        )

                return EXPORTERS["fbx"][1](
                    source,
                    partial,
                    blender=blender,
                    space=space,
                    step=step,
                    make_glb=make_glb,
                    camera=camera,
                )
            if kind == "gltf_animation":
                return EXPORTERS[kind][1](source, partial, space=space, step=step, camera=camera)
            return EXPORTERS[kind][1](source, partial)

        try:
            await asyncio.to_thread(work)
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            partial.unlink(missing_ok=True)
            raise HTTPException(500, str(error)) from None
        os.replace(partial, target)
        return next(f for f in job_files(job_id) if f["name"] == name)

    # -- MMD models (fit a VMD export to a model's proportions) ---------------
    @app.post("/api/mmd-models")
    async def upload_mmd_model(request: Request, name: str):
        """Read an uploaded .pmx into a fit profile; the .pmx itself is not kept."""
        if Path(name).suffix.lower() != ".pmx":
            raise HTTPException(400, "Choose an MMD model (.pmx)")
        mmd_model.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        temporary = mmd_model.MODELS_DIR / f"upload-{secrets.token_hex(4)}.pmx"
        try:
            with temporary.open("wb") as handle:
                async for chunk in request.stream():
                    handle.write(chunk)
            return await asyncio.to_thread(mmd_model.save_model, temporary, Path(name).name)
        except (ValueError, struct.error, IndexError) as error:
            raise HTTPException(400, f"Could not read this model: {error}") from None
        finally:
            temporary.unlink(missing_ok=True)

    @app.get("/api/mmd-models/{model_id}")
    def get_mmd_model(model_id: str):
        try:
            return mmd_model.model_info(model_id)
        except KeyError:
            raise HTTPException(404, "MMD model not found") from None

    @app.post("/api/jobs/{job_id}/exports-upload")
    async def upload_export(job_id: str, request: Request, name: str):
        safe = Path(name).name
        if not safe or Path(safe).suffix.lower() not in {".glb", ".obj", ".png", ".jpg", ".webm"}:
            raise HTTPException(400, "Unsupported export type")
        target = job_dir(job_id) / "exports" / safe
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(await request.body())
        return next(f for f in job_files(job_id) if f["name"] == safe)

    @app.post("/api/jobs/{job_id}/render/start")
    def render_start(job_id: str, payload: dict = Body(...)):
        name = Path(payload.get("name") or "render.mp4").name
        if not name.endswith(".mp4"):
            name += ".mp4"
        target = job_dir(job_id) / "exports" / name
        with render_lock:
            if job_id in renders:
                renders.pop(job_id).abort()
            media = job_dir(job_id) / "preview" / "source.mp4"
            info = json.loads((media.parent / "preview.json").read_text())
            audio = media if info.get("audio") else None
            renders[job_id] = RenderSession(target, float(payload.get("fps") or 30), audio=audio)
        return {"ok": True, "name": name}

    @app.post("/api/jobs/{job_id}/render/frame")
    async def render_frame(job_id: str, request: Request):
        session = renders.get(job_id)
        if session is None:
            raise HTTPException(409, "No render in progress")
        data = await request.body()
        try:
            await asyncio.to_thread(session.write, data)
        except (BrokenPipeError, OSError) as error:
            renders.pop(job_id, None)
            raise HTTPException(500, f"Encoder stopped: {error}") from None
        return {"frames": session.frames}

    @app.post("/api/jobs/{job_id}/render/finish")
    async def render_finish(job_id: str):
        session = renders.pop(job_id, None)
        if session is None:
            raise HTTPException(409, "No render in progress")
        try:
            target = await asyncio.to_thread(session.finish)
        except RuntimeError as error:
            raise HTTPException(500, str(error)) from None
        return next(f for f in job_files(job_id) if f["name"] == target.name)

    @app.post("/api/jobs/{job_id}/render/abort")
    def render_abort(job_id: str):
        session = renders.pop(job_id, None)
        if session is not None:
            session.abort()
        return {"ok": True}

    # -- frontend ----------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index():
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            return HTMLResponse(
                "<h1>GEM-X Studio frontend is not built</h1>"
                "<p>Run <code>npm install &amp;&amp; npm run build</code> in <code>studio/web</code>, "
                "or re-run the installer.</p>",
                status_code=503,
            )
        meta = f'<meta name="studio-token" content="{token}"><meta name="studio-native" content="{int(native)}">'
        return HTMLResponse(page.read_text(encoding="utf-8").replace("<head>", "<head>" + meta, 1))

    if (STATIC_DIR / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{name}")
    def static_root(name: str):
        path = STATIC_DIR / name
        if path.is_file() and path.parent == STATIC_DIR:
            return FileResponse(path)
        raise HTTPException(404)

    app.state.jobs = jobs
    return app
