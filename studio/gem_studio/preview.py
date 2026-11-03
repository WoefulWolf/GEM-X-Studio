"""Convert a saved GEM prediction into a compact bundle for the web viewer.

Layout of ``JOB_DIR/preview``::

    preview.json            frames, instances, intrinsics, 2D keypoints
    source.mp4|source.jpg   browser-playable media on the prediction timeline, with the
                            source's sound when it has any
    faces_<topology>.npy    int32 [F, 3]
    verts_<stream>.npy      float16 [S, V, 3], relative to offsets_<stream>
    offsets_<stream>.npy    float32 [S, 3] per-slot centroid (keeps fp16 precise)

A *stream* is one vertex set (``camera_body``, ``world_body``, ``camera_left``...).
Each frame instance references a stream and a slot (row) within it. Geometry
stays in GEM's coordinates: camera streams are OpenCV camera metres; Motion
world streams are Y-up metres.
"""

import json
import os
import subprocess
import threading
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from . import camera_path


def _np(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def _jsonable(value):
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bool:
            return value.tolist()
        if value.is_floating_point():
            return np.round(value.double().numpy(), 6).tolist()
        return value.tolist()
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


class StreamWriter:
    """Collect per-slot vertices and store them as centred float16."""

    def __init__(self):
        self.streams = {}

    def add(self, stream, vertices):
        vertices = np.asarray(vertices, dtype=np.float32)
        slots = self.streams.setdefault(stream, [])
        slots.append(vertices)
        return len(slots) - 1

    def add_many(self, stream, vertices):
        start = len(self.streams.setdefault(stream, []))
        self.streams[stream].extend(np.asarray(vertices, dtype=np.float32))
        return start

    def save(self, directory):
        info = {}
        for stream, slots in self.streams.items():
            stacked = np.stack(slots)
            offsets = stacked.mean(axis=1)
            np.save(directory / f"offsets_{stream}.npy", offsets.astype(np.float32))
            np.save(
                directory / f"verts_{stream}.npy", (stacked - offsets[:, None]).astype(np.float16)
            )
            info[stream] = {"slots": len(slots), "vertex_count": int(stacked.shape[1])}
        return info


def _run(command):
    """Run a tool, decoding its output as UTF-8 (not the Windows code page)."""
    return subprocess.run(
        command,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _ffmpeg(*args):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args]
    result = _run(command)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, command, result.stdout, result.stderr
        )


AUDIO_CODEC = ("-c:a", "aac", "-b:a", "160k")


def _browser_video(source, target):
    """H.264 copy on the original frame timeline (no frame-rate conversion), with sound."""
    _ffmpeg(
        "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "20", "-pix_fmt", "yuv420p", *AUDIO_CODEC, "-movflags", "+faststart", str(target),
    )  # fmt: skip


def has_audio(path):
    """True when ffmpeg lists an audio stream in ``path``."""
    probe = _run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path)])
    return " Audio:" in (probe.stderr or "")


def add_audio(video, source, *, start=0.0, duration=None):
    """Mux the first audio track of ``source`` into ``video`` (in place), cut to the clip.

    GEM's 30 fps normalization drops sound; the clip starts ``start`` seconds into the
    source. Returns whether ``video`` now has sound.
    """
    video = Path(video)
    if not has_audio(source):
        return False
    partial = video.with_name(video.stem + ".audio.part.mp4")
    limit = ("-t", f"{duration:.6f}") if duration else ()
    try:
        _ffmpeg(
            "-i", str(video), "-ss", f"{start or 0.0:.6f}", "-i", str(source), *limit,
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *AUDIO_CODEC,
            "-movflags", "+faststart", str(partial),
        )  # fmt: skip
        os.replace(partial, video)
    finally:
        partial.unlink(missing_ok=True)
    return True


def _optional_audio(function, *args, **kwargs):
    """Sound is a bonus: a failure here leaves the preview silent instead of failing the run."""
    try:
        return function(*args, **kwargs)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Preview without sound: {type(error).__name__}: {error}", flush=True)
        return False


_audio_lock = threading.Lock()


def ensure_preview_audio(job, job_dir):
    """Previews built before sound support get it once, the first time they are opened."""
    raw = Path(job_dir) / "preview" / "preview.json"
    with _audio_lock:
        if not raw.is_file():
            return
        info = json.loads(raw.read_text())
        if "audio" in info or info.get("media") != "source.mp4":
            return
        source = Path(job["input"]["path"])
        options = job.get("options", {})
        motion = job.get("kind") == "motion-video"
        try:
            audio = source.is_file() and add_audio(
                raw.parent / "source.mp4",
                source,
                start=float(options.get("start") or 0.0) if motion else 0.0,
                duration=options.get("duration") if motion else None,
            )
        except (OSError, subprocess.CalledProcessError):
            return  # e.g. the video is open in the player; try again next time
        for path in (raw, Path(job_dir) / "refined" / "preview.json"):
            if path.is_file():
                data = json.loads(path.read_text())
                path.write_text(json.dumps({**data, "audio": audio}))


def _browser_image(source, target):
    import cv2

    image = cv2.imread(str(source))
    if image is None:
        raise OSError(f"Cannot read image {source}")
    cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 95])


def build_pose_preview(saved, directory, *, progress=None):
    writer = StreamWriter()
    smoothed = saved.get("hand_smoothing") or saved.get("body_smoothing")
    variant = "smoothed" if smoothed else "raw"
    frame_count = int(saved.get("source_frame_count") or 1)
    frames = [{"instances": []} for _ in range(frame_count)]
    image_size = None
    for count, record in enumerate(saved["frames"]):
        index = int(record["frame_id"])
        image_size = image_size or list(record["image_size"])
        frame = frames[index]
        default_k = record.get("intrinsics")
        if default_k is not None:
            frame["K"] = _np(default_k).round(4).tolist()
        for row in record["instances"]:
            geometry = row["smoothed"] if variant == "smoothed" else row
            topology = row["topology"]
            stream = f"camera_{topology}"
            slot = writer.add(stream, _np(geometry["vertices_camera"]))
            if "K" not in frame and row.get("intrinsics") is not None:
                frame["K"] = _np(row["intrinsics"]).round(4).tolist()
            keypoints = _np(geometry["keypoints_2d"])
            valid = np.asarray(_np(geometry["projection_valid"]) > 0.5) & np.isfinite(
                keypoints
            ).all(-1)
            side = row.get("side")
            track = int(row["track_id"]) if row.get("track_id") is not None else 0
            frame["instances"].append(
                {
                    "stream": stream,
                    "slot": slot,
                    "topology": topology,
                    "track": f"{topology}:{track}",
                    "label": f"{side} hand" if side else f"person {track}",
                    "side": side,
                    "bbox": _np(row["bbox"]).round(2).tolist(),
                    "kp2d": np.where(valid[:, None], keypoints, np.nan).round(2).tolist(),
                }
            )
        if progress is not None:
            progress(count + 1, len(saved["frames"]))
    for name, faces in saved["topology"].items():
        np.save(directory / f"faces_{name}.npy", _np(faces).astype(np.int32))
    return {
        "type": "pose",
        "task": saved["task"],
        "mode": "image" if saved["mode"] == "image" else "video",
        "fps": float(saved.get("source_fps") or 0) or None,
        "frame_count": frame_count,
        "image_size": image_size,
        "variant": variant,
        "topologies": list(saved["topology"]),
        "streams": writer.save(directory),
        "frames": frames,
        "views": ["overlay", "camera3d"],
    }


def _facing(params):
    """Per-frame horizontal facing (x, z) of the body; SOMA's rest pose faces +Z."""
    from scipy.spatial.transform import Rotation

    orient = Rotation.from_rotvec(params["global_orient"][0].double().cpu().numpy())
    forward = orient.apply([0.0, 0.0, 1.0])[:, [0, 2]]
    forward /= np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-9)
    return np.round(forward, 3).tolist()


def _camera_frame(saved):
    from .refine import display_camera_frame, gem_camera_to_world

    level = (saved.get("studio_world") or {}).get("camera_to_world_level")
    rotation = np.asarray(level) if level is not None else gem_camera_to_world(saved)
    return np.round(display_camera_frame(rotation), 6).tolist()


def _motion_camera_path(saved, directory, info):
    # Camera pose per frame; smooth rotations from the camera estimate for moving cameras.
    path = camera_path.compute(directory, info)
    rotations = (saved.get("camera") or {}).get("R_w2c")
    if rotations is not None and (info.get("camera") or {}).get("mode") not in (None, "static"):
        import torch

        world_to_camera = torch.as_tensor(rotations, dtype=torch.float64).reshape(-1, 3, 3)
        path = camera_path.with_estimated_rotations(
            directory, {**info, "camera_path": path}, world_to_camera.transpose(1, 2).numpy()
        )
    return path


def build_motion_preview(saved, directory, *, soma_assets, device, progress=None, stage=None):
    import torch

    from gem.inference.timing import prediction_fps
    from gem.motion.geometry import MotionGeometry
    from gem.motion.postprocess.api import ieee_geometry

    processed = bool(saved.get("postprocessing"))
    length = int(torch.as_tensor(saved["length"]).item())
    fps = prediction_fps(saved)
    geometry = MotionGeometry(soma_assets, device=device, chunk_size=32)
    np.save(directory / "faces_body.npy", geometry.faces.numpy().astype(np.int32))
    writer = StreamWriter()
    trajectory = None
    for coordinate, stream in (("incam", "camera_body"), ("global", "world_body")):
        if stage is not None:
            stage("Reconstructing meshes", "camera view" if coordinate == "incam" else "world")
        parameters = saved[f"pred_soma_params_{coordinate}"]
        joints_root = []
        with ieee_geometry() if processed and coordinate == "global" else nullcontext():
            for start, chunk in geometry.iter_chunks(parameters):
                values = chunk["vertices"][0].numpy()
                writer.add_many(stream, values)
                joints_root.append(chunk["joints"][0][:, 0].numpy())
                if progress is not None:
                    progress(start + len(values), length)
        if coordinate == "global":
            trajectory = np.concatenate(joints_root).round(4).tolist()
    camera = torch.as_tensor(saved["conditions"]["K_fullimg"]).cpu().float().numpy()[0]
    frames = [
        {
            "K": camera[index].round(4).tolist(),
            "instances": [
                {"stream": "camera_body", "slot": index, "topology": "body",
                 "track": "body:0", "label": "person"},
            ],
            "world": [
                {"stream": "world_body", "slot": index, "topology": "body",
                 "track": "body:0", "label": "person"},
            ],
        }
        for index in range(length)
    ]  # fmt: skip
    normalized = saved.get("video", {}).get("normalized", {})
    camera_info = {"mode": ((saved.get("release") or {}).get("camera") or {}).get("mode")}
    rotations = (saved.get("camera") or {}).get("R_w2c")
    if rotations is not None:
        # Estimated camera rotation relative to the first frame, in degrees. A
        # large value on a fixed camera explains drifting world trajectories.
        R = torch.as_tensor(rotations, dtype=torch.float64).reshape(-1, 3, 3).numpy()
        relative = np.einsum("tij,kj->tik", R, R[0])
        cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
        camera_info["max_rotation_deg"] = round(float(np.degrees(np.arccos(cosine)).max()), 2)
    return {
        "type": "motion",
        "camera": camera_info,
        # 3D view: camera-space motion shown with gravity up (levelled when refined).
        "camera_frame": _camera_frame(saved),
        # World view: horizontal facing per frame (x, z), to frame her from the front.
        "facing": _facing(saved["pred_soma_params_global"]),
        # Anchored worlds put the floor exactly at y=0; GEM's own world has no fixed floor.
        "ground": 0.0 if saved.get("studio_world") else None,
        "task": "body",
        "mode": "video",
        "fps": fps,
        "frame_count": length,
        "image_size": [normalized.get("width"), normalized.get("height")],
        "variant": "postprocessed" if processed else "raw",
        "topologies": ["body"],
        "streams": writer.save(directory),
        "frames": frames,
        "trajectory": trajectory,
        "views": ["overlay", "world", "camera3d"],
    }


def write_predictions_json(saved, path):
    """A readable companion to predictions.pt (vertices live in the mesh exports)."""
    skip = {"vertices_camera", "pose_token", "keypoints_crop", "topology", "conditions"}

    def prune(value):
        if isinstance(value, dict):
            return {key: prune(item) for key, item in value.items() if key not in skip}
        if isinstance(value, list):
            return [prune(item) for item in value]
        return value

    path.write_text(json.dumps(_jsonable(prune(saved))))


def build_preview(job, prediction_path, job_dir, *, progress=None, stage=None, name="preview"):
    """Write ``JOB_DIR/<name>``; variants other than ``preview`` reuse its media."""
    import torch

    from .paths import SOMA_DIR
    from .worker import resolve_device

    directory = Path(job_dir) / name
    directory.mkdir(parents=True, exist_ok=True)
    saved = torch.load(prediction_path, map_location="cpu", weights_only=False)
    source = Path(job["input"]["path"])
    if name != "preview":
        if saved.get("format") == "gem_pose_predictions_v1":
            info = build_pose_preview(saved, directory, progress=progress)
        else:
            device = resolve_device(job.get("options", {}).get("device"))
            info = build_motion_preview(
                saved,
                directory,
                soma_assets=SOMA_DIR,
                device=device,
                progress=progress,
                stage=stage,
            )
        raw = json.loads((Path(job_dir) / "preview" / "preview.json").read_text())
        info["media"] = raw["media"]
        if "audio" in raw:
            info["audio"] = raw["audio"]
        if info.get("type") == "motion":
            info["camera_path"] = _motion_camera_path(saved, directory, info)
        (directory / "preview.json").write_text(json.dumps(info))
        return info
    if saved.get("format") == "gem_pose_predictions_v1":
        info = build_pose_preview(saved, directory, progress=progress)
        if info["mode"] == "image":
            _browser_image(source, directory / "source.jpg")
            info["media"] = "source.jpg"
        else:
            if stage is not None:
                stage("Preparing preview video")
            _browser_video(source, directory / "source.mp4")
            info["media"] = "source.mp4"
            info["audio"] = _optional_audio(has_audio, directory / "source.mp4")
    else:
        device = resolve_device(job.get("options", {}).get("device"))
        info = build_motion_preview(
            saved, directory, soma_assets=SOMA_DIR, device=device, progress=progress, stage=stage
        )
        if stage is not None:
            stage("Preparing preview video")
        from gem.inference.video import normalize_video

        clip = saved["video"]["source"]
        normalize_video(
            source,
            directory / "source.mp4",
            start=clip["requested_start"],
            duration=clip["selected_duration"],
        )
        info["audio"] = _optional_audio(
            add_audio,
            directory / "source.mp4",
            source,
            start=clip["requested_start"],
            duration=clip["selected_duration"],
        )
        info["media"] = "source.mp4"
    if stage is not None:
        stage("Writing predictions.json")
    write_predictions_json(saved, Path(job_dir) / "predictions.json")
    if info.get("type") == "motion":
        info["camera_path"] = _motion_camera_path(saved, directory, info)
    (directory / "preview.json").write_text(json.dumps(info))
    return info
