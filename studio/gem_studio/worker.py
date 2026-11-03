"""Run one Studio job in its own process: inference, then preview and exports.

``python -m gem_studio.worker JOB_DIR`` reads ``JOB_DIR/job.json``. Structured
events are printed as ``@@{json}`` lines; everything else is plain log output.
Running each job in a fresh process isolates crashes and frees GPU memory.
"""

import json
import sys
import time
import traceback
from pathlib import Path

from .paths import MOTION_CONFIG, SOMA_DIR, WEIGHTS_DIR, ensure_ffmpeg_on_path

_last_progress = [0.0]


def emit(event, **payload):
    print("@@" + json.dumps({"event": event, **payload}), flush=True)


def stage(name, detail=None):
    emit("stage", stage=name, detail=detail)
    print(f"==> {name}" + (f": {detail}" if detail else ""), flush=True)


def progress(current, total, *, force=False):
    now = time.monotonic()
    if force or current >= total or now - _last_progress[0] > 0.25:
        _last_progress[0] = now
        emit("progress", current=int(current), total=int(total))


class ProgressFrames:
    """Sequence proxy that reports the highest frame index requested so far."""

    def __init__(self, frames):
        self._frames = frames
        self._seen = 0

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, index):
        value = self._frames[index]
        last = index.indices(len(self))[1] if isinstance(index, slice) else index + 1
        if last > self._seen:
            self._seen = last
            progress(last, len(self))
        return value

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getattr__(self, name):
        return getattr(self._frames, name)


def resolve_device(device):
    if device not in (None, "", "auto"):
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def onnx_cuda_available():
    """onnxruntime-gpu present with a CUDA provider (torch is imported first so
    ORT reuses torch's already-loaded CUDA/cuDNN libraries)."""
    try:
        import onnxruntime
        import torch  # noqa: F401

        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


def resolve_detector_device(options, device):
    """GEM's detectors default to CPU; on a CUDA run use the GPU when ORT can."""
    requested = options.get("detector_device") or "auto"
    if requested != "auto":
        return requested
    return "cuda" if device.startswith("cuda") and onnx_cuda_available() else "cpu"


DETECTOR_DEVICE = {"value": "cpu"}


def build_argv(job, output):
    kind, task, options = job["kind"], job["task"], job.get("options", {})
    source = job["input"]["path"]
    device = resolve_device(options.get("device"))
    emit("device", device=device)
    detector_device = resolve_detector_device(options, device)
    DETECTOR_DEVICE["value"] = detector_device
    print(f"Detector device: {detector_device}", flush=True)
    soma = str(SOMA_DIR)
    if kind in ("pose-image", "pose-video"):
        argv = [kind, "--task", task, "--input", source, "--soma-assets", soma]
        argv += ["--device", device, "--output", str(output)]
        argv += ["--detector-device", detector_device]
        if task == "body":
            argv += ["--checkpoint", str(WEIGHTS_DIR / "body_s240000.ckpt"), "--release", "body"]
            argv += ["--body-detector", str(WEIGHTS_DIR / "yolox_humanart.onnx")]
        else:
            argv += ["--checkpoint", str(WEIGHTS_DIR / "egohand_s060000.ckpt"), "--release", "hand"]
            argv += ["--hand-detector", str(WEIGHTS_DIR / "hand_detector.onnx")]
            argv += ["--hand-view", options.get("hand_view") or "exo"]
            argv += ["--hands", *(options.get("hands") or ["left", "right"])]
            if kind == "pose-video" and options.get("smooth_hand") is not None:
                argv.append("--smooth-hand" if options["smooth_hand"] else "--no-smooth-hand")
        return argv
    if kind == "motion-video":
        argv = ["motion-video", "--task", "body", "--input", source]
        argv += ["--config", str(MOTION_CONFIG), "--device", device, "--output", str(output)]
        argv += ["--camera-mode", options.get("camera_mode") or "background"]
        if options.get("camera_mode") == "vipe" and options.get("vipe_executable"):
            argv += ["--vipe-executable", options["vipe_executable"]]
        if options.get("start"):
            argv += ["--start", str(float(options["start"]))]
        if options.get("duration"):
            argv += ["--duration", str(float(options["duration"]))]
        if options.get("postprocess", True):
            argv += ["--postprocess", "--soma-assets", soma]
        return argv
    raise ValueError(f"Unknown job kind {kind!r}")


def _cpu_fallback(cls):
    """If the CUDA execution provider cannot start, run that detector on CPU."""
    original = cls.__init__

    def init(self, *args, device="cpu", **kwargs):
        try:
            original(self, *args, device=device, **kwargs)
        except (RuntimeError, ValueError) as error:
            if device == "cpu":
                raise
            print(f"GPU detector unavailable ({error}); falling back to CPU", flush=True)
            original(self, *args, device="cpu", **kwargs)

    cls.__init__ = init


def install_progress_hooks():
    """Wrap GEM stages that iterate frames so the UI gets a live progress bar."""
    import gem.inference.body as body
    import gem.inference.body_video as body_video
    import gem.inference.pose as pose
    from gem.common.detectors.hand import RTDETRHandONNX
    from gem.common.detectors.yolox import YOLOXONNX

    _cpu_fallback(YOLOXONNX)
    _cpu_fallback(RTDETRHandONNX)

    # Motion reads detector.device from its YAML; apply the resolved device.
    load_config = body_video.load_body_video_config

    def load_config_with_detector(path):
        config = load_config(path)
        config.setdefault("detector", {})["device"] = DETECTOR_DEVICE["value"]
        return config

    body_video.load_body_video_config = load_config_with_detector

    detect_track = body.detect_track

    def detect_track_with_progress(frames, *args, **kwargs):
        stage("Detecting and tracking people")
        return detect_track(ProgressFrames(frames), *args, **kwargs)

    body.detect_track = detect_track_with_progress

    run_pose, predict_frame = pose.run_pose, pose.predict_frame
    counter = {"done": 0, "total": 1}

    def run_pose_with_progress(predictor, input_path, annotations, **kwargs):
        counter.update(done=0, total=max(len(annotations), 1))
        stage("Running pose model", f"{counter['total']} frame(s)")
        return run_pose(predictor, input_path, annotations, **kwargs)

    def predict_frame_with_progress(*args, **kwargs):
        result = predict_frame(*args, **kwargs)
        counter["done"] += 1
        progress(counter["done"], counter["total"])
        return result

    pose.run_pose = run_pose_with_progress
    pose.predict_frame = predict_frame_with_progress

    load_predictor = pose.load_predictor

    def load_predictor_with_stage(*args, **kwargs):
        stage("Loading pose model")
        return load_predictor(*args, **kwargs)

    pose.load_predictor = load_predictor_with_stage

    build_conditions = body_video.build_body_conditions

    def build_conditions_with_progress(frames, config, *, device="cpu", progress=None):
        tracked = ProgressFrames(frames)

        def forward(message):
            # Each condition stage makes its own pass over the clip.
            tracked._seen = 0
            (progress or stage)(message)

        return build_conditions(tracked, config, device=device, progress=forward)

    body_video.build_body_conditions = build_conditions_with_progress
    run_body_video = body_video.run_body_video

    def run_body_video_with_stage(*args, progress=None, **kwargs):
        def forward(message):
            stage(message)

        return run_body_video(*args, progress=forward, **kwargs)

    body_video.run_body_video = run_body_video_with_stage

    try:
        import gem.inference.hand as hand

        detect_hands = hand.detect_hand_annotations

        def detect_hands_with_stage(*args, **kwargs):
            stage("Detecting hands")
            return detect_hands(*args, **kwargs)

        hand.detect_hand_annotations = detect_hands_with_stage
    except ImportError:
        pass


def run(job_dir):
    job_dir = Path(job_dir)
    job = json.loads((job_dir / "job.json").read_text())
    output = job_dir / "predictions.pt"
    ensure_ffmpeg_on_path()
    stage("Starting", f"{job['kind']} / {job['task']}")
    argv = build_argv(job, output)
    print("gem-infer " + " ".join(f'"{a}"' if " " in a else a for a in argv), flush=True)
    install_progress_hooks()
    from gem.inference.cli import main as infer

    started = time.time()
    infer(argv)
    emit("inference_done", seconds=round(time.time() - started, 2))
    stage("Building preview")
    from .preview import build_preview

    build_preview(job, output, job_dir, progress=progress, stage=stage)
    options = job.get("options", {})
    if (
        job["kind"] == "motion-video"
        and options.get("camera_mode") == "static"
        and options.get("anchor_world", True)
    ):
        refine_options = {"world": "anchor"}
        report = refine_job(job_dir, job, refine_options)
        emit("refine_done", options=refine_options, report=report)
    emit("done")


def refine_job(job_dir, job, options):
    """Write ``predictions_refined.pt`` and the ``refined/`` preview for a run."""
    import torch

    from gem.inference.serialization import save_predictions

    from .paths import SOMA_DIR
    from .preview import build_preview, write_predictions_json
    from .refine import refine

    device = resolve_device(job.get("options", {}).get("device"))
    stage("Loading predictions")
    saved = torch.load(job_dir / "predictions.pt", map_location="cpu", weights_only=False)
    if saved.get("format") == "gem_pose_predictions_v1":
        stage(
            "Smoothing",
            f"strength {options.get('strength', 2)}, {options.get('detail', 'balanced')}",
        )
    refined = refine(
        saved, options, soma_assets=SOMA_DIR, device=device, progress=progress, stage=stage
    )
    refined["studio_refine"] = options
    target = job_dir / "predictions_refined.pt"
    save_predictions(refined, target)
    stage("Building refined preview")
    build_preview(job, target, job_dir, progress=progress, stage=stage, name="refined")
    write_predictions_json(refined, job_dir / "predictions_refined.json")
    return json.loads(json.dumps(refined.get("studio_world") or {}, default=str))


def run_refine(job_dir):
    """Refine a finished run on request (Refine tab)."""
    job_dir = Path(job_dir)
    job = json.loads((job_dir / "job.json").read_text())
    report = refine_job(job_dir, job, (job.get("refine") or {}).get("options", {}))
    emit("refine_report", report=report)
    emit("done")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if "--refine" in argv:
            run_refine(argv[0])
            return
        run(argv[0])
    except SystemExit as error:
        if error.code not in (0, None):
            emit("error", message=f"Invalid arguments (exit {error.code}); see the log")
            raise
    except BaseException as error:
        traceback.print_exc()
        emit("error", message=f"{type(error).__name__}: {error}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
