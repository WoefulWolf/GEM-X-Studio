"""Model/asset catalog, readiness checks and resumable background downloads.

Each *component* is a group of files that GEM loads together. Each *task*
(Body Pose, Hand Pose, Body Motion) lists the components it needs. Downloads
run in background threads inside the server process; progress is reported by
polling the partially written files on disk.
"""

import shutil
import threading
import time
import traceback
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .paths import SOMA_DIR, WEIGHTS_DIR

YOLOX_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_x_8xb8-300e_humanart-a39d44ed.zip"
)


@dataclass
class Component:
    id: str
    name: str
    description: str
    target: Path
    files: list[str]
    repo: str | None = None
    revision: str | None = None
    url: str | None = None
    approx_mb: int = 0
    license_url: str | None = None

    def paths(self):
        return [self.target / name for name in self.files]

    def missing(self):
        return [path for path in self.paths() if not path.is_file()]


COMPONENTS = [
    Component(
        "body_pose",
        "Body Pose checkpoint",
        "DINOv3 ViT-H/16+ Body Pose model (s240000). Also produces features for Body Motion.",
        WEIGHTS_DIR,
        ["body_s240000.ckpt", "body_config.json"],
        repo="nvidia/GEM-X",
        approx_mb=3540,
        license_url="https://huggingface.co/nvidia/GEM-X",
    ),
    Component(
        "person_detector",
        "Person detector (YOLOX-X HumanArt)",
        "ONNX person detector used for automatic Body Pose and Body Motion boxes.",
        WEIGHTS_DIR,
        ["yolox_humanart.onnx"],
        url=YOLOX_URL,
        approx_mb=380,
        license_url="https://github.com/open-mmlab/mmpose",
    ),
    Component(
        "motion",
        "Body Motion (native77)",
        "Temporal Motion network, statistics and the ViTPose 77-keypoint observation model.",
        WEIGHTS_DIR,
        ["motion_native77_e774_s292175.ckpt", "motion_stats_raw.npz", "vitpose.pth"],
        repo="nvidia/GEM-X",
        approx_mb=3580,
        license_url="https://huggingface.co/nvidia/GEM-X",
    ),
    Component(
        "hand_pose",
        "Hand Pose checkpoint (EgoHand)",
        "EgoHand GA checkpoint (s060000) for left and right hand reconstruction.",
        WEIGHTS_DIR,
        ["egohand_s060000.ckpt"],
        repo="nvidia/EgoHand-v1",
        approx_mb=10660,
        license_url="https://huggingface.co/nvidia/EgoHand-v1",
    ),
    Component(
        "hand_detector",
        "Hand detector (RT-DETR)",
        "ONNX hand detector for automatic left/right hand boxes.",
        WEIGHTS_DIR,
        ["hand_detector.onnx"],
        repo="nvidia/RT-DETR-Hand-Detector-v1",
        approx_mb=170,
        license_url="https://huggingface.co/nvidia/RT-DETR-Hand-Detector-v1",
    ),
    Component(
        "soma_core",
        "SOMA body model (core)",
        "Shared SOMA v0.3.3 rig and shape data.",
        SOMA_DIR,
        ["SOMA_neutral.npz", "SOMA_template_rig.usda", "SOMA_procedural_transforms.json"],
        repo="nvidia/SOMA-X",
        revision="v0.3.3",
        approx_mb=55,
        license_url="https://huggingface.co/nvidia/SOMA-X",
    ),
    Component(
        "soma_correctives",
        "SOMA pose correctives",
        "Corrective weights used by Body Pose and Body Motion meshes.",
        SOMA_DIR,
        ["correctives_model.pt"],
        repo="nvidia/SOMA-X",
        revision="v0.3.3",
        approx_mb=71,
        license_url="https://huggingface.co/nvidia/SOMA-X",
    ),
    Component(
        "soma_hand",
        "SOMA hand + MHR identity",
        "Hand mapping and MHR identity model used by Hand Pose.",
        SOMA_DIR,
        [
            "SOMAHand.npz",
            "MHR/mhr_model_lod1.pt",
            "MHR/base_body_lod1.obj",
            "MHR/SOMA_wrap_lod1.obj",
        ],
        repo="nvidia/SOMA-X",
        revision="v0.3.3",
        approx_mb=700,
        license_url="https://huggingface.co/nvidia/SOMA-X",
    ),
]
COMPONENTS_BY_ID = {component.id: component for component in COMPONENTS}

TASKS = {
    "body_pose": {
        "name": "Body Pose",
        "components": ["body_pose", "person_detector", "soma_core", "soma_correctives"],
    },
    "hand_pose": {
        "name": "Hand Pose",
        "components": ["hand_pose", "hand_detector", "soma_core", "soma_hand"],
    },
    "body_motion": {
        "name": "Body Motion",
        "components": ["body_pose", "motion", "person_detector", "soma_core", "soma_correctives"],
    },
}


@dataclass
class DownloadState:
    component: str
    status: str = "queued"  # queued | downloading | done | error
    file: str | None = None
    downloaded: int = 0
    total: int = 0
    error: str | None = None
    started: float = field(default_factory=time.time)


class DownloadManager:
    """One sequential download thread; requests queue behind each other."""

    def __init__(self):
        self._lock = threading.Lock()
        self._queue = []
        self._states = {}
        self._thread = None

    def status(self):
        with self._lock:
            return {key: vars(state).copy() for key, state in self._states.items()}

    def enqueue(self, component_ids):
        with self._lock:
            for component_id in component_ids:
                if component_id not in COMPONENTS_BY_ID:
                    raise KeyError(component_id)
                current = self._states.get(component_id)
                if current is not None and current.status in ("queued", "downloading"):
                    continue
                if not COMPONENTS_BY_ID[component_id].missing():
                    continue
                self._states[component_id] = DownloadState(component_id)
                self._queue.append(component_id)
            if self._queue and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

    def _run(self):
        while True:
            with self._lock:
                if not self._queue:
                    return
                component_id = self._queue.pop(0)
                state = self._states[component_id]
                state.status = "downloading"
            component = COMPONENTS_BY_ID[component_id]
            try:
                if component.url is not None:
                    self._download_yolox(component, state)
                else:
                    self._download_hf(component, state)
                state.status = "done"
            except Exception as error:  # surfaced in the UI
                traceback.print_exc()
                state.status = "error"
                state.error = f"{type(error).__name__}: {error}"

    def _download_hf(self, component, state):
        from huggingface_hub import HfApi, hf_hub_download

        component.target.mkdir(parents=True, exist_ok=True)
        missing = [name for name in component.files if not (component.target / name).is_file()]
        sizes = {}
        try:
            infos = HfApi().get_paths_info(component.repo, missing, revision=component.revision)
            sizes = {info.path: getattr(info, "size", 0) or 0 for info in infos}
        except Exception:
            pass
        state.total = sum(sizes.values()) or component.approx_mb * 1_000_000
        completed = 0
        for name in missing:
            state.file = name
            done = threading.Event()
            watcher = threading.Thread(
                target=self._watch_partial,
                args=(component.target, name, state, completed, done),
                daemon=True,
            )
            watcher.start()
            try:
                hf_hub_download(
                    component.repo,
                    name,
                    revision=component.revision,
                    local_dir=component.target,
                )
            finally:
                done.set()
                watcher.join()
            completed += sizes.get(name) or (component.target / name).stat().st_size
            state.downloaded = completed
        state.file = None

    @staticmethod
    def _watch_partial(target, name, state, completed, done):
        """huggingface_hub streams to hashed ``*.incomplete`` files under ``.cache``.

        The partial being written is the one that grows; untouched partials left
        by an interrupted session are ignored, while a resumed one is picked up
        as soon as it grows. (NTFS updates mtimes of open files lazily, so size
        growth is the reliable signal.)
        """
        del name  # partial names are hashed, not derived from the filename
        cache = target / ".cache" / "huggingface" / "download"
        sizes, active = {}, None
        while not done.wait(0.5):
            try:
                current = {p: p.stat().st_size for p in cache.rglob("*.incomplete")}
            except OSError:
                continue
            growing = [p for p, size in current.items() if size > sizes.get(p, -1)]
            if sizes and growing:  # first tick only records a baseline
                active = max(growing, key=lambda p: current[p] - sizes.get(p, 0))
            sizes = current
            if active in current:
                state.downloaded = completed + current[active]

    def _download_yolox(self, component, state):
        component.target.mkdir(parents=True, exist_ok=True)
        archive = component.target / "yolox_humanart.zip"
        partial = archive.with_suffix(".zip.part")
        state.file = archive.name
        request = urllib.request.Request(component.url, headers={"User-Agent": "gem-studio"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            state.total = int(response.headers.get("Content-Length") or 0)
            while chunk := response.read(1 << 20):
                out.write(chunk)
                state.downloaded += len(chunk)
        partial.replace(archive)
        with zipfile.ZipFile(archive) as bundle:
            member = next(name for name in bundle.namelist() if name.endswith("/end2end.onnx"))
            with (
                bundle.open(member) as source,
                (component.target / "yolox_humanart.onnx").open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
        archive.unlink()
        state.file = None


def catalog():
    components = []
    for component in COMPONENTS:
        missing = component.missing()
        components.append(
            {
                "id": component.id,
                "name": component.name,
                "description": component.description,
                "files": [str(path) for path in component.paths()],
                "missing": [str(path) for path in missing],
                "ready": not missing,
                "source": component.repo or component.url,
                "approx_mb": component.approx_mb,
                "license_url": component.license_url,
            }
        )
    tasks = {
        task_id: {
            "name": task["name"],
            "components": task["components"],
            "ready": all(not COMPONENTS_BY_ID[c].missing() for c in task["components"]),
        }
        for task_id, task in TASKS.items()
    }
    return {"components": components, "tasks": tasks}


def main(argv=None):
    """``python -m gem_studio.models download body_pose hand_pose ...`` for the installer."""
    import argparse

    parser = argparse.ArgumentParser(description="Download GEM-X Studio models")
    parser.add_argument("action", choices=("download", "status"))
    parser.add_argument("tasks", nargs="*", help=f"Any of: {', '.join(TASKS)}, all")
    args = parser.parse_args(argv)
    if args.action == "status":
        for component in catalog()["components"]:
            print(("[ok]      " if component["ready"] else "[missing] ") + component["name"])
        return
    selected = list(TASKS) if not args.tasks or "all" in args.tasks else args.tasks
    ids = list(dict.fromkeys(c for task in selected for c in TASKS[task]["components"]))
    manager = DownloadManager()
    manager.enqueue(ids)
    last = ""
    while True:
        states = manager.status()
        active = [s for s in states.values() if s["status"] in ("queued", "downloading")]
        for state in states.values():
            if state["status"] == "downloading":
                total = max(state["total"], 1)
                line = (
                    f"  {COMPONENTS_BY_ID[state['component']].name}: "
                    f"{state['downloaded'] / 1e6:,.0f} / {total / 1e6:,.0f} MB"
                )
                if line != last:
                    print(line.ljust(78), end="\r", flush=True)
                    last = line
        if not active:
            break
        time.sleep(1)
    print()
    failed = {k: v for k, v in manager.status().items() if v["status"] == "error"}
    for component_id, state in failed.items():
        print(f"[error] {COMPONENTS_BY_ID[component_id].name}: {state['error']}")
    for component in catalog()["components"]:
        if component["id"] in ids:
            print(("[ok]      " if component["ready"] else "[missing] ") + component["name"])
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
