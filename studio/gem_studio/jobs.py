"""Persistent FIFO job queue; one inference subprocess runs at a time."""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .paths import DATA_DIR, JOBS_DIR, REPO_ROOT, ensure_ffmpeg_on_path

ACTIVE = ("queued", "running")


class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._jobs = {}
        self._process = None
        self._running_id = None
        self._load()
        threading.Thread(target=self._loop, daemon=True).start()

    # -- persistence -------------------------------------------------------
    def _load(self):
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        for path in JOBS_DIR.glob("*/job.json"):
            try:
                job = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") == "running":
                job.update(status="failed", error="Interrupted: Studio was closed during this run")
                self._write(job)
            if (job.get("refine") or {}).get("status") == "running":
                job["refine"].update(status="failed", error="Interrupted: Studio was closed")
                self._write(job)
            self._jobs[job["id"]] = job

    def _write(self, job):
        directory = JOBS_DIR / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "job.json.tmp"
        temporary.write_text(json.dumps(job, indent=2))
        os.replace(temporary, directory / "job.json")

    def directory(self, job_id):
        if job_id not in self._jobs:
            raise KeyError(job_id)
        return JOBS_DIR / job_id

    # -- public API --------------------------------------------------------
    def list(self):
        with self._lock:
            jobs = [dict(job) for job in self._jobs.values()]
        return sorted(jobs, key=lambda job: job["created"], reverse=True)

    def get(self, job_id):
        with self._lock:
            return dict(self._jobs[job_id])

    def create(self, *, kind, task, input_record, options, title=None):
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        job = {
            "id": job_id,
            "title": title or input_record["name"],
            "kind": kind,
            "task": task,
            "input": input_record,
            "options": options,
            "status": "queued",
            "stage": "Waiting in queue",
            "detail": None,
            "progress": None,
            "error": None,
            "device": None,
            "created": time.time(),
            "started": None,
            "finished": None,
            "inference_seconds": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._write(job)
        self._wake.set()
        return dict(job)

    def retry(self, job_id):
        with self._lock:
            old = self._jobs[job_id]
        return self.create(
            kind=old["kind"],
            task=old["task"],
            input_record=old["input"],
            options=old["options"],
            title=old["title"],
        )

    def refine(self, job_id, options):
        """Queue temporal refinement of a finished Pose video run."""
        with self._lock:
            job = self._jobs[job_id]
            if job["status"] != "done" or job["kind"] not in ("pose-video", "motion-video"):
                raise ValueError("Refinement applies to finished video runs")
            if (job.get("refine") or {}).get("status") in ACTIVE:
                raise ValueError("A refinement is already queued for this run")
            job["refine"] = {
                "status": "queued",
                "options": options,
                "stage": "Waiting in queue",
                "progress": None,
                "error": None,
                "created": time.time(),
            }
            self._write(job)
        self._wake.set()
        return self.get(job_id)

    def cancel(self, job_id):
        with self._lock:
            job = self._jobs[job_id]
            refine = job.get("refine") or {}
            if job["status"] == "queued":
                job.update(status="cancelled", stage="Cancelled", finished=time.time())
                self._write(job)
                return
            if refine.get("status") == "queued":
                refine.update(status="cancelled", stage="Cancelled")
                self._write(job)
                return
            if self._running_id != job_id:
                return
            job["cancel_requested"] = True
            process = self._process
        if process is not None:
            _kill_tree(process)

    def delete(self, job_id):
        self.cancel(job_id)
        with self._lock:
            self._jobs.pop(job_id, None)
        # job.json first: even if Windows still holds a file open (e.g. a video
        # being streamed), the run never reappears after a restart.
        (JOBS_DIR / job_id / "job.json").unlink(missing_ok=True)
        shutil.rmtree(JOBS_DIR / job_id, ignore_errors=True)

    def log(self, job_id, offset=0):
        path = JOBS_DIR / job_id / "log.txt"
        if not path.is_file():
            return "", 0
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(512 * 1024)
        return data.decode("utf-8", "replace"), offset + len(data)

    # -- execution ---------------------------------------------------------
    def _next(self):
        """Oldest queued inference or refinement, as (job, action)."""
        with self._lock:
            pending = [
                (job["created"], job, "infer")
                for job in self._jobs.values()
                if job["status"] == "queued"
            ]
            pending += [
                (job["refine"]["created"], job, "refine")
                for job in self._jobs.values()
                if (job.get("refine") or {}).get("status") == "queued"
            ]
            if not pending:
                return None, None
            _, job, action = min(pending, key=lambda item: item[0])
            if action == "infer":
                job.update(status="running", started=time.time(), stage="Starting")
            else:
                job["refine"].update(status="running", started=time.time(), stage="Starting")
            job.pop("cancel_requested", None)
            self._running_id = job["id"]
            self._write(job)
            return dict(job), action

    def _loop(self):
        while True:
            job, action = self._next()
            if job is None:
                self._wake.wait(2)
                self._wake.clear()
                continue
            try:
                self._run(job, action)
            except Exception as error:  # never let the queue thread die
                if action == "infer":
                    self._update(job["id"], status="failed", error=str(error))
                else:
                    self._update_refine(job["id"], status="failed", error=str(error))
            finally:
                self._running_id, self._process = None, None

    def _update(self, job_id, **changes):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(changes)
            self._write(job)

    def _update_refine(self, job_id, **changes):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not job.get("refine"):
                return
            job["refine"].update(changes)
            self._write(job)

    def _run(self, job, action):
        directory = JOBS_DIR / job["id"]
        env = dict(os.environ)
        ensure_ffmpeg_on_path()
        env.update(
            PATH=os.environ["PATH"],
            PYTHONUNBUFFERED="1",
            PYTHONIOENCODING="utf-8",
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
            GEM_STUDIO_DATA=str(DATA_DIR),
        )
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        command = [sys.executable, "-m", "gem_studio.worker", str(directory)]
        if action == "refine":
            command.append("--refine")
        update = self._update if action == "infer" else self._update_refine
        with (directory / "log.txt").open("ab") as log:
            if action == "refine":
                log.write(b"\n===== Refine =====\n")
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            self._process = process
            error = None
            for raw in process.stdout:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line.startswith("@@"):
                    error = self._event(job["id"], line[2:], action) or error
                    continue
                # tqdm-style carriage-return updates: keep only the final state.
                log.write((line.rsplit("\r", 1)[-1] + "\n").encode())
                log.flush()
            code = process.wait()
        result = directory / ("preview" if action == "infer" else "refined") / "preview.json"
        current = self.get(job["id"])
        if current.get("cancel_requested"):
            update(job["id"], status="cancelled", stage="Cancelled", finished=time.time())
        elif code == 0 and result.is_file():
            update(job["id"], status="done", stage="Done", progress=None, finished=time.time())
        else:
            update(
                job["id"],
                status="failed",
                stage="Failed",
                error=error or f"Worker exited with code {code}; see the log",
                finished=time.time(),
            )

    def _event(self, job_id, payload, action="infer"):
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return None
        kind = event.get("event")
        update = self._update if action == "infer" else self._update_refine
        if kind == "stage":
            update(job_id, stage=event["stage"], detail=event.get("detail"), progress=None)
        elif kind == "progress":
            with self._lock:  # frequent; skip the disk write
                job = self._jobs.get(job_id)
                if job is not None:
                    target = job if action == "infer" else job.get("refine") or {}
                    target["progress"] = [event["current"], event["total"]]
        elif kind == "device" and action == "infer":
            self._update(job_id, device=event["device"])
        elif kind == "refine_done":
            # Automatic refinement inside an inference run (fixed-camera Motion).
            self._update(
                job_id,
                refine={
                    "status": "done",
                    "options": event.get("options", {}),
                    "report": event.get("report"),
                    "stage": "Done",
                    "progress": None,
                    "error": None,
                    "created": time.time(),
                    "finished": time.time(),
                },
            )
        elif kind == "refine_report":
            self._update_refine(job_id, report=event.get("report"))
        elif kind == "inference_done":
            self._update(job_id, inference_seconds=event["seconds"])
        elif kind == "error":
            return event["message"]
        return None


def _kill_tree(process):
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        process.kill()


def job_files(job_id):
    """Downloadable outputs, newest exports last."""
    directory = JOBS_DIR / job_id
    files = []
    for name in (
        "predictions.pt",
        "predictions.json",
        "predictions_refined.pt",
        "predictions_refined.json",
    ):
        path = directory / name
        if path.is_file():
            files.append(_file_entry(path, name))
    exports = directory / "exports"
    if exports.is_dir():
        for path in sorted(exports.iterdir(), key=lambda p: p.stat().st_mtime):
            if path.is_file() and ".part." not in path.name and not path.name.endswith(".part"):
                files.append(_file_entry(path, f"exports/{path.name}"))
    return files


def _file_entry(path, relative):
    stat = Path(path).stat()
    return {"name": Path(path).name, "path": relative, "size": stat.st_size, "mtime": stat.st_mtime}
