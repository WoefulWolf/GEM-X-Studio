"""Media library: uploaded copies or references to local files, with thumbnails."""

import json
import shutil
import time
import uuid
from pathlib import Path

from .paths import INPUTS_DIR, media_kind


def _probe(path, kind, thumb):
    import cv2

    info = {}
    if kind == "image":
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image {path.name}")
        info.update(width=image.shape[1], height=image.shape[0])
        frame = image
    else:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video {path.name}")
        try:
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
            info.update(
                width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=round(fps, 4),
                frame_count=frames,
                duration=round(frames / fps, 3) if fps else None,
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(frames // 4, 150))
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode video {path.name}")
        finally:
            capture.release()
    height, width = frame.shape[:2]
    scale = 480 / max(width, height)
    if scale < 1:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), cv2.INTER_AREA)
    cv2.imwrite(str(thumb), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return info


def _create(path, *, owned):
    path = Path(path).resolve()
    kind = media_kind(path)
    if kind is None:
        raise ValueError(f"Unsupported file type: {path.suffix or path.name}")
    if not path.is_file():
        raise FileNotFoundError(str(path))
    input_id = uuid.uuid4().hex[:12]
    directory = INPUTS_DIR / input_id
    directory.mkdir(parents=True, exist_ok=True)
    if owned:
        target = directory / path.name
        shutil.move(str(path), target)
        path = target
    try:
        info = _probe(path, kind, directory / "thumb.jpg")
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    record = {
        "id": input_id,
        "name": path.name,
        "path": str(path),
        "kind": kind,
        "size": path.stat().st_size,
        "owned": owned,
        "added": time.time(),
        **info,
    }
    (directory / "input.json").write_text(json.dumps(record, indent=2))
    return record


def register(path):
    """Reference a local file in place (native file picker)."""
    return _create(path, owned=False)


async def upload(name, stream):
    """Stream a browser upload to disk, then register the stored copy."""
    safe = Path(name).name or "upload"
    staging = INPUTS_DIR / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / f"{uuid.uuid4().hex[:8]}_{safe}"
    with temporary.open("wb") as handle:
        async for chunk in stream:
            handle.write(chunk)
    final = staging / safe
    if final.exists():
        final = staging / f"{uuid.uuid4().hex[:6]}_{safe}"
    temporary.rename(final)
    return _create(final, owned=True)


def get(input_id):
    if not input_id.isalnum():
        raise KeyError(input_id)
    path = INPUTS_DIR / input_id / "input.json"
    if not path.is_file():
        raise KeyError(input_id)
    return json.loads(path.read_text())


def list_inputs():
    records = []
    for path in INPUTS_DIR.glob("*/input.json"):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda record: record["added"], reverse=True)


def delete(input_id):
    get(input_id)
    shutil.rmtree(INPUTS_DIR / input_id, ignore_errors=True)


def thumb_path(input_id):
    get(input_id)
    return INPUTS_DIR / input_id / "thumb.jpg"
