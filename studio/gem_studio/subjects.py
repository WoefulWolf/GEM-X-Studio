"""People on one source frame, for picking the subject of a Body Motion run.

The run's tracker follows everyone and normally keeps the largest person; a picked
subject (frame + box) makes it keep the track that matches that box instead.
"""

import subprocess
import threading

from .paths import WEIGHTS_DIR, ffmpeg_executable

DETECTOR = WEIGHTS_DIR / "yolox_humanart.onnx"
# Same person detector and confidence as the run's tracker (YOLOX defaults).
SCORE_THRESHOLD = 0.5

_detector = None
_lock = threading.Lock()


def _frame_bgr(record, index):
    """Full-resolution source frame ``index``, decoded and seeked exactly like GEM's cut."""
    import cv2
    import numpy as np

    fps = record.get("fps") or 30.0
    start = max(0.0, (index - 0.5) / fps)
    result = subprocess.run(
        [
            ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{start:.6f}", "-i", str(record["path"]), "-frames:v", "1",
            "-f", "image2pipe", "-c:v", "png", "-",
        ],
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )  # fmt: skip
    if result.returncode != 0 or not result.stdout:
        raise ValueError(result.stderr.decode(errors="replace") or "No frame at that position")
    image = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode that frame")
    return image


def detect_people(record, index):
    """Person boxes on source frame ``index``, largest first, in source pixels.

    The detector runs on the CPU (one frame takes a fraction of a second) so picking
    never competes with a run on the GPU.
    """
    global _detector
    if not DETECTOR.is_file():
        raise FileNotFoundError("The person detector is not downloaded yet (Body Motion models)")
    image = _frame_bgr(record, index)
    with _lock:
        if _detector is None:
            from gem.common.detectors.yolox import YOLOXONNX

            _detector = YOLOXONNX(str(DETECTOR), score_thr=SCORE_THRESHOLD, device="cpu")
        boxes, scores = _detector.detect(image)
    height, width = image.shape[:2]
    people = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = (float(value) for value in box)
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        box = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        people.append({"box": box, "score": round(float(score), 3)})

    def area(person):
        x1, y1, x2, y2 = person["box"]
        return (x2 - x1) * (y2 - y1)

    people.sort(key=area, reverse=True)
    return {"frame": index, "width": width, "height": height, "people": people}


def tracking_subject(options, record):
    """The run's subject on GEM's 30 fps clip timeline, or None for the largest person.

    ``options["subject"]`` holds the source frame and box picked in the Clip card. GEM's
    clip starts at ``options["start"]`` and is resampled to 30 fps, without scaling or
    shifting pixels, so only the frame index needs converting.
    """
    subject = options.get("subject")
    if not subject:
        return None
    fps = record.get("fps") or 30.0
    seconds = (int(subject["frame"]) + 0.5) / fps - float(options.get("start") or 0.0)
    return {"frame": max(0, int(seconds * 30)), "box": [float(value) for value in subject["box"]]}
