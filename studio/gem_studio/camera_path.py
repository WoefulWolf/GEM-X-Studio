"""Per-frame capture-camera pose in the world, from a Motion preview's two mesh copies.

A Motion preview stores the same body twice per frame: in OpenCV camera space
(``camera_body``) and in the Y-up world (``world_body``). The rigid transform that
maps one onto the other is the camera's pose for that frame, so the viewer and the
exporters can place a moving camera without reading GEM's camera fields. Seen from
that pose, the world mesh projects exactly like the video (sub-millimetre fit).

numpy only: it runs in the server for previews built before this existed.
"""

import json
import threading
from pathlib import Path

import numpy as np

from .exports import load_slots

# Display copy: a light Gaussian over frames, so the drawn frustum and path don't
# shake with per-frame depth noise. The exact poses stay available for the camera view.
SMOOTHING_FRAMES = 3.0
SAMPLE_EVERY = 8  # vertices used for the fit


def _fit(source, target):
    """Rotation R and translation t minimising |R source + t - target| (Kabsch)."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    covariance = (target - target_mean).T @ (source - source_mean)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    return rotation, target_mean - rotation @ source_mean


def _gaussian(values, sigma):
    """Gaussian smoothing along axis 0 with edge values held (nearest)."""
    radius = int(np.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([values[:1].repeat(radius, 0), values, values[-1:].repeat(radius, 0)])
    return np.stack(
        [np.convolve(padded[:, k], kernel, mode="valid") for k in range(values.shape[1])], axis=1
    )


def _project_rotations(matrices):
    """Nearest rotation to each 3x3 matrix (SVD)."""
    u, _, vt = np.linalg.svd(matrices)
    sign = np.sign(np.linalg.det(u @ vt))
    fix = np.ones((len(matrices), 3))
    fix[:, 2] = sign
    return u @ (fix[:, :, None] * vt)


def compute(directory, info):
    """``camera_path`` for a Motion preview: exact and display (smoothed) poses.

    ``rotation`` maps OpenCV camera coordinates (x right, y down, z forward) to the
    Y-up world, row-major; ``position`` is the camera centre in metres.
    """
    directory = Path(directory)
    frames = info["frames"]
    camera_slots = [frame["instances"][0]["slot"] for frame in frames]
    world_slots = [frame["world"][0]["slot"] for frame in frames]
    camera = load_slots(directory, "camera_body")[:, ::SAMPLE_EVERY].astype(np.float64)
    world = load_slots(directory, "world_body")[:, ::SAMPLE_EVERY].astype(np.float64)
    rotations, positions = [], []
    for camera_slot, world_slot in zip(camera_slots, world_slots):
        rotation, translation = _fit(camera[camera_slot], world[world_slot])
        rotations.append(rotation)
        positions.append(translation)
    rotations, positions = np.asarray(rotations), np.asarray(positions)
    sigma = SMOOTHING_FRAMES if len(frames) > 1 else 0
    if sigma:
        display_positions = _gaussian(positions, sigma)
        display_rotations = _project_rotations(
            _gaussian(rotations.reshape(-1, 9), sigma).reshape(-1, 3, 3)
        )
    else:
        display_positions, display_rotations = positions, rotations
    return {
        "rotation": rotations.reshape(-1, 9).round(5).tolist(),
        "position": positions.round(4).tolist(),
        "display_rotation": display_rotations.reshape(-1, 9).round(5).tolist(),
        "display_position": display_positions.round(4).tolist(),
    }


def with_estimated_rotations(directory, info, camera_to_world):
    """Replace the fitted rotations by the run's own camera estimate (ViPE, background).

    The per-frame fit takes its rotation from GEM's camera-space and world-space body
    orientations, which are predicted separately and disagree by a few degrees from
    frame to frame: the camera then pitches and, seen from metres away, bobs up and
    down. A camera estimate is smooth; one rotation aligns its world with GEM's (the
    chordal mean of fit x estimate^T), and the camera centre is then solved from the
    body (least squares) with that rotation fixed.
    """
    path = info["camera_path"]
    fitted = np.asarray(path["rotation"], dtype=np.float64).reshape(-1, 3, 3)
    count = len(fitted)
    estimate = np.asarray(camera_to_world, dtype=np.float64).reshape(-1, 3, 3)[:count]
    if len(estimate) != count:
        return path
    u, _, vt = np.linalg.svd(np.einsum("tij,tkj->ik", fitted, estimate))
    align = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    rotations = align @ estimate
    frames = info["frames"]
    camera = load_slots(directory, "camera_body")[:, ::SAMPLE_EVERY].astype(np.float64)
    world = load_slots(directory, "world_body")[:, ::SAMPLE_EVERY].astype(np.float64)
    camera_mean = np.stack([camera[frame["instances"][0]["slot"]].mean(0) for frame in frames])
    world_mean = np.stack([world[frame["world"][0]["slot"]].mean(0) for frame in frames])
    positions = world_mean - np.einsum("tij,tj->ti", rotations, camera_mean)
    display_positions = _gaussian(positions, SMOOTHING_FRAMES) if count > 1 else positions
    return {
        **path,
        "rotation": rotations.reshape(-1, 9).round(5).tolist(),
        "position": positions.round(4).tolist(),
        "display_rotation": rotations.reshape(-1, 9).round(5).tolist(),
        "display_position": display_positions.round(4).tolist(),
        "rotation_source": "camera_estimate",
    }


_lock = threading.Lock()


def ensure(directory):
    """Add ``camera_path`` to a Motion ``preview.json`` that lacks it (older runs)."""
    path = Path(directory) / "preview.json"
    with _lock:
        if not path.is_file():
            return
        info = json.loads(path.read_text())
        if info.get("type") != "motion" or "camera_path" in info:
            return
        try:
            info["camera_path"] = compute(directory, info)
        except (OSError, ValueError, KeyError, IndexError):
            return
        path.write_text(json.dumps(info))
