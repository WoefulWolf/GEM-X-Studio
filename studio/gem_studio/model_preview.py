"""Preview of a Body Motion result with an MMD model's limb proportions (Fit to model).

The dancer's own SOMA body is rebuilt once, at rest, with the model's arm and leg
lengths: SOMA's per-segment bone-length ratios (``scale_params``) are set from the
model's bones at the scale the VMD fit uses (matched on leg length). It is then
posed like a normal skinned mesh: the dancer's joint rotations, joint positions
from those rotations and the new lengths (SOMA's joints-only forward pass), and
SOMA's skin weights; the browser does the skinning. Each frame the body is moved
up or down so its lowest point is where the dancer's was, the rule the VMD fit
applies to the model itself. Hip and shoulder width, torso and head stay the
dancer's (SOMA does not expose them), and so do the shoulders: this previews
limb proportions and contacts.

``python -m gem_studio.model_preview JOB_DIR VARIANT MODEL_ID`` writes
``JOB_DIR/model_<variant>/``: ``preview.json`` (the source preview with a
``skinned`` entry), ``faces_body.npy`` and the binary skin files in ``SKIN_FILES``.
"""

import json
import sys
from pathlib import Path

import numpy as np

from . import mmd_model, rigging
from .skeleton import VARIANT_FILES, build_skeleton

# Little-endian binaries the viewer reads (see SkinnedStore in the web app).
SKIN_FILES = {
    "rest": "float32 [V, 3] bind-pose vertices (the preview's body topology)",
    "joints": "uint8 [V, 4] skinning joints",
    "weights": "float32 [V, 4] skinning weights (sum to 1)",
    "world": "float32 [T, J, 12] per-frame joint skin matrices, world (3x4 row-major)",
    "camera": "float32 [T, 12] per-frame world-to-camera transform (OpenCV camera, 3x4)",
}

# SOMA scale segment (parent, child) -> the MMD bones spanning the same segment. The
# shoulder (clavicle) is left out: MMD models have far narrower shoulders, and SOMA
# cannot narrow the chest to match, so the arms would start inside the torso.
_SEGMENTS = {}
for _jp, _en in (("左", "Left"), ("右", "Right")):
    _SEGMENTS[(f"{_en}Arm", f"{_en}ForeArm")] = (f"{_jp}腕", f"{_jp}ひじ")
    _SEGMENTS[(f"{_en}ForeArm", f"{_en}Hand")] = (f"{_jp}ひじ", f"{_jp}手首")
    for _finger, _digits, _joints in rigging._FINGERS:
        for (_a, _b), (_da, _db) in zip(zip(_joints, _joints[1:]), zip(_digits, _digits[1:])):
            _SEGMENTS[(f"{_en}Hand{_a}", f"{_en}Hand{_b}")] = (
                f"{_jp}{_finger}{_da}",
                f"{_jp}{_finger}{_db}",
            )
# Limb ratios are kept within this range (a mis-mapped bone must not explode the body).
RATIO_LIMITS = (0.5, 2.0)


def preview_dir(job_dir, variant):
    return Path(job_dir) / f"model_{variant}"


def segment_ratios(skeleton, profile, segments, scale):
    """Bone-length ratio per SOMA scale segment that gives the body the model's limbs.

    Lengths are compared in metres (model units / ``scale``). SOMA's shin is fixed
    through GEM's body adapter, so the whole leg length is matched on the thigh.
    """
    names = [str(name) for name in skeleton["names"]]
    rest = skeleton["rest_pos"].astype(np.float64)
    model_names = [str(name) for name in profile["names"]]
    model_rest = profile["rest"].astype(np.float64)

    def capture(a, b):
        return float(np.linalg.norm(rest[names.index(a)] - rest[names.index(b)]))

    def model(*bones):
        index = [model_names.index(bone) for bone in bones]
        steps = np.diff(model_rest[index], axis=0)
        return float(np.linalg.norm(steps, axis=1).sum()) / scale

    ratios = np.ones(len(segments))
    for slot, (parent, child) in enumerate(segments):
        if parent not in names or child not in names:
            continue
        if child.endswith("Shin"):  # thigh: absorbs the whole leg (the shin cannot scale)
            jp = "左" if parent.startswith("Left") else "右"
            bones = (f"{jp}足", f"{jp}ひざ", f"{jp}足首")
            if all(bone in model_names for bone in bones):
                shin = capture(child, child.replace("Shin", "Foot"))
                ratios[slot] = (model(*bones) - shin) / max(capture(parent, child), 1e-6)
            continue
        bones = _SEGMENTS.get((parent, child))
        if bones and all(bone in model_names for bone in bones):
            ratios[slot] = model(*bones) / max(capture(parent, child), 1e-6)
    return np.clip(ratios, *RATIO_LIMITS)


def _lowest(directory, stream="world_body"):
    """Per-frame lowest vertex height of a preview stream (what the viewer shows)."""
    verts = np.load(Path(directory) / f"verts_{stream}.npy", mmap_mode="r")
    offsets = np.load(Path(directory) / f"offsets_{stream}.npy")
    lowest = np.empty(len(offsets))
    for start in range(0, len(offsets), 256):
        block = np.asarray(verts[start : start + 256, :, 1], dtype=np.float32)
        lowest[start : start + 256] = block.min(axis=1) + offsets[start : start + 256, 1]
    return lowest


def _rotations(orient):
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(orient[0].double().cpu().numpy()).as_matrix()


def build(job_dir, variant, model_id, *, soma_assets, device="cpu"):
    """Write the model-proportion preview of ``variant`` ("raw" or "refined")."""
    import torch

    from .refine import _joints
    from .skeleton import skeleton_path

    job_dir = Path(job_dir)
    source_dir = job_dir / VARIANT_FILES[variant][0]
    source_info = json.loads((source_dir / "preview.json").read_text())
    profile = mmd_model.load_profile(model_id)
    model = mmd_model.model_info(model_id)
    saved = torch.load(job_dir / VARIANT_FILES[variant][1], map_location="cpu", weights_only=False)
    # The dancer's own skeleton (cached for the skeleton exports) gives the current lengths.
    cached = skeleton_path(job_dir, variant, "world")
    dancer = (
        rigging.load_skeleton(cached)
        if cached.is_file()
        else build_skeleton(saved, "world", soma_assets=soma_assets, device=device)
    )
    scale = rigging.model_scale(dancer, profile)

    from gem.motion.geometry import MotionGeometry

    runtime = MotionGeometry(soma_assets, device=device).body_model.runtime
    count = saved["pred_soma_params_global"]["scale_params"].shape[-1] - 1  # [0]: global scale
    segments = list(runtime.scale_param_segments)[:count]
    ratios = segment_ratios(dancer, profile, segments, scale)
    factor = torch.ones(count + 1)
    factor[1:] = torch.as_tensor(ratios, dtype=factor.dtype)

    def proportioned(params):
        out = {name: value.clone() for name, value in params.items()}
        out["scale_params"] = out["scale_params"] * factor.to(out["scale_params"].dtype)
        return out

    world = proportioned(saved["pred_soma_params_global"])
    camera = proportioned(saved["pred_soma_params_incam"])
    skeleton = build_skeleton(
        {**saved, "pred_soma_params_global": world}, "world", soma_assets=soma_assets, device=device
    )
    # Same rule as the VMD fit: the lowest point goes where the dancer's was.
    shift = _lowest(source_dir) - rigging._capture_lowest(skeleton)
    position = skeleton["pos"].astype(np.float64)
    position[..., 1] += shift[:, None]
    turn = skeleton["rot"].astype(np.float64) @ np.transpose(skeleton["rest_rot"], (0, 2, 1))
    place = position - np.einsum("tjab,jb->tja", turn, skeleton["rest_pos"].astype(np.float64))
    world_matrices = np.concatenate((turn, place[..., None]), axis=-1)  # [T, J, 3, 4]

    # World to camera per frame, from the root: the same body in both spaces.
    to_camera = _rotations(camera["global_orient"]) @ np.transpose(
        _rotations(world["global_orient"]), (0, 2, 1)
    )
    camera_hips = _joints(camera, soma_assets=soma_assets, device=device)[:, 0]
    world_hips = skeleton["pos"][:, 0].astype(np.float64)  # before the shift
    offset = camera_hips - np.einsum("tab,tb->ta", to_camera, world_hips)
    camera_matrices = np.concatenate((to_camera, offset[..., None]), axis=-1)  # [T, 3, 4]

    weights = skeleton["weights"]
    top = np.argsort(-weights, axis=1)[:, :4]
    top_weights = np.take_along_axis(weights, top, axis=1)
    top_weights /= np.maximum(top_weights.sum(axis=1, keepdims=True), 1e-9)

    target = preview_dir(job_dir, variant)
    target.mkdir(parents=True, exist_ok=True)
    blobs = {
        "rest": skeleton["rest_vertices"].astype("<f4"),
        "joints": top.astype(np.uint8),
        "weights": top_weights.astype("<f4"),
        "world": world_matrices.astype("<f4"),
        "camera": camera_matrices.astype("<f4"),
    }
    for name, data in blobs.items():
        (target / f"skin_{name}.bin").write_bytes(np.ascontiguousarray(data).tobytes())
    np.save(target / "faces_body.npy", np.load(source_dir / "faces_body.npy"))
    vertex_count, joint_count = len(skeleton["rest_vertices"]), len(skeleton["names"])
    trajectory = position[:, 0]
    info = {
        **source_info,
        "streams": {
            stream: {"slots": len(position), "vertex_count": vertex_count}
            for stream in ("camera_body", "world_body")
        },
        "skinned": {"joints": joint_count, "frames": len(position), "vertices": vertex_count},
        "trajectory": trajectory.round(4).tolist(),
        "variant": "model",
        "model": {
            "id": model_id,
            "name": model["name"],
            "scale": round(scale, 3),
            "ratios": {
                f"{parent}>{child}": round(float(ratio), 3)
                for (parent, child), ratio in zip(segments, ratios)
                if abs(ratio - 1.0) > 1e-3
            },
        },
        "model_source": {
            "variant": variant,
            "mtime": (source_dir / "preview.json").stat().st_mtime,
        },
    }
    (target / "preview.json").write_text(json.dumps(info))
    return info


def is_current(job_dir, variant, model_id):
    """Whether the stored model preview matches this model and the current source."""
    path = preview_dir(job_dir, variant) / "preview.json"
    source = Path(job_dir) / VARIANT_FILES[variant][0] / "preview.json"
    if not path.is_file() or not source.is_file():
        return False
    try:
        info = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (info.get("model") or {}).get("id") == model_id and (
        info.get("model_source") or {}
    ).get("mtime") == source.stat().st_mtime


def main(argv=None):
    from .paths import SOMA_DIR
    from .power import opt_out_of_power_throttling

    opt_out_of_power_throttling()
    job_dir, variant, model_id = (sys.argv[1:] if argv is None else argv)[:3]
    build(job_dir, variant, model_id, soma_assets=SOMA_DIR)
    print(preview_dir(job_dir, variant))


if __name__ == "__main__":
    main()
