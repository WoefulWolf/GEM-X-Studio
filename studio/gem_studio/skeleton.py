"""Animated SOMA skeleton for a Motion result, cached next to its preview bundle.

``python -m gem_studio.skeleton JOB_DIR VARIANT SPACE`` writes
``JOB_DIR/<preview|refined>/skeleton_<space>.npz`` with:

    names [J]           public SOMA joint names (Hips ... RightToeEnd)
    parents [J]         parent index, -1 for Hips
    rest_pos [J, 3]     bind-pose joint positions (T-pose, Hips at origin, Y-up, facing +Z)
    rest_rot [J, 3, 3]  bind-pose joint world rotations
    pos [T, J, 3]       animated joint world positions (Y-up metres)
    rot [T, J, 3, 3]    animated joint world rotations
    rest_vertices [V,3] bind-pose mesh (MID topology, the preview's faces)
    weights [V, J]      SOMA linear-blend skinning weights (Root folded into Hips)
    fps []              frame rate

Camera space is converted from OpenCV to Y-up with (x, -y, -z), as other exports.
Only joints are evaluated per frame (no mesh), so this runs quickly on the CPU.
"""

import sys
from pathlib import Path

import numpy as np
import torch

CV_TO_YUP = np.diag([1.0, -1.0, -1.0])
VARIANT_FILES = {
    "raw": ("preview", "predictions.pt"),
    "refined": ("refined", "predictions_refined.pt"),
}


def skeleton_path(job_dir, variant, space):
    return Path(job_dir) / VARIANT_FILES[variant][0] / f"skeleton_{space}.npz"


def build_skeleton(result, space, *, soma_assets, device="cpu"):
    from gem.inference.timing import prediction_fps
    from gem.motion.geometry import MotionGeometry
    from gem.motion.postprocess import contact
    from gem.motion.postprocess.api import PostprocessBody

    params = result["pred_soma_params_global" if space == "world" else "pred_soma_params_incam"]
    geometry = MotionGeometry(soma_assets, device=device, chunk_size=128)
    body = PostprocessBody(geometry.body_model)
    names = list(body.soma.public_joint_names)[1:]
    parents = np.array([parent for parent in body.parents], dtype=np.int32)
    length = params["transl"].shape[1]
    rotations, positions = [], []
    with torch.inference_mode():
        for start in range(0, length, 128):
            chunk = {
                name: value[:, start : start + 128].to(device) for name, value in params.items()
            }
            transforms = (
                contact._soma_forward(body, chunk)["transforms"][0, :, 1:].double().cpu().numpy()
            )
            rotations.append(transforms[..., :3, :3])
            positions.append(transforms[..., :3, 3])
        rest = {name: value[:, :1].clone().to(device) for name, value in params.items()}
        for name in ("body_pose", "global_orient", "transl"):
            rest[name].zero_()
        rest_transforms = (
            contact._soma_forward(body, rest)["transforms"][0, 0, 1:].double().cpu().numpy()
        )
        _, rest_chunk = next(iter(geometry.iter_chunks(rest)))
        rest_vertices = rest_chunk["vertices"][0, 0].double().numpy()
        weights = body.soma.public_skinning_weights().double().cpu().numpy()
    weights = np.concatenate((weights[:, 1:2] + weights[:, :1], weights[:, 2:]), axis=1)
    rot = np.concatenate(rotations)
    pos = np.concatenate(positions)
    if space == "camera":
        rot = CV_TO_YUP @ rot
        pos = pos @ CV_TO_YUP.T
    return {
        "names": np.array(names),
        "parents": parents,
        "rest_pos": rest_transforms[:, :3, 3].astype(np.float32),
        "rest_rot": rest_transforms[:, :3, :3].astype(np.float32),
        "pos": pos.astype(np.float32),
        "rot": rot.astype(np.float32),
        "rest_vertices": rest_vertices.astype(np.float32),
        "weights": weights.astype(np.float32),
        "fps": np.float32(prediction_fps(result)),
    }


def ensure_skeleton(job_dir, variant, space, *, soma_assets, device="cpu"):
    """Build (or reuse) the cached skeleton; rebuilt when predictions are newer."""
    target = skeleton_path(job_dir, variant, space)
    source = Path(job_dir) / VARIANT_FILES[variant][1]
    if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    result = torch.load(source, map_location="cpu", weights_only=False)
    if "pred_soma_params_global" not in result:
        raise ValueError("Skeleton exports are available for Body Motion runs")
    skeleton = build_skeleton(result, space, soma_assets=soma_assets, device=device)
    partial = target.with_name(target.stem + ".part.npz")
    np.savez(partial, **skeleton)
    partial.replace(target)
    return target


def main(argv=None):
    from .paths import SOMA_DIR
    from .power import opt_out_of_power_throttling

    opt_out_of_power_throttling()
    job_dir, variant, space = (sys.argv[1:] if argv is None else argv)[:3]
    print(ensure_skeleton(job_dir, variant, space, soma_assets=SOMA_DIR))


if __name__ == "__main__":
    main()
