"""Offline temporal refinement of framewise Pose videos.

Both tasks use GEM's motion-adaptive parameter smoother
(``gem.pose.smoothing``): rotations, camera translation and shape are filtered
per track, then geometry is decoded again through SOMA. Vertices are never
averaged. Hands reuse ``gem.pose.postprocess.smooth_hand_predictions``; bodies
follow the same contract here with the Body geometry and parameter layout.
Raw predictions are kept; filtered geometry is stored in each row's ``smoothed``.
"""

from collections import defaultdict
from dataclasses import asdict, replace

import numpy as np
import torch

from gem.pose.postprocess import _numpy, _project, _restore
from gem.pose.smoothing import (
    AdaptiveSmoothingConfig,
    smooth_rotations,
    smooth_shape,
    smooth_translation,
)

BODY_PARAMETERS = ("global_orient", "body_pose", "identity_coeffs", "scale_params")
BODY_SHAPE = ("identity_coeffs", "scale_params")
GEOMETRY_FIELDS = (
    "camera_translation",
    "joints_camera",
    "vertices_camera",
    "keypoints_2d",
    "keypoints_crop",
    "projection_valid",
)
# How strongly fast motion is protected from smoothing (reference-speed scale).
DETAIL_SCALE = {"smooth": 3.0, "balanced": 1.0, "sharp": 0.35}
FK_BATCH = 32
# Motion static-contact channels: left ankle/toe, right ankle/toe (then two hands).
FOOT_CONTACTS = 4


def smoothing_config(options):
    """Map the Studio controls onto GEM's AdaptiveSmoothingConfig.

    ``strength`` is the widest Gaussian in 30 fps frames (GEM default: 2).
    ``detail`` scales the speeds above which smoothing backs off.
    ``lock_shape`` holds identity/scale nearly constant across each track.
    """
    base = AdaptiveSmoothingConfig()
    scale = DETAIL_SCALE.get(options.get("detail", "balanced"), 1.0)
    return replace(
        base,
        max_sigma_seconds=max(float(options.get("strength", 2.0)), 0.25) / 30.0,
        translation_speed_mps=base.translation_speed_mps * scale,
        angular_speed_radps=base.angular_speed_radps * scale,
        shape_sigma_seconds=600.0 if options.get("lock_shape", True) else base.shape_sigma_seconds,
    )


def _body_segments(entries, fps, config):
    segment = []
    for entry in entries:
        frame_id, row, _ = entry
        values = [row["camera_translation"], *(row["parameters"][n] for n in BODY_PARAMETERS)]
        valid = all(np.isfinite(_numpy(value)).all() for value in values)
        if segment and (
            not valid
            or frame_id != segment[-1][0] + 1
            or (frame_id - segment[-1][0]) / fps > config.max_gap_seconds
        ):
            yield segment
            segment = []
        if valid:
            segment.append(entry)
    if segment:
        yield segment


def smooth_body_predictions(result, *, geometry, config, progress=None):
    if result["task"] != "body" or result["mode"] != "framewise_video":
        raise ValueError("Body smoothing requires framewise body video predictions")
    fps = float(result["source_fps"])
    frames = [
        dict(frame, instances=[dict(row) for row in frame["instances"]])
        for frame in result["frames"]
    ]
    groups = defaultdict(list)
    for frame in frames:
        for row in frame["instances"]:
            # Rows outside any valid segment keep their raw geometry.
            row["smoothed"] = {
                **{name: row[name] for name in GEOMETRY_FIELDS},
                "parameters": dict(row["parameters"]),
            }
            groups[row["track_id"]].append(
                (frame["frame_id"], row, row.get("intrinsics", frame["intrinsics"]))
            )
    minimum_depth = (result.get("release") or {}).get("inference", {}).get("minimum_depth", 1e-4)
    device = geometry._device_anchor.device
    total = sum(len(entries) for entries in groups.values())
    done = 0
    for entries in groups.values():
        entries.sort(key=lambda item: item[0])
        for segment in _body_segments(entries, fps, config):
            times = np.array([entry[0] / fps for entry in segment], dtype=np.float64)
            rows = [entry[1] for entry in segment]
            translations = smooth_translation(
                np.stack([_numpy(row["camera_translation"]) for row in rows]), times, config
            )
            parameters = {
                name: np.stack([_numpy(row["parameters"][name]) for row in rows])
                for name in BODY_PARAMETERS
            }
            rotations = smooth_rotations(
                np.concatenate((parameters["global_orient"], parameters["body_pose"]), axis=1),
                times,
                config,
            )
            parameters["global_orient"], parameters["body_pose"] = (
                rotations[:, :1],
                rotations[:, 1:],
            )
            for name in BODY_SHAPE:
                parameters[name] = smooth_shape(parameters[name], times, config.shape_sigma_seconds)
            for index, row in enumerate(rows):
                row["smoothed"]["parameters"] = {
                    name: _restore(parameters[name][index], row["parameters"][name])
                    for name in BODY_PARAMETERS
                }
                row["smoothed"]["camera_translation"] = _restore(
                    translations[index], row["camera_translation"]
                )
            for start in range(0, len(rows), FK_BATCH):
                batch = rows[start : start + FK_BATCH]
                tensors = {
                    name: torch.as_tensor(
                        np.stack([_numpy(row["smoothed"]["parameters"][name]) for row in batch]),
                        dtype=torch.float32,
                        device=device,
                    )
                    for name in BODY_PARAMETERS
                }
                with torch.inference_mode():
                    decoded = geometry(tensors)
                joints = _numpy(decoded["joints"])
                vertices = _numpy(decoded["vertices"])
                for offset, row in enumerate(batch):
                    intrinsics = segment[start + offset][2]
                    smoothed = row["smoothed"]
                    translation = _numpy(smoothed["camera_translation"]).astype(np.float64)
                    points, pixels, crop, valid = _project(
                        joints[offset],
                        translation,
                        _numpy(intrinsics),
                        _numpy(row["bbox"]),
                        minimum_depth,
                    )
                    smoothed.update(
                        joints_camera=_restore(points, row["joints_camera"]),
                        vertices_camera=_restore(
                            vertices[offset].astype(np.float64) + translation[None],
                            row["vertices_camera"],
                        ),
                        keypoints_2d=_restore(pixels, row["keypoints_2d"]),
                        keypoints_crop=_restore(crop, row["keypoints_crop"]),
                        projection_valid=_restore(valid, row["projection_valid"]),
                    )
                done += len(batch)
                if progress is not None:
                    progress(done, total)
    return {**result, "frames": frames, "body_smoothing": asdict(config)}


def _mean_rotation(matrices):
    """Chordal L2 mean of rotation matrices (nearest rotation to their sum)."""
    u, _, vt = np.linalg.svd(np.sum(matrices, axis=0))
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


MOTION_DEFAULTS = {
    "world": "anchor",  # "anchor": fixed-camera world from camera space; "gem": GEM's integrated world
    "strength": 0.0,  # temporal smoothing width in 30 fps frames; 0 = off (Motion is already temporal)
    "detail": "balanced",
    "lock_shape": True,
    "plant_feet": True,  # hip height + leg IK so planted feet rest flat on the floor
    "contact_threshold": 0.5,  # GEM foot-contact probability that counts as planted
    "floor_offset_cm": 0.0,  # final vertical shift (positive raises the body)
    "level": "auto",  # "auto": level the world on the planted-feet floor plane (balance fallback); "off"
    "tilt_deg": 0.0,  # manual extra tilt toward (+) / away from (-) the camera
    "roll_deg": 0.0,  # manual extra sideways tilt
    "depth_smoothing": 4.0,  # Gaussian width (30 fps frames) on distance-to-camera only; 0 = off
    "lock_feet": True,  # cancel horizontal sliding of planted feet by moving the root
    # GEM world: let the floor follow the planted feet (smoothed over ~1 s), removing the
    # slow vertical drift GEM's integrated trajectory can build up with a moving camera.
    "follow_floor": False,
}

# Smoothing width (30 fps frames) of the followed floor: slow drift goes, jumps stay.
FOLLOW_FLOOR_FRAMES = 30.0

# 77-joint indices: contact joints (ankle, ball) and the non-finger joints used as a COM proxy.
CONTACT_JOINTS = (69, 70, 74, 75)
BODY_JOINTS = tuple(range(0, 15)) + tuple(range(39, 43)) + tuple(range(67, 77))
MAX_AUTO_LEVEL_DEG = 20.0


def _rotations(params):
    """Axis-angle Motion params -> [T, 77, 3, 3] (global orient + 76 body joints)."""
    from scipy.spatial.transform import Rotation

    length = params["transl"].shape[1]
    orient = Rotation.from_rotvec(params["global_orient"][0].double().cpu().numpy()).as_matrix()
    body = Rotation.from_rotvec(params["body_pose"][0].double().cpu().numpy().reshape(-1, 3))
    return np.concatenate((orient[:, None], body.as_matrix().reshape(length, -1, 3, 3)), axis=1)


def smooth_motion_params(params, times, config):
    """GEM's adaptive smoother on one Motion parameter set (axis-angle layout)."""
    from scipy.spatial.transform import Rotation

    length = params["transl"].shape[1]
    out = {name: value.clone() for name, value in params.items()}
    rotations = smooth_rotations(_rotations(params), times, config)
    dtype = params["global_orient"].dtype
    out["global_orient"][0] = torch.as_tensor(
        Rotation.from_matrix(rotations[:, 0]).as_rotvec(), dtype=dtype
    )
    out["body_pose"][0] = torch.as_tensor(
        Rotation.from_matrix(rotations[:, 1:].reshape(-1, 3, 3)).as_rotvec().reshape(length, -1),
        dtype=dtype,
    )
    out["transl"][0] = torch.as_tensor(
        smooth_translation(params["transl"][0].double().cpu().numpy(), times, config), dtype=dtype
    )
    for name in ("identity_coeffs", "scale_params"):
        if params[name].dim() == 3:
            out[name][0] = torch.as_tensor(
                smooth_shape(
                    params[name][0].double().cpu().numpy(), times, config.shape_sigma_seconds
                ),
                dtype=params[name].dtype,
            )
    return out


def stabilize_depth(camera, sigma_frames):
    """Smooth only the distance to the camera, keeping each frame's image position.

    Monocular depth is the weakest-observed quantity: the root jitters along the
    line of sight while its image position is pinned by the pixels. The camera-
    space translation is moved along its own viewing ray to a temporally smoothed
    depth, so the projection (and the overlay) is unchanged.
    """
    from scipy.ndimage import gaussian_filter1d

    if sigma_frames <= 0:
        return camera, None
    transl = camera["transl"][0].double().cpu().numpy()
    depth = transl[:, 2]
    smoothed = gaussian_filter1d(depth, float(sigma_frames), mode="nearest")
    scale = smoothed / np.where(np.abs(depth) > 1e-6, depth, 1e-6)
    out = {name: value.clone() for name, value in camera.items()}
    out["transl"][0] = torch.as_tensor(transl * scale[:, None], dtype=camera["transl"].dtype)
    residual = depth - smoothed
    return out, {
        "sigma_frames": float(sigma_frames),
        "depth_jitter_removed_cm": round(float(np.std(residual)) * 100, 2),
    }


def lock_feet(world, conf, *, threshold, soma_assets, device, window_frames=30, ease_frames=2.0):
    """Stop planted feet from sliding horizontally by shifting the root.

    Each frame, the confidence-weighted horizontal motion of contact joints that
    are planted in both this and the previous frame is subtracted from the root.
    The accumulated offset is eased over ``ease_frames`` (no jolts when a contact
    starts or ends) and high-pass filtered (``window_frames``) so it only
    removes short-term skating and never pulls the motion away from its
    (anchored) long-term position. Limbs are unchanged.
    """
    from scipy.ndimage import gaussian_filter1d

    joints = _joints(world, soma_assets=soma_assets, device=device)
    planted = conf > threshold
    held = (planted[1:] & planted[:-1]) * conf[1:]
    step = np.diff(joints[:, list(CONTACT_JOINTS)][..., [0, 2]], axis=0)  # [T-1, 4, (x, z)]
    weight = held.sum(axis=1)
    slide = np.where(
        weight[:, None] > 0,
        (step * held[..., None]).sum(axis=1) / np.maximum(weight, 1e-9)[:, None],
        0.0,
    )
    offset = np.concatenate([np.zeros((1, 2)), -np.cumsum(slide, axis=0)])
    # Ease corrections in/out at contact changes, then keep only the short-term part.
    offset = gaussian_filter1d(offset, ease_frames, axis=0, mode="nearest")
    offset -= gaussian_filter1d(offset, window_frames, axis=0, mode="nearest")
    out = {name: value.clone() for name, value in world.items()}
    out["transl"][0, :, 0] += torch.as_tensor(offset[:, 0], dtype=world["transl"].dtype)
    out["transl"][0, :, 2] += torch.as_tensor(offset[:, 1], dtype=world["transl"].dtype)
    before = np.linalg.norm(step, axis=-1)[held > 0].mean() if (held > 0).any() else 0.0
    return out, {
        "planted_slide_mm_per_frame_before": round(float(before) * 1000, 2),
        "max_root_shift_cm": round(float(np.abs(offset).max()) * 100, 2),
    }


def _rotate_params(source, template, rotation):
    """``source`` motion under one extra world rotation (translation finished by ``_place``)."""
    from scipy.spatial.transform import Rotation

    out = {name: value.clone() for name, value in template.items()}
    dtype = template["global_orient"].dtype
    out["global_orient"][0] = torch.as_tensor(
        Rotation.from_matrix(rotation @ _rotations(source)[:, 0]).as_rotvec(), dtype=dtype
    )
    out["transl"][0] = torch.as_tensor(
        source["transl"][0].double().cpu().numpy() @ rotation.T, dtype=dtype
    )
    return out


def _joints(params, *, soma_assets, device):
    """[T, 77, 3] joint positions in the parameters' own coordinates (joints-only FK)."""
    from gem.motion.geometry import MotionGeometry
    from gem.motion.postprocess import contact
    from gem.motion.postprocess.api import PostprocessBody

    body = PostprocessBody(MotionGeometry(soma_assets, device=device).body_model)
    length = params["transl"].shape[1]
    chunks = []
    with torch.inference_mode():
        for start in range(0, length, 128):
            chunk = {
                name: value[:, start : start + 128].to(device) for name, value in params.items()
            }
            chunks.append(contact._soma_forward(body, chunk)["joints"][0].double().cpu().numpy())
    return np.concatenate(chunks)


def _camera_axes(rotation):
    """Horizontal viewing direction and sideways axis of the camera, in the world."""
    view = rotation @ np.array([0.0, 0.0, 1.0])  # OpenCV +Z: where the camera looks
    view_h = np.array([view[0], 0.0, view[2]])
    view_h /= max(np.linalg.norm(view_h), 1e-9)
    return view_h, np.cross([0.0, 1.0, 0.0], view_h)


def level_rotation(joints, conf, *, threshold, view_h, side_h):
    """Rotation (pitch/roll only) that makes the measured 'up' vertical.

    Primary: the plane through planted contact joints (ankle and ball fitted
    with their own height offsets) is the floor, so its normal is gravity.
    Fallback when the feet cover too little ground to fit a plane: balance, i.e.
    on average the body's centre of mass sits above its planted feet.
    """
    from scipy.spatial.transform import Rotation

    planted = conf > threshold
    points = np.concatenate([joints[planted[:, k], j] for k, j in enumerate(CONTACT_JOINTS)])
    weights = np.concatenate([conf[planted[:, k], k] for k in range(len(CONTACT_JOINTS))])
    # One height offset per joint *type* (ankle, ball) shared by both feet: both
    # feet stand on the same floor, so a left/right height difference is roll.
    labels = np.concatenate(
        [np.full(planted[:, k].sum(), k % 2) for k in range(len(CONTACT_JOINTS))]
    )
    up, method, residual = None, None, None
    if len(points) >= 30 and min(np.std(points[:, 0]), np.std(points[:, 2])) >= 0.06:
        design = np.column_stack(
            [points[:, 0], points[:, 2]] + [(labels == k).astype(float) for k in range(2)]
        )
        root = np.sqrt(weights)[:, None]
        coef, *_ = np.linalg.lstsq(design * root, points[:, 1] * root[:, 0], rcond=None)
        residual = float(np.sqrt(np.mean((design @ coef - points[:, 1]) ** 2)))
        up, method = np.array([-coef[0], 1.0, -coef[1]]), "floor_plane"
    any_planted = planted.any(axis=1)
    if up is None and any_planted.sum() >= 15:
        com = joints[:, BODY_JOINTS].mean(axis=1)
        support = np.array(
            [
                joints[t, [j for k, j in enumerate(CONTACT_JOINTS) if planted[t, k]]].mean(axis=0)
                for t in np.flatnonzero(any_planted)
            ]
        )
        up, method = (com[any_planted] - support).mean(axis=0), "balance"
    if up is None:
        return np.eye(3), {"method": "insufficient_contact"}
    up /= np.linalg.norm(up)
    horizontal = np.array([up[0], 0.0, up[2]])
    info = {
        "method": method,
        "measured_tilt_deg": round(float(np.degrees(np.arccos(np.clip(up[1], -1, 1)))), 2),
        "toward_camera_deg": round(
            float(np.degrees(np.arcsin(np.clip(horizontal @ -view_h, -1, 1)))), 2
        ),
        "sideways_deg": round(float(np.degrees(np.arcsin(np.clip(horizontal @ side_h, -1, 1)))), 2),
    }
    if residual is not None:
        info["floor_fit_residual_cm"] = round(residual * 100, 2)
    axis = np.cross(up, [0.0, 1.0, 0.0])
    angle = np.arcsin(np.clip(np.linalg.norm(axis), 0, 1))
    if angle < 1e-6:
        return np.eye(3), info
    angle = min(angle, np.radians(MAX_AUTO_LEVEL_DEG))
    info["applied_deg"] = round(float(np.degrees(angle)), 2)
    return Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix(), info


def display_camera_frame(camera_to_world):
    """Camera-to-display rotation for the 3D view: gravity up, heading removed.

    ``camera_to_world`` maps OpenCV camera coordinates to the Y-up world. The
    result keeps its pitch and roll but turns it so the camera looks along -Z
    (into the screen); for a level camera it equals the plain (x, -y, -z) flip.
    """
    from scipy.spatial.transform import Rotation

    rotation = np.asarray(camera_to_world, dtype=np.float64)
    forward = rotation @ np.array([0.0, 0.0, 1.0])
    heading = np.arctan2(forward[0], forward[2])
    return Rotation.from_euler("y", np.pi - heading).as_matrix() @ rotation


def gem_camera_to_world(result):
    """GEM's own camera-to-world rotation estimate (chordal mean over the clip)."""
    camera = _rotations(result["pred_soma_params_incam"])[:, 0]
    world = _rotations(result["pred_soma_params_global"])[:, 0]
    return _mean_rotation(world @ camera.transpose(0, 2, 1))


def anchor_world(camera, world):
    """Fixed-camera world: camera-space motion under one camera-to-world rotation.

    GEM's world trajectory integrates predicted root velocity, which can drift
    (and contact/IK can stack heights) on in-place motion such as dance. With a
    fixed camera the exact world motion is the camera-space result under a single
    rotation: the chordal mean of ``R_world(t) R_camera(t)^T``, i.e. GEM's own
    gravity-aligned estimate. Translation is completed later by ``_place``.
    """
    from scipy.spatial.transform import Rotation

    camera_orient = _rotations(camera)[:, 0]
    world_orient = _rotations(world)[:, 0]
    relative = world_orient @ camera_orient.transpose(0, 2, 1)
    rotation = _mean_rotation(relative)
    spread = np.degrees(Rotation.from_matrix(relative @ rotation.T).magnitude())
    anchored = {name: value.clone() for name, value in world.items()}
    dtype = world["global_orient"].dtype
    anchored["global_orient"][0] = torch.as_tensor(
        Rotation.from_matrix(rotation @ camera_orient).as_rotvec(), dtype=dtype
    )
    anchored["transl"][0] = torch.as_tensor(
        camera["transl"][0].double().cpu().numpy() @ rotation.T, dtype=dtype
    )
    return anchored, rotation, float(np.percentile(spread, 95))


def _place(camera, world, rotation, planted, *, soma_assets, device, progress=None, follow=False):
    """Exact translation for anchored worlds and a contact-based floor at y=0.

    Anchored: decoded world vertices must equal ``rotation @ camera vertices``;
    the per-frame residual is absorbed into translation (SOMA's root-offset
    convention). Floor: median lowest point over frames with a planted foot, or with
    ``follow``, that lowest point per planted frame, interpolated and smoothed over
    ``FOLLOW_FLOOR_FRAMES`` (removes slow vertical drift, keeps jumps).
    """
    from gem.motion.geometry import MotionGeometry

    geometry = MotionGeometry(soma_assets, device=device, chunk_size=64)
    length = world["transl"].shape[1]
    correction = np.zeros((length, 3))
    lowest = np.zeros(length)
    first_centre = None
    pairs = zip(geometry.iter_chunks(camera), geometry.iter_chunks(world))
    for (start, cam_chunk), (_, world_chunk) in pairs:
        decoded = world_chunk["vertices"][0].double().numpy()
        if rotation is not None:
            target = cam_chunk["vertices"][0].double().numpy() @ rotation.T
            correction[start : start + len(target)] = (target - decoded).mean(axis=1)
        else:
            target = decoded
        stop = start + len(target)
        lowest[start:stop] = target[..., 1].min(axis=1)
        if first_centre is None:
            first_centre = target[0].mean(axis=0)
        if progress is not None:
            progress(stop, length)
    if follow and planted.sum() >= 10:
        floor = _fill_and_smooth(lowest, planted, sigma=FOLLOW_FLOOR_FRAMES)
        source = "foot_contact_follow"
    elif planted.sum() >= 10:
        floor, source = float(np.median(lowest[planted])), "foot_contact_median"
    else:
        floor, source = float(np.percentile(lowest, 10)), "lowest_point_p10"
    shift = np.zeros((length, 3))
    shift[:, 1] = -np.asarray(floor)
    if rotation is not None:
        shift[:, 0], shift[:, 2] = -first_centre[0], -first_centre[2]
    placed = {name: value.clone() for name, value in world.items()}
    placed["transl"][0] = torch.as_tensor(
        world["transl"][0].double().cpu().numpy() + correction + shift, dtype=world["transl"].dtype
    )
    return placed, source


def _fill_and_smooth(values, known, sigma=2.0):
    """Interpolate a per-frame signal across unknown frames, then Gaussian-smooth it."""
    from scipy.ndimage import gaussian_filter1d

    frames = np.arange(len(values))
    if not known.any():
        return np.zeros_like(values)
    filled = np.interp(frames, frames[known], values[known])
    return gaussian_filter1d(filled, sigma, mode="nearest")


def _flat_foot_heights(params, feet_ids, *, soma_assets, device):
    """Height of each contact joint above the sole when standing (rest pose, this body)."""
    from gem.motion.geometry import MotionGeometry

    geometry = MotionGeometry(soma_assets, device=device)
    rest = {name: value[:, :1].clone().to(device) for name, value in params.items()}
    for name in ("body_pose", "global_orient", "transl"):
        rest[name].zero_()
    _, chunk = next(iter(geometry.iter_chunks(rest)))
    sole = chunk["vertices"][0, 0, :, 1].min().item()
    joints = chunk["joints"][0, 0].double().numpy()
    return np.array([joints[j, 1] - sole for j in feet_ids])


def plant_feet(world, camera, logits, *, soma_assets, device, threshold=0.5):
    """Flat, planted soles on y=0 using GEM's contact rig and CCD leg IK.

    1. Hips: a smoothed vertical shift so planted contact joints reach their
       flat-foot height: the ankle / ball height above the sole in SOMA's rest
       pose for this body (anatomical, identical for both feet).
    2. Legs: GEM's two-joint CCD chains toward targets that keep GEM's
       anti-slide horizontal hold, put planted ankle *and* ball-of-foot joints at
       their flat-foot heights (a flat sole), and never let a foot below them.
    The shared body pose is written to both camera and world parameters.
    """
    from soma.geometry.rig_utils import remove_joint_orient_local

    from gem.common.geometry.rotations import matrix_to_axis_angle
    from gem.motion.postprocess import contact
    from gem.motion.postprocess.api import PostprocessBody, SomaContactPostprocessor, ieee_geometry

    device = torch.device(device)
    body = PostprocessBody(SomaContactPostprocessor(soma_assets, device=device).body_model)
    feet_ids = contact.SOMA_CONTACT_JOINT_IDS[:FOOT_CONTACTS]
    conf = torch.as_tensor(logits)[0, :, :FOOT_CONTACTS].float().sigmoid().numpy()
    planted = conf > threshold
    params = {name: value.detach().to(device) for name, value in world.items()}
    with torch.inference_mode(), ieee_geometry(), torch.autocast(device.type, enabled=False):
        joints = contact._soma_forward(body, params)["joints"][0].cpu().numpy()
        heights = joints[:, feet_ids, 1]
        rest = _flat_foot_heights(world, feet_ids, soma_assets=soma_assets, device=device)
        error = rest[None] - heights  # positive: the joint sits too low and must rise
        known = planted.any(axis=1)
        before = float(np.abs(error[planted]).mean()) if planted.any() else 0.0
        if known.any():
            # Mean vertical correction over each frame's planted joints.
            per_frame = np.zeros(len(heights))
            per_frame[known] = (error * planted).sum(axis=1)[known] / planted.sum(axis=1)[known]
            shift = _fill_and_smooth(per_frame, known)
            params["transl"][0, :, 1] += torch.as_tensor(
                shift, dtype=params["transl"].dtype, device=device
            )
        forward = contact._soma_forward(body, params)
        joints_t = forward["joints"]
        world_mat = forward["transforms"][..., 1:, :, :]
        targets = joints_t.clone()
        weight = torch.as_tensor(conf, device=device)
        rest_t = torch.as_tensor(rest, dtype=targets.dtype, device=device)
        feet_index = list(feet_ids)
        for frame in range(1, targets.shape[1]):
            previous = targets[:, frame - 1, feet_index]
            current = joints_t[:, frame, feet_index]
            hold = weight[frame - 1, :, None].to(targets.dtype)  # GEM's anti-slide blend
            blended = previous * hold + current * (1 - hold)
            targets[:, frame, feet_index, 0] = blended[..., 0]
            targets[:, frame, feet_index, 2] = blended[..., 2]
        y = targets[0, :, feet_index, 1]
        planted_t = torch.as_tensor(planted, device=device)
        y = torch.where(planted_t, rest_t[None].expand_as(y), torch.maximum(y, rest_t[None]))
        targets[0, :, feet_index, 1] = y
        parents = list(body.parents)
        local = contact._world_to_local(world_mat, parents)
        local = contact._solve_chain(
            local, parents, targets[..., (69, 70), :], (3, 4), contact.SOMA_LEFT_LEG_CHAIN
        )
        local = contact._solve_chain(
            local, parents, targets[..., (74, 75), :], (3, 4), contact.SOMA_RIGHT_LEG_CHAIN
        )
        procedural = body.soma.procedural_transforms
        absolute = contact.matrix.get_rotation(local).reshape(-1, 77, 3, 3)
        relative = remove_joint_orient_local(
            absolute,
            procedural.source_joint_orient[1:].to(absolute),
            procedural.source_joint_orient_parent_t[1:].to(absolute),
        ).reshape(*local.shape[:2], 77, 3, 3)
        params["body_pose"] = (
            matrix_to_axis_angle(relative)[..., 1:, :].flatten(-2).to(params["body_pose"])
        )
        after_heights = (
            contact._soma_forward(body, params)["joints"][0].cpu().numpy()[:, feet_ids, 1]
        )
        after = (
            float(
                np.mean(
                    np.abs(after_heights[planted] - np.broadcast_to(rest, planted.shape)[planted])
                )
            )
            if planted.any()
            else 0.0
        )
    planted_world = {name: value.cpu() for name, value in params.items()}
    planted_camera = {**camera, "body_pose": planted_world["body_pose"].clone()}
    stats = {
        "planted_height_error_cm_before": round(before * 100, 2),
        "planted_height_error_cm_after": round(after * 100, 2),
        "planted_frame_fraction": round(float(planted.any(axis=1).mean()), 3),
    }
    return planted_world, planted_camera, stats


def refine_motion(result, options, *, soma_assets, device, progress=None, stage=None):
    """Smoothing, world anchoring, floor placement, foot planting and floor offset."""
    options = {**MOTION_DEFAULTS, **{k: v for k, v in options.items() if v is not None}}
    say = stage or (lambda *_: None)
    camera = result["pred_soma_params_incam"]
    world = result["pred_soma_params_global"]
    length = world["transl"].shape[1]
    logits = result.get("static_conf_logits")
    times = np.arange(length, dtype=np.float64) / 30.0
    report = {"options": options}
    if float(options["strength"]) > 0:
        say("Smoothing motion", f"strength {options['strength']}")
        config = smoothing_config(options)
        camera = smooth_motion_params(camera, times, config)
        world = smooth_motion_params(world, times, config)
    if float(options["depth_smoothing"]) > 0:
        say("Stabilizing depth", f"{options['depth_smoothing']} frames")
        stabilized, report["depth"] = stabilize_depth(camera, float(options["depth_smoothing"]))
        if options["world"] != "anchor":
            # GEM-world mode: carry the same camera-space change into the world, through
            # each frame's camera rotation (the camera may turn during the clip).
            delta = (stabilized["transl"][0] - camera["transl"][0]).double().cpu().numpy()
            per_frame = _rotations(world)[:, 0] @ np.transpose(_rotations(camera)[:, 0], (0, 2, 1))
            world = {name: value.clone() for name, value in world.items()}
            world["transl"][0] += torch.as_tensor(
                np.einsum("tij,tj->ti", per_frame, delta), dtype=world["transl"].dtype
            )
        camera = stabilized
    threshold = float(options["contact_threshold"])
    conf = (
        torch.as_tensor(logits)[0, :, :FOOT_CONTACTS].float().sigmoid().cpu().numpy()
        if logits is not None
        else np.zeros((length, FOOT_CONTACTS))
    )
    planted = (conf > threshold).any(-1)
    # GEM's own gravity-aligned camera estimate (also defines "toward the camera").
    _, gem_rotation, spread = anchor_world(camera, world)
    anchored = options["world"] == "anchor"
    source, base = (camera, gem_rotation) if anchored else (world, np.eye(3))
    if anchored:
        say("Anchoring world motion to the fixed camera")
        report.update(rotation_spread_deg_p95=round(spread, 2))
    view_h, side_h = _camera_axes(gem_rotation)
    level = np.eye(3)
    if options["level"] == "auto" and logits is not None:
        say("Levelling the world on the floor")
        joints = _joints(source, soma_assets=soma_assets, device=device) @ base.T
        level, report["level"] = level_rotation(
            joints, conf, threshold=threshold, view_h=view_h, side_h=side_h
        )
    from scipy.spatial.transform import Rotation

    manual = (
        Rotation.from_rotvec(-side_h * np.radians(float(options["tilt_deg"])))
        * Rotation.from_rotvec(view_h * np.radians(float(options["roll_deg"])))
    ).as_matrix()
    rotation = manual @ level @ base
    # Levelled camera-to-world rotation (both modes): the 3D view's camera frame.
    report["camera_to_world_level"] = (manual @ level @ gem_rotation).tolist()
    if anchored or not np.allclose(rotation, np.eye(3)):
        world = _rotate_params(source, world, rotation)
        report.update(camera_to_world=rotation.tolist())
    else:
        rotation = None
    follow = bool(options["follow_floor"]) and not anchored
    say("Placing the floor", "following the planted feet" if follow else None)
    world, report["floor"] = _place(
        source,
        world,
        rotation,
        planted,
        soma_assets=soma_assets,
        device=device,
        progress=progress,
        follow=follow,
    )
    if options["lock_feet"] and logits is not None:
        say("Locking planted feet")
        world, report["lock_feet"] = lock_feet(
            world, conf, threshold=threshold, soma_assets=soma_assets, device=device
        )
    if options["plant_feet"] and logits is not None:
        say("Planting feet (hip height + leg IK)")
        world, camera, report["plant_feet"] = plant_feet(
            world, camera, logits, soma_assets=soma_assets, device=device, threshold=threshold
        )
        # Planting moves the soles to their anatomical height; settle the floor again.
        world, _ = _place(
            world, world, None, planted, soma_assets=soma_assets, device=device, follow=follow
        )
    offset = float(options["floor_offset_cm"]) / 100.0
    if offset:
        world = {**world, "transl": world["transl"].clone()}
        world["transl"][0, :, 1] += offset
    report["mode"] = "anchored_to_fixed_camera" if anchored else "gem_world"
    return {
        **result,
        "pred_soma_params_incam": camera,
        "pred_soma_params_global": world,
        "studio_world": report,
    }


def refine(result, options, *, soma_assets, device, progress=None, stage=None):
    """Return a refined copy of a Pose video or Motion result."""
    if (
        result.get("format") == "gem_body_video_predictions_v1"
        or "pred_soma_params_global" in result
    ):
        return refine_motion(
            result, options, soma_assets=soma_assets, device=device, progress=progress, stage=stage
        )
    if result.get("format") != "gem_pose_predictions_v1" or result["mode"] != "framewise_video":
        raise ValueError("Refinement applies to Body/Hand Pose video runs")
    config = smoothing_config(options)
    if result["task"] == "hand":
        from gem.pose.geometry.hand import HandGeometry
        from gem.pose.postprocess import smooth_hand_predictions

        geometry = HandGeometry(soma_assets).to(device)
        return smooth_hand_predictions(result, geometry=geometry, config=config)
    from gem.pose.geometry.body import BodyGeometry

    scales = next(
        row["parameters"]["scale_params"].shape[-1]
        for frame in result["frames"]
        for row in frame["instances"]
    )
    geometry = BodyGeometry(
        soma_assets, device=device, scale_parameterization="packed" if scales == 57 else "local56"
    )
    return smooth_body_predictions(result, geometry=geometry, config=config, progress=progress)
