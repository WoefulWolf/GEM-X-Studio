"""Skeletal exports from a cached SOMA skeleton (see skeleton.py): skinned glTF and MMD VMD.

Bones use world-aligned rest frames: every joint's bind rotation is identity, so
an animated joint's world rotation is ``D_j(t) = R_j(t) R_j_rest^T`` (its delta
from the T-pose bind) and its local rotation is ``D_parent^T D_j``. This is
independent of SOMA's internal joint orientations, which also makes retargeting
to other skeletons (MMD) a matter of picking joints: rotations of skipped
in-between joints fold into the next mapped bone automatically.
"""

import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .exports import TRACK_COLORS, _Glb, _vertex_normals, add_capture_camera, load_faces

# -- MMD naming --------------------------------------------------------------
# (MMD bone, SOMA joint, MMD parent, A-pose arm side). センター has no SOMA joint.
# MMD finger bones (thumb ０-２, others １-３) and the SOMA joints that drive them.
_FINGERS = (
    ("親指", ("０", "１", "２"), ("Thumb1", "Thumb2", "Thumb3")),
    ("人指", ("１", "２", "３"), ("Index2", "Index3", "Index4")),
    ("中指", ("１", "２", "３"), ("Middle2", "Middle3", "Middle4")),
    ("薬指", ("１", "２", "３"), ("Ring2", "Ring3", "Ring4")),
    ("小指", ("１", "２", "３"), ("Pinky2", "Pinky3", "Pinky4")),
)


def _mmd_bones():
    bones = [
        ("センター", None, None, None),
        ("下半身", "Hips", "センター", None),
        ("上半身", "Spine1", "センター", None),
        ("上半身2", "Chest", "上半身", None),
        ("首", "Neck1", "上半身2", None),
        ("頭", "Head", "首", None),
    ]
    for jp, en in (("左", "Left"), ("右", "Right")):
        side = en.lower()
        bones += [
            (f"{jp}肩", f"{en}Shoulder", "上半身2", None),
            (f"{jp}腕", f"{en}Arm", f"{jp}肩", side),
            (f"{jp}ひじ", f"{en}ForeArm", f"{jp}腕", side),
            (f"{jp}手首", f"{en}Hand", f"{jp}ひじ", side),
        ]
        for finger, digits, joints in _FINGERS:
            parent = f"{jp}手首"
            for digit, joint in zip(digits, joints):
                bones.append((f"{jp}{finger}{digit}", f"{en}Hand{joint}", parent, side))
                parent = f"{jp}{finger}{digit}"
        bones += [
            (f"{jp}足", f"{en}Leg", "下半身", None),
            (f"{jp}ひざ", f"{en}Shin", f"{jp}足", None),
            (f"{jp}足首", f"{en}Foot", f"{jp}ひざ", None),
            (f"{jp}足先EX", f"{en}ToeBase", f"{jp}足首", None),
        ]
    return bones


MMD_BONES = _mmd_bones()
SOMA_TO_MMD = {soma: mmd for mmd, soma, _, _ in MMD_BONES if soma}
MMD_LEG_IK = ("左足ＩＫ", "右足ＩＫ", "左つま先ＩＫ", "右つま先ＩＫ")


def joint_names(skeleton, naming):
    names = [str(name) for name in skeleton["names"]]
    if naming == "mmd":
        return [SOMA_TO_MMD.get(name, name) for name in names]
    return names


# -- shared math -------------------------------------------------------------
def load_skeleton(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def world_deltas(skeleton):
    """[T, J, 3, 3] rotation of each joint relative to its T-pose bind, in world."""
    return skeleton["rot"].astype(np.float64) @ np.transpose(skeleton["rest_rot"], (0, 2, 1))


def _yaw_to_face_forward(deltas, hips=0):
    """Yaw rotation making the body face +Z at the first frame."""
    forward = deltas[0, hips] @ np.array([0.0, 0.0, 1.0])
    angle = np.arctan2(forward[0], forward[2])
    return Rotation.from_euler("y", -angle).as_matrix()


def _continuous(quats):
    """Flip signs so consecutive quaternions stay in the same hemisphere."""
    quats = quats.copy()
    for index in range(1, len(quats)):
        flip = np.sum(quats[index] * quats[index - 1], axis=-1) < 0
        quats[index][flip] *= -1
    return quats


def _sample(skeleton, step):
    frames = np.arange(0, len(skeleton["pos"]), max(1, int(step)))
    return frames, frames / float(skeleton["fps"])


# -- skinned glTF ------------------------------------------------------------
def export_skinned_gltf(
    skeleton_file,
    preview_dir,
    target,
    *,
    step=1,
    naming="soma",
    face_forward=False,
    camera=False,
    space="world",
):
    """Skinned mesh + animated skeleton (rotations for every joint, Hips translation).

    The mesh is SOMA's bind pose with its own skinning weights (top-4 per
    vertex). Linear skinning omits SOMA's pose correctives (typically a few mm);
    use the shape-key export when the exact surface matters.
    """
    skeleton = load_skeleton(skeleton_file)
    frames, times = _sample(skeleton, step)
    parents = skeleton["parents"]
    names = joint_names(skeleton, naming)
    deltas = world_deltas(skeleton)[frames]
    positions = skeleton["pos"][frames].astype(np.float64)
    yaw = None
    if face_forward:
        yaw = _yaw_to_face_forward(deltas)
        deltas = yaw @ deltas
        positions = positions @ yaw.T
    rest = skeleton["rest_pos"].astype(np.float64)
    glb = _Glb()
    gltf = glb.json
    gltf.update(scene=0, scenes=[{"name": "GEM-X", "nodes": []}], nodes=[], meshes=[], skins=[])
    for index, name in enumerate(names):
        parent = parents[index]
        node = {
            "name": name,
            "translation": (rest[index] - (rest[parent] if parent >= 0 else 0)).tolist(),
        }
        gltf["nodes"].append(node)
    for index, parent in enumerate(parents):
        if parent >= 0:
            gltf["nodes"][parent].setdefault("children", []).append(index)
    root = int(np.flatnonzero(parents < 0)[0])
    inverse_bind = np.repeat(np.eye(4)[None], len(names), axis=0)
    inverse_bind[:, :3, 3] = -rest
    # glTF matrices are column-major.
    ibm = glb.accessor(
        np.transpose(inverse_bind, (0, 2, 1)).astype(np.float32).reshape(len(names), 16), "MAT4"
    )
    gltf["skins"].append(
        {
            "name": "GEM-X skeleton",
            "joints": list(range(len(names))),
            "skeleton": root,
            "inverseBindMatrices": ibm,
        }
    )

    vertices = skeleton["rest_vertices"].astype(np.float32)
    faces = load_faces(preview_dir, "body").astype(np.uint32)
    weights = skeleton["weights"]
    top = np.argsort(-weights, axis=1)[:, :4]
    top_weights = np.take_along_axis(weights, top, 1)
    top_weights /= top_weights.sum(1, keepdims=True)
    top[top_weights <= 0] = 0  # unused influence slots must point at joint 0
    gltf["materials"] = [
        {
            "name": "body",
            "pbrMetallicRoughness": {
                "baseColorFactor": [*TRACK_COLORS[None], 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.65,
            },
            "doubleSided": True,
        }
    ]
    primitive = {
        "attributes": {
            "POSITION": glb.accessor(vertices, "VEC3", target=glb.ARRAY_BUFFER, bounds=True),
            "NORMAL": glb.accessor(
                _vertex_normals(vertices, faces).astype(np.float32), "VEC3", target=glb.ARRAY_BUFFER
            ),
            "JOINTS_0": glb.accessor(top.astype(np.uint16), "VEC4", target=glb.ARRAY_BUFFER),
            "WEIGHTS_0": glb.accessor(
                top_weights.astype(np.float32), "VEC4", target=glb.ARRAY_BUFFER
            ),
        },
        "indices": glb.accessor(faces.reshape(-1), "SCALAR", target=glb.ELEMENT_ARRAY_BUFFER),
        "material": 0,
    }
    gltf["meshes"].append({"name": "body", "primitives": [primitive]})
    gltf["nodes"].append({"name": "body", "mesh": 0, "skin": 0})
    gltf["scenes"][0]["nodes"] = [root, len(gltf["nodes"]) - 1]

    time_accessor = glb.accessor(times.astype(np.float32), "SCALAR", bounds=True)
    channels, samplers = [], []
    for index, parent in enumerate(parents):
        local = (
            deltas[:, index]
            if parent < 0
            else np.transpose(deltas[:, parent], (0, 2, 1)) @ deltas[:, index]
        )
        quats = _continuous(Rotation.from_matrix(local).as_quat()[:, None])[:, 0]
        samplers.append(
            {
                "input": time_accessor,
                "output": glb.accessor(quats.astype(np.float32), "VEC4"),
                "interpolation": "LINEAR",
            }
        )
        channels.append(
            {"sampler": len(samplers) - 1, "target": {"node": index, "path": "rotation"}}
        )
    samplers.append(
        {
            "input": time_accessor,
            "output": glb.accessor(positions[:, root].astype(np.float32), "VEC3"),
            "interpolation": "LINEAR",
        }
    )
    channels.append({"sampler": len(samplers) - 1, "target": {"node": root, "path": "translation"}})
    if camera:
        info = json.loads((Path(preview_dir) / "preview.json").read_text())
        add_capture_camera(glb, info, space, frames, time_accessor, channels, samplers, yaw=yaw)
    gltf["animations"] = [{"name": "GEM-X", "channels": channels, "samplers": samplers}]
    glb.write(target)
    return target


# -- MMD VMD -----------------------------------------------------------------
# MikuMikuDance's default linear interpolation block (X, Y, Z, rotation curves).
_VMD_LINEAR = bytes(
    [20, 20, 0, 0, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107,
     20, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0,
     20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0, 0,
     20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0, 0, 0]
)  # fmt: skip
_TO_MMD = np.diag([1.0, 1.0, -1.0])  # right-handed Y-up -> MMD's left-handed (z flipped)


def _sjis(text, size):
    data = text.encode("shift_jis")[:size]
    return data + b"\0" * (size - len(data))


def mmd_motion(skeleton, *, arm_angle_deg=35.0, scale=12.5, face_forward=True):
    """Per-bone MMD local rotations [T, 4] (xyzw, MMD space) and センター positions [T, 3].

    MMD bones rest axis-aligned in an A-pose: arms rotated down by ``arm_angle_deg``.
    A bone's world rotation is ``W = D_soma(t) A^T`` (A = rest A-pose turn for
    arm-chain bones), so its direction matches SOMA's; local = W_parent^T W.
    ``scale`` converts metres to MMD units (12.5: 1 unit = 8 cm).
    """
    names = [str(name) for name in skeleton["names"]]
    index = {name: i for i, name in enumerate(names)}
    deltas = world_deltas(skeleton)
    positions = skeleton["pos"].astype(np.float64)
    hips = index["Hips"]
    if face_forward:
        yaw = _yaw_to_face_forward(deltas, hips)
        deltas = yaw @ deltas
        positions = positions @ yaw.T
    arm = np.radians(arm_angle_deg)
    a_pose = {
        "left": Rotation.from_euler("z", -arm).as_matrix(),
        "right": Rotation.from_euler("z", arm).as_matrix(),
    }
    worlds = {"センター": np.repeat(np.eye(3)[None], len(deltas), axis=0)}
    rotations = {}
    for mmd, soma, parent, side in MMD_BONES:
        if soma is None:
            continue
        world = deltas[:, index[soma]]
        if side:
            world = world @ a_pose[side].T
        worlds[mmd] = world
        local = np.transpose(worlds[parent], (0, 2, 1)) @ world
        local = _TO_MMD @ local @ _TO_MMD
        rotations[mmd] = _continuous(Rotation.from_matrix(local).as_quat()[:, None])[:, 0]
    # センター: hips displacement from standing at the origin.
    stand = skeleton["rest_pos"][hips, 1] - skeleton["rest_vertices"][:, 1].min()
    start = positions[0, hips]
    centre = positions[:, hips] - np.array([start[0], stand, start[2]])
    rotations["センター"] = np.tile([0.0, 0.0, 0.0, 1.0], (len(deltas), 1))
    return rotations, (centre @ _TO_MMD.T) * scale


def write_vmd(target, rotations, centre, *, fps=30.0, model_name="GEM-X"):
    """VMD 0002 with one keyframe per frame (resampled to MMD's 30 fps) and leg IK off."""
    count = len(centre)
    frames = np.round(np.arange(count) * 30.0 / fps).astype(int)
    keep = np.concatenate(([True], np.diff(frames) > 0))
    records = []
    zero = (0.0, 0.0, 0.0)
    for name, quats in rotations.items():
        encoded = _sjis(name, 15)
        for i in np.flatnonzero(keep):
            position = tuple(centre[i]) if name == "センター" else zero
            records.append(
                encoded + struct.pack("<I3f4f", int(frames[i]), *position, *quats[i]) + _VMD_LINEAR
            )
    with open(target, "wb") as out:
        out.write(_sjis("Vocaloid Motion Data 0002", 30))
        out.write(_sjis(model_name, 20))
        out.write(struct.pack("<I", len(records)))
        out.write(b"".join(records))
        out.write(struct.pack("<IIII", 0, 0, 0, 0))  # morph, camera, light, self-shadow
        out.write(struct.pack("<I", 1))  # one show/IK frame at 0: model shown, leg IK off
        out.write(struct.pack("<IBI", 0, 1, len(MMD_LEG_IK)))
        for name in MMD_LEG_IK:
            out.write(_sjis(name, 20) + struct.pack("<B", 0))
    return target


def read_vmd(path):
    """Minimal reader (bone + IK sections) used for verification."""
    data = Path(path).read_bytes()
    assert data[:25] == b"Vocaloid Motion Data 0002"
    offset = 50
    (count,) = struct.unpack_from("<I", data, offset)
    offset += 4
    bones = []
    for _ in range(count):
        name = data[offset : offset + 15].split(b"\0")[0].decode("shift_jis")
        frame, *values = struct.unpack_from("<I3f4f", data, offset + 15)
        bones.append((name, frame, values[:3], values[3:]))
        offset += 111
    morph, camera, light, shadow = struct.unpack_from("<IIII", data, offset)
    offset += 16
    (ik_frames,) = struct.unpack_from("<I", data, offset)
    offset += 4
    frame, show, ik_count = struct.unpack_from("<IBI", data, offset)
    offset += 9
    iks = []
    for _ in range(ik_count):
        iks.append(
            (data[offset : offset + 20].split(b"\0")[0].decode("shift_jis"), data[offset + 20])
        )
        offset += 21
    return {
        "bones": bones,
        "counts": (morph, camera, light, shadow),
        "ik": iks,
        "size": len(data),
        "end": offset,
    }


def export_vmd(skeleton_file, target, *, arm_angle_deg=35.0, scale=12.5, face_forward=True):
    skeleton = load_skeleton(skeleton_file)
    rotations, centre = mmd_motion(
        skeleton, arm_angle_deg=arm_angle_deg, scale=scale, face_forward=face_forward
    )
    return write_vmd(target, rotations, centre, fps=float(skeleton["fps"]))
