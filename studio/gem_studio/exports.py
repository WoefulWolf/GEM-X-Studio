"""Mesh exports built from a job's preview bundle (NumPy only, no torch).

Exported geometry is Y-up metres for DCC tools: OpenCV camera streams are
converted with (x, -y, -z); Motion world streams are already Y-up.
"""

import io
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

_CV_TO_YUP = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def load_slots(directory, stream, start=0, count=None):
    """Return float32 [N, V, 3] in GEM coordinates for slots [start, start+count).

    The memory map is released before returning: Windows cannot delete a run
    folder while any of its files is still mapped by the server.
    """
    directory = Path(directory)
    if not stream.replace("_", "").isalnum():
        raise ValueError("Invalid stream name")
    verts = np.load(directory / f"verts_{stream}.npy", mmap_mode="r")
    try:
        stop = len(verts) if count is None else min(len(verts), start + count)
        data = np.array(verts[start:stop], dtype=np.float32)
    finally:
        verts._mmap.close()
        del verts
    offsets = np.load(directory / f"offsets_{stream}.npy")
    return data + offsets[start:stop, None]


def load_faces(directory, topology):
    if not topology.isalnum():
        raise ValueError("Invalid topology name")
    return np.load(Path(directory) / f"faces_{topology}.npy")


def _yup(stream, vertices):
    return vertices * _CV_TO_YUP if stream.startswith("camera_") else vertices


def _obj_text(vertices, faces, name):
    out = io.StringIO()
    out.write(f"# GEM-X Studio export, Y-up metres\no {name}\n")
    np.savetxt(out, vertices, fmt="v %.6f %.6f %.6f")
    np.savetxt(out, faces + 1, fmt="f %d %d %d")
    return out.getvalue()


def _instances(info):
    """Yield (frame_index, instance) across camera and world instance lists."""
    for index, frame in enumerate(info["frames"]):
        for key in ("instances", "world"):
            for instance in frame.get(key, []):
                yield index, instance


def _safe(text):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def export_obj_sequence(preview_dir, target):
    preview_dir = Path(preview_dir)
    info = json.loads((preview_dir / "preview.json").read_text())
    faces = {name: load_faces(preview_dir, name) for name in info["topologies"]}
    cache = {}
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for index, instance in _instances(info):
            stream = instance["stream"]
            if stream not in cache:
                cache[stream] = load_slots(preview_dir, stream)
            vertices = _yup(stream, cache[stream][instance["slot"]])
            name = f"{_safe(stream)}_{_safe(instance['track'])}"
            archive.writestr(
                f"{name}/frame_{index:05d}.obj",
                _obj_text(vertices, faces[instance["topology"]], name),
            )
    return target


def _tracks(info):
    """Group instances into (stream, track) sequences spanning every frame."""
    tracks = {}
    for index, instance in _instances(info):
        tracks.setdefault((instance["stream"], instance["track"], instance["topology"]), {})[
            index
        ] = instance["slot"]
    return tracks


def _held_slots(slots, frames):
    """Slot per requested frame; gaps hold the previous (or first) shape."""
    current = slots[min(slots)]
    ordered = []
    for index in frames:
        current = slots.get(index, current)
        ordered.append(current)
    return np.asarray(ordered)


def export_point_cache(preview_dir, target):
    """Rest-pose OBJ + Point Cache 2 (.pc2) per track, for Blender/Maya/3ds Max.

    Frames where a track is absent hold its previous (or first) shape.
    """
    preview_dir = Path(preview_dir)
    info = json.loads((preview_dir / "preview.json").read_text())
    fps = float(info.get("fps") or 30.0)
    frame_count = len(info["frames"])
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for (stream, track, topology), slots in _tracks(info).items():
            vertices = _yup(stream, load_slots(preview_dir, stream))
            faces = load_faces(preview_dir, topology)
            samples = vertices[_held_slots(slots, range(frame_count))]
            name = f"{_safe(stream)}_{_safe(track)}"
            archive.writestr(f"{name}.obj", _obj_text(samples[0], faces, name))
            header = struct.pack(
                "<12siiffi", b"POINTCACHE2\0", 1, samples.shape[1], 0.0, 1.0, len(samples)
            )
            archive.writestr(f"{name}.pc2", header + samples.astype("<f4").tobytes())
        archive.writestr(
            "README.txt",
            "Point Cache 2 export from GEM-X Studio (Y-up metres, "
            f"{fps:g} fps).\nBlender: import the .obj, add a Mesh Cache modifier, "
            "choose PC2 and the matching .pc2 file.\n",
        )
    return target


def export_npz(preview_dir, target):
    preview_dir = Path(preview_dir)
    info = json.loads((preview_dir / "preview.json").read_text())
    arrays = {f"faces_{name}": load_faces(preview_dir, name) for name in info["topologies"]}
    for stream in info["streams"]:
        arrays[f"vertices_{stream}"] = load_slots(preview_dir, stream)
    for (stream, track, _), slots in _tracks(info).items():
        frames = np.asarray(sorted(slots), dtype=np.int32)
        key = f"{_safe(stream)}_{_safe(track)}"
        arrays[f"frames_{key}"] = frames
        arrays[f"slots_{key}"] = np.asarray([slots[i] for i in frames], dtype=np.int32)
    np.savez_compressed(target, **arrays, coordinates=np.array("GEM: OpenCV camera / Y-up world"))
    return target


TRACK_COLORS = {
    "left": (0.37, 0.88, 0.54),
    "right": (0.37, 0.70, 1.0),
    None: (0.61, 0.88, 0.36),
}


def _vertex_normals(vertices, faces):
    corners = vertices[faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12)


class _Glb:
    """Minimal binary glTF 2.0 writer: one buffer, tightly packed views."""

    ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
    FLOAT, UNSIGNED_INT = 5126, 5125
    COMPONENT_TYPES = {"u4": 5125, "u2": 5123, "u1": 5121, "f4": 5126}

    def __init__(self):
        self.binary = bytearray()
        self.json = {
            "asset": {"version": "2.0", "generator": "GEM-X Studio"},
            "buffers": [],
            "bufferViews": [],
            "accessors": [],
        }

    def accessor(self, array, kind, *, target=None, bounds=False):
        array = np.ascontiguousarray(array)
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        view = {"buffer": 0, "byteOffset": len(self.binary), "byteLength": array.nbytes}
        if target:
            view["target"] = target
        self.binary.extend(array.tobytes())
        self.json["bufferViews"].append(view)
        accessor = {
            "bufferView": len(self.json["bufferViews"]) - 1,
            "componentType": self.COMPONENT_TYPES.get(array.dtype.str[1:], self.FLOAT),
            "count": len(array),
            "type": kind,
        }
        if bounds:
            flat = array.reshape(len(array), -1)
            accessor["min"] = flat.min(axis=0).tolist()
            accessor["max"] = flat.max(axis=0).tolist()
        self.json["accessors"].append(accessor)
        return len(self.json["accessors"]) - 1

    def write(self, path):
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        self.json["buffers"] = [{"byteLength": len(self.binary)}]
        text = json.dumps(self.json, separators=(",", ":")).encode()
        text += b" " * (-len(text) % 4)
        with open(path, "wb") as out:
            out.write(struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(self.binary)))
            out.write(struct.pack("<II", len(text), 0x4E4F534A) + text)
            out.write(struct.pack("<II", len(self.binary), 0x004E4942) + bytes(self.binary))


# OpenCV camera (looks down +Z, y down) -> glTF camera (looks down -Z, y up).
_CV_TO_GLTF_CAMERA = np.diag([1.0, -1.0, -1.0])


def _continuous_quats(quats):
    """Flip signs so consecutive quaternions stay in one hemisphere (smooth slerp)."""
    quats = np.array(quats, dtype=np.float64)
    for index in range(1, len(quats)):
        if np.dot(quats[index], quats[index - 1]) < 0:
            quats[index] *= -1
    return quats


def add_capture_camera(glb, info, space, frames, time_accessor, channels, samplers, yaw=None):
    """Add the capture camera as a glTF perspective camera node; returns whether it was added.

    World space: keyed per frame from ``camera_path`` (the camera moves with the run).
    Camera space: the camera is the origin, looking down -Z (static). The field of
    view is the clip's median vertical FOV. ``yaw`` rotates it with a turned skeleton.
    """
    from scipy.spatial.transform import Rotation

    gltf = glb.json
    path = info.get("camera_path")
    if space == "world" and not path:
        return False
    width, height = info.get("image_size") or [0, 0]
    focal = [frame["K"][1][1] for frame in info["frames"] if frame.get("K")]
    if not (width and height and focal):
        return False
    gltf.setdefault("cameras", []).append(
        {
            "name": "capture_camera",
            "type": "perspective",
            "perspective": {
                "yfov": float(2 * np.arctan(height / 2 / float(np.median(focal)))),
                "aspectRatio": float(width / height),
                "znear": 0.05,
                "zfar": 1000.0,
            },
        }
    )
    node = {"name": "capture_camera", "camera": len(gltf["cameras"]) - 1}
    gltf["nodes"].append(node)
    index = len(gltf["nodes"]) - 1
    gltf["scenes"][0]["nodes"].append(index)
    if space != "world":
        return True
    rotations = np.asarray(path["rotation"], dtype=np.float64)[frames].reshape(-1, 3, 3)
    positions = np.asarray(path["position"], dtype=np.float64)[frames]
    if yaw is not None:
        rotations = yaw @ rotations
        positions = positions @ yaw.T
    quats = _continuous_quats(Rotation.from_matrix(rotations @ _CV_TO_GLTF_CAMERA).as_quat())
    node["translation"] = positions[0].tolist()
    node["rotation"] = quats[0].tolist()
    for values, kind, target in (
        (positions, "VEC3", "translation"),
        (quats, "VEC4", "rotation"),
    ):
        samplers.append(
            {
                "input": time_accessor,
                "output": glb.accessor(values.astype(np.float32), kind),
                "interpolation": "LINEAR",
            }
        )
        channels.append({"sampler": len(samplers) - 1, "target": {"node": index, "path": target}})
    return True


def _animation_tracks(info, space):
    tracks = _tracks(info)
    prefix = "world_" if space == "world" else "camera_"
    selected = {key: slots for key, slots in tracks.items() if key[0].startswith(prefix)}
    return selected or tracks


def export_gltf_animation(preview_dir, target, *, space="auto", step=1, camera=False):
    """Animated GLB: per track a base mesh plus one morph target per sampled frame.

    Weights are keyed one-hot per frame with LINEAR interpolation, so playback
    blends vertex positions between frames. Morph targets reproduce GEM's
    meshes exactly (including pose correctives) and import as shape keys in
    Blender and blend shapes in Unity/Unreal/Maya. Gaps hold the previous shape.
    """
    preview_dir = Path(preview_dir)
    info = json.loads((preview_dir / "preview.json").read_text())
    fps = float(info.get("fps") or 30.0)
    frames = list(range(0, len(info["frames"]), max(1, int(step))))
    if len(frames) < 2:
        raise ValueError("Animation export needs a video result with at least two frames")
    if space == "auto":
        space = "world" if "world" in info.get("views", []) else "camera"
    times = (np.asarray(frames, dtype=np.float32) / fps).astype(np.float32)
    glb = _Glb()
    gltf = glb.json
    gltf.update(scene=0, scenes=[{"name": "GEM-X", "nodes": []}], nodes=[], meshes=[], materials=[])
    channels, samplers = [], []
    time_accessor = glb.accessor(times, "SCALAR", bounds=True)
    for (stream, track, topology), slots in _animation_tracks(info, space).items():
        vertices = _yup(stream, load_slots(preview_dir, stream))[_held_slots(slots, frames)]
        faces = load_faces(preview_dir, topology).astype(np.uint32)
        base = vertices[0]
        side = topology if topology in ("left", "right") else None
        gltf["materials"].append(
            {
                "name": f"{track}_material",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [*TRACK_COLORS[side], 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.65,
                },
                "doubleSided": True,
            }
        )
        targets = [
            {"POSITION": glb.accessor(frame - base, "VEC3", target=glb.ARRAY_BUFFER, bounds=True)}
            for frame in vertices[1:]
        ]
        primitive = {
            "attributes": {
                "POSITION": glb.accessor(base, "VEC3", target=glb.ARRAY_BUFFER, bounds=True),
                "NORMAL": glb.accessor(
                    _vertex_normals(base, faces).astype(np.float32), "VEC3", target=glb.ARRAY_BUFFER
                ),
            },
            "indices": glb.accessor(faces.reshape(-1), "SCALAR", target=glb.ELEMENT_ARRAY_BUFFER),
            "material": len(gltf["materials"]) - 1,
            "targets": targets,
        }
        name = _safe(f"{track.replace(':', '_')}_{space}")
        gltf["meshes"].append(
            {
                "name": name,
                "primitives": [primitive],
                "weights": [0.0] * len(targets),
                "extras": {"targetNames": [f"frame_{index:05d}" for index in frames[1:]]},
            }
        )
        gltf["nodes"].append({"name": name, "mesh": len(gltf["meshes"]) - 1})
        node = len(gltf["nodes"]) - 1
        gltf["scenes"][0]["nodes"].append(node)
        weights = np.zeros((len(frames), len(targets)), dtype=np.float32)
        weights[np.arange(1, len(frames)), np.arange(len(targets))] = 1.0
        samplers.append(
            {
                "input": time_accessor,
                "output": glb.accessor(weights.reshape(-1), "SCALAR"),
                "interpolation": "LINEAR",
            }
        )
        channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": "weights"}})
    if camera:
        add_capture_camera(glb, info, space, frames, time_accessor, channels, samplers)
    gltf["animations"] = [{"name": "GEM-X", "channels": channels, "samplers": samplers}]
    glb.write(target)
    return target


# -- FBX through a local Blender ------------------------------------------------

_BLENDER_SCRIPT = r"""
import sys, bpy
source, target, fps, step = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
# Set the rate first: the glTF importer converts seconds with the scene FPS.
scene.render.fps = max(1, round(float(fps)))
scene.render.fps_base = scene.render.fps / float(fps)
bpy.ops.import_scene.gltf(filepath=source)
# Shape-key and armature animations both live in bpy.data.actions.
end = max([1] + [int(round(action.frame_range[1])) for action in bpy.data.actions])
scene.frame_start, scene.frame_end = 0, end
# The glTF shape-key animation lives on the mesh's Key datablock (as an NLA strip),
# which per-strip/per-action export skips; bake one global take over the scene range.
bpy.ops.export_scene.fbx(
    filepath=target, use_selection=False, bake_anim=True,
    bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False,
    bake_anim_simplify_factor=0.0, bake_anim_step=float(step), add_leaf_bones=False,
    axis_forward="-Z", axis_up="Y", object_types={"MESH", "EMPTY", "ARMATURE", "CAMERA"},
)
"""


def find_blender(configured=None):
    """Settings path, then PATH, then the newest standard Windows install."""
    import glob
    import os
    import shutil

    if configured and Path(configured).is_file():
        return str(configured)
    found = shutil.which("blender")
    if found:
        return found
    roots = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("LOCALAPPDATA", ""),
    ]
    candidates = []
    for root in filter(None, roots):
        candidates += glob.glob(os.path.join(root, "Blender Foundation", "Blender*", "blender.exe"))
    candidates += glob.glob(r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe")
    if sys.platform == "darwin":
        candidates.append("/Applications/Blender.app/Contents/MacOS/Blender")
    candidates = [path for path in candidates if Path(path).is_file()]
    return max(candidates, key=_blender_version) if candidates else None


def _blender_version(path):
    import re

    match = re.search(r"Blender\s*(\d+)\.(\d+)", str(path))
    return tuple(map(int, match.groups())) if match else (0, 0)


def export_fbx(preview_dir, target, *, blender, space="auto", step=1, make_glb=None, camera=False):
    """Animated GLB converted to FBX by headless Blender.

    By default the GLB is the shape-key animation (FBX blend shapes);
    ``make_glb(path)`` supplies another one, e.g. the skinned skeleton export.
    """
    import subprocess
    import tempfile

    if not blender:
        raise RuntimeError("FBX export needs Blender; install it or set its path in Settings")
    info = json.loads((Path(preview_dir) / "preview.json").read_text())
    # Import at the clip's own rate (Blender converts glTF seconds with the integer
    # scene FPS) and let the FBX baker sample every ``step`` frames.
    fps = float(info.get("fps") or 30.0)
    with tempfile.TemporaryDirectory(prefix="gem-fbx-") as temporary:
        source = Path(temporary) / "animation.glb"
        script = Path(temporary) / "to_fbx.py"
        if make_glb is None:
            export_gltf_animation(preview_dir, source, space=space, step=step, camera=camera)
        else:
            make_glb(source)
        script.write_text(_BLENDER_SCRIPT)
        result = subprocess.run(
            [blender, "--background", "--factory-startup", "--python", str(script), "--",
             str(source), str(target), str(fps), str(max(1, int(step)))],
            capture_output=True, encoding="utf-8", errors="replace", timeout=1800,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )  # fmt: skip
        if result.returncode != 0 or not Path(target).is_file():
            tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-8:])
            raise RuntimeError(f"Blender FBX conversion failed:\n{tail}")
    return target


EXPORTERS = {
    "obj_sequence": ("meshes_obj_sequence.zip", export_obj_sequence),
    "point_cache": ("meshes_point_cache_pc2.zip", export_point_cache),
    "npz": ("meshes.npz", export_npz),
    "gltf_animation": ("animation.glb", export_gltf_animation),
    "fbx": ("animation.fbx", export_fbx),
}
