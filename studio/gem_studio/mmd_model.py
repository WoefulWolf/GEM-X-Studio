"""MMD models (.pmx) for fitting a VMD export to the model's own proportions.

Only what the fit needs is kept: the bone tree (rest positions, parents, rotation
append) and the body surface (vertices on body bones with their skin weights). The
uploaded .pmx itself is discarded after reading; nothing else (textures, morphs,
physics) is read.
"""

import json
import re
import struct
import uuid

import numpy as np

from .paths import DATA_DIR
from .rigging import _model_arm_angle

MODELS_DIR = DATA_DIR / "mmd_models"

# A vertex belongs to the body surface (can touch the floor) when its main bone is
# a body bone: limbs, torso, neck and head with their twist/helper/D variants...
_BODY_WORDS = (
    "足", "ひざ", "膝", "つま先", "手", "指", "ひじ", "肘", "腕", "上半身", "下半身",
    "首", "頭", "肩", "腰", "センター", "グルーブ", "胸",
)  # fmt: skip
# ...and not hair, clothing or accessories (usually physics-driven, they hang loose).
_LOOSE_WORDS = (
    "髪", "もみあげ", "アホ毛", "ｽｶｰﾄ", "スカート", "ﾈｸﾀｲ", "ネクタイ", "ベルト", "袖", "リボン",
    "ﾘﾎﾞﾝ", "しっぽ", "尻尾", "羽", "帽", "服", "飾", "揺", "パーツ", "ﾊﾟｰﾂ", "紐", "裾", "コート",
    "マント", "ﾏﾝﾄ", "フード", "耳",
)  # fmt: skip


class _Reader:
    def __init__(self, data):
        self.data, self.at = data, 0

    def take(self, fmt):
        values = struct.unpack_from("<" + fmt, self.data, self.at)
        self.at += struct.calcsize("<" + fmt)
        return values

    def skip(self, size):
        self.at += size

    def text(self, encoding):
        (size,) = self.take("i")
        raw = self.data[self.at : self.at + size]
        self.at += size
        return raw.decode("utf-16-le" if encoding == 0 else "utf-8", errors="replace")

    def index(self, size):
        return self.take({1: "b", 2: "h", 4: "i"}[size])[0]


def read_pmx(path):
    """Bones and skinned vertex positions of a PMX 2.0/2.1 model (MMD units, MMD axes)."""
    reader = _Reader(open(path, "rb").read())  # noqa: SIM115
    if reader.data[:4] != b"PMX ":
        raise ValueError("Not a PMX model (MMD .pmx file)")
    reader.skip(4)
    reader.take("f")
    (count,) = reader.take("B")
    encoding, extra_uv, vertex_size, texture_size, _, bone_size = reader.take(f"{count}B")[:6]
    name = reader.text(encoding)
    name_en = reader.text(encoding)
    reader.text(encoding), reader.text(encoding)

    (vertex_count,) = reader.take("i")
    positions = np.zeros((vertex_count, 3), np.float32)
    bone_index = np.zeros((vertex_count, 4), np.int32)
    weights = np.zeros((vertex_count, 4), np.float32)
    for vertex in range(vertex_count):
        positions[vertex] = reader.take("3f")
        reader.skip(4 * (5 + 4 * extra_uv))  # normal, uv, extra uvs
        (kind,) = reader.take("B")
        if kind == 0:  # BDEF1
            bone_index[vertex, 0], weights[vertex, 0] = reader.index(bone_size), 1.0
        elif kind in (1, 3):  # BDEF2, SDEF (as linear blend)
            bone_index[vertex, :2] = reader.index(bone_size), reader.index(bone_size)
            (weight,) = reader.take("f")
            weights[vertex, :2] = weight, 1.0 - weight
            if kind == 3:
                reader.skip(36)
        elif kind in (2, 4):  # BDEF4, QDEF
            bone_index[vertex] = [reader.index(bone_size) for _ in range(4)]
            weights[vertex] = reader.take("4f")
        else:
            raise ValueError(f"Unsupported PMX vertex weight type {kind}")
        reader.skip(4)  # edge scale
    (index_count,) = reader.take("i")
    reader.skip(index_count * vertex_size)
    (texture_count,) = reader.take("i")
    for _ in range(texture_count):
        reader.text(encoding)
    (material_count,) = reader.take("i")
    for _ in range(material_count):
        reader.text(encoding), reader.text(encoding)
        reader.skip(4 * 4 + 4 * 3 + 4 + 4 * 3 + 1 + 4 * 4 + 4)  # colours, flags, edge
        reader.skip(2 * texture_size + 1)  # texture, sphere, sphere mode
        (shared_toon,) = reader.take("B")
        reader.skip(texture_size if shared_toon == 0 else 1)
        reader.text(encoding)
        reader.skip(4)  # face count

    (bone_count,) = reader.take("i")
    bones = []
    for _ in range(bone_count):
        bone = {"name": reader.text(encoding)}
        reader.text(encoding)
        bone["position"] = reader.take("3f")
        bone["parent"] = reader.index(bone_size)
        reader.skip(4)  # transform level
        (flags,) = reader.take("H")
        if flags & 0x0001:
            reader.index(bone_size)
        else:
            reader.skip(12)
        bone["append"] = None
        if flags & 0x0300:
            source = reader.index(bone_size)
            (ratio,) = reader.take("f")
            if flags & 0x0100:  # rotation append (付与回転)
                bone["append"] = (source, ratio)
        if flags & 0x0400:
            reader.skip(12)
        if flags & 0x0800:
            reader.skip(24)
        if flags & 0x2000:
            reader.skip(4)
        if flags & 0x0020:  # IK: target, loops, limit angle, links
            reader.index(bone_size)
            reader.skip(8)
            (links,) = reader.take("i")
            for _ in range(links):
                reader.index(bone_size)
                (limited,) = reader.take("B")
                if limited:
                    reader.skip(24)
        bones.append(bone)
    return {
        "name": name.strip() or name_en.strip(),
        "positions": positions,
        "bone_index": bone_index,
        "weights": weights,
        "bones": bones,
    }


def _is_body_bone(name):
    return any(word in name for word in _BODY_WORDS) and not any(
        word in name for word in _LOOSE_WORDS
    )


def build_profile(pmx):
    """The fit profile: bone tree plus the body-surface vertices."""
    bones = pmx["bones"]
    names = [bone["name"] for bone in bones]
    for required in ("センター", "左足", "左ひざ", "左足首", "左腕", "左ひじ"):
        if required not in names:
            raise ValueError(f"The model has no standard bone {required}; cannot fit to it")
    rows = np.arange(len(pmx["positions"]))
    main = pmx["bone_index"][rows, np.argmax(pmx["weights"], axis=1)]
    body_bones = np.array([_is_body_bone(name) for name in names])
    keep = body_bones[np.clip(main, 0, len(names) - 1)] & (main >= 0)
    if keep.sum() < 100:
        raise ValueError("Could not find the model's body surface (too few body vertices)")
    rest = np.array([bone["position"] for bone in bones], np.float32)
    append = np.array([bone["append"][0] if bone["append"] else -1 for bone in bones], np.int32)
    ratio = np.array([bone["append"][1] if bone["append"] else 0.0 for bone in bones], np.float32)
    return {
        "names": np.array(names),
        "parents": np.array([bone["parent"] for bone in bones], np.int32),
        "rest": rest,
        "append": append,
        "append_ratio": ratio,
        "vertices": pmx["positions"][keep],
        "vertex_bones": np.clip(pmx["bone_index"][keep], 0, len(names) - 1),
        "vertex_weights": pmx["weights"][keep],
    }


def summary(profile, model_id, name):
    """Size and proportions shown in Studio (MMD units)."""
    names = list(profile["names"])
    rest = profile["rest"]

    def at(bone):
        return rest[names.index(bone)]

    def length(*chain):
        return float(sum(np.linalg.norm(at(a) - at(b)) for a, b in zip(chain, chain[1:])))

    return {
        "id": model_id,
        "name": name,
        "height": round(float(np.ptp(profile["vertices"][:, 1])), 2),
        "leg_length": round(length("左足", "左ひざ", "左足首"), 2),
        "arm_length": round(length("左腕", "左ひじ", "左手首") if "左手首" in names else 0.0, 2),
        "arm_angle_deg": round(_model_arm_angle(profile), 1),
        "body_vertices": int(len(profile["vertices"])),
    }


def save_model(pmx_path, original_name):
    """Read an uploaded .pmx, store its fit profile and summary, and return the summary."""
    pmx = read_pmx(pmx_path)
    profile = build_profile(pmx)
    model_id = uuid.uuid4().hex[:12]
    name = pmx["name"] or re.sub(r"\.pmx$", "", original_name, flags=re.IGNORECASE)
    info = summary(profile, model_id, name)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODELS_DIR / f"{model_id}.npz", **profile)
    text = json.dumps(info, ensure_ascii=False, indent=2)
    (MODELS_DIR / f"{model_id}.json").write_text(text, encoding="utf-8")
    return info


def model_info(model_id):
    if not model_id.isalnum():
        raise KeyError(model_id)
    path = MODELS_DIR / f"{model_id}.json"
    if not path.is_file():
        raise KeyError(model_id)
    return json.loads(path.read_text(encoding="utf-8"))


def load_profile(model_id):
    model_info(model_id)
    with np.load(MODELS_DIR / f"{model_id}.npz") as data:
        return {name: data[name] for name in data.files}
