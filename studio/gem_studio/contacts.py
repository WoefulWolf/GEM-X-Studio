"""Floor contacts for Body Motion: palms, heels and toes on the floor (Refine).

Detection. Each frame, every contact patch (each palm; each foot's heel and toes)
is measured against the floor (y = 0): the height of its lowest point and the
horizontal speed of its centre. A patch is *planted* when it is within reach of
the floor and nearly still, or when GEM predicts that joint as static and it is
close (GEM's "static" also covers a hand on a table, hence the distance). A foot
whose heel and toes are both near the floor but moving together is *sliding*: it
stays on the floor without being pinned. Very short contacts are dropped and
short gaps closed.

Planting. A planted patch is pinned where it touched down (the median of its
positions over the contact) and put on the floor, eased in and out. Palms lie
flat; a foot with heel and toes both planted lies flat, while toes alone (heel
rotating) or heel alone keep the foot's own rotation about the pinned point.
Arms and legs reach their targets with exact two-bone IK; the body moves only
where a limb cannot reach. The camera-space motion is left as GEM had it (the
overlay keeps matching the video); the pose itself is shared.
"""

import numpy as np

# Patch -> (end joint of its limb, joint the patch is rigid on, GEM static channel).
# 77-joint indices: LeftHand 14, RightHand 42, LeftFoot 69, LeftToeBase 70,
# RightFoot 74, RightToeBase 75; GEM channels: L ankle, L toe, R ankle, R toe, L hand, R hand.
PATCHES = {
    "left_hand": (14, 14, 4),
    "right_hand": (42, 42, 5),
    "left_heel": (69, 69, 0),
    "left_toes": (69, 70, 1),
    "right_heel": (74, 74, 2),
    "right_toes": (74, 75, 3),
}
# Limb chains (base, middle, end) for two-bone IK.
LIMBS = {14: (12, 13, 14), 42: (40, 41, 42), 69: (67, 68, 69), 74: (72, 73, 74)}
TOES = {69: 70, 74: 75}

# A patch slower than this (smoothed, m/s) counts as still, by contact sensitivity (0-1):
# GEM's estimate of a planted hand drifts much more than of a planted foot, and a hand
# repositions in fast bursts, while a foot swinging low over the floor must not be pinned.
HAND_STILL_SPEED = (0.3, 1.3)  # at sensitivity 0 and 1 (0.8 m/s at the default 0.5)
FOOT_STILL_SPEED = (0.2, 0.5)
GEM_REACH = 0.15  # m: a GEM-static patch closer than this to the floor is planted
# A foot tilted more than this (heel above toes, or toes above heel) is on one end only:
# the raised end rotates about the planted one instead of being laid flat.
PIVOT_TILT_DEG = 15.0
MIN_CONTACT_FRAMES = 3  # shorter contacts are noise
GAP_FRAMES = 2  # gaps up to this long inside a contact are closed
EASE_FRAMES = 3  # pins ease in and out over this many frames
REACH_MARGIN = 0.995  # a limb never straightens completely
# A hand closer to the floor than this turns its palm flat, fully at the floor, whether
# it is planted or just passing (landing, pushing off): a hand on the floor lies flat.
PALM_FLAT_REACH = 0.06
PALM_FLAT_FULL = 0.02  # fully flat this close
# A flat hand points its fingers the way the forearm leans (as when bracing on a hand), so
# the wrist bends 90 degrees minus the lean instead of past its range; below this lean
# (a vertical forearm) any finger direction is equally natural and GEM's is kept.
PALM_LEAN_MIN_DEG = 8.0


PATCH_POINTS = 120  # surface points kept per patch (evenly spread)


def patch_vertices(skeleton):
    """Per patch: rest-pose surface points with their skinning (4 joints, weights).

    Palms: the palm side of the hand (the T-pose palms face down). Heels and toes: the
    sole under the ankle joint and under the toe joint. Points are skinned like GEM's
    mesh (SOMA's weights), so a bent wrist lowers the heel of the hand as it does there.
    """
    names = [str(name) for name in skeleton["names"]]
    rest = skeleton["rest_vertices"].astype(np.float64)
    rest_pos = skeleton["rest_pos"].astype(np.float64)
    main = np.argmax(skeleton["weights"], axis=1)
    sole = rest[:, 1].min()
    out = {}
    for patch, (_, joint, _) in PATCHES.items():
        side = "Left" if patch.startswith("left") else "Right"
        if patch.endswith("hand"):
            hand = [i for i, name in enumerate(names) if name.startswith(f"{side}Hand")]
            wrist = rest_pos[joint]
            mask = np.isin(main, hand) & (rest[:, 1] < wrist[1] - 0.005)
            # Palm and finger bases only: the fingertips curl and should not lift the palm.
            reach = np.abs(rest[:, 0] - wrist[0]) < 0.11
            mask &= reach
        elif patch.endswith("heel"):
            mask = (main == joint) & (rest[:, 1] < sole + 0.02)
        else:
            toes = [names.index(f"{side}ToeBase"), names.index(f"{side}ToeEnd")]
            mask = np.isin(main, toes) & (rest[:, 1] < sole + 0.02)
        index = np.flatnonzero(mask)
        index = index[np.linspace(0, len(index) - 1, min(len(index), PATCH_POINTS)).astype(int)]
        weights = skeleton["weights"][index]
        top = np.argsort(-weights, axis=1)[:, :4]
        top_weights = np.take_along_axis(weights, top, axis=1).astype(np.float64)
        top_weights /= np.maximum(top_weights.sum(axis=1, keepdims=True), 1e-9)
        out[patch] = {"rest": rest[index], "joints": top, "weights": top_weights}
    return out


def _skin(patch, world, pos, rest_rot, rest_pos):
    """[T, N, 3] posed patch points (linear blend skinning) for world joint rotations."""
    joints, weights, rest = patch["joints"], patch["weights"], patch["rest"]
    points = 0.0
    for k in range(joints.shape[1]):
        joint = joints[:, k]
        delta = world[:, joint] @ np.transpose(rest_rot[joint], (0, 2, 1))  # [T, N, 3, 3]
        local = rest - rest_pos[joint]
        posed = np.einsum("tnab,nb->tna", delta, local) + pos[:, joint]
        points = points + weights[None, :, k, None] * posed
    return points


def _positions(pos, old_world, new_world, parents):
    """Joint positions after world rotations change (bone offsets stay in their parent)."""
    out = pos.copy()
    for joint in range(len(parents)):
        parent = parents[joint]
        if parent < 0:
            continue
        turn = new_world[:, parent] @ np.transpose(old_world[:, parent], (0, 2, 1))
        out[:, joint] = out[:, parent] + np.einsum(
            "tab,tb->ta", turn, pos[:, joint] - pos[:, parent]
        )
    return out


def palm_frame(skeleton, patches, patch):
    """A hand's palm normal (out of the palm) and finger direction in the rest pose.

    The normal is the palm surface's own plane (it is tilted ~9 degrees from the hand
    joint's frame in SOMA's rest pose); the fingers run from the wrist to the middle knuckle.
    """
    names = [str(name) for name in skeleton["names"]]
    side = "Left" if patch.startswith("left") else "Right"
    points = patches[patch]["rest"]
    _, _, vt = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    normal = vt[-1] if vt[-1][1] < 0 else -vt[-1]
    rest = skeleton["rest_pos"].astype(np.float64)
    fingers = rest[names.index(f"{side}HandMiddle1")] - rest[names.index(f"{side}Hand")]
    fingers -= normal * (fingers @ normal)  # within the palm plane
    return normal, fingers / np.linalg.norm(fingers)


def _yaw_between(a, b):
    """Signed angle about +y taking the horizontal part of ``a`` to that of ``b``."""
    a = a * np.array([1.0, 0.0, 1.0])
    b = b * np.array([1.0, 0.0, 1.0])
    return np.arctan2(np.cross(a, b)[:, 1], np.sum(a * b, axis=-1))


def _deltas(skeleton):
    """[T, J, 3, 3] world rotation of each joint relative to its rest (T-pose) rotation."""
    return skeleton["rot"].astype(np.float64) @ np.transpose(skeleton["rest_rot"], (0, 2, 1))


def patch_state(skeleton, patches, world=None, pos=None):
    """Per patch: lowest-point height [T] and centre position [T, 3]."""
    world = skeleton["rot"].astype(np.float64) if world is None else world
    pos = skeleton["pos"].astype(np.float64) if pos is None else pos
    rest_rot = skeleton["rest_rot"].astype(np.float64)
    rest_pos = skeleton["rest_pos"].astype(np.float64)
    heights, centres = {}, {}
    for patch, data in patches.items():
        points = _skin(data, world, pos, rest_rot, rest_pos)
        heights[patch] = points[..., 1].min(axis=1)
        centres[patch] = points.mean(axis=1)
    return heights, centres


def _runs(mask):
    """(start, stop) of each run of True."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def _clean(mask):
    """Close short gaps, then drop short runs."""
    mask = mask.copy()
    for start, stop in _runs(~mask):
        if start > 0 and stop < len(mask) and stop - start <= GAP_FRAMES:
            mask[start:stop] = True
    for start, stop in _runs(mask):
        if stop - start < MIN_CONTACT_FRAMES:
            mask[start:stop] = False
    return mask


def still_speed(patch, sensitivity):
    low, high = HAND_STILL_SPEED if patch.endswith("hand") else FOOT_STILL_SPEED
    return low + (high - low) * float(np.clip(sensitivity, 0.0, 1.0))


def detect(skeleton, patches, conf, *, reach, threshold, fps, hands=True, feet=True):
    """Contact states per patch: ``planted`` and ``sliding`` [T] booleans, with heights.

    ``threshold`` is GEM's static-confidence threshold; its complement is the contact
    sensitivity, which also sets how still a patch must be.
    """
    from scipy.ndimage import gaussian_filter1d

    heights, centres = patch_state(skeleton, patches)
    states = {}
    for patch in patches:
        is_hand = patch.endswith("hand")
        if (is_hand and not hands) or (not is_hand and not feet):
            continue
        height = heights[patch]
        smooth = gaussian_filter1d(centres[patch][:, [0, 2]], 1.5, axis=0, mode="nearest")
        speed = np.linalg.norm(np.gradient(smooth, axis=0), axis=1) * fps
        near = height < reach
        static = conf[:, PATCHES[patch][2]] > threshold if conf is not None else False
        still = speed < still_speed(patch, 1.0 - threshold)
        planted = _clean((near & still) | (static & (height < GEM_REACH)))
        states[patch] = {"planted": planted, "near": near, "speed": speed, "height": height}
    for side in ("left", "right"):
        heel, toes = states.get(f"{side}_heel"), states.get(f"{side}_toes")
        if heel is None or toes is None:
            continue
        # A clearly tilted foot pivots on its lower end (heel raise, toe lift).
        span = np.linalg.norm(centres[f"{side}_heel"] - centres[f"{side}_toes"], axis=1)
        rise = heel["height"] - toes["height"]
        limit = span * np.tan(np.radians(PIVOT_TILT_DEG))
        heel["planted"] = _clean(heel["planted"] & ~(rise > limit))
        toes["planted"] = _clean(toes["planted"] & ~(-rise > limit))
        # Heel and toes both near the floor but not planted: the foot slides on it.
        sliding = _clean(heel["near"] & toes["near"] & ~heel["planted"] & ~toes["planted"])
        heel["sliding"] = toes["sliding"] = sliding
    for state in states.values():
        state.setdefault("sliding", np.zeros_like(state["planted"]))
    return states, centres


def _ease(mask):
    """Weight 1 inside the runs of ``mask``, ramping to 0 over EASE_FRAMES outside them."""
    weight = mask.astype(float)
    for start, stop in _runs(mask):
        for k in range(1, EASE_FRAMES + 1):
            ramp = 1.0 - k / (EASE_FRAMES + 1)
            if start - k >= 0:
                weight[start - k] = max(weight[start - k], ramp)
            if stop - 1 + k < len(weight):
                weight[stop - 1 + k] = max(weight[stop - 1 + k], ramp)
    return weight


def pins(states, centres):
    """Per patch: pinned horizontal position [T, 2] (nan off contact) and its weight [T]."""
    out = {}
    for patch, state in states.items():
        target = np.full((len(state["planted"]), 2), np.nan)
        for start, stop in _runs(state["planted"]):
            spot = np.median(centres[patch][start:stop][:, [0, 2]], axis=0)
            lo, hi = max(0, start - EASE_FRAMES), min(len(target), stop + EASE_FRAMES)
            target[lo:hi] = spot
        out[patch] = (target, _ease(state["planted"]))
    return out


def _align(normals, up=np.array([0.0, -1.0, 0.0])):
    """[T, 3, 3] smallest rotations taking each normal to ``up``."""
    from scipy.spatial.transform import Rotation

    normals = normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-9)
    axis = np.cross(normals, up)
    sine = np.linalg.norm(axis, axis=-1)
    angle = np.arctan2(sine, np.clip(normals @ up, -1, 1))
    axis = axis / np.maximum(sine, 1e-9)[:, None]
    return Rotation.from_rotvec(axis * angle[:, None]).as_matrix()


def _rotation_between(a, b):
    """[T, 3, 3] smallest rotations taking vectors a to b."""
    from scipy.spatial.transform import Rotation

    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
    axis = np.cross(a, b)
    sine = np.linalg.norm(axis, axis=-1)
    angle = np.arctan2(sine, np.clip(np.sum(a * b, axis=-1), -1, 1))
    axis = axis / np.maximum(sine, 1e-9)[:, None]
    return Rotation.from_rotvec(axis * angle[:, None]).as_matrix()


def _slerp_rotation(rotation, weight):
    """Rotation matrices scaled by a per-frame weight in [0, 1] (along their own axis)."""
    from scipy.spatial.transform import Rotation

    vector = Rotation.from_matrix(rotation).as_rotvec() * weight[:, None]
    return Rotation.from_rotvec(vector).as_matrix()


def solve(skeleton, patches, states, pin):
    """New world rotations for limb joints and a root shift that plant the contacts.

    Returns ({joint: [T, 3, 3] new world rotation} for changed joints, [T, 3] root shift,
    stats). Rotations of joints not listed are unchanged.
    """
    from scipy.ndimage import gaussian_filter1d

    rest_rot = skeleton["rest_rot"].astype(np.float64)
    world = skeleton["rot"].astype(np.float64).copy()
    pos = skeleton["pos"].astype(np.float64)
    deltas = _deltas(skeleton)
    original = skeleton["rot"].astype(np.float64)
    parents = [int(parent) for parent in skeleton["parents"]]
    names = [str(name) for name in skeleton["names"]]
    frames = len(pos)
    ends = sorted({PATCHES[patch][0] for patch in states})
    changed = {}

    def hand_joints(patch):
        """Finger joints of a hand (all its descendants), in hierarchy order."""
        side = "Left" if patch.startswith("left") else "Right"
        return [i for i, n in enumerate(names) if n.startswith(f"{side}Hand") and n != f"{side}Hand"]

    # 1. Flatten: palms near the floor, feet with heel and toes both planted.
    near_heights, _ = patch_state(skeleton, {p: patches[p] for p in states})
    flat = {}
    for end in ends:
        if end in (14, 42):
            patch = "left_hand" if end == 14 else "right_hand"
            closeness = np.clip(
                (PALM_FLAT_REACH - near_heights[patch]) / (PALM_FLAT_REACH - PALM_FLAT_FULL), 0.0, 1.0
            )
            weight = np.maximum(pin[patch][1], closeness)
            flat[patch] = weight
        else:
            side = "left" if end == 69 else "right"
            heel, toes = f"{side}_heel", f"{side}_toes"
            if heel not in pin or toes not in pin:
                continue
            weight = np.minimum(pin[heel][1], pin[toes][1])
        if not weight.any():
            continue
        if end in (14, 42):
            palm, _ = palm_frame(skeleton, patches, patch)
            turn = _slerp_rotation(_align(deltas[:, end] @ palm), weight)
        else:
            normal = deltas[:, end] @ np.array([0.0, -1.0, 0.0])
            turn = _slerp_rotation(_align(normal), weight)
        world[:, end] = turn @ world[:, end]
        changed[end] = world[:, end]
        if end in TOES:  # the toes ride on the foot
            world[:, TOES[end]] = turn @ world[:, TOES[end]]
        if end in (14, 42):
            # The palm is skinned mostly to the knuckle joints (HandIndex1 ... HandPinky1), not
            # the hand joint alone: straighten the fingers to their rest pose relative to the
            # hand, or a curled or bent-back estimate tilts the palm. The thumb keeps its pose.
            for joint in hand_joints(patch):
                parent = parents[joint]
                posed = np.transpose(original[:, parent], (0, 2, 1)) @ original[:, joint]
                if "Thumb" in names[joint]:
                    world[:, joint] = world[:, parent] @ posed
                    continue
                rest = rest_rot[parent].T @ rest_rot[joint]
                back = _slerp_rotation(np.transpose(posed, (0, 2, 1)) @ rest, weight)
                world[:, joint] = world[:, parent] @ posed @ back
                changed[joint] = world[:, joint]
    flattened = _positions(pos, original, world, parents)

    # 2. Where each limb's end joint must go: planted patches on their pins and the floor.
    heights, centres = patch_state(
        skeleton, {p: patches[p] for p in states}, world=world, pos=flattened
    )
    move = {end: np.zeros((frames, 3)) for end in ends}
    count = {end: np.zeros(frames) for end in ends}
    weight_end = {end: np.zeros(frames) for end in ends}
    for patch, state in states.items():
        end = PATCHES[patch][0]
        target, weight = pin[patch]
        wanted = np.zeros((frames, 3))
        pinned = ~np.isnan(target[:, 0])
        wanted[pinned, 0] = target[pinned, 0] - centres[patch][pinned, 0]
        wanted[pinned, 2] = target[pinned, 1] - centres[patch][pinned, 2]
        wanted[:, 1] = -heights[patch]
        slide = state["sliding"].astype(float)
        both = np.maximum(weight, slide)
        wanted[:, [0, 2]] *= weight[:, None]
        wanted[:, 1] *= both
        if patch in flat:
            # A flattened palm reaches below the wrist: lift it out of the floor if needed.
            lift = np.maximum(-heights[patch], 0.0)
            wanted[:, 1] = np.where(both > 0, wanted[:, 1], lift)
            both = np.maximum(both, (flat[patch] > 0) | (lift > 0))
        active = both > 0
        move[end][active] += wanted[active]
        count[end][active] += 1
        weight_end[end] = np.maximum(weight_end[end], both)
    targets = {}
    for end in ends:
        active = count[end] > 0
        shift = np.zeros((frames, 3))
        shift[active] = move[end][active] / count[end][active, None]
        targets[end] = pos[:, end] + shift

    # 3. Root shift where a limb cannot reach its target (all planted limbs at once).
    root = np.zeros((frames, 3))
    lengths = {
        end: np.linalg.norm(pos[:, LIMBS[end][1]] - pos[:, LIMBS[end][0]], axis=1)
        + np.linalg.norm(pos[:, LIMBS[end][2]] - pos[:, LIMBS[end][1]], axis=1)
        for end in ends
    }
    for _ in range(12):
        step = np.zeros((frames, 3))
        used = np.zeros(frames)
        for end in ends:
            active = weight_end[end] > 0
            reach = targets[end] - (pos[:, LIMBS[end][0]] + root)
            distance = np.linalg.norm(reach, axis=1)
            excess = distance - REACH_MARGIN * lengths[end]
            over = active & (excess > 0)
            step[over] += reach[over] / distance[over, None] * excess[over, None]
            used[over] += 1
        if not used.any():
            break
        root[used > 0] += step[used > 0] / used[used > 0, None]
    root = gaussian_filter1d(root, 1.5, axis=0, mode="nearest")

    # 4. Two-bone IK toward the targets (frames where the limb is planted or easing).
    elbows = {end: pos[:, LIMBS[end][1]] + root for end in ends}
    wrists = {end: pos[:, end] + root for end in ends}
    for end in ends:
        active = weight_end[end] > 0
        if not active.any():
            continue
        base_j, mid_j, end_j = LIMBS[end]
        base = pos[active, base_j] + root[active]
        mid = pos[active, mid_j] + root[active]
        tip = pos[active, end_j] + root[active]
        target = targets[end][active]
        upper = np.linalg.norm(mid - base, axis=1)
        lower = np.linalg.norm(tip - mid, axis=1)
        offset = target - base
        distance = np.clip(
            np.linalg.norm(offset, axis=1),
            np.abs(upper - lower) + 1e-4,
            REACH_MARGIN * (upper + lower),
        )
        direction = offset / np.maximum(np.linalg.norm(offset, axis=1), 1e-9)[:, None]
        # Keep the knee / elbow bending the way it did: its side of the base-target line.
        bend = (mid - base) - np.sum((mid - base) * direction, axis=1)[:, None] * direction
        fallback = np.cross(direction, np.array([1.0, 0.0, 0.0]))
        small = np.linalg.norm(bend, axis=1) < 1e-6
        bend[small] = fallback[small]
        bend /= np.maximum(np.linalg.norm(bend, axis=1), 1e-9)[:, None]
        cosine = np.clip((upper**2 + distance**2 - lower**2) / (2 * upper * distance), -1, 1)
        new_mid = base + upper[:, None] * (
            cosine[:, None] * direction + np.sqrt(1 - cosine**2)[:, None] * bend
        )
        new_tip = base + direction * distance[:, None]
        first = _rotation_between(mid - base, new_mid - base)
        second = _rotation_between(np.einsum("tab,tb->ta", first, tip - mid), new_tip - new_mid)
        upper_rot = world[active, base_j].copy()
        lower_rot = world[active, mid_j].copy()
        world[active, base_j] = first @ upper_rot
        world[active, mid_j] = second @ first @ lower_rot
        elbows[end][active] = new_mid
        wrists[end][active] = new_tip
        changed[base_j] = world[:, base_j]
        changed[mid_j] = world[:, mid_j]
        changed[end_j] = world[:, end_j]  # its world rotation is kept (or flattened)

    # 5. Flat hands point their fingers the way the forearm (as solved) leans, turning
    # about the vertical so the palm stays flat: the wrist then bends 90 degrees minus the
    # lean, as when bracing on a hand, instead of past its range.
    from scipy.spatial.transform import Rotation

    for patch, weight in flat.items():
        end = PATCHES[patch][0]
        if not weight.any():
            continue
        _, fingers = palm_frame(skeleton, patches, patch)
        delta = world[:, end] @ rest_rot[end].T
        pointing = np.einsum("tab,b->ta", delta, fingers)
        lean = wrists[end] - elbows[end]
        horizontal = lean * np.array([1.0, 0.0, 1.0])
        angle = np.degrees(np.arctan2(np.linalg.norm(horizontal, axis=1), np.abs(lean[:, 1])))
        # The lean's direction is unreliable when the forearm is almost vertical: fade in.
        trust = np.clip((angle - PALM_LEAN_MIN_DEG) / PALM_LEAN_MIN_DEG, 0.0, 1.0) * weight
        direction = gaussian_filter1d(horizontal / np.maximum(
            np.linalg.norm(horizontal, axis=1, keepdims=True), 1e-9) * trust[:, None], 2.0,
            axis=0, mode="nearest")
        yaw = _yaw_between(pointing, direction) * np.clip(np.linalg.norm(direction, axis=1), 0, 1)
        spin = Rotation.from_rotvec(np.outer(yaw, [0.0, 1.0, 0.0])).as_matrix()
        for joint in [end] + [j for j in hand_joints(patch) if j in changed]:
            world[:, joint] = spin @ world[:, joint]
            changed[joint] = world[:, joint]

    stats = {}
    for group in ("feet", "hands"):
        members = [
            state["planted"]
            for patch, state in states.items()
            if patch.endswith("hand") == (group == "hands")
        ]
        if members:
            stats[f"{group}_planted_fraction"] = round(float(np.any(members, axis=0).mean()), 3)
    for patch, state in states.items():
        stats[f"{patch}_planted_fraction"] = round(float(state["planted"].mean()), 3)
    stats["max_root_shift_cm"] = round(float(np.linalg.norm(root, axis=1).max()) * 100, 2)
    return changed, root, stats
