import * as THREE from "three"
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js"
import { OBJExporter } from "three/addons/exporters/OBJExporter.js"
import { OrbitControls } from "three/addons/controls/OrbitControls.js"

import type { Preview, PreviewInstance } from "@/lib/api"
import type { VertexSource } from "./mesh-store"

export type ViewMode = "overlay" | "camera3d" | "world"

export interface MeshStyle {
  showMesh: boolean
  opacity: number
  color: string
}

const SIDE_COLORS: Record<string, string> = { left: "#5fe08a", right: "#5fb2ff" }
// OpenCV camera (x right, y down, z forward) -> three.js (y up, z back).
const CV_TO_GL = new THREE.Vector3(1, -1, -1)
const CV_TO_GL_MATRIX = new THREE.Matrix3().set(1, 0, 0, 0, -1, 0, 0, 0, -1)
const IDENTITY = new THREE.Matrix3()
const DEFAULT_FOV = 40
// Floor grid: metres across and cells (0.5 m each).
const GRID_SIZE = 30
const GRID_DIVISIONS = 60

/** Pinhole projection matching OpenCV intrinsics on a W x H raster. */
function intrinsicsProjection(K: number[][], width: number, height: number, near = 0.05, far = 200) {
  const [fx, cx] = [K[0][0], K[0][2] + 0.5]
  const [fy, cy] = [K[1][1], K[1][2] + 0.5]
  const m = new THREE.Matrix4()
  m.set(
    (2 * fx) / width, 0, 1 - (2 * cx) / width, 0,
    0, (2 * fy) / height, (2 * cy) / height - 1, 0,
    0, 0, -(far + near) / (far - near), (-2 * far * near) / (far - near),
    0, 0, -1, 0,
  ) // prettier-ignore
  return m
}

/** Radial alpha ramp: opaque centre fading to transparent at the rim. */
function radialFade() {
  const size = 256
  const canvas = document.createElement("canvas")
  canvas.width = canvas.height = size
  const context = canvas.getContext("2d")!
  const gradient = context.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
  gradient.addColorStop(0, "rgba(255,255,255,1)")
  gradient.addColorStop(0.45, "rgba(255,255,255,0.55)")
  gradient.addColorStop(1, "rgba(255,255,255,0)")
  context.fillStyle = gradient
  context.fillRect(0, 0, size, size)
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  return texture
}

interface Slot {
  geometry: THREE.BufferGeometry
  depth: THREE.Mesh
  color: THREE.Mesh
  material: THREE.MeshStandardMaterial
  topology: string
}

export class ViewerEngine {
  readonly renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private root = new THREE.Group()
  private overlayCamera = new THREE.Camera()
  private orbitCamera = new THREE.PerspectiveCamera(DEFAULT_FOV, 1, 0.02, 500)
  private controls: OrbitControls
  private keyLight = new THREE.DirectionalLight(0xffffff, 2.2)
  // Overhead light whose only job is the contact shadow on the floor.
  private shadowLight = new THREE.DirectionalLight(0xffffff, 0.35)
  private floorDisc: THREE.Mesh
  private shadowCatcher: THREE.Mesh
  private grid: THREE.GridHelper
  private trajectory: THREE.Line | null = null
  private cameraPath: THREE.Line | null = null
  private frustum: THREE.LineSegments | null = null
  private slots: Slot[] = []
  private mode: ViewMode = "overlay"
  private frame = 0
  private style: MeshStyle = { showMesh: true, opacity: 0.85, color: "#9be15d" }
  private dirty = true
  private raf = 0
  private width = 1
  private height = 1
  private floor: number | null = null
  private framedFor: ViewMode | null = null
  // A fixed capture camera's single pose in the world (the 3D view of a refined result).
  private fixedPose: { rotation: THREE.Matrix3; position: THREE.Vector3 } | null | undefined
  follow = true
  /** Called when the view stops following (the user took over the camera), so the UI can show it. */
  onFollowChange?: (follow: boolean) => void

  constructor(
    canvas: HTMLCanvasElement,
    private preview: Preview,
    private store: VertexSource,
  ) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
    })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.scene.add(this.root)
    this.scene.add(new THREE.HemisphereLight(0xf2f6ff, 0x30343c, 1.25))
    this.scene.add(this.keyLight)
    this.scene.add(this.keyLight.target)
    // A soft shadow under the body is the cue that shows whether the feet touch the floor;
    // grid lines alone make a grounded body look like it hovers.
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap
    this.shadowLight.castShadow = true
    this.shadowLight.shadow.mapSize.set(2048, 2048)
    this.shadowLight.shadow.bias = -0.0004
    this.shadowLight.shadow.radius = 3
    const shadowCamera = this.shadowLight.shadow.camera
    shadowCamera.left = shadowCamera.bottom = -2.5
    shadowCamera.right = shadowCamera.top = 2.5
    shadowCamera.near = 0.5
    shadowCamera.far = 12
    this.scene.add(this.shadowLight)
    this.scene.add(this.shadowLight.target)
    // On the dark viewer a shadow needs something lighter to darken: a faint disc under
    // the body, fading out at the rim, with a shadow-only layer on top.
    const disc = new THREE.CircleGeometry(3, 96).rotateX(-Math.PI / 2)
    this.floorDisc = new THREE.Mesh(
      disc,
      new THREE.MeshBasicMaterial({
        color: 0xb4bcc8,
        map: radialFade(),
        transparent: true,
        opacity: 0.16,
        depthWrite: false,
      }),
    )
    this.shadowCatcher = new THREE.Mesh(
      disc,
      new THREE.ShadowMaterial({ opacity: 0.55, transparent: true, depthWrite: false }),
    )
    this.shadowCatcher.receiveShadow = true
    this.shadowCatcher.renderOrder = 1
    this.scene.add(this.floorDisc, this.shadowCatcher)
    this.grid = new THREE.GridHelper(GRID_SIZE, GRID_DIVISIONS, 0x5d6470, 0x3a3f48)
    ;(this.grid.material as THREE.Material).transparent = true
    ;(this.grid.material as THREE.Material).opacity = 0.55
    this.scene.add(this.grid)
    this.controls = new OrbitControls(this.orbitCamera, canvas)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.12
    this.controls.addEventListener("change", () => (this.dirty = true))
    this.controls.addEventListener("start", () => {
      if (!this.follow) return
      this.follow = false
      this.onFollowChange?.(false)
    })
    if (preview.trajectory?.length) this.buildTrajectory(preview.trajectory)
    this.buildCameraPath()
    store.onLoad = () => {
      this.dirty = true
    }
    const loop = () => {
      this.raf = requestAnimationFrame(loop)
      if (this.mode !== "overlay" && this.controls.update()) this.dirty = true
      if (this.dirty) this.draw()
    }
    loop()
  }

  /** Swap to another vertex source (e.g. Raw <-> Refined) keeping camera and view. */
  setSource(preview: Preview, store: VertexSource) {
    for (const slot of this.slots) {
      if (!slot) continue
      this.root.remove(slot.depth, slot.color)
      slot.geometry.dispose()
      slot.material.dispose()
    }
    this.slots = []
    this.preview = preview
    this.store = store
    // The floor is the lowest point seen so far (where no ground is known): start it again on
    // the new motion, or a previous result that dipped lower leaves this one hovering above it.
    this.floor = null
    this.fixedPose = undefined
    store.onLoad = () => {
      this.dirty = true
    }
    if (this.trajectory) {
      this.scene.remove(this.trajectory)
      this.trajectory.geometry.dispose()
      this.trajectory = null
    }
    if (preview.trajectory?.length) this.buildTrajectory(preview.trajectory)
    this.buildCameraPath()
    this.setFrame(this.frame)
  }

  dispose() {
    cancelAnimationFrame(this.raf)
    this.controls.dispose()
    for (const slot of this.slots) {
      slot.geometry.dispose()
      slot.material.dispose()
    }
    this.shadowCatcher.geometry.dispose()
    const discMaterial = this.floorDisc.material as THREE.MeshBasicMaterial
    discMaterial.map?.dispose()
    discMaterial.dispose()
    ;(this.shadowCatcher.material as THREE.Material).dispose()
    this.shadowLight.shadow.dispose()
    this.renderer.dispose()
  }

  setSize(width: number, height: number) {
    this.width = Math.max(1, Math.round(width))
    this.height = Math.max(1, Math.round(height))
    this.renderer.setSize(this.width, this.height, false)
    this.orbitCamera.aspect = this.width / this.height
    this.orbitCamera.updateProjectionMatrix()
    this.dirty = true
  }

  setMode(mode: ViewMode) {
    this.mode = mode
    this.controls.enabled = mode !== "overlay"
    this.follow = true
    this.onFollowChange?.(true)
    this.framedFor = null
    if (mode !== "camera3d") this.setOrbitFov(DEFAULT_FOV)
    this.dirty = true
  }

  setStyle(style: MeshStyle) {
    this.style = style
    this.dirty = true
  }

  setFrame(frame: number) {
    this.frame = Math.max(0, Math.min(frame, this.preview.frame_count - 1))
    this.dirty = true
    this.prefetch()
  }

  resetCamera() {
    this.framedFor = null
    this.follow = true
    this.dirty = true
  }

  /** Instances drawn in the current view. */
  private instances(frame = this.frame): PreviewInstance[] {
    const record = this.preview.frames[frame]
    if (!record) return []
    return (this.inWorld() ? record.world : record.instances) ?? []
  }

  private prefetch() {
    const byStream = new Map<string, number[]>()
    const end = Math.min(this.preview.frame_count, this.frame + 48)
    for (let index = this.frame; index < end; index++) {
      for (const instance of this.instances(index)) {
        const list = byStream.get(instance.stream) ?? []
        list.push(instance.slot)
        byStream.set(instance.stream, list)
      }
    }
    for (const [stream, slots] of byStream) this.store.prefetch(stream, slots)
  }

  private intrinsics(frame = this.frame) {
    for (let index = frame; index >= 0; index--) {
      const K = this.preview.frames[index]?.K
      if (K) return K
    }
    return null
  }

  private slot(index: number, topology: string) {
    let slot = this.slots[index]
    if (slot && slot.topology !== topology) {
      this.root.remove(slot.depth, slot.color)
      slot.geometry.dispose()
      slot = undefined as unknown as Slot
    }
    if (!slot) {
      const faces = this.store.faces.get(topology)!
      const geometry = new THREE.BufferGeometry()
      geometry.setIndex(new THREE.BufferAttribute(faces, 1))
      const material = new THREE.MeshStandardMaterial({
        roughness: 0.62,
        metalness: 0.04,
        side: THREE.DoubleSide,
        transparent: true,
      })
      // Depth pre-pass so a translucent mesh never shows its own back faces.
      const depth = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ colorWrite: false }))
      depth.renderOrder = 0
      const color = new THREE.Mesh(geometry, material)
      color.renderOrder = 1
      color.castShadow = true
      material.depthFunc = THREE.LessEqualDepth
      slot = { geometry, depth, color, material, topology }
      this.slots[index] = slot
      this.root.add(depth, color)
    }
    return slot
  }

  private updateMeshes(frame: number) {
    const instances = this.instances(frame)
    let complete = true
    const centroid = new THREE.Vector3()
    let counted = 0
    let minY = Infinity
    instances.forEach((instance, index) => {
      const vertices = this.store.get(instance.stream, instance.slot)
      const slot = this.slot(index, instance.topology)
      if (!vertices) {
        complete = false
        return
      }
      const attribute = slot.geometry.getAttribute("position") as THREE.BufferAttribute | undefined
      if (attribute && attribute.array.length === vertices.length) {
        ;(attribute.array as Float32Array).set(vertices)
        attribute.needsUpdate = true
      } else {
        slot.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(vertices), 3))
      }
      slot.geometry.computeVertexNormals()
      slot.geometry.computeBoundingSphere()
      const color = instance.side ? SIDE_COLORS[instance.side] : this.style.color
      slot.material.color.set(color)
      slot.material.opacity = this.mode === "overlay" ? this.style.opacity : 1
      slot.material.transparent = slot.material.opacity < 1
      slot.depth.visible = slot.color.visible = this.style.showMesh
      // Centroid / floor in display coordinates.
      const display = this.displayMatrix()
      const sample = new THREE.Vector3()
      for (let i = 0; i < vertices.length; i += 3 * 97) {
        sample.set(vertices[i], vertices[i + 1], vertices[i + 2]).applyMatrix3(display)
        centroid.add(sample)
        counted++
        if (sample.y < minY) minY = sample.y
      }
    })
    for (let index = instances.length; index < this.slots.length; index++) {
      const slot = this.slots[index]
      if (slot) slot.depth.visible = slot.color.visible = false
    }
    if (counted) centroid.divideScalar(counted)
    return { complete, centroid: counted ? centroid : null, minY }
  }

  private draw() {
    this.dirty = false
    const { centroid, minY } = this.updateMeshes(this.frame)
    const frame = this.cameraFrame()
    const moving = this.movingCamera()
    const inWorld = this.inWorld()
    if (inWorld) {
      this.root.scale.set(1, 1, 1)
      this.root.quaternion.identity()
    } else if (this.mode === "camera3d" && frame) {
      // Gravity up: the camera keeps its real tilt, the floor stays flat.
      this.root.scale.set(1, 1, 1)
      this.root.quaternion.setFromRotationMatrix(new THREE.Matrix4().setFromMatrix3(frame))
    } else {
      this.root.scale.copy(CV_TO_GL)
      this.root.quaternion.identity()
    }

    if (this.mode === "overlay") {
      const K = this.intrinsics()
      const [width, height] = this.preview.image_size
      if (K) {
        this.overlayCamera.projectionMatrix.copy(intrinsicsProjection(K, width, height))
        this.overlayCamera.projectionMatrixInverse.copy(this.overlayCamera.projectionMatrix).invert()
      }
      this.keyLight.position.set(-0.6, 0.8, 1.5)
      this.keyLight.target.position.set(0, 0, -3)
      this.grid.visible = false
      this.floorDisc.visible = this.shadowCatcher.visible = false
      if (this.trajectory) this.trajectory.visible = false
      if (this.cameraPath) this.cameraPath.visible = false
      if (this.frustum) this.frustum.visible = false
      this.scene.background = null
      this.renderer.render(this.scene, this.overlayCamera)
      return
    }

    // Orbit views
    // A known floor (anchored worlds: y=0) beats the lowest point seen so far,
    // which sits below the soles whenever a frame dips slightly.
    const ground = inWorld ? this.preview.ground : null
    if (centroid && Number.isFinite(minY)) {
      if (this.floor == null || this.framedFor !== this.mode) this.floor = minY
      this.floor = Math.min(this.floor, minY)
      if (this.framedFor !== this.mode) {
        this.framedFor = this.mode
        this.floor = minY
        if (this.mode === "camera3d") {
          this.frameCaptureCamera(centroid)
        } else {
          const target = centroid.clone()
          this.orbitCamera.position.copy(target).add(this.worldViewOffset())
          this.controls.target.copy(target)
          this.controls.update()
        }
      } else if (this.follow && this.mode === "world") {
        const delta = centroid.clone().sub(this.controls.target)
        this.controls.target.add(delta)
        this.orbitCamera.position.add(delta)
      } else if (this.follow && moving) {
        // Locked to the capture camera: ride along with it.
        this.frameCaptureCamera(centroid)
      }
    }
    if (ground != null) this.floor = ground
    this.grid.visible = true
    // Fixed in the world: the grid recentres near the view only in whole cells, so its lines
    // never slide while orbiting or following.
    const cell = GRID_SIZE / GRID_DIVISIONS
    this.grid.position.set(
      Math.round(this.controls.target.x / cell) * cell,
      (this.floor ?? 0) - 0.002,
      Math.round(this.controls.target.z / cell) * cell,
    )
    this.floorDisc.visible = this.shadowCatcher.visible = true
    // Light from almost straight above the body, so the shadow sits right under the feet.
    const over = centroid ?? this.controls.target
    this.floorDisc.position.set(over.x, (this.floor ?? 0) - 0.001, over.z)
    this.shadowCatcher.position.copy(this.floorDisc.position)
    this.shadowLight.target.position.set(over.x, this.floor ?? 0, over.z)
    this.shadowLight.position.set(over.x + 0.35, (this.floor ?? 0) + 6, over.z + 0.6)
    // Looking through a moving camera, its own frustum and path would only clutter the frame edges.
    const throughCamera = moving && this.follow
    if (this.cameraPath) {
      this.cameraPath.visible = moving && !throughCamera
    }
    if (this.trajectory) {
      this.trajectory.visible = inWorld
      this.trajectory.position.y = (this.floor ?? 0) + 0.004
      this.trajectory.geometry.setDrawRange(0, this.frame + 1)
    }
    if (this.mode === "camera3d" && !throughCamera) this.updateFrustum()
    else if (this.frustum) this.frustum.visible = false
    this.keyLight.position.copy(this.orbitCamera.position).add(new THREE.Vector3(1, 2.5, 1))
    this.keyLight.target.position.copy(this.controls.target)
    this.scene.background = null
    this.renderer.render(this.scene, this.orbitCamera)
  }

  private buildTrajectory(points: number[][]) {
    const positions = new Float32Array(points.length * 3)
    points.forEach(([x, , z], index) => positions.set([x, 0, z], index * 3))
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))
    this.trajectory = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color: 0x5fb2ff, transparent: true, opacity: 0.9 }),
    )
    this.scene.add(this.trajectory)
  }

  /** The capture camera's path (smoothed), drawn in the 3D view when the camera moves. */
  private buildCameraPath() {
    if (this.cameraPath) {
      this.scene.remove(this.cameraPath)
      this.cameraPath.geometry.dispose()
      ;(this.cameraPath.material as THREE.Material).dispose()
      this.cameraPath = null
    }
    const points = this.preview.camera_path?.display_position
    if (!points?.length) return
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(points.flat()), 3))
    this.cameraPath = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color: 0xffc857, transparent: true, opacity: 0.7 }),
    )
    this.cameraPath.visible = false
    this.scene.add(this.cameraPath)
  }

  /**
   * True in the 3D view for a moving camera (ViPE or background estimate): the scene is the
   * world and the camera moves through it.
   */
  private movingCamera() {
    const mode = this.preview.camera?.mode
    return this.mode === "camera3d" && !!this.preview.camera_path && !!mode && mode !== "static"
  }

  /**
   * Whether the current view draws the world copy of the motion. World always; 3D for a
   * moving camera, and for a fixed one once the result is refined (a known floor): the floor,
   * planting and contact fixes live in the world copy only, so the camera-space copy would
   * bob against the floor. A raw fixed-camera result keeps the camera-centred copy, which
   * GEM's integrated world can drift away from.
   */
  private inWorld() {
    if (this.mode === "world" || this.movingCamera()) return true
    return this.mode === "camera3d" && this.preview.ground != null && !!this.staticPose()
  }

  /**
   * One pose for a fixed camera: the median of its per-frame fit (which also absorbs the
   * refine's small per-frame body shifts) and the mean of its rotations.
   */
  private staticPose() {
    if (this.fixedPose !== undefined) return this.fixedPose
    const path = this.preview.camera_path
    this.fixedPose = null
    if (!path?.rotation?.length || !path.position?.length) return null
    const median = (axis: number) => {
      const values = path.position.map((point) => point[axis]).sort((a, b) => a - b)
      return values[Math.floor(values.length / 2)]
    }
    const sum = new THREE.Quaternion(0, 0, 0, 0)
    let first: THREE.Quaternion | null = null
    for (const rows of path.rotation) {
      const q = new THREE.Quaternion().setFromRotationMatrix(
        new THREE.Matrix4().setFromMatrix3(
          new THREE.Matrix3().set(...(rows as [number, number, number, number, number, number, number, number, number])),
        ),
      )
      first ??= q.clone()
      const sign = first.dot(q) < 0 ? -1 : 1
      sum.set(sum.x + sign * q.x, sum.y + sign * q.y, sum.z + sign * q.z, sum.w + sign * q.w)
    }
    const rotation = new THREE.Matrix3().setFromMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(sum.normalize()))
    this.fixedPose = { rotation, position: new THREE.Vector3(median(0), median(1), median(2)) }
    return this.fixedPose
  }

  /** Where the capture camera sits in the drawn scene (null: the camera-centred origin). */
  private capturePose(display = false) {
    if (this.movingCamera()) return this.cameraPose(this.frame, display)
    return this.inWorld() && this.mode === "camera3d" ? this.staticPose() : null
  }

  /** Capture-camera pose at ``frame``: exact, or the smoothed copy used for drawing. */
  private cameraPose(frame: number, display = false) {
    const path = this.preview.camera_path
    const rotation = (display ? path?.display_rotation : path?.rotation)?.[frame]
    const position = (display ? path?.display_position : path?.position)?.[frame]
    if (!rotation || !position) return null
    return {
      rotation: new THREE.Matrix3().set(
        ...(rotation as [number, number, number, number, number, number, number, number, number]),
      ),
      position: new THREE.Vector3(position[0], position[1], position[2]),
    }
  }

  /** Three-quarter view from the subject's front (35° off her facing), slightly above. */
  private worldViewOffset() {
    const facing = this.preview.facing?.[this.frame]
    if (!facing) return new THREE.Vector3(2.4, 1.5, 3.2)
    const front = new THREE.Vector3(facing[0], 0, facing[1]).normalize()
    const side = new THREE.Vector3(0, 1, 0).cross(front)
    const angle = THREE.MathUtils.degToRad(35)
    return front
      .multiplyScalar(Math.cos(angle))
      .add(side.multiplyScalar(Math.sin(angle)))
      .multiplyScalar(4)
      .setY(1.5)
  }

  private setOrbitFov(fov: number) {
    if (this.orbitCamera.fov === fov) return
    this.orbitCamera.fov = fov
    this.orbitCamera.updateProjectionMatrix()
  }

  /**
   * Look through the capture camera: same position (the origin), direction and
   * vertical field of view as the video, orbiting around the subject's depth.
   */
  private frameCaptureCamera(centroid: THREE.Vector3) {
    // A moving camera is placed at its exact pose for this frame, so the subject lines up
    // with the video; a fixed one at its pose in the world, or the camera-centred origin.
    const pose = this.capturePose()
    const origin = pose?.position ?? new THREE.Vector3()
    const forward = new THREE.Vector3(0, 0, 1).applyMatrix3(pose?.rotation ?? this.displayMatrix()).normalize()
    const distance = Math.max(0.5, centroid.clone().sub(origin).dot(forward))
    const K = this.intrinsics()
    if (K) this.setOrbitFov(THREE.MathUtils.radToDeg(2 * Math.atan(this.preview.image_size[1] / 2 / K[1][1])))
    this.orbitCamera.position.copy(origin)
    this.controls.target.copy(origin).addScaledVector(forward, distance)
    this.orbitCamera.lookAt(this.controls.target)
    this.controls.update()
  }

  /** Camera-to-display rotation for the 3D view (levelled), when the result provides one. */
  private cameraFrame(): THREE.Matrix3 | null {
    const rows = this.preview.camera_frame
    if (!rows) return null
    return new THREE.Matrix3().set(...(rows.flat() as [number, number, number, number, number, number, number, number, number]))
  }

  /** Maps stored vertex coordinates to the current view's display coordinates. */
  private displayMatrix(): THREE.Matrix3 {
    if (this.inWorld()) return IDENTITY
    if (this.mode === "camera3d") return this.cameraFrame() ?? CV_TO_GL_MATRIX
    return CV_TO_GL_MATRIX
  }

  private updateFrustum() {
    const K = this.intrinsics()
    if (!K) return
    if (!this.frustum) {
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(18 * 3), 3))
      this.frustum = new THREE.LineSegments(
        geometry,
        new THREE.LineBasicMaterial({ color: 0xffc857, transparent: true, opacity: 0.85 }),
      )
      this.scene.add(this.frustum)
    }
    const [width, height] = this.preview.image_size
    const depth = 0.6
    // A moving camera sits at its (smoothed) pose; a fixed one at its pose or the origin.
    const pose = this.capturePose(true)
    const rotation = pose?.rotation ?? this.displayMatrix()
    const origin = pose?.position ?? new THREE.Vector3()
    const corners = [
      [0, 0],
      [width, 0],
      [width, height],
      [0, height],
    ].map(([u, v]) =>
      new THREE.Vector3(((u - K[0][2]) / K[0][0]) * depth, ((v - K[1][2]) / K[1][1]) * depth, depth)
        .applyMatrix3(rotation)
        .add(origin),
    )
    // Plus a drop line to the floor: shows the camera's height above the ground.
    const foot = new THREE.Vector3(origin.x, this.floor ?? 0, origin.z)
    const segments = [
      origin, corners[0], origin, corners[1], origin, corners[2], origin, corners[3],
      corners[0], corners[1], corners[1], corners[2], corners[2], corners[3], corners[3], corners[0],
      origin, foot,
    ] // prettier-ignore
    const attribute = this.frustum.geometry.getAttribute("position") as THREE.BufferAttribute
    segments.forEach((point, index) => attribute.setXYZ(index, point.x, point.y, point.z))
    attribute.needsUpdate = true
    this.frustum.visible = true
  }

  // -- export helpers -----------------------------------------------------
  /** Load every mesh for ``frame`` and draw synchronously (used by video export). */
  async renderFrameNow(frame: number) {
    this.frame = frame
    await Promise.all(this.instances(frame).map((item) => this.store.ensure(item.stream, item.slot)))
    this.prefetch()
    this.draw()
  }

  /** Current-frame meshes in Y-up metres, as GLB or OBJ. */
  async exportFrame(format: "glb" | "obj"): Promise<Blob> {
    const scene = new THREE.Scene()
    const instances = this.instances()
    for (const instance of instances) {
      const vertices = await this.store.ensure(instance.stream, instance.slot)
      const geometry = new THREE.BufferGeometry()
      const positions = new Float32Array(vertices)
      if (instance.stream.startsWith("camera_")) {
        for (let i = 0; i < positions.length; i += 3) {
          positions[i + 1] *= -1
          positions[i + 2] *= -1
        }
      }
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))
      geometry.setIndex(new THREE.BufferAttribute(this.store.faces.get(instance.topology)!, 1))
      geometry.computeVertexNormals()
      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({
          color: instance.side ? SIDE_COLORS[instance.side] : this.style.color,
          roughness: 0.6,
        }),
      )
      mesh.name = instance.label.replace(/\s+/g, "_")
      scene.add(mesh)
    }
    if (format === "obj") {
      return new Blob([new OBJExporter().parse(scene)], { type: "text/plain" })
    }
    const result = await new GLTFExporter().parseAsync(scene, { binary: true })
    return new Blob([result as ArrayBuffer], { type: "model/gltf-binary" })
  }
}
