import { request, urls, type Preview, type PreviewVariant } from "@/lib/api"

import type { VertexSource } from "./mesh-store"

const CACHE_FRAMES = 16 // skinned frames kept (both streams); skinning one takes ~1 ms

/**
 * Vertices for a skinned preview: one bind mesh posed per frame by linear-blend skinning
 * (4 joints per vertex) in the browser, instead of streaming every frame's vertices.
 * World frames use the per-frame joint skin matrices; the camera stream is the same
 * body under that frame's world-to-camera transform.
 */
export class SkinnedStore implements VertexSource {
  readonly faces = new Map<string, Uint32Array>()
  onLoad: (() => void) | null = null
  private cache = new Map<string, Float32Array>()
  private data: {
    rest: Float32Array
    joints: Uint8Array
    weights: Float32Array
    world: Float32Array
    camera: Float32Array
  } | null = null
  private loading: Promise<void> | null = null
  private jobId: string
  private preview: Preview
  private variant: PreviewVariant

  constructor(jobId: string, preview: Preview, variant: PreviewVariant) {
    this.jobId = jobId
    this.preview = preview
    this.variant = variant
  }

  async loadFaces() {
    await Promise.all([
      ...this.preview.topologies.map(async (topology) => {
        const buffer = await request<ArrayBuffer>(urls.faces(this.jobId, topology, this.variant))
        this.faces.set(topology, new Uint32Array(new Int32Array(buffer)))
      }),
      this.load(),
    ])
  }

  vertexCount() {
    return this.preview.skinned?.vertices ?? 0
  }

  get(stream: string, slot: number): Float32Array | null {
    if (!this.data) {
      void this.load()
      return null
    }
    return this.pose(stream, slot)
  }

  async ensure(stream: string, slot: number): Promise<Float32Array> {
    await this.load()
    return this.pose(stream, slot)
  }

  prefetch() {
    void this.load()
  }

  private load() {
    this.loading ??= (async () => {
      const fetchBinary = (name: string) => request<ArrayBuffer>(urls.modelSkin(this.jobId, name, this.variant))
      const [rest, joints, weights, world, camera] = await Promise.all(
        ["rest", "joints", "weights", "world", "camera"].map(fetchBinary),
      )
      this.data = {
        rest: new Float32Array(rest),
        joints: new Uint8Array(joints),
        weights: new Float32Array(weights),
        world: new Float32Array(world),
        camera: new Float32Array(camera),
      }
      this.onLoad?.()
    })()
    return this.loading
  }

  private pose(stream: string, slot: number): Float32Array {
    const key = `${stream}:${slot}`
    const cached = this.cache.get(key)
    if (cached) {
      this.cache.delete(key)
      this.cache.set(key, cached)
      return cached
    }
    const { rest, joints, weights, world, camera } = this.data!
    const count = this.vertexCount()
    const jointCount = this.preview.skinned!.joints
    const base = slot * jointCount * 12
    const out = new Float32Array(count * 3)
    for (let v = 0; v < count; v++) {
      const x = rest[v * 3]
      const y = rest[v * 3 + 1]
      const z = rest[v * 3 + 2]
      let px = 0
      let py = 0
      let pz = 0
      for (let k = 0; k < 4; k++) {
        const w = weights[v * 4 + k]
        if (w === 0) continue
        const m = base + joints[v * 4 + k] * 12
        px += w * (world[m] * x + world[m + 1] * y + world[m + 2] * z + world[m + 3])
        py += w * (world[m + 4] * x + world[m + 5] * y + world[m + 6] * z + world[m + 7])
        pz += w * (world[m + 8] * x + world[m + 9] * y + world[m + 10] * z + world[m + 11])
      }
      out[v * 3] = px
      out[v * 3 + 1] = py
      out[v * 3 + 2] = pz
    }
    if (stream.startsWith("camera")) {
      const c = slot * 12
      for (let v = 0; v < count; v++) {
        const x = out[v * 3]
        const y = out[v * 3 + 1]
        const z = out[v * 3 + 2]
        out[v * 3] = camera[c] * x + camera[c + 1] * y + camera[c + 2] * z + camera[c + 3]
        out[v * 3 + 1] = camera[c + 4] * x + camera[c + 5] * y + camera[c + 6] * z + camera[c + 7]
        out[v * 3 + 2] = camera[c + 8] * x + camera[c + 9] * y + camera[c + 10] * z + camera[c + 11]
      }
    }
    this.cache.set(key, out)
    while (this.cache.size > CACHE_FRAMES) this.cache.delete(this.cache.keys().next().value!)
    return out
  }
}
