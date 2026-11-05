import { request, urls, type Preview, type PreviewVariant } from "@/lib/api"

const CHUNK = 32
const MAX_CHUNKS = 48 // per store; ~350 MB worst case for 20k-vertex bodies

/** Where the viewer gets each frame's vertices: streamed (MeshStore) or skinned (SkinnedStore). */
export interface VertexSource {
  readonly faces: Map<string, Uint32Array>
  onLoad: (() => void) | null
  loadFaces(): Promise<void>
  vertexCount(stream: string): number
  /** Vertices for one slot if available; otherwise starts loading and returns null. */
  get(stream: string, slot: number): Float32Array | null
  ensure(stream: string, slot: number): Promise<Float32Array>
  prefetch(stream: string, slots: Iterable<number>): void
}

/** Chunked, LRU-cached vertex streaming for one job's preview bundle. */
export class MeshStore implements VertexSource {
  readonly faces = new Map<string, Uint32Array>()
  onLoad: (() => void) | null = null
  private cache = new Map<string, Float32Array>()
  private pending = new Map<string, Promise<Float32Array>>()
  private jobId: string
  private preview: Preview
  private variant: PreviewVariant

  constructor(jobId: string, preview: Preview, variant: PreviewVariant = "raw") {
    this.jobId = jobId
    this.preview = preview
    this.variant = variant
  }

  async loadFaces() {
    await Promise.all(
      this.preview.topologies.map(async (topology) => {
        const buffer = await request<ArrayBuffer>(urls.faces(this.jobId, topology, this.variant))
        this.faces.set(topology, new Uint32Array(new Int32Array(buffer)))
      }),
    )
  }

  vertexCount(stream: string) {
    return this.preview.streams[stream]?.vertex_count ?? 0
  }

  /** Vertices for one slot if cached; otherwise starts loading and returns null. */
  get(stream: string, slot: number): Float32Array | null {
    const chunk = Math.floor(slot / CHUNK)
    const key = `${stream}:${chunk}`
    const data = this.cache.get(key)
    if (!data) {
      void this.load(stream, chunk)
      return null
    }
    // Refresh LRU position.
    this.cache.delete(key)
    this.cache.set(key, data)
    return this.slice(stream, data, slot - chunk * CHUNK)
  }

  async ensure(stream: string, slot: number): Promise<Float32Array> {
    const chunk = Math.floor(slot / CHUNK)
    const data = this.cache.get(`${stream}:${chunk}`) ?? (await this.load(stream, chunk))
    return this.slice(stream, data, slot - chunk * CHUNK)
  }

  prefetch(stream: string, slots: Iterable<number>) {
    const chunks = new Set<number>()
    for (const slot of slots) chunks.add(Math.floor(slot / CHUNK))
    for (const chunk of chunks) {
      if (!this.cache.has(`${stream}:${chunk}`)) void this.load(stream, chunk)
    }
  }

  private slice(stream: string, data: Float32Array, local: number) {
    const size = this.vertexCount(stream) * 3
    return data.subarray(local * size, (local + 1) * size)
  }

  private load(stream: string, chunk: number): Promise<Float32Array> {
    const key = `${stream}:${chunk}`
    let promise = this.pending.get(key)
    if (promise) return promise
    promise = request<ArrayBuffer>(urls.mesh(this.jobId, stream, chunk * CHUNK, CHUNK, this.variant))
      .then((buffer) => {
        const data = new Float32Array(buffer)
        this.cache.set(key, data)
        while (this.cache.size > MAX_CHUNKS) {
          this.cache.delete(this.cache.keys().next().value!)
        }
        this.onLoad?.()
        return data
      })
      .finally(() => this.pending.delete(key))
    this.pending.set(key, promise)
    return promise
  }
}
