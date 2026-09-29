// Batch terrain sampler: evaluates the GPU height field at many directions in one pass and reads
// the results back. Per point: [height, land, mountain mask, roughness, slope, temperature, moisture, 0].
import * as THREE from 'three'
import { GLSL_HEAD } from '../config.js'
import { NOISE } from '../glsl/noise.js'
import { TERRAIN } from '../glsl/terrain.glsl.js'

const COLS = 1024            // points per row (two output texels each)
const MAX_ROWS = 256         // points per pass = COLS * MAX_ROWS

const FRAG = /* glsl */`${GLSL_HEAD}
uniform highp sampler2D uDirs;
uniform float uMaxF;
uniform int uCount, uUseCiv;
out vec4 outS;
${NOISE}
${TERRAIN}
void main(){
  ivec2 px = ivec2(gl_FragCoord.xy);
  int idx = px.y*${COLS} + px.x/2;
  if (idx >= uCount){ outS = vec4(0.0); return; }
  vec3 p = normalize(texelFetch(uDirs, ivec2(px.x/2, px.y), 0).xyz);
  vec3 info;
  vec4 t = terrain(p, uMaxF, info);     // before the world exists the civ lookup is empty
  if ((px.x & 1) == 0) outS = vec4(t.x, info);
  else {
    vec3 g = t.yzw - p*dot(t.yzw, p);
    outS = vec4(length(g)/PLANET_R, climate(p, t.x, info.x), 0.0);
  }
}`

export class TerrainSampler {
  constructor(renderer, shared) {
    this.renderer = renderer
    this.dirTex = new THREE.DataTexture(new Float32Array(COLS * MAX_ROWS * 4), COLS, MAX_ROWS, THREE.RGBAFormat, THREE.FloatType)
    this.target = new THREE.WebGLRenderTarget(COLS * 2, MAX_ROWS, { type: THREE.FloatType, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter })
    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: shared.quadVert, fragmentShader: FRAG,
      uniforms: { uDirs: { value: this.dirTex }, uMaxF: { value: 1e6 }, uCount: { value: 0 }, uUseCiv: { value: 0 }, ...shared.civ },
    })
    this.quad = shared.makeQuad(this.material)
    this.waiting = []            // async requests not yet drawn
    this.busy = false
  }

  // copy n directions (from 'start' in dirs) into the direction texture from row 'row' on
  put(dirs, start, n, row) {
    const src = this.dirTex.image.data, o = row * COLS * 4
    for (let i = 0; i < n; i++) {
      const j = (start + i) * 3, k = o + i * 4
      src[k] = dirs[j]; src[k + 1] = dirs[j + 1]; src[k + 2] = dirs[j + 2]
    }
  }

  // one pass over groups [{ row, n, maxF, useCiv }] (each from its own row), then the target holds 2 texels per point
  draw(groups) {
    this.dirTex.needsUpdate = true
    const u = this.material.uniforms, prev = this.renderer.getRenderTarget()
    for (const g of groups) {
      u.uMaxF.value = g.maxF; u.uCount.value = g.row * COLS + g.n; u.uUseCiv.value = g.useCiv ? 1 : 0
      this.target.viewport.set(0, g.row, COLS * 2, Math.ceil(g.n / COLS))
      this.renderer.setRenderTarget(this.target)   // (applies the viewport)
      this.renderer.render(this.quad.scene, this.quad.camera)
    }
    this.renderer.setRenderTarget(prev)
  }

  // synchronous: dirs is a flat xyz array; returns Float32Array(n * 8)
  sample(dirs, maxF = 1.2e6, useCiv = false) {
    const t0 = performance.now()
    const total = dirs.length / 3, out = new Float32Array(total * 8)
    const buf = new Float32Array(COLS * 2 * MAX_ROWS * 4)
    for (let start = 0; start < total; start += COLS * MAX_ROWS) {
      const n = Math.min(COLS * MAX_ROWS, total - start), rows = Math.ceil(n / COLS)
      this.put(dirs, start, n, 0)
      this.draw([{ row: 0, n, maxF, useCiv }])
      this.renderer.readRenderTargetPixels(this.target, 0, 0, COLS * 2, rows, buf)
      out.set(buf.subarray(0, n * 8), start * 8)
    }
    this.renderer.setRenderTarget(null)
    console.info(`sampler: ${total} points in ${Math.round(performance.now() - t0)} ms`)
    return out
  }

  // asynchronous version for runtime use (building heights etc.). Requests made while a pass is in flight wait and
  // then go together: one pass (each request in its own rows, with its own maxF) and one readback for all of them,
  // so a module's small requests don't queue a GPU round trip each in front of another's
  sampleAsync(dirs, maxF = 1.2e6, useCiv = true) {
    const p = new Promise((resolve, reject) => this.waiting.push({ dirs, maxF, useCiv, resolve, reject }))
    if (!this.busy) { this.busy = true; queueMicrotask(() => this.pump()) }   // after the frame's other requests
    return p
  }

  async pump() {
    while (this.waiting.length) {
      const batch = []
      let rows = 0
      while (this.waiting.length) {
        const need = Math.ceil(this.waiting[0].dirs.length / 3 / COLS)
        if (batch.length && rows + need > MAX_ROWS) break
        batch.push(this.waiting.shift())
        rows += need
      }
      try {
        if (rows > MAX_ROWS) batch[0].resolve(await this.large(batch[0]))   // one request bigger than a pass: in chunks
        else {
          let row = 0
          const groups = batch.map(r => { const n = r.dirs.length / 3, g = { row, n, maxF: r.maxF, useCiv: r.useCiv }; this.put(r.dirs, 0, n, row); row += Math.ceil(n / COLS); return g })
          this.draw(groups)
          const buf = new Float32Array(COLS * 2 * Math.max(rows, 1) * 4)
          await this.renderer.readRenderTargetPixelsAsync(this.target, 0, 0, COLS * 2, Math.max(rows, 1), buf)
          groups.forEach((g, i) => batch[i].resolve(buf.slice(g.row * COLS * 8, g.row * COLS * 8 + g.n * 8)))
        }
      } catch (e) { for (const r of batch) r.reject(e) }
    }
    this.busy = false
  }

  async large({ dirs, maxF, useCiv }) {
    const total = dirs.length / 3, out = new Float32Array(total * 8)
    for (let start = 0; start < total; start += COLS * MAX_ROWS) {
      const n = Math.min(COLS * MAX_ROWS, total - start), rows = Math.ceil(n / COLS), buf = new Float32Array(COLS * 2 * rows * 4)
      this.put(dirs, start, n, 0)
      this.draw([{ row: 0, n, maxF, useCiv }])
      await this.renderer.readRenderTargetPixelsAsync(this.target, 0, 0, COLS * 2, rows, buf)
      out.set(buf.subarray(0, n * 8), start * 8)
    }
    return out
  }
}
