// Stations: side platforms with coping and a yellow line, canopies on posts, lamps, a station building with a
// clock and a name board and, at city termini, a glazed train shed and a clock tower. Every piece is a box (or a
// barrel vault) spanning a stretch [s0, s1] of its route with lateral and vertical extents, so it follows the
// track; the vertex shader lifts it onto the rail bed with one terrain() call. The timetable is in trains.js.
import * as THREE from 'three'
import { R } from '../config.js'
import { hash2i } from './spec.js'
import { objUniforms } from './objects.glsl.js'
import { STATION_VERT, STATION_FRAG } from './stations.glsl.js'

const RANGE = 6500
const MAX = [6000, 400]                                 // box and vault instances
export const PIECE = { PLATFORM: 0, ROOF: 1, POST: 2, LAMP: 3, WALL: 4, CLOCK: 5, SIGN: 6, VAULT: 7 }

// unit shapes: x along the track 0..1, y across 0..1, z up 0..1, normals in unit space
function boxGeometry() {
  const pos = [], nrm = [], idx = []
  for (const [a, s] of [[0, 1], [0, -1], [1, 1], [1, -1], [2, 1], [2, -1]]) {
    const u = (a + 1) % 3, v = (a + 2) % 3, base = pos.length / 3
    for (const [i, j] of [[0, 0], [1, 0], [1, 1], [0, 1]]) {
      const p = [0, 0, 0], n = [0, 0, 0]
      p[a] = s > 0 ? 1 : 0; p[u] = i; p[v] = j; n[a] = s
      pos.push(...p); nrm.push(...n)
    }
    idx.push(...(s > 0 ? [0, 1, 2, 0, 2, 3] : [0, 2, 1, 0, 3, 2]).map(k => k + base))
  }
  return finish(pos, nrm, idx)
}
function vaultGeometry(n = 14) {
  const pos = [], nrm = [], idx = []
  for (let i = 0; i <= n; i++) {
    const a = Math.PI * i / n, y = 0.5 - 0.5 * Math.cos(a), z = Math.sin(a), l = Math.hypot(2 * Math.cos(a), Math.sin(a))
    for (const x of [0, 1]) { pos.push(x, y, z); nrm.push(0, -2 * Math.cos(a) / l, Math.sin(a) / l) }
    if (i) idx.push(2 * i - 2, 2 * i - 1, 2 * i + 1, 2 * i - 2, 2 * i + 1, 2 * i)
  }
  return finish(pos, nrm, idx)
}
function finish(pos, nrm, idx) {
  const g = new THREE.InstancedBufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3))
  g.setIndex(idx)
  return g
}

export class Stations {
  constructor(ctx, list) {
    this.world = ctx.world
    this.byRoute = new Map()
    for (const st of list) { if (!this.byRoute.has(st.ri)) this.byRoute.set(st.ri, []); this.byRoute.get(st.ri).push(st) }
    const uniforms = { ...objUniforms(ctx.shared, ctx.FAR), uCamPos: { value: new THREE.Vector3() }, uHour: { value: 0 } }
    this.uniforms = uniforms
    this.meshes = [boxGeometry(), vaultGeometry()].map((geo, k) => {
      const material = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: STATION_VERT, fragmentShader: STATION_FRAG, uniforms, side: k ? THREE.DoubleSide : THREE.FrontSide })
      const attr = (name, size) => {
        const a = new THREE.InstancedBufferAttribute(new Float32Array(MAX[k] * size), size)
        a.setUsage(THREE.DynamicDrawUsage)
        geo.setAttribute(name, a)
        return a
      }
      const m = { A: attr('iA', 3), B: attr('iB', 3), X: attr('iBox', 4), M: attr('iMisc', 3), n: 0, mesh: new THREE.Mesh(geo, material), max: MAX[k] }
      m.mesh.frustumCulled = false
      m.mesh.visible = false
      geo.instanceCount = 0
      ctx.scene.add(m.mesh)
      return m
    })
  }

  // the pieces of one station, built the first time it comes near: { a, b (sea-level planet xyz), box [y0, y1, z0, z1], kind, vault }
  pieces(st) {
    const w = this.world, out = [], L = st.len, s0 = st.s - L / 2, big = st.big, side = st.side
    const at = s => w.routePoint(st.route, s).dir.map(v => v * R)
    const add = (sa, sb, y0, y1, z0, z1, kind, vault = false) => out.push({ a: at(sa), b: at(sb), box: [Math.min(y0, y1), Math.max(y0, y1), z0, z1], kind, vault })
    const along = (sa, sb, step, fn) => { const n = Math.max(1, Math.round((sb - sa) / step)); for (let i = 0; i < n; i++) fn(sa + (sb - sa) * i / n, sa + (sb - sa) * (i + 1) / n) }
    const halt = st.kind === 'halt'
    for (const sd of [-1, 1]) {
      along(s0, s0 + L, 20, (a, b) => add(a, b, sd * 1.72, sd * 5.2, -1.8, 1.0, PIECE.PLATFORM))
      for (let x = 10; x < L - 4; x += 24) {            // lamps
        add(s0 + x - 0.08, s0 + x + 0.08, sd * 4.62, sd * 4.78, 1.0, 5.0, PIECE.POST)
        add(s0 + x - 0.18, s0 + x + 0.18, sd * 4.1, sd * 4.8, 4.8, 5.05, PIECE.LAMP)
      }
      if (big) {                                         // shed columns
        for (let x = 0; x <= L + 0.1; x += 20) add(s0 + x - 0.3, s0 + x + 0.3, sd * 7.1, sd * 7.7, -1.8, 7.0, PIECE.POST)
        continue
      }
      const cl = halt ? 14 : Math.min(L * 0.55, 90), c0 = st.s - cl / 2      // canopy on posts
      along(c0, c0 + cl, 20, (a, b) => add(a, b, sd * 2.0, sd * 5.4, 3.7, 3.95, PIECE.ROOF))
      for (let x = 3; x < cl; x += 10) add(c0 + x - 0.12, c0 + x + 0.12, sd * 4.28, sd * 4.52, 1.0, 3.7, PIECE.POST)
    }
    if (big) along(s0, s0 + L, 20, (a, b) => add(a, b, -7.4, 7.4, 7.0, 15.5, PIECE.VAULT, true))
    // station building beside the platform on `side`: hall, roof slab, clock and name board facing the track
    const u = k => hash2i(st.seed, k, 1), y0 = big ? 7.9 : 5.3
    const depth = big ? 16 : halt ? 5 + 2 * u(4) : 8 + 4 * u(4), H = big ? 15 : halt ? 4.2 + u(5) : 6.5 + 2 * u(5)
    const bl = big ? 80 : halt ? 9 + 5 * u(6) : 22 + 12 * u(6), b0 = st.s + (halt ? 18 : -bl / 2)
    add(b0, b0 + bl, side * y0, side * (y0 + depth), -6, H, PIECE.WALL)
    add(b0 - 1, b0 + bl + 1, side * (y0 - 0.8), side * (y0 + depth + 0.8), H, H + 0.6, PIECE.ROOF)
    const m = b0 + bl / 2, cs = Math.min(2.2, H - 3.2)
    if (!halt) add(m - cs / 2, m + cs / 2, side * (y0 - 0.25), side * y0, H - 0.8 - cs, H - 0.8, PIECE.CLOCK)
    add(halt ? m - 3.5 : m + 3, halt ? m + 3.5 : m + 12, side * (y0 - 0.2), side * y0, 2.9, 3.6, PIECE.SIGN)
    if (big) {                                           // clock tower at the city end of the hall
      const t0 = b0 - 10
      add(t0, t0 + 10, side * (y0 + 2), side * (y0 + 12), -6, 36, PIECE.WALL)
      add(t0 - 0.3, t0 + 10.3, side * (y0 + 1.7), side * (y0 + 12.3), 29, 33.5, PIECE.CLOCK)
      add(t0 - 0.8, t0 + 10.8, side * (y0 + 1.2), side * (y0 + 12.8), 36, 36.8, PIECE.ROOF)
    }
    return out
  }

  update(frame, nearRoutes, fwd) {
    for (const m of this.meshes) m.n = 0
    const cam = frame.camPos
    this.uniforms.uCamPos.value.copy(cam)
    this.uniforms.uHour.value = (((frame.simTime / 3600 + frame.geo.lon * 12 / Math.PI) % 24) + 24) % 24
    for (const [ri] of nearRoutes) for (const st of this.byRoute.get(ri) ?? []) {
      const p = st.pos ??= this.world.routePoint(st.route, st.s).dir.map(v => v * R)
      const d = [p[0] - cam.x, p[1] - cam.y, p[2] - cam.z]
      if (Math.hypot(...d) > RANGE + st.len / 2 || d[0] * fwd.x + d[1] * fwd.y + d[2] * fwd.z < -st.len) continue
      for (const pc of st.parts ??= this.pieces(st)) {
        const m = this.meshes[pc.vault ? 1 : 0]
        if (m.n >= m.max) break
        const i = m.n++
        m.A.array.set([pc.a[0] - cam.x, pc.a[1] - cam.y, pc.a[2] - cam.z], i * 3)
        m.B.array.set([pc.b[0] - cam.x, pc.b[1] - cam.y, pc.b[2] - cam.z], i * 3)
        m.X.array.set(pc.box, i * 4)
        m.M.array.set([pc.kind, st.seed % 4096, st.side], i * 3)
      }
    }
    for (const m of this.meshes) {
      m.mesh.visible = m.n > 0
      m.mesh.geometry.instanceCount = m.n
      if (m.n) for (const a of [m.A, m.B, m.X, m.M]) { a.clearUpdateRanges(); a.addUpdateRange(0, m.n * a.itemSize); a.needsUpdate = true }
    }
  }
}
