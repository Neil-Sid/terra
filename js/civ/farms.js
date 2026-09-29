// Farms: season and precision uniforms for the field painting (farm.glsl.js), and farmsteads (house, barns,
// sheds, silos, grain bins) on the farmyards the painting leaves in some parcel corners. The district and
// parcel layout mirrors farmPaint() exactly; notes/farm.md lists the shared hashes. Contracts in notes/CIV.md.
import * as THREE from 'three'
import { R } from '../config.js'
import { hash2i, HALF } from './spec.js'
import { objUniforms } from './objects.glsl.js'
import { FARM_CELL, FARM_RANGE, FARM_REG0, FARM_VERT, FARM_FRAG } from './farm.glsl.js'

const RANGE = FARM_RANGE      // farmsteads within this distance of the camera (m); the shader dithers them out by 0.9 of it
const REBUILD = 300           // re-collect after the camera moves this far (m)
const YARDS = 2000            // new yards per height request (one request at a time; the sampler batches it with the rest)
const MAX_INST = 2000
const FADE_IN = 0.4            // s: new farmsteads start this far into FARM_FRAG's 0.8 s dither-in
const STRIDE = 17
const HOUSE = 0, BARN = 1, SHED = 2, SILO = 3, BIN = 4

const clamp01 = x => Math.min(1, Math.max(0, x))
const smooth = (a, b, x) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t) }
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const noise2 = (x, y, seed) => {            // farmVN(vec2) in farm.glsl.js
  const a = Math.floor(x), b = Math.floor(y), u = (x - a) * (x - a) * (3 - 2 * (x - a)), v = (y - b) * (y - b) * (3 - 2 * (y - b))
  const h = (i, j) => hash2i(a + i, b + j, seed)
  return (h(0, 0) * (1 - u) + h(1, 0) * u) * (1 - v) + (h(0, 1) * (1 - u) + h(1, 1) * u) * v
}
const valueNoise = (x, y, z, seed) => {     // farmVN(vec3): z folds into the seed
  const c = Math.floor(z), w = (z - c) * (z - c) * (3 - 2 * (z - c))
  return noise2(x, y, seed + c * 1013) * (1 - w) + noise2(x, y, seed + c * 1013 + 1013) * w
}
const VN = (v, s, seed, o = 0) => valueNoise(v[0] / s + o, v[1] / s + o, v[2] / s + o, seed)       // farmVN(v/s + o, seed)
const rot = v => [0.8 * v[1] + 0.6 * v[2], -0.8 * v[0] + 0.36 * v[1] - 0.48 * v[2], -0.6 * v[0] - 0.48 * v[1] + 0.64 * v[2]]   // farmRot
const farmLL = dir => [dir[0] * R, dir[1] * R, dir[2] * R]                            // metres on the sphere

// ---- where farmland is, mirrored from farm.glsl.js (farmCover, farmLocal, farmShare) ----
function cover(ll, farmD, terr, T, M, slope, h) {
  const hill = terr[1] - 1.1 * terr[0]
  const ok = smooth(0.3, 0.08, terr[0]) * (1 - 0.7 * smooth(0.35, 0.55, hill)) * smooth(0.4, 0.25, slope)
    * smooth(1500, 600, h) * smooth(2, 7, T)
    * (1 - 0.8 * smooth(0.72, 0.88, M) * smooth(60, 10, h)) * (1 - 0.68 * smooth(0.7, 0.85, M) * smooth(20, 24, T))
  if (ok * farmD <= 0) return 0
  let x = ll.map(v => v / 150e3)
  x = [x[0] + 1.6 * VN(x, 1 / 0.35, 41) - 0.8, x[1] + 1.6 * VN(x, 1 / 0.35, 42, 5.2) - 0.8, x[2] + 1.6 * VN(x, 1 / 0.35, 47, 9.7) - 0.8]
  const r = rot(x)
  const reg = 0.5 * VN(x, 1, 43) + 0.3 * VN(r, 1 / 2.7, 44) + 0.2 * VN(x, 1 / 6.3, 46, 3.3)
  const val = 1 - smooth(0.012, 0.035, Math.abs(VN(r, 1 / 1.4, 45, 9.1) - 0.5))
  const dry = Math.max(smooth(0.1, 0.28, M), val * smooth(0.02, 0.07, M))
  const suit = ok * dry, lvl = reg + 0.12 * smooth(0.05, 0.7, farmD) + 0.3 * (suit - 0.6) + 0.2 * val
  return 0.53 * smooth(0.25, 0.55, suit) * smooth(0, 0.12, farmD) * smooth(FARM_REG0, FARM_REG0 + 0.08, lvl)
}
function local(ll, aa = 1e-3) {
  const f1 = smooth(0.5, 1.5, 2000 / aa), f2 = smooth(0.5, 1.5, 6500 / aa)
  const n = 0.55 * f1 * (VN(ll, 2000, 17) - 0.5) + 0.45 * f2 * (VN(rot(ll), 6500, 18) - 0.5)
  return [clamp01(0.5 + 1.6 * n), 0.9 * (0.55 * (1 - f1) + 0.45 * (1 - f2))]
}
const share = (c, nb) => clamp01((c - 0.8 * nb[0] + 0.4 * nb[1]) / (0.2 + 0.8 * nb[1]))
const norm = v => { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l] }

// ---- layout, mirrored from farm.glsl.js ----
const CELL_A = FARM_CELL / R
const rowCells = j => Math.max(Math.floor(2 * Math.PI * Math.cos((j + 0.5) * CELL_A) * R / FARM_CELL), 1)
function centre(i, j, n) {
  const lat = (j + 0.15 + 0.7 * hash2i(i, j, 72)) * CELL_A, lon = ((i + 0.15 + 0.7 * hash2i(i, j, 71)) / n - 0.5) * 2 * Math.PI
  return [Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon)]
}
const tangentDist = (p, D) => { const v = [p[0] - D[0], p[1] - D[1], p[2] - D[2]], k = dot(v, D); return Math.hypot(v[0] - D[0] * k, v[1] - D[1] * k, v[2] - D[2] * k) * R }
// district cells whose centres lie within 'radius' of dir: [{ i, j, D }]
function districtsNear(dir, radius) {
  const lat = Math.asin(dir[1]), lon = Math.atan2(dir[0], dir[2]), out = []
  const dj = Math.ceil(radius / FARM_CELL) + 1
  for (let j = Math.floor(lat / CELL_A) - dj; j <= Math.floor(lat / CELL_A) + dj; j++) {
    const n = rowCells(j), fx = (lon / (2 * Math.PI) + 0.5) * n, di = Math.ceil(radius / (2 * Math.PI * R * Math.cos(lat) / n)) + 1
    for (let i = Math.floor(fx) - di; i <= Math.floor(fx) + di; i++) {
      const iw = ((i % n) + n) % n, D = centre(iw, j, n)
      if (tangentDist(dir, D) < radius) out.push({ i: iw, j, D })
    }
  }
  return out
}
// nearest district centre to dir (the Voronoi cell dir belongs to)
function districtAt(dir) {
  let best = null, bd = Infinity
  for (const d of districtsNear(dir, FARM_CELL * 1.6)) { const t = tangentDist(dir, d.D); if (t < bd) { bd = t; best = d } }
  return best
}
// farmPivotP / farmRiceP
const style = (hS, T, M, h) => hS < 0.8 * smooth(0.35, 0.15, M) * smooth(10, 16, T) ? 1 : hS < smooth(0.55, 0.7, M) * smooth(20, 24, T) * smooth(400, 150, h) ? 2 : 0
// district frame and parcel size for a layout style
function plan(d, st) {
  const ds = d.i * 1013 + d.j * 7919, llD = farmLL(d.D), hA = hash2i(ds, 2, 73)
  const h1 = clamp01(0.5 + 1.5 * (0.7 * VN(llD, 60e3, 52) + 0.3 * hash2i(ds, 3, 75) - 0.5))
  const h2 = clamp01(0.5 + 1.5 * (0.6 * VN(rot(llD), 45e3, 53) + 0.4 * hash2i(ds, 4, 76) - 0.5))
  const ang = st === 1 ? (hA - 0.5) * 0.06 : Math.PI * (1.6 * VN(llD, 40e3, 54) + 0.25 * hA)
  const e = norm([d.D[2], 0, -d.D[0]]), n = [d.D[1] * e[2] - d.D[2] * e[1], d.D[2] * e[0] - d.D[0] * e[2], d.D[0] * e[1] - d.D[1] * e[0]]
  const c = Math.cos(ang), s = Math.sin(ang)
  const ex = [0, 1, 2].map(k => e[k] * c + n[k] * s), ny = [0, 1, 2].map(k => -e[k] * s + n[k] * c)
  const Sx = st === 1 ? (h1 < 0.7 ? 805 : 402) : 260 + 560 * h1, Sy = st === 1 ? Sx : Sx * (0.35 + 0.55 * h2)
  return { ds, ex, ny, Sx, Sy, hS: 0.75 * VN(llD, 30e3, 51) + 0.25 * hash2i(ds, 1, 74) }
}
const toDir = (d, pl, x, y) => norm([0, 1, 2].map(k => d.D[k] + (pl.ex[k] * x + pl.ny[k] * y) / R))

// buildings of one farmstead in yard coordinates (x along the district axis, origin at the yard corner)
function design(W, H, h) {
  const out = [{ kind: HOUSE, x: 8, y: H - 7, sx: 11, sy: 8, wall: h(1) < 0.5 ? 5.6 : 3.2, rise: 3.2 }]
  const bw = 20 + 8 * h(2), bd = 11 + 4 * h(3)
  out.push({ kind: BARN, x: W - 2 - bw / 2, y: 2 + bd / 2, sx: bw, sy: bd, wall: 5 + 2 * h(4), rise: 4 + 2 * h(5) })
  if (h(6) < 0.6 && W > 50) out.push({ kind: SHED, x: W - 16, y: H - 8, sx: 24, sy: 11, wall: 5, rise: 1.3 })
  const silos = h(7) < 0.3 ? 0 : 1 + Math.floor(h(8) * 3), bins = h(9) < 0.5
  for (let i = 0; i < silos; i++) {
    const r = bins ? 4.5 : 3 + h(10), x = W - bw - 4 - r - i * (2 * r + 1.5)
    if (x - r < 18) break
    out.push({ kind: bins ? BIN : SILO, x, y: 2 + r, r, dome: bins ? 0.35 : 0.8, wall: bins ? 7 + 2 * h(11) : 14 + 7 * h(11) })
  }
  return out
}

// unit gabled box: walls z 0..1, gable ends and roof up to the ridge at z = 2 (roof normals flagged with z = 1)
function gableGeometry() {
  const pos = [], nor = [], idx = []
  const poly = (pts, n) => {
    const b = pos.length / 3
    for (const p of pts) { pos.push(...p); nor.push(...n) }
    const [a, c, d] = pts, e1 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]], e2 = [d[0] - a[0], d[1] - a[1], d[2] - a[2]]
    const f = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]
    const flip = dot(f, n[2] > 0.5 ? [0, n[1], 1] : n) < 0
    for (let k = 1; k < pts.length - 1; k++) flip ? idx.push(b, b + k + 1, b + k) : idx.push(b, b + k, b + k + 1)
  }
  for (const s of [-0.5, 0.5]) {
    poly([[s, -0.5, 0], [s, 0.5, 0], [s, 0.5, 1], [s, -0.5, 1]], [Math.sign(s), 0, 0])
    poly([[-0.5, s, 0], [0.5, s, 0], [0.5, s, 1], [-0.5, s, 1]], [0, Math.sign(s), 0])
    poly([[s, -0.5, 1], [s, 0.5, 1], [s, 0, 2]], [Math.sign(s), 0, 0])
    poly([[-0.5, s, 1], [0.5, s, 1], [0.5, 0, 2], [-0.5, 0, 2]], [0, Math.sign(s), 1])
  }
  return { pos, nor, idx, around: new Array(pos.length / 3).fill(0) }
}
// unit cylinder (z 0..1) with a dome (z 1..2); 'around' runs 0..1 so the seam has its own vertices
function siloGeometry(N = 16, RINGS = 4) {
  const pos = [], nor = [], idx = [], around = []
  const ring = (z, rr, nz, nr) => { const b = pos.length / 3; for (let i = 0; i <= N; i++) { const a = i / N * 2 * Math.PI; pos.push(Math.cos(a) * rr, Math.sin(a) * rr, z); nor.push(Math.cos(a) * nr, Math.sin(a) * nr, nz); around.push(i / N) } return b }
  const strip = (a, b) => { for (let i = 0; i < N; i++) idx.push(a + i, a + i + 1, b + i + 1, a + i, b + i + 1, b + i) }
  strip(ring(0, 1, 0, 1), ring(1, 1, 0, 1))
  let prev = ring(1, 1, 0, 1)
  for (let k = 1; k <= RINGS; k++) {
    const phi = k / RINGS * Math.PI / 2, next = ring(1 + Math.sin(phi), Math.cos(phi), Math.sin(phi), Math.cos(phi))
    strip(prev, next)
    prev = next
  }
  return { pos, nor, idx, around }
}

export class Farms {
  constructor(ctx) {
    this.ctx = ctx
    const fm = ctx.world.textures.farm.image
    this.farmMap = { data: fm.data, n: 1, k: 1 / 255, W: fm.width, H: fm.height }
    this.terrMap = this.terrainMap(ctx.sampler)
    this.uniforms = { uFarmDay: { value: 0 }, uFarmOn: { value: 1 }, uFarmA: { value: new THREE.Vector3() }, uFarmCamA: { value: new THREE.Vector3() }, uFarmTerr: { value: this.terrMap.tex } }
    // the terrain material exists already (its uniforms spread shared.civ at creation): hand it ours before it first compiles
    for (const m of ctx.scene.children) if (m.material?.fragmentShader?.includes('farmPaint(')) Object.assign(m.material.uniforms, this.uniforms)

    this.sites = new Map()          // parcel key -> farmstead | null (rejected) | 0 (height pending)
    this.visible = []
    this.lastPos = new THREE.Vector3(1e9, 0, 0)
    this.lastPre = new THREE.Vector3(9, 0, 0)   // where the last prefetch collected (a direction)
    this.anchor = new THREE.Vector3()
    this.pending = false
    this.dirty = true
    this.count = 0
    this.time = 0

    this.objUniforms = { ...objUniforms(ctx.shared, ctx.FAR), uAnchor: { value: new THREE.Vector3() }, uPixAngle: ctx.shared.uPixAngle }
    this.meshes = [gableGeometry(), siloGeometry()].map((g, shape) => this.makeMesh(g, shape))
  }

  makeMesh({ pos, nor, idx, around }, shape) {
    const geo = new THREE.InstancedBufferGeometry()
    geo.setIndex(idx)
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3))
    geo.setAttribute('around', new THREE.Float32BufferAttribute(around, 1))
    const buf = new THREE.InstancedInterleavedBuffer(new Float32Array(MAX_INST * STRIDE), STRIDE).setUsage(THREE.DynamicDrawUsage)
    const at = (name, size, off) => geo.setAttribute(name, new THREE.InterleavedBufferAttribute(buf, size, off))
    at('iPos', 3, 0); at('iUp', 3, 3); at('iEx', 3, 6); at('iSize', 4, 9); at('iMisc', 4, 13)
    geo.instanceCount = 0
    const mat = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: FARM_VERT, fragmentShader: FARM_FRAG, uniforms: { ...this.objUniforms, uShape: { value: shape } } })
    const mesh = new THREE.Mesh(geo, mat)
    mesh.frustumCulled = false
    mesh.visible = false
    mesh.userData.buf = buf
    this.ctx.scene.add(mesh)
    return mesh
  }

  get enabled() { return this.uniforms.uFarmOn.value > 0 }
  set enabled(v) { this.uniforms.uFarmOn.value = v ? 1 : 0 }

  update(frame) {
    const u = this.uniforms, c = frame.camPos, d = c.length()
    this.time = frame.time
    u.uFarmDay.value = (frame.simTime / 86400) % 365
    // anchor = camera direction rounded to float32; the remainder is computed in doubles
    const A = u.uFarmA.value.set(Math.fround(c.x / d), Math.fround(c.y / d), Math.fround(c.z / d))
    u.uFarmCamA.value.set(c.x / d - A.x, c.y / d - A.y, c.z / d - A.z)

    if (frame.geo.agl > 0.9 * RANGE || !this.enabled) {
      for (const m of this.meshes) m.visible = false
      if (this.enabled && frame.dest?.agl < 0.9 * RANGE) this.prefetch(frame.dest.dir)   // heading down: fetch yards there
      return
    }
    if (this.dirty || frame.camPos.distanceTo(this.lastPos) > REBUILD) this.rebuild(frame)
    this.objUniforms.uAnchor.value.subVectors(this.anchor, frame.camPos)
    for (const m of this.meshes) m.visible = m.geometry.instanceCount > 0
  }

  rebuild(frame) {
    this.dirty = false
    this.lastPos.copy(frame.camPos)
    const cam = [frame.camDir.x, frame.camDir.y, frame.camDir.z]
    this.visible = []
    const todo = []
    for (const d of districtsNear(cam, RANGE + FARM_CELL)) {
      for (const st of [0, 1]) this.collect(d, plan(d, st), st, cam, todo)
    }
    if (todo.length && !this.pending) this.requestHeights(todo)
    this.fill()
  }

  // yards around where the camera is heading (camera.js destination()) before it's low enough to see them: heights only,
  // the drawn set is left alone; one request at a time, the next once it's back
  prefetch(dir) {
    if (this.pending || (!this.dirty && this.lastPre.distanceTo(dir) * R < REBUILD)) return
    this.dirty = false
    this.lastPre.copy(dir)
    const cam = [dir.x, dir.y, dir.z], todo = [], shown = this.visible
    this.visible = []                     // collect() lists known sites here
    for (const d of districtsNear(cam, RANGE + FARM_CELL)) for (const st of [0, 1]) this.collect(d, plan(d, st), st, cam, todo)
    this.visible = shown
    if (todo.length) this.requestHeights(todo)
  }

  // farmyards of one district layout within range: known ones go to this.visible, new ones to todo
  collect(d, pl, st, cam, todo) {
    const v = [cam[0] - d.D[0], cam[1] - d.D[1], cam[2] - d.D[2]], cx = dot(v, pl.ex) * R, cy = dot(v, pl.ny) * R
    if (Math.hypot(cx, cy) > FARM_CELL * 1.6 + RANGE) return
    const p = Math.min(0.35, pl.Sx * pl.Sy / 6e5), C = FARM_CELL * 1.6     // the district lies within C of its centre
    for (let ry = Math.floor(Math.max(cy - RANGE, -C) / pl.Sy); ry <= Math.floor(Math.min(cy + RANGE, C) / pl.Sy); ry++) {
      const off = st === 1 ? 0 : hash2i(ry, pl.ds, 78) * pl.Sx
      for (let rx = Math.floor((Math.max(cx - RANGE, -C) + off) / pl.Sx); rx <= Math.floor((Math.min(cx + RANGE, C) + off) / pl.Sx); rx++) {
        if (hash2i(rx, ry, pl.ds + 14) >= p) continue
        const W = 45 + 30 * hash2i(rx, ry, pl.ds + 15), H = 35 + 25 * hash2i(rx, ry, pl.ds + 16)
        const x0 = rx * pl.Sx - off + 10, y0 = ry * pl.Sy + 10
        if (Math.hypot(x0 + W / 2 - cx, y0 + H / 2 - cy) > RANGE) continue
        const key = `${pl.ds}:${st}:${rx}:${ry}`
        const site = this.sites.get(key)
        if (site) this.visible.push(site)
        else if (site === undefined) todo.push({ key, d, pl, st, rx, ry, x0, y0, W, H })
      }
    }
  }

  // ground height, slope and climate at each new yard (centre + corners), then keep the ones the painting farms
  requestHeights(todo) {
    todo = todo.slice(0, YARDS)
    const dirs = new Float32Array(todo.length * 15)
    todo.forEach((t, i) => {
      t.dir = toDir(t.d, t.pl, t.x0 + t.W / 2, t.y0 + t.H / 2)
      const pts = [t.dir, toDir(t.d, t.pl, t.x0, t.y0), toDir(t.d, t.pl, t.x0 + t.W, t.y0), toDir(t.d, t.pl, t.x0, t.y0 + t.H), toDir(t.d, t.pl, t.x0 + t.W, t.y0 + t.H)]
      pts.forEach((p, k) => dirs.set(p, i * 15 + k * 3))
      this.sites.set(t.key, 0)
    })
    this.pending = true
    this.ctx.sampler.sampleAsync(dirs, 1.2e6, true).then(s => {
      todo.forEach((t, i) => {
        const o = i * 40
        let lo = Infinity, hi = -Infinity
        for (let k = 0; k < 5; k++) { lo = Math.min(lo, s[o + k * 8]); hi = Math.max(hi, s[o + k * 8]) }
        this.sites.set(t.key, this.accept(t, s[o], s[o + 4], s[o + 5], s[o + 6], lo) ? this.farmstead(t, hi, hi - lo + 1.5) : null)
      })
      this.pending = false
      this.dirty = true
    }, () => { this.pending = false; for (const t of todo) this.sites.delete(t.key) })
  }

  // same tests as farmPaint(): this layout's style, and the yard's field (strip 0) is farmed; off towns, roads,
  // lakes and the grass strip along rivers (farmBank in farm.glsl.js: 15 m + 0.3 half widths from the channel)
  accept(t, h, slope, T, M, lo) {
    if (lo < 1 || slope > 0.2) return false
    const r = Math.hypot(t.W, t.H) / 2, w = this.ctx.world.nearestWater(t.dir, r + 200)
    if (w && (w.type === 'river' ? w.d < r + 15 + 0.15 * w.width : w.d < r && lo < w.level + 1)) return false
    const own = districtAt(t.dir)
    if (style(t.pl.hS, T, M, h) !== t.st || !own || own.i * 1013 + own.j * 7919 !== t.pl.ds) return false
    const k = t.st === 1 ? 1 : 1 + Math.floor(hash2i(t.rx, t.ry, t.pl.ds + 7) * 4)
    const llC = farmLL(toDir(t.d, t.pl, t.x0 - 10 + t.pl.Sx / k / 2, t.y0 - 10 + t.pl.Sy / 2))
    if (0.2 * hash2i(t.rx * 8, t.ry, t.pl.ds + 5) + 0.8 * local(llC)[0] > this.cover(t.dir, T, M, slope, h) - 0.03) return false
    const city = this.ctx.world.nearestCity(t.dir, 60000)
    if (city && city.distance < city.city.radius * 1.25) return false
    const seg = this.ctx.world.nearestSegment(t.dir, 90)
    return !(seg && seg.d < HALF[seg.type] + 45)
  }

  // debug: expected farmed share of the land from n random points, optionally within 'radius' m of 'centre' ([x, y, z]);
  // maxF = terrain detail of the slope test (8e3 ~ the 400 km view, 7e5 close up)
  async coverage(n = 20000, maxF = 8e3, centre = null, radius = 0) {
    const dirs = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) {
      let d
      if (centre) {
        const e = norm([centre[2], 0, -centre[0]]), f = [centre[1] * e[2] - centre[2] * e[1], centre[2] * e[0] - centre[0] * e[2], centre[0] * e[1] - centre[1] * e[0]]
        const r = radius * Math.sqrt(hash2i(i, 1, 99)) / R, a = 2 * Math.PI * hash2i(i, 2, 99)
        d = norm([0, 1, 2].map(k => centre[k] + (e[k] * Math.cos(a) + f[k] * Math.sin(a)) * r))
      } else {
        const z = 2 * hash2i(i, 1, 99) - 1, a = 2 * Math.PI * hash2i(i, 2, 99), s = Math.sqrt(1 - z * z)
        d = [s * Math.sin(a), z, s * Math.cos(a)]
      }
      dirs.set(d, i * 3)
    }
    const s = await this.ctx.sampler.sampleAsync(dirs, maxF, false)
    let land = 0, farmed = 0
    for (let i = 0; i < n; i++) {
      const o = i * 8, dir = [dirs[i * 3], dirs[i * 3 + 1], dirs[i * 3 + 2]], ll = farmLL(dir)
      if (s[o] <= 0) continue
      land++
      farmed += share(this.cover(dir, s[o + 5], s[o + 6], s[o + 4], s[o]), local(ll))
    }
    return { land, farmed: +(farmed / Math.max(land, 1)).toFixed(4) }
  }

  // farmCover() at dir with the painting's texture inputs
  cover(dir, T, M, slope, h) {
    return cover(farmLL(dir), this.at(this.farmMap, dir, 0), [this.at(this.terrMap, dir, 0), this.at(this.terrMap, dir, 1) * 1.5], T, M, slope, h)
  }

  // regional terrain for the painting (r = mountain mask, g = roughness / 1.5), sampled at the texel centres of a
  // lat/lon grid like the farm map: the per-pixel slope depends on the view distance, these don't
  terrainMap(sampler, W = 1024, H = 512) {
    const dirs = new Float32Array(W * H * 3)
    for (let y = 0; y < H; y++) {
      const lat = ((y + 0.5) / H - 0.5) * Math.PI
      for (let x = 0; x < W; x++) {
        const lon = ((x + 0.5) / W - 0.5) * 2 * Math.PI
        dirs.set([Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon)], (y * W + x) * 3)
      }
    }
    const s = sampler.sample(dirs, 500, false), data = new Uint8Array(W * H * 2)
    for (let i = 0; i < W * H; i++) { data[i * 2] = Math.round(255 * clamp01(s[i * 8 + 2])); data[i * 2 + 1] = Math.round(255 * clamp01(s[i * 8 + 3] / 1.5)) }
    const tex = new THREE.DataTexture(data, W, H, THREE.RGFormat, THREE.UnsignedByteType)
    tex.wrapS = THREE.RepeatWrapping
    tex.minFilter = THREE.LinearMipmapLinearFilter
    tex.magFilter = THREE.LinearFilter
    tex.generateMipmaps = true
    tex.needsUpdate = true
    return { data, n: 2, k: 1 / 255, W, H, tex }
  }

  // bilinear texel lookup (channel ch) at dir, as the GPU filters the base level
  at({ data, n, k, W, H }, dir, ch) {
    const u = (Math.atan2(dir[0], dir[2]) / (2 * Math.PI) + 0.5) * W - 0.5, v = (Math.asin(dir[1]) / Math.PI + 0.5) * H - 0.5
    const x = Math.floor(u), y = Math.max(0, Math.min(H - 2, Math.floor(v))), fx = u - x, fy = v - y
    const t = (i, j) => data[(j * W + ((i % W) + W) % W) * n + ch] * k
    return (t(x, y) * (1 - fx) + t(x + 1, y) * fx) * (1 - fy) + (t(x, y + 1) * (1 - fx) + t(x + 1, y + 1) * fx) * fy
  }

  farmstead(t, base, found) {
    const h = k => hash2i(t.rx * 31 + k, t.ry, t.pl.ds + 20)
    const parts = design(t.W, t.H, h).map((b, i) => ({ ...b, dir: toDir(t.d, t.pl, t.x0 + b.x, t.y0 + b.y), seed: (h(40 + i) * 997) | 0 }))
    return { dir: t.dir, base, found, ex: t.pl.ex, ny: t.pl.ny, parts, born: this.time - FADE_IN }
  }

  // roof height above sea level + 1 m inside a loaded farm building's footprint, else null (camera collision)
  heightAt(dir) {
    const d = dir.isVector3 ? [dir.x, dir.y, dir.z] : dir
    let top = null
    for (const site of this.visible) {
      if (tangentDist(d, site.dir) > 150) continue
      for (const b of site.parts) {
        const v = [d[0] - b.dir[0], d[1] - b.dir[1], d[2] - b.dir[2]], x = dot(v, site.ex) * R, y = dot(v, site.ny) * R
        const inside = b.kind >= SILO ? Math.hypot(x, y) <= b.r : Math.abs(x) <= b.sx / 2 && Math.abs(y) <= b.sy / 2
        if (inside) top = Math.max(top ?? -Infinity, site.base + b.wall + (b.kind >= SILO ? b.r * b.dome : b.rise) + 1)
      }
    }
    return top
  }

  // instance data relative to a fresh anchor at the camera, split by shape
  fill() {
    const A = this.anchor.copy(this.lastPos), n = [0, 0]
    for (const site of this.visible) {
      for (const b of site.parts) {
        const shape = b.kind >= SILO ? 1 : 0
        if (n[shape] >= MAX_INST) continue
        const D = this.meshes[shape].userData.buf.array, o = n[shape]++ * STRIDE, r = R + site.base
        D[o] = b.dir[0] * r - A.x; D[o + 1] = b.dir[1] * r - A.y; D[o + 2] = b.dir[2] * r - A.z
        D.set(b.dir, o + 3); D.set(site.ex, o + 6)
        if (shape) D.set([b.r, b.dome, b.wall, 0], o + 9); else D.set([b.sx, b.sy, b.wall, b.rise], o + 9)
        D.set([b.kind, b.seed, site.found, site.born], o + 13)
      }
    }
    this.meshes.forEach((m, i) => {
      m.geometry.instanceCount = n[i]
      m.userData.buf.needsUpdate = true
    })
    this.count = n[0] + n[1]
  }
}
