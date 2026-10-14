// Factories: industrial districts (zone 1 blocks) in 3D near the camera. Contracts in notes/CIV.md.
// Blocks come from the INDUSTRY districts (district blocks cached as in buildings.js, notes/layout.md).
// Each industrial block is split into 1, 2 or 4 plots, and each plot gets a plan from its hash: a factory
// hall (sawtooth or flat roof), a distribution shed with dock doors, a tank farm, a silo battery or heavy
// works with chimneys and a gantry crane, plus offices and yards. industry.glsl.js paints the same plans
// on the ground, so the painted roofs sit under the 3D ones and carry the look beyond the 3D range.
// Like buildings.js: blocks are cached with GPU-sampled ground heights, instances are rebuilt relative to
// an anchor when the camera has moved, one instanced mesh per shape. Chimneys also get smoke plumes.
// No popping (as in buildings.js): each part has a range; over the last 20% of it the shader slides the part
// into the ground (roof parts into their roof) from the true distance, and the set holds everything within
// range + the camera travel until the next set. Rebuilds run BUDGET ms per frame (a generator) into back
// buffers that are swapped in when done. A shape over its CAP eases the uLod range scale down (the farthest
// sink away), and it creeps back up once there is room.
import * as THREE from 'three'
import { R } from '../config.js'
import { TYPE, HALF, DKIND, hash2i, localToGrid, localToDir, dirToLocal, dirToGrid, CITY_REACH } from './spec.js'
import { districtRecs, enumBlocks, blockAt, frameAt, clearOf, rectPts, TRUNK_GAP } from './buildings.js'
import { objUniforms } from './objects.glsl.js'
import { VERT, FRAG, SMOKE_VERT, SMOKE_FRAG } from './factories.glsl.js'
import { LAKE_DRY, cityWater, waterAt } from './shore.js'

const NEAR = 3000, FAR = 12000, TALL = 20000  // full detail / main structures (big sheds) / chimneys (3D distance, m)
const FINE = 1200                              // small parts
const SMOKE = 3500                             // chimneys with plumes (m; less while MAX_PUFFS is hit)
const PUFFS = 12
const STRIDE = 24, SMOKE_STRIDE = 10
const BUDGET = 2                               // rebuild CPU per frame (ms)
const FADE_IN = 0.4                            // s: new plots start this far into the shader's 0.8 s dither-in
const PAD = 600                                // grid distances differ from metres by less than this (the warp is bounded)
const CAP = [40000, 12000, 16000, 2000]        // instances per shape: box, tooth, cylinder, sphere
const MAX_PUFFS = 3000
const DESIGN = 1500                            // block designs per rebuild, nearest first
const ENUM = 20                                // ms of new districts enumerated per rebuild (nearest first)
const SAMPLE_PTS = 40000
const BOX = 0, TOOTH = 1, CYL = 2, SPHERE = 3
const KIND = { HALL: 0, SHED: 1, TANKS: 2, SILOS: 3, WORKS: 4 }
const ST = { HALL: 0, SHED: 1, TANK: 2, SILO: 3, WORKS: 4, OFFICE: 5, TOOTH: 6, STACK: 7, BANDED: 8, STEEL: 9, PLANT: 10 }
const RAIL_CLEAR = 13, ROAD_CLEAR = 3          // keep-out beyond the rail centreline / road edge (m), as painted

const clamp = (x, a, b) => Math.min(b, Math.max(a, x))
const neg = a => [-a[0], -a[1], -a[2]]
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
function segDist(x, y, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, t = clamp(((x - ax) * dx + (y - ay) * dy) / Math.max(dx * dx + dy * dy, 1e-9), 0, 1)
  return Math.hypot(x - ax - dx * t, y - ay - dy * t)
}
// distance between a rect [x0, x1, y0, y1] and a segment (the box distance is convex along the segment)
function rectSegDist(b, s) {
  const f = t => {
    const x = s[0] + (s[2] - s[0]) * t, y = s[1] + (s[3] - s[1]) * t
    return Math.hypot(Math.max(b[0] - x, 0, x - b[1]), Math.max(b[2] - y, 0, y - b[3]))
  }
  let a = 0, c = 1
  for (let i = 0; i < 20; i++) { const m1 = a + (c - a) / 3, m2 = c - (c - a) / 3; f(m1) < f(m2) ? c = m2 : a = m1 }
  return f((a + c) / 2)
}

// ---- layout: mirrors indPlot / indPlan / indGrid in industry.glsl.js ----
function plots(b) {
  const L = b.info.lot, s = b.D.hseed, { kx, ky } = b
  const hb = hash2i(kx, ky, s + 41), f = 0.35 + 0.3 * hash2i(kx, ky, s + 43)
  const nx = hb < 0.4 ? 1 : hb < 0.8 ? (hash2i(kx, ky, s + 42) < 0.5 ? 2 : 1) : 2
  const ny = hb < 0.4 ? 1 : hb < 0.8 ? 3 - nx : 2
  const mx = L.x0 + (L.x1 - L.x0) * f, my = L.y0 + (L.y1 - L.y0) * f
  const xs = nx === 2 ? [[L.x0, mx - 2], [mx + 2, L.x1]] : [[L.x0, L.x1]]
  const ys = ny === 2 ? [[L.y0, my - 2], [my + 2, L.y1]] : [[L.y0, L.y1]]
  return ys.flatMap((y, j) => xs.map((x, i) => ({ id: [kx * 4 + i, ky * 4 + j], x0: x[0], x1: x[1], y0: y[0], y1: y[1] })))
}
function plan(seed, p) {
  const h = n => hash2i(p.id[0], p.id[1], seed + 45 + n)
  const w = p.x1 - p.x0, d = p.y1 - p.y0, L = Math.max(w, d), S = Math.min(w, d), ph = h(0)
  const kind = S < 34 ? (ph < 0.6 ? KIND.HALL : KIND.SHED) : ph < 0.3 ? KIND.HALL : ph < 0.55 ? KIND.SHED : ph < 0.68 ? KIND.TANKS : ph < 0.77 ? KIND.SILOS : KIND.WORKS
  const side = h(1) < 0.5 ? -1 : 1, yard = S * (0.24 + 0.14 * h(2))
  return {
    id: p.id, h, kind, ax: w > d - 0.5, cx: (p.x0 + p.x1) / 2, cy: (p.y0 + p.y1) / 2, L, S, side, yard,
    lmin: 4 - L / 2, lmax: L / 2 - 4, smin: side > 0 ? 4 - S / 2 : yard - S / 2, smax: side > 0 ? S / 2 - yard : S / 2 - 4,
  }
}
function grid(P) {
  const Lm = P.lmax - P.lmin, Sm = P.smax - P.smin, h = P.h
  if (P.kind === KIND.TANKS) {
    const D = Math.min(12 + 14 * h(3), Sm - 2), pitch = D + 5, sphere = h(4) < 0.35
    const cols = clamp(Math.floor((Lm + 5) / pitch), 1, 6), rows = clamp(Math.floor((Sm + 5) / pitch), 1, 2)
    return { l0: P.lmin, s0: P.smin, pl: Lm / cols, ps: Sm / rows, cols, rows, D: sphere ? Math.min(D, 18) : D, sphere }
  }
  const D = 7 + 3 * h(3), cols = clamp(Math.floor(Lm * 0.7 / D), 2, 10), rows = clamp(Math.floor(Sm / D), 1, 3)
  return { l0: P.lmin, s0: (P.smin + P.smax) / 2 - rows * D / 2, pl: D, ps: D, cols, rows, D: D - 0.4, sphere: false }
}

// the structures of a plan, in plan coordinates (l along the long axis, s across). Each part: shape, centre
// (l, s), size (a along its own x, b across), height H, base z0, style, extras e0/e1, lod (0 main, 1 detail,
// 2 fine), alongS (its x runs along s), found (false = no foundation), stack (chimney mouth for smoke)
function design(P) {
  const parts = [], h = P.h, Lm = P.lmax - P.lmin, Sm = P.smax - P.smin, lm = (P.lmin + P.lmax) / 2, sm = (P.smin + P.smax) / 2
  const edge = P.side > 0 ? P.smax : P.smin, out = P.side, dock = P.ax ? P.side : -P.side
  const tone = h(6)
  const add = (shape, l, s, a, b, H, style, o = {}) => parts.push({ shape, l, s, a, b, H, style, z0: 0, tone, e0: 0, e1: 0, lod: 0, alongS: false, found: true, ...o })
  const stack = (l, s, H, steam) => {
    const D = 1.8 + H * 0.035
    add(CYL, l, s, D, D, H, H > 80 ? ST.BANDED : ST.STACK, { e0: 0.35, stack: { H, D: D * 0.65, steam }, tall: true })
  }
  if (P.kind === KIND.HALL || P.kind === KIND.SHED) {
    const hall = P.kind === KIND.HALL, H = hall ? 8 + 5 * h(3) : 10 + 4 * h(3)
    const teeth = hall && h(4) < 0.65, n = Math.max(2, Math.round(Lm / (8 + 4 * h(5)))), pitch = Lm / n
    add(BOX, lm, sm, Lm, Sm, H, hall ? ST.HALL : ST.SHED, { e0: dock, e1: teeth ? pitch : 0 })
    if (teeth) for (let k = 0; k < n; k++) add(TOOTH, P.lmin + (k + 0.5) * pitch, sm, pitch, Sm, 2.2 + 1.3 * h(7), ST.TOOTH, { z0: H, lod: 1, found: false, roof: true })
    else for (let k = 0; k < 4; k++) {
      const hk = m => hash2i(P.id[0] * 5 + k, P.id[1], 60 + m)
      add(BOX, lm + (hk(1) - 0.5) * Lm * 0.7, sm + (hk(2) - 0.5) * Sm * 0.6, 3 + 5 * hk(3), 2.5 + 3 * hk(4), 2 + 1.5 * hk(5), ST.PLANT, { z0: H, lod: 2, found: false, roof: true })
    }
    const od = Math.min(P.yard - (hall ? 8 : 10), hall ? 12 : 8), ow = Math.min(hall ? 14 + 12 * h(5) : 12 + 8 * h(5), Lm * 0.45)
    if (od > 5) add(BOX, hall ? P.lmax - ow / 2 : P.lmin + ow / 2, edge + out * od / 2, ow, od, hall ? (h(7) > 0.5 ? 3 : 2) * 3.6 + 0.8 : 7.6, ST.OFFICE)
    if (hall && h(8) < 0.2) stack(P.lmin + 3, edge + out * 3, 22 + 20 * h(3), true)
  } else if (P.kind === KIND.TANKS) {
    const g = grid(P)
    for (let r = 0; r < g.rows; r++) for (let c = 0; c < g.cols; c++) {
      const l = g.l0 + (c + 0.5) * g.pl, s = g.s0 + (r + 0.5) * g.ps, hk = hash2i(P.id[0] * 8 + c, P.id[1] * 8 + r, 5)
      if (g.sphere) {
        add(SPHERE, l, s, g.D, g.D, g.D, ST.TANK, { z0: 3, found: false, tone: h(7) })
        for (const [dl, ds] of [[1, 0], [-1, 0], [0, 1], [0, -1]])   // legs up to the equator
          add(BOX, l + dl * g.D * 0.42, s + ds * g.D * 0.42, 0.9, 0.9, 3 + g.D / 2, ST.STEEL, { tone: 0.8, lod: 1 })
      } else add(CYL, l, s, g.D, g.D, 8 + 12 * hk, ST.TANK, { tone: h(7) })
    }
    if (g.rows === 2) add(BOX, lm, g.s0 + g.ps, Lm, 1.6, 4.5, ST.STEEL, { tone: 0.8, lod: 1 })   // pipe rack
    add(BOX, P.lmin + 6, edge + out * Math.min(P.yard / 2, 7), 10, 7, 4.5, ST.PLANT)             // pump house
  } else if (P.kind === KIND.SILOS) {
    const g = grid(P), Hs = 22 + 18 * h(5)
    for (let r = 0; r < g.rows; r++) for (let c = 0; c < g.cols; c++) add(CYL, g.l0 + (c + 0.5) * g.pl, g.s0 + (r + 0.5) * g.ps, g.D, g.D, Hs, ST.SILO)
    add(BOX, g.l0 + g.cols * g.D / 2, sm, g.cols * g.D, 3, 2.5, ST.STEEL, { z0: Hs, tone: 0.8, lod: 1, found: false, roof: true })   // top gallery
    const lh = g.l0 + g.cols * g.D + 1, wh = Math.min(10, P.lmax - lh), sh = Math.min(Sm, 12)
    if (wh > 4) {
      add(BOX, lh + wh / 2, sm, wh, sh, Hs + 9, ST.SILO)                                           // head house
      // conveyor from the truck pit in the yard up to the head house
      const hi = sm + out * (sh / 2 - 1), lo = edge + out * (P.yard - 6), top = Hs + 5
      add(BOX, lh + wh / 2, (hi + lo) / 2, Math.abs(lo - hi), 2.4, 2.2, ST.STEEL,
        { alongS: true, z0: out > 0 ? top : 1.5, e1: out > 0 ? 1.5 - top : top - 1.5, tone: 0.3, lod: 1, found: false })
    }
  } else {
    const lw = P.lmin + 0.58 * Lm, H = 16 + 10 * h(3)
    add(BOX, (P.lmin + lw) / 2, sm, lw - P.lmin, Sm, H, ST.WORKS, { e0: dock })
    const Hc = 40 + 70 * h(5) ** 2, n = h(4) < 0.3 ? 2 : 1
    for (let k = 0; k < n; k++) stack(lw + 5 + k * 9, sm + (k ? 0.25 * Sm : 0), Hc * (k ? 0.7 : 1), h(7) < 0.3)
    // gantry crane over the stock yard
    const l0 = lw + 22, l1 = P.lmax - 3
    if (l1 - l0 > 6) {
      const lc = l0 + (l1 - l0) * h(8), Hg = 14 + 6 * h(6), o = { tone: 0.2, lod: 1 }
      add(BOX, lc, P.smin + 0.7, 1.4, 1.4, Hg, ST.STEEL, o)
      add(BOX, lc, P.smax - 0.7, 1.4, 1.4, Hg, ST.STEEL, o)
      add(BOX, lc, sm, Sm + 2, 2, 2, ST.STEEL, { ...o, alongS: true, z0: Hg, found: false })
    }
  }
  return parts
}

export class Factories {
  constructor(ctx) {
    this.ctx = ctx
    this.world = ctx.world
    this.recs = new Set()                      // district records whose blocks are cached
    this.nBlocks = 0
    this.visible = []
    this.cities = []
    this.anchor = new THREE.Vector3()
    this.lastPos = new THREE.Vector3(1e9, 0, 0)
    this.dirty = true
    this.enabled = true
    this.pending = false
    this.tick = 0
    this.job = null                            // the rebuild in progress (generator)
    this.counts = [0, 0, 0, 0]
    this.lod = this.lodTarget = this.lodSet = 1   // range scale: drawn, wanted, the drawn set's
    this.reach = this.reachTarget = this.reachSet = SMOKE   // plume reach, the same way
    this.uniforms = { ...objUniforms(ctx.shared), uAnchor: { value: new THREE.Vector3() }, uShift: { value: 0.3 }, uLod: { value: 1 } }
    this.meshes = [unitBox(), unitTooth(), unitCylinder(16), unitSphere(14, 9)].map((g, shape) => this.makeMesh(g, shape))
    this.makeSmoke()
    this.stats = { rebuilds: 0, cpuMs: 0, sliceMs: 0, frames: 0, instances: 0, puffs: 0, lod: 1, gpuMs: [0, 0, 0, 0, 0] }
    const gl = ctx.renderer.getContext(), ext = gl.getExtension('EXT_disjoint_timer_query_webgl2')
    if (ext) [...this.meshes, this.smoke].forEach((m, i) => this.gpuTimer(gl, ext, m, i))
  }

  // debug: GPU time per mesh (ms, smoothed) in stats.gpuMs: box, tooth, cylinder, sphere, smoke
  gpuTimer(gl, ext, mesh, i) {
    let q = null, active = false
    mesh.onBeforeRender = () => {
      if (q && !active && gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) {
        this.stats.gpuMs[i] += (gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6 - this.stats.gpuMs[i]) * 0.1
        gl.deleteQuery(q); q = null
      }
      if (!q) { q = gl.createQuery(); gl.beginQuery(ext.TIME_ELAPSED_EXT, q); active = true }
    }
    mesh.onAfterRender = () => { if (active) { gl.endQuery(ext.TIME_ELAPSED_EXT); active = false } }
  }

  makeMesh({ pos, nor, u, idx }, shape) {
    const geo = new THREE.InstancedBufferGeometry()
    geo.setIndex(idx)
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3))
    geo.setAttribute('aU', new THREE.Float32BufferAttribute(u, 1))
    const stage = [0, 1].map(() => new Float32Array(CAP[shape] * STRIDE))   // drawn, being filled
    const buf = new THREE.InstancedInterleavedBuffer(stage[0], STRIDE).setUsage(THREE.DynamicDrawUsage)
    const at = (name, size, off) => geo.setAttribute(name, new THREE.InterleavedBufferAttribute(buf, size, off))
    at('iPos', 3, 0); at('iUp', 3, 3); at('iEx', 3, 6); at('iSize', 3, 9); at('iMisc', 4, 12); at('iExt', 2, 16); at('iEy', 3, 18); at('iFade', 3, 21)
    geo.instanceCount = 0
    const mat = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: VERT, fragmentShader: FRAG, uniforms: { ...this.uniforms, uShape: { value: shape } } })
    const mesh = new THREE.Mesh(geo, mat)
    mesh.frustumCulled = false
    mesh.userData = { stage, buf }
    this.ctx.scene.add(mesh)
    return mesh
  }

  makeSmoke() {
    const geo = new THREE.InstancedBufferGeometry()
    geo.setIndex([0, 1, 2, 0, 2, 3])
    geo.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3))
    this.smokeStage = [0, 1].map(() => new Float32Array(MAX_PUFFS * SMOKE_STRIDE))
    this.smokeBuf = new THREE.InstancedInterleavedBuffer(this.smokeStage[0], SMOKE_STRIDE).setUsage(THREE.DynamicDrawUsage)
    const at = (name, size, off) => geo.setAttribute(name, new THREE.InterleavedBufferAttribute(this.smokeBuf, size, off))
    at('iTop', 3, 0); at('iUp', 3, 3); at('iP', 4, 6)
    geo.instanceCount = 0
    this.smokeU = { ...objUniforms(this.ctx.shared), uAnchor: this.uniforms.uAnchor, uWind: { value: new THREE.Vector3() }, uSmoke: { value: SMOKE } }
    const mat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: SMOKE_VERT, fragmentShader: SMOKE_FRAG, uniforms: this.smokeU,
      transparent: true, depthWrite: false, blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor, blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor,
    })
    this.smoke = new THREE.Mesh(geo, mat)
    this.smoke.frustumCulled = false
    this.ctx.scene.add(this.smoke)
  }

  update(frame) {
    const step = 40 + 0.05 * Math.max(frame.geo.agl, 0), moved = frame.camPos.distanceTo(this.lastPos)
    if (!this.job && (this.dirty || moved > step)) {
      this.urgent = this.dirty || this.counts.every(c => c === 0) || moved > 4 * step   // streaming a new area (civ/index.js)
      this.job = this.build(frame, step)
    }
    if (this.job) {
      const t0 = performance.now(), budget = this.urgent ? Math.max(BUDGET, 0.3 * (frame.budget ?? 0)) : BUDGET
      let r
      do r = this.job.next(); while (!r.done && performance.now() - t0 < budget)
      const ms = performance.now() - t0, s = this.stats
      this.jobMs += ms; this.jobMax = Math.max(this.jobMax, ms); this.jobFrames++
      if (r.done) { this.job = null; s.cpuMs = this.jobMs; s.sliceMs = this.jobMax; s.frames = this.jobFrames }
    }
    const ease = Math.min(1, frame.dt * 1.5)
    this.lod += (this.lodSet - this.lod) * ease
    this.reach += (this.reachSet - this.reach) * ease
    this.uniforms.uLod.value = this.stats.lod = this.lod
    this.smokeU.uSmoke.value = this.reach
    this.uniforms.uAnchor.value.subVectors(this.anchor, frame.camPos)
    // light wind veering slowly through the day
    const a = 0.8 + 0.6 * Math.sin(frame.simTime / 30000), sp = 3.5 + 1.5 * Math.sin(frame.simTime / 17000)
    this.smokeU.uWind.value.copy(frame.geo.east).multiplyScalar(Math.cos(a) * sp).addScaledVector(frame.geo.north, Math.sin(a) * sp)
    for (const m of this.meshes) m.visible = this.enabled && m.geometry.instanceCount > 0
    this.smoke.visible = this.enabled && this.smoke.geometry.instanceCount > 0
  }

  // one rebuild, a slice per frame: collect blocks, request heights, fill the back buffers, swap
  *build(frame, step) {
    this.jobMs = this.jobMax = this.jobFrames = 0
    this.dirty = false
    const A = frame.camPos.clone(), agl = Math.max(frame.geo.agl, 0), now = frame.time
    this.lastPos.copy(A)
    this.tick++
    const c = frame.around ?? frame.camDir         // the destination while too high to see any (civ/index.js)
    yield* this.collect([c.x, c.y, c.z])
    this.requestHeights()
    yield* this.fill(A, agl, now, 2 * step + 60)       // margin: travel until the next set is drawn
    if (this.nBlocks > 60000) {   // forget districts out of sight (their blocks are enumerated again when needed)
      for (const r of this.recs) if (r.seen < this.tick - 3 && !r.blocks.some(b => b.state === 1)) { this.nBlocks -= r.blocks.length; r.blocks = r.byKey = null; this.recs.delete(r) }
      yield
    }
    this.stats.rebuilds++
  }

  // industrial blocks around the camera, district by district (INDUSTRY districts only); designs the nearest first
  *collect(cam) {
    const near = this.world.near(cam, TALL), visible = [], cities = []
    const todo = []                            // blocks to design, nearest first
    let work = 0, enumerated = 0
    for (const city of near.cities) {
      if (city.tier < 1) continue
      const [lx, ly] = dirToLocal(city, cam), reach = city.radius * CITY_REACH + PAD
      if (Math.hypot(lx, ly) > reach + TALL) continue
      cities.push(city)
      const [cu, cv] = dirToGrid(city, cam)
      let segs = null
      const recs = yield* districtRecs(city, cu, cv, TALL / 0.85 + PAD, D => D.kind === DKIND.INDUSTRY)
      for (const r of recs) r.dist = Math.hypot(r.x - lx, r.y - ly) - r.rad
      recs.sort((a, b) => a.dist - b.dist)
      for (const rec of recs) {
        if (rec.dist > TALL + 300) break
        rec.seen = this.tick
        if (!rec.blocks) {
          if (enumerated > ENUM) { this.dirty = true; continue }
          const gen = enumBlocks(city, rec, i => i.zone === 1, b => Object.assign(b, { state: 4, plots: [], dist: 0, seen: 0 }))
          for (;;) { const t0 = performance.now(), r = gen.next(); enumerated += performance.now() - t0; if (r.done) break; yield }
          this.nBlocks += rec.blocks.length
          this.recs.add(rec)
        }
        for (const b of rec.blocks) {
          if (++work > 400) { work = 0; yield }
          const d = Math.hypot(b.x - lx, b.y - ly)        // metres, from the cached block centre
          if (d > TALL + 300) continue                      // + block radius: parts at the edge are faded out
          b.dist = d; b.seen = this.tick
          if (b.state === 4) todo.push([b, segs ??= this.localSegs(near.segs, city, lx, ly)])
          else visible.push(b)
        }
      }
    }
    todo.sort((a, b) => a[0].dist - b[0].dist)
    if (todo.length > DESIGN) this.dirty = true
    for (const [b, segs] of todo.slice(0, DESIGN)) {
      designBlock(b, segs, this.world)
      visible.push(b)
      if (++work > 10) { work = 0; yield }
    }
    this.visible = visible.sort((a, b) => a.dist - b.dist)
    this.cities = cities
  }

  // road/rail segments near the camera in the city's local plane (metres): [ax, ay, bx, by, clearance]
  localSegs(ids, city, cx, cy) {
    const S = this.world.seg, out = []
    const loc = o => dirToLocal(city, [S[o], S[o + 1], S[o + 2]])
    for (const k of ids) {
      const [ax, ay] = loc(k * 8), [bx, by] = loc(k * 8 + 4), t = S[k * 8 + 3]
      if (segDist(cx, cy, ax, ay, bx, by) < TALL + 500) out.push([ax, ay, bx, by, t === TYPE.RAIL ? RAIL_CLEAR : HALF[t] + ROAD_CLEAR])
    }
    return out
  }

  // roof height (m above sea level, plus 1 m) of a loaded structure whose footprint contains dir, else null
  heightAt(dir) {
    const d = dir.isVector3 ? [dir.x, dir.y, dir.z] : dir
    let top = null
    for (const city of this.cities) {
      const [lx, ly] = dirToLocal(city, d)
      if (Math.hypot(lx, ly) > city.radius * CITY_REACH + PAD) continue
      const at = blockAt(city, ...localToGrid(city, lx, ly)), b = at?.b
      if (!b || b.state !== 2 || !b.plots) continue
      const { x, y } = at
      for (const p of b.plots) {
        if (p.wet) continue
        for (const q of p.parts) {   // cranes and conveyors can be walked under
          const [x0, x1, y0, y1] = q.box          // district coords
          if (q.style !== ST.STEEL && x >= x0 && x <= x1 && y >= y0 && y <= y1) top = Math.max(top ?? -Infinity, p.base + q.z0 + q.H + 1)
        }
      }
    }
    return top
  }

  requestHeights() {
    if (this.pending) return
    const todo = []
    let n = 0
    for (const b of this.visible) {
      if (b.state !== 0) continue
      todo.push(b); n += b.plots.length * 5
      if (n >= SAMPLE_PTS) break
    }
    if (!todo.length) return
    const dirs = new Float32Array(n * 3)
    let o = 0
    for (const b of todo) for (const p of b.plots) { dirs.set(p.probe, o); o += 15 }
    for (const b of todo) b.state = 1
    this.pending = true
    this.ctx.sampler.sampleAsync(dirs, 1.2e6, true).then(s => {
      let i = 0
      for (const b of todo) {
        for (const p of b.plots) {
          let lo = Infinity, hi = -Infinity
          for (let k = 0; k < 5; k++, i++) { lo = Math.min(lo, s[i * 8]); hi = Math.max(hi, s[i * 8]) }
          p.base = hi; p.found = hi - lo + 3; p.wet = lo < Math.max(0.3, p.lake + LAKE_DRY)
        }
        b.state = 2
      }
      this.pending = false
      this.dirty = true
    }, () => { this.pending = false; for (const b of todo) b.state = 0 })
  }

  // write the parts of ready blocks within range x lod + margin (relative to anchor A) and the smoke puffs
  // into the back buffers, nearest blocks first, then swap them in
  *fill(A, agl, now, margin) {
    const n = this.counts.fill(0), data = this.meshes.map(m => m.userData.stage[1]), SD = this.smokeStage[1]
    if (this.room) this.lodTarget = Math.min(1, this.lodTarget * 1.1)   // room: reach further
    if (this.smokeRoom) this.reachTarget = Math.min(SMOKE, this.reachTarget * 1.1)
    const k = Math.max(this.lod, this.lodTarget), reach = Math.max(this.reach, this.reachTarget)
    let puffs = 0, work = 0, over = false
    for (const b of this.visible) {
      if (b.state !== 2) continue
      const d3 = Math.hypot(b.dist, agl) - 400          // at most this far (block radius)
      if (d3 > TALL * k + margin) break
      if (++work > 32) { work = 0; yield }
      for (const p of b.plots) {
        if (p.wet) continue
        for (const q of p.parts) {   // sorted by range, farthest-seen first
          if (d3 > q.range * k + margin) break
          const sh = q.shape, r = R + p.base + q.z0, x = q.dir[0] * r - A.x, y = q.dir[1] * r - A.y, z = q.dir[2] * r - A.z
          const d = Math.sqrt(x * x + y * y + z * z)
          if (d > q.range * k + margin) continue
          if (n[sh] >= CAP[sh]) { over = true; continue }
          const D = data[sh], o = n[sh]++ * STRIDE
          D[o] = x; D[o + 1] = y; D[o + 2] = z
          D[o + 3] = q.dir[0]; D[o + 4] = q.dir[1]; D[o + 5] = q.dir[2]
          D.set(q.ex, o + 6)
          D[o + 9] = q.a; D[o + 10] = q.b; D[o + 11] = q.H
          D[o + 12] = q.found ? p.found : -1; D[o + 13] = q.seed; D[o + 14] = q.style; D[o + 15] = q.tone
          D[o + 16] = q.e0; D[o + 17] = q.e1
          D.set(q.ey, o + 18)
          D[o + 21] = q.range; D[o + 22] = p.born ??= now - FADE_IN   // fades in once when first drawn (a new block)
          D[o + 23] = q.sink
          if (q.stack && d < reach + margin && puffs + PUFFS + 1 > MAX_PUFFS) this.reachTarget = Math.min(this.reachTarget, d * 0.95)
          else if (q.stack && d < reach + margin) {
            const rt = R + p.base + q.H, m = q.stack, count = PUFFS + (m.H > 45 ? 1 : 0)
            for (let k = 0; k < count; k++) {
              const s = puffs++ * SMOKE_STRIDE
              SD[s] = q.dir[0] * rt - A.x; SD[s + 1] = q.dir[1] * rt - A.y; SD[s + 2] = q.dir[2] * rt - A.z
              SD[s + 3] = q.dir[0]; SD[s + 4] = q.dir[1]; SD[s + 5] = q.dir[2]
              SD[s + 6] = q.seed; SD[s + 7] = m.D; SD[s + 8] = k === PUFFS ? 2 : m.steam ? 1 : 0; SD[s + 9] = k / PUFFS + hash2i(q.seed, 1, 9)
            }
          }
        }
      }
    }
    // over a cap: the farthest were dropped, so pull the range in (they sink away as uLod eases down)
    if (over) this.lodTarget *= 0.85
    this.room = !over && n.every((c, i) => c < 0.7 * CAP[i])
    this.smokeRoom = puffs < 0.7 * MAX_PUFFS
    this.reachSet = this.reachTarget
    // swap: the back buffers become the drawn ones
    this.lodSet = this.lodTarget
    this.anchor.copy(A)
    const swap = (buf, stage, count, stride) => {
      stage.reverse(); buf.array = stage[0]
      buf.clearUpdateRanges(); buf.addUpdateRange(0, Math.max(count, 1) * stride); buf.needsUpdate = true
    }
    this.meshes.forEach((m, i) => { swap(m.userData.buf, m.userData.stage, n[i], STRIDE); m.geometry.instanceCount = n[i] })
    swap(this.smokeBuf, this.smokeStage, puffs, SMOKE_STRIDE)
    this.smoke.geometry.instanceCount = puffs
    this.stats.instances = n.reduce((a, b) => a + b, 0)
    this.stats.puffs = puffs
  }
}


// plots of one block (record b) with their parts placed in district coords (the painted plots, industry.glsl.js);
// parts in a road or rail corridor or reaching a trunk street (district edge, boulevard, ring) are dropped, and plots a
// river runs through (lake plots: requestHeights). Each plot is laid out in the district frame at its centre
// (districtAxes): district units -> metres along ex / ey (not always square, so boxes are parallelograms lining up
// with the streets; round shapes stay round).
function designBlock(b, segs, world) {
  const { city, D } = b
  const wet = cityWater(world, city, city.radius * CITY_REACH + PAD) && waterAt(world, localToDir(city, b.x, b.y), b.hd) !== -Infinity
  const near = segs.filter(s => segDist(b.x, b.y, s[0], s[1], s[2], s[3]) < b.hd + s[4] + 3)
  b.plots = []
  for (const pl of plots(b)) {
    const P = plan(D.hseed, pl), parts = [], f = frameAt(city, D, P.cx, P.cy), [ja, jb, jc, jd] = f.J
    const lake = wet ? waterAt(world, localToDir(city, f.lx, f.ly), Math.hypot(pl.x1 - pl.x0, pl.y1 - pl.y0) / 2 * Math.max(f.sx, f.sy)) : -Infinity
    if (lake === 'river') continue
    const det = ja * jd - jb * jc, f2 = ja * ja + jb * jb + jc * jc + jd * jd
    const smin = Math.sqrt(Math.max((f2 - Math.sqrt(Math.max(f2 * f2 - 4 * det * det, 0))) / 2, 1e-6))   // least metres per district unit
    // plot-centred district offsets <-> local metres (linear frame of the plot)
    const toDir = (gx, gy) => localToDir(city, ...f.toLocal(gx - P.cx, gy - P.cy))
    const toD = (px, py) => [P.cx + (jd * (px - f.lx) - jb * (py - f.ly)) / det, P.cy + (ja * (py - f.ly) - jc * (px - f.lx)) / det]
    const gsegs = near.map(s => [...toD(s[0], s[1]), ...toD(s[2], s[3]), s[4] / smin])
    const toXY = (l, s) => P.ax ? [P.cx + l, P.cy + s] : [P.cx + s, P.cy + l]
    let k = 0
    // keep-out: road / rail corridors and trunk streets; raised parts over a dropped ground part go too
    // (roof plant, teeth, galleries, gantry beams, spheres on their legs: nothing floats over a missing hall)
    const qs = design(P)
    for (const q of qs) {
      const [gx, gy] = toXY(q.l, q.s), alongX = P.ax !== q.alongS
      const hx = (alongX ? q.a : q.b) / 2, hy = (alongX ? q.b : q.a) / 2
      q.box = [gx - hx, gx + hx, gy - hy, gy + hy]  // footprint in district coords
      q.drop = gsegs.some(s => rectSegDist(q.box, s) < s[4]) || (!b.clear && !clearOf(city, b.rec, f.W, rectPts(...q.box), TRUNK_GAP))
    }
    const lost = qs.filter(q => q.drop && q.z0 === 0), over = (p, q) => p.box[0] < q.box[1] && q.box[0] < p.box[1] && p.box[2] < q.box[3] && q.box[2] < p.box[3]
    for (const q of qs) {
      if (q.drop || (q.z0 > 0 && lost.some(p => over(p, q)))) continue
      const [gx, gy] = toXY(q.l, q.s), alongX = P.ax !== q.alongS
      const ka = alongX ? f.sx : f.sy, kb = alongX ? f.sy : f.sx
      q.dir = toDir(gx, gy)
      q.ex = alongX ? f.ex : f.ey
      if (q.shape >= CYL) { q.a = q.b = q.a * Math.min(f.sx, f.sy); q.ey = cross(q.dir, q.ex) }   // round: square frame
      else { q.a *= ka; q.b *= kb; q.ey = alongX ? f.ey : neg(f.ex) }                            // right-handed
      if (q.style !== ST.STEEL) q.e1 *= ka          // sawtooth pitch runs along x (steel: vertical rise)
      q.seed = ((P.id[0] * 131 + P.id[1] * 977 + D.hseed) * 16 + k++) & 0x3fffff
      // by lod, and beyond NEAR out to (height / 0.0012) only; big main structures further (a 150 m shed reads from afar)
      const size = q.lod ? 0 : 0.08 * Math.sqrt(q.a * q.b)
      q.range = Math.min(q.lod === 2 ? FINE : q.lod === 1 ? NEAR : q.tall ? TALL : FAR, Math.max(NEAR, (q.z0 + q.H + size) / 0.0012))
      if (q.tall) q.range = Math.min(q.range, Math.max(NEAR, q.a / 0.0003))   // chimneys: gone before thinner than ~1/3 px (no shimmer)
      // sink depth at the range edge: roof parts into their roof, the rest (and a conveyor's rise) into the ground
      q.sink = q.roof ? q.H + 0.5 : q.z0 + q.H + (q.style === ST.STEEL ? Math.abs(q.e1) : 0) + 1
      parts.push(q)
    }
    if (!parts.length) continue
    parts.sort((a, b) => b.range - a.range)
    const hx = (pl.x1 - pl.x0) / 2 - 3, hy = (pl.y1 - pl.y0) / 2 - 3, cx = (pl.x0 + pl.x1) / 2, cy = (pl.y0 + pl.y1) / 2
    const probe = [[cx, cy], [cx + hx, cy + hy], [cx - hx, cy + hy], [cx + hx, cy - hy], [cx - hx, cy - hy]].flatMap(([gx, gy]) => toDir(gx, gy))
    b.plots.push({ parts, probe, base: 0, found: 3, wet: false, lake, born: null })
  }
  b.state = b.plots.length ? 0 : 2
}

// ---- unit shapes: x, y in [-0.5, 0.5], z in [0, 1] ----
function mesh() {
  const pos = [], nor = [], u = [], idx = []
  const quad = (pts, n, us = [0, 0, 0, 0]) => {
    const b = pos.length / 3
    pts.forEach((p, i) => { pos.push(...p); nor.push(...n); u.push(us[i]) })
    for (let k = 1; k < pts.length - 1; k++) idx.push(b, b + k, b + k + 1)
  }
  return { pos, nor, u, idx, quad }
}
// polygons are wound counter-clockwise seen from outside
function unitBox() {
  const m = mesh()
  m.quad([[0.5, -0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 1], [0.5, -0.5, 1]], [1, 0, 0])
  m.quad([[-0.5, 0.5, 0], [-0.5, -0.5, 0], [-0.5, -0.5, 1], [-0.5, 0.5, 1]], [-1, 0, 0])
  m.quad([[0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, 0.5, 1], [0.5, 0.5, 1]], [0, 1, 0])
  m.quad([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, -0.5, 1], [-0.5, -0.5, 1]], [0, -1, 0])
  m.quad([[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]], [0, 0, 1])
  return m
}
// one sawtooth bay: glazed vertical face at x = -0.5, roof sloping down to x = +0.5 (normal marked (1, 0, 1))
function unitTooth() {
  const m = mesh()
  m.quad([[-0.5, 0.5, 0], [-0.5, -0.5, 0], [-0.5, -0.5, 1], [-0.5, 0.5, 1]], [-1, 0, 0])
  m.quad([[-0.5, -0.5, 1], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 1]], [1, 0, 1])
  m.quad([[-0.5, -0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 1]], [0, -1, 0])
  m.quad([[0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, 0.5, 1]], [0, 1, 0])
  return m
}
function unitCylinder(n) {
  const m = mesh()
  for (let i = 0; i < n; i++) {
    const a0 = i / n * 2 * Math.PI, a1 = (i + 1) / n * 2 * Math.PI, c0 = Math.cos(a0), s0 = Math.sin(a0), c1 = Math.cos(a1), s1 = Math.sin(a1)
    const b = m.pos.length / 3
    m.pos.push(0.5 * c0, 0.5 * s0, 0, 0.5 * c1, 0.5 * s1, 0, 0.5 * c1, 0.5 * s1, 1, 0.5 * c0, 0.5 * s0, 1)
    m.nor.push(c0, s0, 0, c1, s1, 0, c1, s1, 0, c0, s0, 0)
    m.u.push(i / n, (i + 1) / n, (i + 1) / n, i / n)
    m.idx.push(b, b + 1, b + 2, b, b + 2, b + 3)
  }
  m.quad(Array.from({ length: n }, (_, i) => [0.5 * Math.cos(i / n * 2 * Math.PI), 0.5 * Math.sin(i / n * 2 * Math.PI), 1]), [0, 0, 1], new Array(n).fill(0))
  return m
}
function unitSphere(nu, nv) {
  const m = mesh()
  for (let j = 0; j <= nv; j++) for (let i = 0; i <= nu; i++) {
    const th = j / nv * Math.PI, ph = i / nu * 2 * Math.PI, x = Math.sin(th) * Math.cos(ph), y = Math.sin(th) * Math.sin(ph), z = -Math.cos(th)
    m.pos.push(0.5 * x, 0.5 * y, 0.5 + 0.5 * z); m.nor.push(x, y, z); m.u.push(i / nu)
  }
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
    const a = j * (nu + 1) + i, b = a + nu + 1
    m.idx.push(a, a + 1, b + 1, a, b + 1, b)
  }
  return m
}
