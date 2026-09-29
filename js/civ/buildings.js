// Buildings: 3D buildings on city lots near the camera. Contracts in notes/CIV.md, layout in notes/layout.md.
// Blocks come from the district layout (spec.js): each district's blocks are enumerated once in its own coords and
// cached (shared helpers below, also used by factories.js); each lot gets one building designed from the painted
// lot hash (the largest on screen first), ground heights come from a coarse grid
// sampled once per city (refined by exact GPU probes per block), and all visible buildings are drawn as one
// instanced mesh. The instance set is rebuilt only when the camera
// has moved: positions are stored relative to an anchor and shifted by a uniform every frame.
// No popping: every part has a range (by its lot's height); over the last 20% of it the shader slides the
// part into the ground (rooftop detail into its roof) from the true distance every frame, so a building
// shrinks onto its painted roof. The set holds everything within range + a margin for the camera travel
// until the next set. Rebuilds run a few ms per frame (a generator) into a back buffer and swap when done.
// Over MAX_INST the uLod range scale eases down, so the smallest on screen (range / distance) sink away
// first; it creeps back up once there is room.
import * as THREE from 'three'
import { R } from '../config.js'
import {
  TIER, HALF, DKIND, PROFILES, hash2i, warpGrid, localToGrid, localToDir, dirToLocal, dirToGrid, cityFrame, CITY_REACH,
  district, districtAt, districtPitch, districtEdges, toLattice, fromLattice, toDistrict, fromDistrict, rowLine, rowAt,
  rowConnectors, blockInfo, boulevards, ringRadius, BOULEVARD_END,
} from './spec.js'
import { objUniforms } from './objects.glsl.js'
import { VERT, FRAG } from './buildings.glsl.js'
import { LAKE_DRY, cityWater, waterAt } from './shore.js'

const NEAR = 2500           // every building within this (3D) distance (m)
const FAR = 9000            // beyond NEAR a building of height h out to 250 h, up to this
const SKY = 22000           // dense city blocks: tall buildings out to 250 h, up to this (cities and metros)
const MASS_K = 600          // far tier: a block's mid-rise mass out to 600 x its mean height (up to SKY)
const MASS_MIN = 12         // no mass for blocks lower than this (houses and gardens stay paint)
const MASS_LOT = 50         // lots this tall stay separate (towers and high-rises make the skyline)
const TOWER_LOT = 24        // metro towers only on lots at least this wide (painted tower roofs on smaller lots sit under a mid-rise)
const MAX_INST = 75000      // drawn instances (soft: uLod keeps about this many in range)
const STAGE = 100000        // hard cap per set: beyond it only the largest on screen are kept
const STRIDE = 24
const PAD = 600             // grid distances differ from metres by less than this (the warp is bounded)
const NEW_BLOCKS = 4000     // block designs per rebuild, largest on screen first (the rest follow in the next one;
                            // x3 once a first set is drawn while streaming: fewer rescans and refills per block)
const ENUM = 40             // ms of new districts enumerated per rebuild (nearest first)
const SAMPLE_PTS = 60000    // points per exact height request
const GRID = 300            // city ground grid spacing (m): city ground is smoothed at ~2.5 km (civ.glsl.js terrainCiv)
const CURV = 30             // grid nodes whose second difference exceeds this (m) are not trusted (rough ground)
const EXACT = 1500          // closer than this, blocks wait for exact heights; farther they start from the grid
const BUDGET = 2.5          // rebuild CPU per frame (ms)
const WAIT = 1              // yielded by a rebuild to wait for the next frame (a ground grid on its way)
const FADE_IN = 0.4         // s: new blocks start this far into the shader's 0.8 s dither-in (so they fade in over 0.4 s)
const WARM_PTS = 250000     // ground grid points per warm-up request (big cities' grids are sampled before they're needed)
const HOUSE = 0, BRICK = 1, CONCRETE = 2, GLASS = 3, RIBBON = 4, PLANT = 5
const DETAIL = 1500         // rooftop plant rooms and masts only this close
const FINE = 600            // rooftop HVAC units only this close
// share of windows lit by local hour (0..23): homes peak in the evening, offices in working hours
const HOMES = [0.12, 0.08, 0.06, 0.05, 0.05, 0.07, 0.14, 0.22, 0.16, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08, 0.1, 0.14, 0.22, 0.3, 0.34, 0.34, 0.32, 0.26, 0.18]
const OFFICES = [0.04, 0.03, 0.03, 0.03, 0.03, 0.04, 0.08, 0.25, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.55, 0.42, 0.26, 0.16, 0.11, 0.08, 0.06, 0.05]
const TOWERS = [0.1, 0.08, 0.07, 0.07, 0.07, 0.08, 0.12, 0.3, 0.55, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.6, 0.5, 0.42, 0.35, 0.29, 0.24, 0.2, 0.15]   // cleaners, late shifts, lights left on
const SHOPS = [0.25, 0.2, 0.15, 0.15, 0.15, 0.2, 0.4, 0.7, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.8, 0.6, 0.45, 0.3]
const curve = (t, h) => { const i = Math.floor(h), f = h - i; return t[i % 24] + (t[(i + 1) % 24] - t[i % 24]) * f }

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const neg = a => [-a[0], -a[1], -a[2]]
// distance from (x, y) to segment a-b in 2D
function segDist(x, y, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, t = Math.min(1, Math.max(0, ((x - ax) * dx + (y - ay) * dy) / Math.max(dx * dx + dy * dy, 1e-9)))
  return Math.hypot(x - ax - dx * t, y - ay - dy * t)
}

// ---- district blocks (shared with factories.js; layout in notes/layout.md) ----
// A district record holds the district's Voronoi cell (lattice-space edges with outward normals), its bounds in
// district coords, the trunk streets that may cut it, and (once enumerated by the one consumer of its kind) its blocks.
const LAT_R1 = 0.8                                   // spec.js log-radial lattice scale (radii)
const EDGE_HALF = PROFILES[2].half, TRUNK_HALF = PROFILES[3].half
export const TRUNK_GAP = 2.5                       // footprints keep this far past a trunk's half width (m)
const RECS = new WeakMap()                          // district -> record
const frameOf = city => city._frame ??= cityFrame(city.dir, city.angle)   // same as spec.js
const bkey = (kx, ky) => (kx + 2048) * 4096 + ky + 2048

function makeRec(city, D) {
  const [sa, sb] = toLattice(city, D.ou, D.ov), edges = [], pts = []
  for (const e of districtEdges(city, D)) {
    let nx = e.b[1] - e.a[1], ny = e.a[0] - e.b[0]
    const l = Math.hypot(nx, ny) || 1
    nx /= l; ny /= l
    if (nx * (sa - e.a[0]) + ny * (sb - e.a[1]) > 0) { nx = -nx; ny = -ny }   // outward, towards the neighbour
    edges.push([e.a[0], e.a[1], nx, ny])
    for (let t = 0; t < 8; t++) pts.push(fromLattice(city, e.a[0] + (e.b[0] - e.a[0]) * t / 8, e.a[1] + (e.b[1] - e.a[1]) * t / 8))
  }
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, rad = 0
  for (const [u, v] of pts) {
    const [x, y] = toDistrict(D, u, v)
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y)
    rad = Math.max(rad, Math.hypot(u - D.ou, v - D.ov))
  }
  const [x, y] = warpGrid(city, D.ou, D.ov), bl = boulevards(city)
  const rec = {
    D, city, edges, x0, x1, y0, y1, x, y, rad: rad * 1.15 + 50, blocks: null, byKey: null, seen: 0,
    bl: bl.map(a => [Math.cos(a), Math.sin(a)]),
    lined: [...bl.map((a, i) => hash2i(i, 0, city.seed + 25) < 0.6), hash2i(9, 1, city.seed + 25) < 0.5],   // trunks lined with high-rises
  }
  RECS.set(D, rec)
  return rec
}

// district records (want(D) only) whose cells may come within rho grid units of grid (cu, cv), clamped to the city's
// reach. The log-radial lattice map is a contraction (lattice distance <= grid distance), so cells are picked in lattice space
export function* districtRecs(city, cu, cv, rho, want) {
  const P = districtPitch(city), [a, b] = toLattice(city, cu, cv), out = []
  const Lr = toLattice(city, city.radius * CITY_REACH + PAD, 0)[0] + 1.5 * P, rl = rho + 1.5 * P
  let n = 0
  for (let j = Math.floor(Math.max(b - rl, -Lr) / P); j <= Math.floor(Math.min(b + rl, Lr) / P); j++) {
    for (let i = Math.floor(Math.max(a - rl, -Lr) / P); i <= Math.floor(Math.min(a + rl, Lr) / P); i++) {
      const D = district(city, i, j)
      if (!want(D)) continue
      let r = RECS.get(D)
      if (!r) { r = makeRec(city, D); if ((++n & 15) === 0) yield }
      out.push(r)
    }
    yield
  }
  return out
}

// signed metres from grid (u, v) to the trunks that may cut district rec: its cell edges (> 0 inside), then each boulevard
// and the ring (signed across). W = warp Jacobian [a, b, c, d] near the point. Same maths as spec.js trunkStreets
export function trunkDist(city, rec, u, v, W, out) {
  const [a, b, c, d] = W, det = a * d - b * c
  const m = (gx, gy) => 1 / Math.hypot((d * gx - c * gy) / det, (-b * gx + a * gy) / det)
  const R1 = LAT_R1 * city.radius, r = Math.hypot(u, v), gu = r > 1e-3 ? u / r : 1, gv = r > 1e-3 ? v / r : 0
  const s = r < 1e-3 ? 1 : R1 * Math.log1p(r / R1) / r, s1 = 1 / (1 + r / R1), la = u * s, lb = v * s
  let k = 0
  for (const [ax, ay, nx, ny] of rec.edges) {
    const gn = (s1 - s) * (gu * nx + gv * ny)
    out[k++] = (nx * (ax - la) + ny * (ay - lb)) * m(s * nx + gn * gu, s * ny + gn * gv)
  }
  for (const [ex, ey] of rec.bl) {
    const t = u * ex + v * ey
    out[k++] = t > 0 && t < BOULEVARD_END * city.radius ? (-u * ey + v * ex) * m(-ey, ex) : 1e9
  }
  if (city.tier >= 2) {
    const [Rr, dR] = ringRadius(city, Math.atan2(v, u)), q = dR / Math.max(r, 1)
    out[k++] = (r - Rr) * m(gu + q * gv, gv - q * gu)
  }
  return k
}
// is the district-coords polygon pts clear of every trunk of rec by margin (m)? all corners inside the cell and on one
// side of each boulevard / the ring (a convex footprint then stays clear, to first order)
export function clearOf(city, rec, W, pts, margin) {
  const D = rec.D, T = pts.map(([x, y]) => { const o = []; trunkDist(city, rec, ...fromDistrict(D, x, y), W, o); return o })
  const nE = rec.edges.length
  for (let k = 0; k < T[0].length; k++) {
    let lo = Infinity, pos = 0, neg = 0
    for (const d of T) { lo = Math.min(lo, Math.abs(d[k])); if (d[k] > 0) pos++; else neg++ }
    if (lo < (k < nE ? EDGE_HALF : TRUNK_HALF) + margin || (k < nE ? neg : pos && neg)) return false
  }
  return true
}
export const rectPts = (x0, x1, y0, y1) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

// enumerate the blocks of district rec once: blocks whose rect overlaps the cell, built up and keep(info). make(fields)
// returns the consumer's block record. Fields: city, D, rec, kx, ky, info (blockInfo), dens, x, y (local metres of the
// centre), W (warp Jacobian there), hd (half diagonal, m), clear (no trunk within hd: lots need no trunk test), avenue
export function* enumBlocks(city, rec, keep, make) {
  const D = rec.D, blocks = [], byKey = new Map(), m = Math.abs(D.A) + 30, nE = rec.edges.length, d = []
  for (let ky = Math.floor((rec.y0 - m - D.oy) / D.by) - 1; ky <= Math.floor((rec.y1 + m - D.oy) / D.by) + 1; ky++) {
    const { bx, off } = rowConnectors(D, ky), y0 = rowLine(D, ky), y1 = rowLine(D, ky + 1)
    const hdl = 0.5 * Math.hypot(bx, y1 - y0) + Math.abs(D.A)
    for (let kx = Math.floor((rec.x0 - m - off) / bx); kx <= Math.floor((rec.x1 + m - off) / bx); kx++) {
      const rect = { x0: kx * bx + off, x1: (kx + 1) * bx + off, y0, y1 }
      const [u, v] = fromDistrict(D, (rect.x0 + rect.x1) / 2, (y0 + y1) / 2), [la, lb] = toLattice(city, u, v)
      let e = Infinity
      for (const [ax, ay, nx, ny] of rec.edges) e = Math.min(e, nx * (ax - la) + ny * (ay - lb))
      const r = Math.hypot(u, v), s = r < 1e-3 ? 1 : LAT_R1 * city.radius * Math.log1p(r / (LAT_R1 * city.radius)) / r
      if (e < -hdl * s) continue                              // wholly outside the cell (lattice metres <= s grid metres)
      const info = blockInfo(city, D, kx, ky, rect)
      if (info.density <= 0 || !keep(info)) continue
      const [x, y, a, b, c, dd] = warpGrid(city, u, v), W = [a, b, c, dd], hd = hdl * 1.2
      const n = trunkDist(city, rec, u, v, W, d)
      let clear = true, avenue = false
      for (let k = 0; k < n; k++) {
        const g = k < nE ? d[k] : Math.abs(d[k]), half = k < nE ? EDGE_HALF : TRUNK_HALF
        if (g < hd + half + TRUNK_GAP + 1) clear = false
        if (k >= nE && g < hd + half + 25 && rec.lined[k - nE]) avenue = true
      }
      const blk = make({ city, D, rec, kx, ky, info, dens: info.density, x, y, W, hd, clear, avenue })
      blocks.push(blk); byKey.set(bkey(kx, ky), blk)
    }
    yield
  }
  rec.blocks = blocks; rec.byKey = byKey
}
// the cached block record under grid (u, v), if its district was enumerated
export function blockAt(city, u, v) {
  const D = districtAt(city, u, v).D, rec = RECS.get(D)
  if (!rec?.byKey) return null
  const [x, y] = toDistrict(D, u, v), row = rowAt(D, y), { bx, off } = rowConnectors(D, row.ky)
  return { b: rec.byKey.get(bkey(Math.floor((x - off) / bx), row.ky)), x, y }
}
// frame at district point (x, y): grid (u, v), local metres (lx, ly), warp Jacobian W, local metres per district unit
// J = [dlx/dx, dlx/dy, dly/dx, dly/dy], world unit tangents ex / ey of district +x / +y, metres per unit sx / sy
// (spec.js districtAxes with one warp)
export function frameAt(city, D, x, y) {
  const [u, v] = fromDistrict(D, x, y), [lx, ly, a, b, c, d] = warpGrid(city, u, v), q = D.A * D.k * Math.cos(D.k * x)
  const tx0 = D.c - q * D.s, tx1 = D.s + q * D.c
  const jxx = a * tx0 + b * tx1, jyx = c * tx0 + d * tx1, jxy = -a * D.s + b * D.c, jyy = -c * D.s + d * D.c
  const sx = Math.hypot(jxx, jyx), sy = Math.hypot(jxy, jyy), [fe, fn] = frameOf(city)
  return {
    u, v, lx, ly, W: [a, b, c, d], J: [jxx, jxy, jyx, jyy], sx, sy,
    ex: fe.map((e, i) => (e * jxx + fn[i] * jyx) / sx), ey: fe.map((e, i) => (e * jxy + fn[i] * jyy) / sy),
    toLocal: (ox, oy) => [lx + jxx * ox + jxy * oy, ly + jyx * ox + jyy * oy],   // district-unit offsets -> local metres
  }
}
// how far district rows bend away from a straight frame over a half width w (district units): the shear's
// A k^2 w^2 / 2 (the base warp adds < 0.5 m over a lot)
export const shearBend = (D, w) => Math.abs(D.A) * D.k * D.k * w * w / 2

export class Buildings {
  constructor(ctx) {
    this.ctx = ctx
    this.world = ctx.world
    this.recs = new Set()        // district records whose blocks (designs, heights) are cached
    this.nBlocks = 0
    this.grounds = new Map()     // city id -> coarse ground grid
    this.visible = []
    this.cities = []             // cities near the camera at the last rebuild
    this.anchor = new THREE.Vector3()
    this.lastPos = new THREE.Vector3(1e9, 0, 0)
    this.dirty = true
    this.enabled = true          // debug toggle
    this.pending = false
    this.tick = 0
    this.job = null              // the rebuild in progress (generator)
    this.lod = this.lodTarget = this.lodSet = 1   // range scale: drawn, wanted, the drawn set's
    this.stage = [0, 1].map(() => new Float32Array(STAGE * STRIDE))   // front (drawn) and back (being filled)
    this.rank = new Float32Array(STAGE)
    this.mesh = this.makeMesh()
    ctx.scene.add(this.mesh)
    this.stats = { rebuilds: 0, cpuMs: 0, sliceMs: 0, frames: 0, gpuMs: 0, lod: 1 }
    this.gpuTimer(ctx.renderer.getContext())
  }

  // debug: GPU time of the building draw (EXT_disjoint_timer_query_webgl2 when available)
  gpuTimer(gl) {
    const ext = gl.getExtension('EXT_disjoint_timer_query_webgl2')
    if (!ext) return
    let q = null, active = false
    this.mesh.onBeforeRender = () => {
      if (q && !active && gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) {
        this.stats.gpuMs += (gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6 - this.stats.gpuMs) * 0.1
        q = null
      }
      if (!q) { q = gl.createQuery(); gl.beginQuery(ext.TIME_ELAPSED_EXT, q); active = true }
    }
    this.mesh.onAfterRender = () => { if (active) { gl.endQuery(ext.TIME_ELAPSED_EXT); active = false } }
  }

  makeMesh() {
    const geo = new THREE.InstancedBufferGeometry()
    const { pos, nor, idx } = unitHouse()
    geo.setIndex(idx)
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3))
    this.buf = new THREE.InstancedInterleavedBuffer(this.stage[0], STRIDE).setUsage(THREE.DynamicDrawUsage)
    const at = (name, size, off) => geo.setAttribute(name, new THREE.InterleavedBufferAttribute(this.buf, size, off))
    at('iPos', 3, 0); at('iUp', 3, 3); at('iEx', 3, 6); at('iSize', 4, 9); at('iMisc', 4, 13); at('iEy', 3, 17); at('iFade', 4, 20)
    geo.instanceCount = this.count = 0
    this.uniforms = { ...objUniforms(this.ctx.shared, this.ctx.FAR), uAnchor: { value: new THREE.Vector3() }, uAct: { value: new THREE.Vector4() }, uLod: { value: 1 } }
    const mat = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: VERT, fragmentShader: FRAG, uniforms: this.uniforms })
    const mesh = new THREE.Mesh(geo, mat)
    mesh.frustumCulled = false
    return mesh
  }

  update(frame) {
    this.now = frame.time
    const step = 40 + 0.05 * Math.max(frame.geo.agl, 0), moved = frame.camPos.distanceTo(this.lastPos)
    if (!this.job && (this.dirty || moved > step)) {
      this.urgent = this.dirty || this.count === 0 || moved > 4 * step   // streaming a new area: may use the frame's room
      this.job = this.build(frame, step)
    }
    else if (!this.job && this.warm) this.stepWarm()
    if (this.job) {
      const t0 = performance.now(), budget = this.urgent ? Math.max(BUDGET, 0.7 * (frame.budget ?? 0)) : BUDGET
      let r
      do r = this.job.next(); while (!r.done && r.value !== WAIT && performance.now() - t0 < budget)
      const ms = performance.now() - t0, s = this.stats
      this.jobMs += ms; this.jobMax = Math.max(this.jobMax, ms); this.jobFrames++
      if (r.done) { this.job = null; s.cpuMs = this.jobMs; s.sliceMs = this.jobMax; s.frames = this.jobFrames }
    }
    this.lod += (this.lodSet - this.lod) * Math.min(1, frame.dt * 1.5)   // only as far as the drawn set reaches
    this.uniforms.uLod.value = this.stats.lod = this.lod
    this.uniforms.uAnchor.value.subVectors(this.anchor, frame.camPos)
    const hour = ((frame.simTime / 3600 + frame.geo.lon * 12 / Math.PI) % 24 + 24) % 24
    this.uniforms.uAct.value.set(curve(HOMES, hour), curve(OFFICES, hour), curve(SHOPS, hour), curve(TOWERS, hour))
    this.mesh.visible = this.enabled && this.mesh.geometry.instanceCount > 0
  }

  // one rebuild, a slice per frame: collect blocks, request heights, fill the back buffer, swap
  *build(frame, step) {
    this.jobMs = this.jobMax = this.jobFrames = 0
    this.dirty = false
    const A = frame.camPos.clone(), agl = Math.max(frame.geo.agl, 0)
    this.lastPos.copy(A)
    this.tick++
    const c = frame.around ?? frame.camDir         // the destination while too high to see any (civ/index.js)
    yield* this.collect([c.x, c.y, c.z])
    for (let i = 0; i < 60 && this.cities.some(c => this.grounds.get(c.id)?.h === null); i++) yield WAIT   // grids on their way
    let work = 0
    for (const b of this.visible) {
      if (b.state !== 0 || Math.hypot(b.dist, agl) < EXACT) continue
      settle(b, this.grounds.get(b.city.id))
      if (++work > 100) { work = 0; yield }
    }
    yield* this.requestHeights()
    if (this.grounds.size > 40) for (const [id, g] of this.grounds) if (g.h && !g.keep && !this.cities.some(c => c.id === id)) this.grounds.delete(id)
    yield* this.fill(A, agl, this.now, 2 * step + 60)  // margin: travel until the next set is drawn; new parts fade in from now
    if (this.nBlocks > 150000) {   // forget districts out of sight (their blocks are enumerated again when needed)
      for (const r of this.recs) if (r.seen < this.tick - 3 && !r.blocks.some(b => b.state === 1)) { this.nBlocks -= r.blocks.length; r.blocks = r.byKey = null; this.recs.delete(r) }
      yield
    }
    this.stats.rebuilds++
  }

  // built-up blocks around the camera, district by district (nearest first); enumerates new districts within ENUM
  // blocks and designs new blocks within NEW_BLOCKS, largest on screen first
  *collect(cam) {
    const near = this.world.near(cam, SKY), visible = [], cities = [], grids = []
    const todo = Array.from({ length: 64 }, () => [])   // blocks to design, binned by expected size on screen
    let work = 0, left = 0, enumerated = 0
    for (const city of near.cities) {
      const [lx, ly] = dirToLocal(city, cam), reach = city.radius * CITY_REACH + PAD
      if (Math.hypot(lx, ly) > reach + SKY) continue
      cities.push(city)
      if (!this.grounds.has(city.id)) yield* this.groundGrid(city, grids)
      const range = city.tier >= TIER.CITY ? SKY : FAR
      const [cu, cv] = dirToGrid(city, cam), segs = this.localSegs(near.segs, city, lx, ly, range)
      const recs = yield* districtRecs(city, cu, cv, range / 0.85 + PAD, D => D.kind !== DKIND.INDUSTRY)   // factories.js builds industry
      for (const r of recs) r.dist = Math.hypot(r.x - lx, r.y - ly) - r.rad
      recs.sort((a, b) => a.dist - b.dist)
      for (const rec of recs) {
        if (rec.dist > range + 300) break
        rec.seen = this.tick
        if (!rec.blocks) {
          if (enumerated > ENUM) { left++; continue }
          const gen = enumBlocks(city, rec, i => i.zone === 0 && !i.park, newBlock)
          for (;;) { const t0 = performance.now(), r = gen.next(); enumerated += performance.now() - t0; if (r.done) break; yield }
          this.nBlocks += rec.blocks.length
          this.recs.add(rec)
        }
        for (const b of rec.blocks) {
          if (++work > 400) { work = 0; yield }
          const d = Math.hypot(b.x - lx, b.y - ly)        // metres, from the cached block centre
          if (d > range + 300) continue                     // + block radius: lots at the edge are faded out
          b.dist = d; b.seen = this.tick
          if (b.state !== 4) visible.push(b)
          else if (d < b.reach + PAD) { todo[bin(b.guess / Math.max(d, 100) * 64)].push([b, segs]); left++ }   // as far as its buildings or mass may be drawn
        }
      }
    }
    if (grids.length) this.requestGrids(grids)
    // design new blocks, the largest on screen first, so the skyline comes before the houses at your feet
    let t0 = performance.now()
    this.stats.designMs = 0
    for (let i = 63, quota = this.urgent && this.count ? 3 * NEW_BLOCKS : NEW_BLOCKS; i >= 0 && quota > 0; i--) {
      for (const [b, segs] of todo[i].slice(0, quota)) {
        designBlock(b, segs, this.world)
        visible.push(b)
        quota--; left--
        if (++work > 10 || performance.now() - t0 > 0.8) { work = 0; this.stats.designMs += performance.now() - t0; yield; t0 = performance.now() }
      }
    }
    if (left > 0) this.dirty = true
    this.visible = visible
    this.cities = cities
  }

  // coarse ground grid over a city's reach in its local plane, sampled once. City ground is smoothed at ~2.5 km:
  // bilinear heights are off by 0.3 m (median), 1 m (p90), 2.6 m (p99), so they only place blocks beyond EXACT
  // until the exact probes arrive; never near the sea (h < 3 m: wet lots) or on rough ground (outside the city).
  *groundGrid(city, grids, register = true) {
    const half = city.radius * CITY_REACH + PAD, n = Math.ceil(2 * half / GRID) + 1
    const g = { x0: -half, n, h: null, bad: null, dirs: new Float32Array(n * n * 3) }
    if (register) this.grounds.set(city.id, g)
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) g.dirs.set(localToDir(city, g.x0 + i * GRID, g.x0 + j * GRID), (j * n + i) * 3)
      if ((j & 31) === 31) yield
    }
    grids.push([city.id, g])
  }
  // all new grids in one sampler request (a rebuild follows unless it's a warm-up)
  requestGrids(grids, rebuild = true) {
    const dirs = new Float32Array(grids.reduce((n, [, g]) => n + g.dirs.length, 0))
    let o = 0
    for (const [, g] of grids) { dirs.set(g.dirs, o); o += g.dirs.length; g.dirs = null }
    this.ctx.sampler.sampleAsync(dirs, 1.2e6, true).then(s => {
      let o = 0
      for (const [, g] of grids) {
        const n = g.n, h = g.h = new Float32Array(n * n), bad = g.bad = new Uint8Array(n * n).fill(1)
        for (let k = 0; k < n * n; k++, o += 8) h[k] = s[o]
        for (let j = 1; j < n - 1; j++) for (let i = 1; i < n - 1; i++) {
          const k = j * n + i, c = 2 * h[k]
          const curv = Math.max(Math.abs(h[k - 1] + h[k + 1] - c), Math.abs(h[k - n] + h[k + n] - c), Math.abs(h[k + n + 1] - h[k + n - 1] - h[k - n + 1] + h[k - n - 1]) / 2)
          bad[k] = h[k] < 3 || curv > CURV ? 1 : 0
        }
      }
      if (rebuild) this.dirty = true
    }, () => { for (const [id] of grids) this.grounds.delete(id) })
  }

  // warm-up (main.js, while the loading curtain is up and after): the ground grids of every city and metro, biggest
  // first, a slice per idle frame, in batched requests, so a zoom into one doesn't wait for its grid. They are kept.
  prewarm() {
    this.warm ??= this.warmGrids()
  }
  stepWarm() {
    const t0 = performance.now()
    while (performance.now() - t0 < BUDGET) if (this.warm.next().done) { this.warm = false; break }
  }
  *warmGrids() {
    const big = this.world.cities.filter(c => c.tier >= TIER.CITY).sort((a, b) => b.pop - a.pop)
    let grids = [], pts = 0, total = 0
    // registered only when requested (a rebuild waits for registered grids); one a rebuild made meanwhile wins
    const send = () => {
      grids = grids.filter(([id]) => !this.grounds.has(id))
      for (const [id, g] of grids) { g.keep = true; this.grounds.set(id, g) }
      if (grids.length) this.requestGrids(grids, false)
      grids = []; pts = 0
    }
    for (const city of big) {
      if (this.grounds.has(city.id)) continue
      yield* this.groundGrid(city, grids, false)
      const n = grids[grids.length - 1][1].n ** 2
      total += n
      if ((pts += n) >= WARM_PTS) send()
      yield
    }
    send()
    console.info(`buildings: ground grids of ${big.length} cities (${total} points) requested at ${Math.round(performance.now())} ms`)
  }

  // road/rail segments near the camera in the city's local plane (metres): [ax, ay, bx, by, half]
  localSegs(ids, city, cx, cy, range) {
    const S = this.world.seg, out = []
    const loc = o => dirToLocal(city, [S[o], S[o + 1], S[o + 2]])
    for (const k of ids) {
      const [ax, ay] = loc(k * 8), [bx, by] = loc(k * 8 + 4)
      if (segDist(cx, cy, ax, ay, bx, by) < range + 500) out.push([ax, ay, bx, by, HALF[S[k * 8 + 3]]])
    }
    return out
  }

  // roof height (m above sea level, plus 1 m) of a loaded building whose footprint contains dir, else null
  heightAt(dir) {
    const d = dir.isVector3 ? [dir.x, dir.y, dir.z] : dir
    let top = null
    for (const city of this.cities) {
      const [lx, ly] = dirToLocal(city, d)
      if (Math.hypot(lx, ly) > city.radius * CITY_REACH + PAD) continue
      const at = blockAt(city, ...localToGrid(city, lx, ly)), b = at?.b
      if (!b?.lots || b.state !== 2) continue   // (industrial blocks are factories.js records)
      const { x, y } = at
      for (const l of b.lots) {
        if (l.wet) continue
        for (const q of l.parts) {
          const [x0, x1, y0, y1] = q.box          // district coords
          if (x >= x0 && x <= x1 && y >= y0 && y <= y1) top = Math.max(top ?? -Infinity, l.base + q.z0 + q.H + q.roof + 1)
        }
      }
    }
    return top
  }

  // exact ground heights (5 probes per lot), nearest first, one request at a time: blocks waiting for them
  // (near, or where the grid isn't trusted) and blocks drawn on grid heights (refined in place)
  // (a generator: the nearest-first pick is binned by distance and the probes are built in slices)
  *requestHeights() {
    if (this.pending) return
    const bins = Array.from({ length: 64 }, () => [])
    let w = 0
    for (const b of this.visible) {
      if (b.state === 0 || b.refine) bins[Math.min(63, Math.floor(b.dist / 350))].push(b)
      if (++w > 5000) { w = 0; yield }
    }
    const todo = []
    let n = 0
    for (const bin of bins) for (const b of bin) if (n < SAMPLE_PTS) { todo.push(b); n += b.lots.length * 5 }
    if (!todo.length || this.pending) return
    const dirs = new Float32Array(n * 3)
    let o = 0
    for (const b of todo) {
      for (const l of b.lots) for (let k = 0; k < 10; k += 2, o += 3) dirs.set(localToDir(b.city, l.pxy[k], l.pxy[k + 1]), o)
      if (++w > 300) { w = 0; yield }
    }
    for (const b of todo) if (b.refine) b.refine = false; else b.state = 1
    this.pending = true
    this.ctx.sampler.sampleAsync(dirs, 1.2e6, true).then(s => {
      let i = 0
      for (const b of todo) {
        for (const l of b.lots) {
          let lo = Infinity, hi = -Infinity
          for (let k = 0; k < 5; k++, i++) { lo = Math.min(lo, s[i * 8]); hi = Math.max(hi, s[i * 8]) }
          setGround(l, lo, hi)
        }
        b.state = 2
      }
      this.pending = false
      this.dirty = true
    }, () => { this.pending = false; for (const b of todo) if (b.state === 1) b.state = 0; else b.refine = true })
  }

  // write the parts of ready blocks within range x lod + margin into the back buffer, relative to anchor A,
  // then swap. rank = range / distance (range grows with height, so it ranks by size on screen): a part is
  // drawn while rank > 1 / uLod. Over MAX_INST, lodTarget drops to where they fit and uLod eases there, so
  // the smallest sink away; under 80% it rises again. A full STAGE keeps only the largest (rank >= floor).
  *fill(A, agl, now, margin) {
    if (this.count < 0.8 * MAX_INST) this.lodTarget = Math.min(1, this.lodTarget * 1.1)   // room: reach further
    const D = this.stage[1], rank = this.rank, hist = new Uint32Array(64), k = Math.max(this.lod, this.lodTarget), k0 = Math.min(this.lod, this.lodTarget)
    let n = 0, work = 0, floor = 0
    // count the instance just written; a full STAGE keeps only the largest on screen
    const put = function* (s) {
      rank[n] = s; hist[bin(s)]++
      if (++n < STAGE) return
      floor = cutAt(hist, STAGE * 0.75)
      let m = 0
      for (let i = 0; i < n; i++) if (rank[i] >= floor) { if (m !== i) { D.copyWithin(m * STRIDE, i * STRIDE, (i + 1) * STRIDE); rank[m] = rank[i] } m++ }
      hist.fill(0, 0, bin(floor))
      n = m
      yield
    }
    for (const b of this.visible) {
      const d3 = Math.hypot(b.dist, agl) - 400
      if (b.state !== 2 || d3 > b.range * k + margin) continue
      if (++work > 64) { work = 0; yield }
      const m = b.mass
      if (m) {   // far tier: the block's mid-rise as one mass, rising over [0.8, 1] x r0 while its buildings sink
        let base = 0, found = 0, dry = 0
        for (const l of m.lots) if (!l.wet) { base += l.base; found = Math.max(found, l.found); dry++ }
        const r = R + base / Math.max(dry, 1), x = m.dir[0] * r - A.x, y = m.dir[1] * r - A.y, z = m.dir[2] * r - A.z
        const d = Math.sqrt(x * x + y * y + z * z), s = m.range / Math.max(d, 1)
        if (2 * dry > m.lots.length && d <= m.range * k + margin && d >= 0.8 * m.r0 * k0 - margin && s >= floor) {
          const o = n * STRIDE
          D[o] = x; D[o + 1] = y; D[o + 2] = z
          D.set(m.dir, o + 3); D.set(m.ex, o + 6)
          D[o + 9] = m.sx; D[o + 10] = m.sy; D[o + 11] = m.H; D[o + 12] = 0
          D[o + 13] = found; D[o + 14] = m.seed; D[o + 15] = m.style; D[o + 16] = 0.5
          D.set(m.ey, o + 17)
          D[o + 20] = m.range; D[o + 21] = m.born ??= now - FADE_IN; D[o + 22] = m.H + 2; D[o + 23] = m.r0
          yield* put(s)
        }
      }
      if (d3 > b.lotRange * k + margin) continue
      for (const l of b.lots) {
        if (l.wet) continue
        for (const q of l.parts) {
          const r = R + l.base + q.z0, x = q.dir[0] * r - A.x, y = q.dir[1] * r - A.y, z = q.dir[2] * r - A.z
          const d = Math.sqrt(x * x + y * y + z * z), s = q.range / Math.max(d, 1)
          if (d > q.range * k + margin || s < floor) continue
          const o = n * STRIDE
          D[o] = x; D[o + 1] = y; D[o + 2] = z
          D[o + 3] = q.dir[0]; D[o + 4] = q.dir[1]; D[o + 5] = q.dir[2]
          D.set(q.ex, o + 6)
          D[o + 9] = q.sx; D[o + 10] = q.sy; D[o + 11] = q.H; D[o + 12] = q.roof
          D[o + 13] = q.z0 > 0 ? -1 - q.z0 : l.found; D[o + 14] = q.seed; D[o + 15] = q.style; D[o + 16] = l.lh
          D.set(q.ey, o + 17)
          D[o + 20] = q.range; D[o + 21] = l.born ??= now - FADE_IN   // fades in once when first drawn (a new block)
          D[o + 22] = q.range < l.range ? q.H + q.roof + 0.3 : l.top + 2   // sink: rooftop detail into the roof, podiums and the rest into the ground
          D[o + 23] = 0
          yield* put(s)
        }
      }
    }
    if (n > MAX_INST) this.lodTarget = Math.min(this.lodTarget, 1 / cutAt(hist, MAX_INST))
    // swap: the back buffer becomes the drawn one
    this.lodSet = this.lodTarget
    this.stage.reverse()
    this.anchor.copy(A)
    this.buf.array = this.stage[0]
    this.buf.clearUpdateRanges()
    this.buf.addUpdateRange(0, Math.max(n, 1) * STRIDE)
    this.buf.needsUpdate = true
    this.mesh.geometry.instanceCount = this.count = n
  }
}

// rank histogram: 4 bins per octave of range / distance
const bin = s => Math.min(63, Math.max(0, Math.floor(Math.log2(s) * 4) + 16))
// the lowest rank (a bin edge) at which at most max are kept
function cutAt(hist, max) {
  let kept = 0, i = 63
  while (i >= 0 && kept + hist[i] <= max) kept += hist[i--]
  return i < 0 ? 0 : 2 ** ((i + 1 - 16) / 4)
}

const RING = [[1, 1], [1, -1], [-1, 1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]]

// block record (not designed yet: state 4) from enumBlocks fields
function newBlock(f) {
  const b = Object.assign(f, { state: 4, lots: [], range: 0, dist: 0, seen: 0 })
  b.suburb = f.D.kind === DKIND.SUBURB || f.dens < (f.city.tier === TIER.METRO ? 0.12 : 0.2)
  b.tc = f.city.tier === TIER.METRO && !b.suburb ? towerChance(f.city, f.D, f.dens) : 0
  b.hr = b.suburb ? 0 : highRise(f.city, f.D, f.kx, f.ky, f.dens, f.avenue)
  b.guess = guessTop(b)
  b.reach = reachGuess(b)
  return b
}

// how far the base warp strays from its straight tangent frame (metres per grid metre squared) around grid (u, v)
function bend(city, u, v, h) {
  const [x, y, a, b, c, d] = warpGrid(city, u, v)
  let e = 0
  for (const [s, t] of RING) {
    const [px, py] = warpGrid(city, u + s * h, v + t * h)
    e = Math.max(e, Math.hypot(px - x - (a * s + b * t) * h, py - y - (c * s + d * t) * h) / ((s * s + t * t) * h * h))
  }
  return e
}

// design the lots of block record b (blockInfo().lot split nx x ny, as painted), skipping those a road, railway or
// river runs through (lake lots: setGround) and, in blocks a trunk street may cut, those whose footprint would reach
// it or lie outside the district. Each lot is laid out in the district frame at its centre (districtAxes): district
// units x sx / sy are metres, the parts are parallelograms along both local streets.
function designBlock(b, segs, world) {
  const { city, D, info } = b, L = info.lot, W = (L.x1 - L.x0) / info.lots.nx, H = (L.y1 - L.y0) / info.lots.ny
  const wet = cityWater(world, city, city.radius * CITY_REACH + PAD) && waterAt(world, localToDir(city, b.x, b.y), b.hd) !== -Infinity
  const near = segs.filter(s => segDist(b.x, b.y, s[0], s[1], s[2], s[3]) < b.hd + s[4] + 3)
  const far = city.tier >= TIER.CITY ? SKY : FAR   // collect() designs blocks within reachGuess() only
  b.lots = []
  const dc = []
  for (let j = 0; j < info.lots.ny; j++) for (let i = 0; i < info.lots.nx; i++) {
    const cx = L.x0 + (i + 0.5) * W, cy = L.y0 + (j + 0.5) * H, f = frameAt(city, D, cx, cy)
    const hr = Math.hypot(W * f.sx, H * f.sy) / 2
    // roads and railways (world segments) into lot district units through the inverse lot Jacobian; clearance / the
    // least metres per unit, so a part is dropped if it could come within half + 1 m in metres
    const segs = near.length ? lotSegs(near, f, cx, cy, hr) : NONE
    if (segs.some(s => s[5] < 0)) continue                       // a road across the lot centre
    const lake = wet ? waterAt(world, localToDir(city, f.lx, f.ly), hr) : -Infinity
    if (lake === 'river') continue
    if (!b.clear) {                                              // wholly outside the district or under a trunk
      const n = trunkDist(city, b.rec, f.u, f.v, f.W, dc), nE = b.rec.edges.length
      let out = false
      for (let k = 0; k < n; k++) if (k < nE ? dc[k] < -hr : Math.abs(dc[k]) + hr < TRUNK_HALF) out = true
      if (out) continue
    }
    let l = design(city, D, b, i, j, W, H)
    // a trunk street (district edge, boulevard, ring) or a road through the lot: build on the part away from it (same
    // hashes, the largest of 0.75 .. 0.3 of the lot from one side that fits), else not at all
    if ((!b.clear || segs.length) && !fits(city, b, f, l, cx, cy, segs)) {
      l = null
      const sides = [[1, 0], [-1, 0], [0, 1], [0, -1]].filter(([ax, ay]) => {
        const c = cut(city, D, b, i, j, W, H, ax, ay, 0.3)
        return c && fits(city, b, f, c.l, cx + c.dx, cy + c.dy, segs)
      })
      for (const k of [0.75, 0.55, 0.4, 0.3]) {
        for (const [ax, ay] of sides) {
          const c = cut(city, D, b, i, j, W, H, ax, ay, k)
          if (c && fits(city, b, f, c.l, cx + c.dx, cy + c.dy, segs)) { l = c.l; for (const q of l.parts) { q.ox += c.dx; q.oy += c.dy } break }
        }
        if (l || !sides.length) break
      }
      if (!l) continue
    }
    l.lake = lake
    for (const q of l.parts) {
      q.box = [cx + q.ox - q.sx / 2, cx + q.ox + q.sx / 2, cy + q.oy - q.sy / 2, cy + q.oy + q.sy / 2]   // footprint, district coords
      q.dir = localToDir(city, ...f.toLocal(q.ox, q.oy))
      q.sx *= f.sx; q.sy *= f.sy                    // metres
      const swap = q.sy > q.sx                      // the gable ridge runs along the long axis
      if (swap) [q.sx, q.sy] = [q.sy, q.sx]
      q.ex = swap ? f.ey : f.ex
      q.ey = swap ? neg(f.ex) : f.ey                // keeps the frame right-handed
    }
    // ground probes (local metres): the lot centre and the corners of the built area
    const hx = W / 2 - 2.5, hy = H / 2 - 2.5
    l.pxy = [[0, 0], [hx, hy], [-hx, hy], [hx, -hy], [-hx, -hy]].flatMap(([ox, oy]) => f.toLocal(ox, oy))
    l.top = Math.max(...l.parts.map(q => q.z0 + q.H + q.roof))
    // drawn out to 250 x height (NEAR..far: SKY for dense city blocks); rooftop detail and podiums no farther than q.near
    // (the far cap varies per lot, so the tallest don't all end on one ring)
    const range = l.range = Math.min(Math.max(l.top * 250, NEAR), far * (0.7 + 0.3 * l.lh))
    for (const q of l.parts) q.range = Math.min(q.near, range)
    b.range = Math.max(b.range, range)
    b.lots.push(l)
  }
  b.lotRange = b.range
  b.mass = mass(b, far)
  if (b.mass) b.range = Math.max(b.range, b.mass.range)
  b.state = b.lots.length ? 0 : 2
}

// do the parts of lot design l centred at district (cx, cy) keep clear of the trunks of block b (unless clear) and of
// the lot's road segments (lotSegs)? f = lot frame
function fits(city, b, f, l, cx, cy, segs) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
  for (const q of l.parts) { x0 = Math.min(x0, q.ox - q.sx / 2); x1 = Math.max(x1, q.ox + q.sx / 2); y0 = Math.min(y0, q.oy - q.sy / 2); y1 = Math.max(y1, q.oy + q.sy / 2) }
  const box = [cx + x0, cx + x1, cy + y0, cy + y1]
  if (segs.some(s => rectSegDist(box, s) < s[4])) return false
  return b.clear || clearOf(city, b.rec, f.W, rectPts(...box), TRUNK_GAP)
}
// the design on fraction k of lot (i, j) along x (ax = +-1) or y (ay = +-1), away from the other side: { l, dx, dy }
function cut(city, D, b, i, j, W, H, ax, ay, k) {
  const w = ax ? W * k : W, h = ay ? H * k : H
  return Math.min(w, h) < 7 ? null : { l: design(city, D, b, i, j, w, h), dx: ax * (W - w) / 2, dy: ay * (H - h) / 2 }
}
const NONE = []
// segments [ax, ay, bx, by, clearance (local m)] near lot frame f (centre cx, cy) in district coords:
// [ax, ay, bx, by, clearance (units), metres from the lot centre to the road edge]
function lotSegs(near, f, cx, cy, hr) {
  const [ja, jb, jc, jd] = f.J, det = ja * jd - jb * jc, f2 = ja * ja + jb * jb + jc * jc + jd * jd
  const smin = Math.sqrt(Math.max((f2 - Math.sqrt(Math.max(f2 * f2 - 4 * det * det, 0))) / 2, 1e-6))
  const toD = (px, py) => [cx + (jd * (px - f.lx) - jb * (py - f.ly)) / det, cy + (ja * (py - f.ly) - jc * (px - f.lx)) / det]
  const out = []
  for (const s of near) {
    const d = segDist(f.lx, f.ly, s[0], s[1], s[2], s[3])
    if (d > hr + s[4] + 1) continue
    const c = (s[4] + 1) / smin
    out.push([...toD(s[0], s[1]), ...toD(s[2], s[3]), c, d - s[4]])
  }
  return out
}
// distance between a rect [x0, x1, y0, y1] and a segment (the box distance is convex along the segment)
export function rectSegDist(b, s) {
  const f = t => {
    const x = s[0] + (s[2] - s[0]) * t, y = s[1] + (s[3] - s[1]) * t
    return Math.hypot(Math.max(b[0] - x, 0, x - b[1]), Math.max(b[2] - y, 0, y - b[3]))
  }
  let a = 0, c = 1
  for (let i = 0; i < 20; i++) { const m1 = a + (c - a) / 3, m2 = c - (c - a) / 3; f(m1) < f(m2) ? c = m2 : a = m1 }
  return f((a + c) / 2)
}

// far tier: the block's mid-rise lots (not towers or houses) merged into one flat-roofed mass over their bounds (district
// coords), at their footprint-weighted mean height. Drawn from r0 (their mean range, where they have sunk away) out
// to MASS_K x height, so whole blocks keep their relief and window texture far beyond the single buildings. In blocks
// a trunk may cut, lots join (clearest first) only while the mass stays clear of it
function mass(b, far) {
  const { city, D } = b
  let lots = b.lots.filter(l => l.top < MASS_LOT && l.parts[0].style !== HOUSE)
  if (!lots.length) return null
  const bounds = ls => {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
    for (const l of ls) for (const p of l.parts) { x0 = Math.min(x0, p.box[0]); x1 = Math.max(x1, p.box[1]); y0 = Math.min(y0, p.box[2]); y1 = Math.max(y1, p.box[3]) }
    return [x0, x1, y0, y1]
  }
  if (!b.clear) {
    const cen = l => [(l.parts[0].box[0] + l.parts[0].box[1]) / 2, (l.parts[0].box[2] + l.parts[0].box[3]) / 2]
    const [mx, my] = lots.map(cen).reduce((s, c) => [s[0] + c[0] / lots.length, s[1] + c[1] / lots.length], [0, 0])
    lots.sort((p, q) => { const a = cen(p), c = cen(q); return Math.hypot(a[0] - mx, a[1] - my) - Math.hypot(c[0] - mx, c[1] - my) })
    const kept = []
    for (const l of lots) {
      const [x0, x1, y0, y1] = bounds([...kept, l])
      if (clearOf(city, b.rec, b.W, rectPts(x0, x1, y0, y1), TRUNK_GAP)) kept.push(l)
    }
    if (!kept.length) return null
    lots = kept
  }
  let a = 0, h = 0, r0 = 0, big = null
  for (const l of lots) {
    const q = l.parts[0], s = q.sx * q.sy
    a += s; h += s * l.top; r0 += l.range / lots.length
    if (!big || s > big.sx * big.sy) big = q
  }
  const H = h / a, range = Math.min(MASS_K * H, far * (0.7 + 0.3 * hash2i(b.kx, b.ky, D.hseed + 29)))
  if (H < MASS_MIN || range < 1.3 * r0) return null
  const [x0, x1, y0, y1] = bounds(lots), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, f = frameAt(city, D, cx, cy)
  const hw = (x1 - x0) / 2, inset = Math.min(15, bend(city, f.u, f.v, Math.max(x1 - x0, y1 - y0) / 2) * ((x1 - x0) ** 2 + (y1 - y0) ** 2) / 4 + 2 * shearBend(D, hw))
  return {
    lots, H, range, r0, style: big.style, seed: (b.kx * 131 + b.ky * 977 + D.hseed + 0x2a5) & 0x3fffff, born: null,
    dir: localToDir(city, f.lx, f.ly), ex: f.ex, ey: f.ey, sx: Math.max((x1 - x0) * f.sx - inset, 10), sy: Math.max((y1 - y0) * f.sy - inset, 10),
  }
}

const setGround = (l, lo, hi) => { l.base = hi; l.found = hi - lo + 3; l.wet = lo < Math.max(0.3, l.lake + LAKE_DRY) }

// bilinear ground height at local (x, y) from a city grid, NaN where the grid isn't trusted
function gridAt(g, x, y) {
  const fx = (x - g.x0) / GRID, fy = (y - g.x0) / GRID, i = Math.floor(fx), j = Math.floor(fy), n = g.n
  if (i < 0 || j < 0 || i >= n - 1 || j >= n - 1) return NaN
  const k = j * n + i, h = g.h, tx = fx - i, ty = fy - j
  if (g.bad[k] | g.bad[k + 1] | g.bad[k + n] | g.bad[k + n + 1]) return NaN
  return (h[k] * (1 - tx) + h[k + 1] * tx) * (1 - ty) + (h[k + n] * (1 - tx) + h[k + n + 1] * tx) * ty
}
// ground heights of block b from its city grid: drawn now (state 2), refined by exact probes later. Where the
// grid isn't trusted it stays in state 0 and waits for them
function settle(b, g) {
  if (!g?.h) return
  for (const l of b.lots) {
    let lo = Infinity, hi = -Infinity
    for (let k = 0; k < 10; k += 2) {
      const h = gridAt(g, l.pxy[k], l.pxy[k + 1])
      if (Number.isNaN(h)) return
      lo = Math.min(lo, h); hi = Math.max(hi, h)
    }
    setGround(l, lo, hi)
  }
  b.state = 2; b.refine = true
}

// how far a block's buildings (or its far-tier mass) may be drawn, guessed before designing it: the mass of
// mid-rise at the mean height design() gives that density, out to MASS_K x that height
function reachGuess(b) {
  const city = b.city
  if (city.tier < TIER.CITY) return FAR
  const mid = b.suburb ? 0 : (3 + Math.pow(b.dens, 1.2) * (city.tier === TIER.METRO ? 15 : 9) * 0.67) * 3.2 + 1
  return Math.min(SKY, Math.max(FAR, 250 * b.guess, mid >= MASS_MIN ? MASS_K * mid : 0))
}

// roof height of a block's tallest building, guessed before designing it (ranks design work by size on screen)
function guessTop(b) {
  const t = b.city.tier
  if (b.tc > 0.02) return 150
  if (b.hr > 0.02) return 80
  if (t <= TIER.TOWN || b.suburb) return 8
  return 4 + Math.pow(b.dens, 1.2) * (t === TIER.METRO ? 30 : 18)
}

// the building(s) on lot (i, j) of block b, deterministic from the painted lot hash (paint.glsl.js civLots:
// hash2i(kx*8 + i, ky*8 + j, D.hseed + 3 + n)). W x Dp = lot size (district units ~ m). Parts are boxes in district
// units from the lot centre: offset (ox, oy), size, base above ground z0, wall height H, gable rise, style.
// Footprints follow the painted roofs: lot - 5 (old core - 1.5), houses 8-16 x 7-12, towers x 0.65-0.95 (max 58)
function design(city, D, b, i, j, W, Dp) {
  const { kx, ky, dens } = b, h = n => hash2i(kx * 8 + i, ky * 8 + j, D.hseed + 3 + n)   // n = 0 is the painted roof hash
  const lh = h(0), r1 = h(1), r2 = h(2), r3 = h(3), r4 = h(4), t = city.tier, core = D.kind === DKIND.CORE
  const seed = (kx * 131 + ky * 977 + i * 7 + j * 3 + D.hseed) & 0x3fffff
  const parts = []
  const add = (ox, oy, sx, sy, z0, H, roof, style, near = Infinity) =>
    parts.push({ ox, oy, sx: Math.max(sx, 3), sy: Math.max(sy, 3), z0, H, roof, style, near, seed: seed + parts.length * 0x100000 })
  // a few HVAC units on a flat roof of size w x d at height z
  const hvac = (w, d, z) => {
    for (let k = 0; k < 1 + Math.floor(r3 * 3); k++) {
      const hk = m => hash2i(seed, k, 40 + m)
      add((hk(1) - 0.5) * (w - 6), (hk(2) - 0.5) * (d - 6), 1.5 + 2 * hk(3), 1.5 + 1.5 * hk(4), z, 1.2 + 0.8 * hk(5), 0, PLANT, FINE)
    }
  }
  // the rows bend away from a straight frame (curved districts): keep the corners on the lot
  const sx = W - (core ? 1.5 : 5), sy = Dp - (core ? 2.1 : 5) - 2 * shearBend(D, W / 2)
  const pitch = core && r2 < 0.65          // painted pitched roofs in the old core
  if (t <= TIER.TOWN || b.suburb) {
    if (t === TIER.TOWN && dens > 0.45 && !b.suburb && r2 < 0.7) {   // town centre: a small block
      const floors = Math.min(3, 1 + Math.floor(r1 * 2.2 * (0.6 + dens) + 1)), k = r3 < 0.25 ? 0.8 : 1
      add(0, 0, sx * k, sy * k, 0, floors * 3.2 + 1, pitch ? Math.min(6, 0.28 * Math.min(sx, sy) * k) : 0, r3 < 0.5 ? BRICK : CONCRETE)
    } else {                                // one house per lot (the painted roof)
      const floors = Math.min(3, 1 + Math.floor(r1 * (t === TIER.VILLAGE ? 1.6 : 2.2) * (0.6 + dens)))
      const w = Math.min(sx, 8 + 8 * r3), d = Math.min(sy, 7 + 5 * r4)
      add(0, 0, w, d, 0, floors * 2.9 + 0.3, r2 < 0.8 ? Math.min(w, d) * (0.25 + 0.2 * hash2i(seed, 0, 5)) : 0, HOUSE)
    }
  } else if (r2 < b.tc && Math.min(sx, sy) >= TOWER_LOT) {   // (no needles: small lots get the mid-rise footprint over the painted tower roof)
    let floors = 20 + Math.round(60 * Math.pow(r3, 2.2))
    if (r4 > 0.96) floors += 30 + Math.round(40 * r1)                // a few supertalls
    const r5 = h(5), style = r5 < 0.55 ? GLASS : r5 < 0.8 ? RIBBON : CONCRETE, k = 0.65 + 0.3 * r4   // concrete / stone: punched windows
    const tx = Math.min(sx * k, 58), ty = Math.min(sy * k, 58), H = floors * 3.6 + 1
    const ph = (2 + Math.floor(r4 * 4)) * 3.2 + 1
    if (r3 > 0.3 && sx * sy > 900) add(0, 0, sx, sy, 0, ph, 0, r4 < 0.5 ? CONCRETE : BRICK, Math.max(NEAR, ph * 250))   // podium: sinks before its tower
    if (r1 > 0.35 && r1 < 0.8) {                                     // setback crown
      const low = Math.round(floors * (0.55 + 0.25 * r4)) * 3.6
      add(0, 0, tx, ty, 0, low, 0, style)
      add(0, 0, tx * 0.7, ty * 0.7, low, H - low, 0, style)
    } else add(0, 0, tx, ty, 0, H, 0, style)
    add(0, 0, tx * 0.3, ty * 0.3, H, 4, 0, PLANT, DETAIL)            // roof plant
  } else if (Math.max(sx, sy) < 56 && r2 < b.hr) {
    // high-rise on the mid-rise footprint (the painted roof stays right): 12-40 floors, a few taller
    const hf = t === TIER.METRO ? 1.2 : 0.6 + 0.7 * hash2i(city.seed, 8, 31)                 // per-city height
    let floors = Math.round((12 + 28 * Math.pow(r1, 1.6)) * hf)
    if (r4 > 0.95) floors += 15 + Math.round(25 * r3)
    const k = r3 < 0.25 ? 0.8 : 1, H = floors * 3.6 + 1, style = r4 < 0.4 ? GLASS : r4 < 0.7 ? RIBBON : CONCRETE
    if (floors > 20 && r4 > 0.5 && r4 < 0.8) {                       // setback crown
      const low = Math.round(floors * 0.7) * 3.6
      add(0, 0, sx * k, sy * k, 0, low, 0, style)
      add(0, 0, sx * k * 0.72, sy * k * 0.72, low, H - low, 0, style)
    } else add(0, 0, sx * k, sy * k, 0, H, 0, style)
    add(0, 0, sx * k * 0.3, sy * k * 0.3, H, 4, 0, PLANT, DETAIL)
  } else {
    const floors = 3 + Math.round(Math.pow(dens, 1.2) * (t === TIER.METRO ? 15 : 9) * (0.35 + 0.65 * r1))
    const k = r3 < 0.25 ? 0.8 : 1, H = floors * 3.2 + 1                 // sometimes set back
    add(0, 0, sx * k, sy * k, 0, H, pitch ? Math.min(7, 0.28 * Math.min(sx, sy) * k) : 0, r2 < 0.45 ? BRICK : CONCRETE)
    if (!pitch) {
      if (r4 > 0.35) add((r1 - 0.5) * sx * 0.4, (r2 - 0.5) * sy * 0.4, 3 + 5 * r3, 3 + 4 * r4, H, 2.5 + 1.5 * r1, 0, PLANT, DETAIL)  // plant room
      if (sx * k > 12 && sy * k > 12) hvac(sx * k, sy * k, H)
    }
  }
  return { parts, lh, base: 0, found: 3, wet: false }
}

// metro towers cluster downtown and in hashed districts (paint.glsl.js civLots mirrors this)
const smooth = x => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x) }
function towerChance(city, D, dens) {
  return 0.65 * smooth((dens - 0.55) / 0.4) * (0.25 + 0.75 * hash2i(D.key, 5, city.seed + 21))
}
// high-rises on smaller lots: a tier-2 downtown (dense and tall in some cities, modest or almost none in others), in
// hashed patches of a few blocks, and lining some boulevards and ring roads (avenue: the block fronts one) in cities and metros
function highRise(city, D, kx, ky, dens, avenue) {
  if (city.tier < TIER.CITY) return 0
  const cbd = city.tier === TIER.CITY ? Math.max(0, hash2i(city.seed, 7, 31) - 0.15) / 0.85 : 0
  const cl = hash2i(Math.floor(kx / 4), Math.floor(ky / 3), D.hseed + 22)
  const core = 0.7 * cbd * smooth((dens - 0.55) / 0.35) * (0.15 + 0.85 * cl * cl)
  return core + (avenue ? 0.3 * smooth((dens - 0.35) / 0.3) * (city.tier === TIER.CITY ? 0.2 + cbd : 1) : 0)
}

// unit building: 4 walls, gable ends on x = +-0.5, two roof halves meeting at the ridge (z = 2)
function unitHouse() {
  const pos = [], nor = [], idx = []
  const poly = (pts, n) => {
    const b = pos.length / 3
    for (const p of pts) { pos.push(...p); nor.push(...n) }
    const [a, c, d] = pts, e1 = sub(c, a), e2 = sub(d, a)
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
  return { pos, nor, idx }
}
