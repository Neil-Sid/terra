// Traffic: cars, trucks and buses on highways, country roads and city streets near the camera.
// Contracts in notes/CIV.md, notes/traffic.md. Nothing is simulated: each lane (or closed lane loop in a city)
// holds vehicle slots at a fixed spacing that slide along at the lane's speed, so a vehicle's position is a pure
// function of time. Every frame we enumerate only the slots near the camera and write them into instance buffers.
import * as THREE from 'three'
import { R } from '../config.js'
import { TYPE, LANES, SKIND, PROFILES, profileOf, CITY_REACH, cityFrame, warpGrid, dirToGrid, gridToDir, hash2i, districtsNear, districtEdges,
  toDistrict, fromDistrict, fromLattice, rowConnectors, blockInfo, districtAt, boulevards, BOULEVARD_END, ringRadius, cityCell } from './spec.js'
import { objUniforms } from './objects.glsl.js'
import { PART, TRAFFIC_VERT, TRAFFIC_FRAG } from './traffic.glsl.js'
import { cityWater, waterAt, LakeTest } from './shore.js'

const ROUTE_RANGE = 3200     // m, vehicles on highways and roads
const STREET_RANGE = 700     // m, vehicles on city streets
const GRID_SLACK = 1.15      // grid units per metre at most (round-5 warp)
const BUILD_BUDGET = 1500    // city loop points built per frame at most

// per route type and lane: speed (m/s), slot spacing (m), share of empty slots, trucks, buses
const FLOW = [
  [{ speed: 33, gap: 70, skip: 0.35, truck: 0.05, bus: 0.01 }, { speed: 25, gap: 85, skip: 0.3, truck: 0.35, bus: 0.03 }],
  [{ speed: 19, gap: 130, skip: 0.45, truck: 0.12, bus: 0.03 }],
]
// city loops: block lane loops per local profile (core lane, street, super-block edge profile), industry blocks,
// district edge loops (inner lane), boulevards and the ring road
const LOCAL_FLOWS = [{ speed: 6, gap: 60, skip: 0.6, truck: 0.01, bus: 0 }, { speed: 9, gap: 50, skip: 0.5, truck: 0.03, bus: 0.01 },
  { speed: 11, gap: 45, skip: 0.45, truck: 0.05, bus: 0.02 }]
const IND_FLOW = { speed: 8, gap: 60, skip: 0.5, truck: 0.35, bus: 0 }
const EDGE_FLOW = { speed: 12, gap: 40, skip: 0.4, truck: 0.05, bus: 0.04 }
const TRUNK_FLOW = { speed: 14, gap: 34, skip: 0.3, truck: 0.04, bus: 0.07 }
const DLOOP = new WeakMap()  // district -> its edge loop (inner lane) or null

// linear paint colours, weighted towards the white / black / grey of real traffic
const PAINT = [[0.7, 0.7, 0.68], [0.7, 0.7, 0.68], [0.03, 0.03, 0.03], [0.03, 0.03, 0.035], [0.4, 0.42, 0.44],
  [0.4, 0.42, 0.44], [0.12, 0.12, 0.13], [0.45, 0.03, 0.02], [0.03, 0.07, 0.28], [0.05, 0.14, 0.07],
  [0.5, 0.4, 0.26], [0.2, 0.05, 0.03], [0.7, 0.42, 0.02], [0.08, 0.2, 0.4]]

// ---- meshes (local metres: x right, y up, z forward) ----

// a box whose top face may be narrower / shifted (cabins, windscreens); no bottom face
function block(g, w, y0, y1, z0, z1, part, o = {}) {
  const wt = o.wt ?? w, t0 = o.t0 ?? z0, t1 = o.t1 ?? z1, sides = o.sides ?? part
  const B = (sx, z) => [sx * w, y0, z], T = (sx, z) => [sx * wt, y1, z]
  const c = [0, (y0 + y1) / 2, (z0 + z1 + t0 + t1) / 4]
  quad(g, c, [T(-1, t0), T(1, t0), T(1, t1), T(-1, t1)], o.top ?? part)
  quad(g, c, [B(-1, z1), B(1, z1), T(1, t1), T(-1, t1)], o.front ?? sides)
  quad(g, c, [B(-1, z0), B(1, z0), T(1, t0), T(-1, t0)], o.back ?? sides)
  quad(g, c, [B(1, z0), B(1, z1), T(1, t1), T(1, t0)], sides)
  quad(g, c, [B(-1, z0), B(-1, z1), T(-1, t1), T(-1, t0)], sides)
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

// flat quad facing away from the block centre c
function quad(g, c, p, part) {
  let n = cross3(sub(p[1], p[0]), sub(p[2], p[0]))
  const l = Math.hypot(...n), mid = p.reduce((m, q) => m.map((v, i) => v + q[i] / 4), [0, 0, 0])
  const out = dot3(n, sub(mid, c)) > 0
  n = n.map(v => (out ? v : -v) / l)
  const base = g.pos.length / 3
  for (const q of p) { g.pos.push(...q); g.nrm.push(...n); g.info.push(part, 0, 0) }
  // local (right, up, forward) maps to a left-handed world frame, so emit clockwise-in-local
  if (out) g.idx.push(base, base + 2, base + 1, base, base + 3, base + 2)
  else g.idx.push(base, base + 1, base + 2, base, base + 2, base + 3)
}

// a pair of lamps at (+-x, y, z) facing forward (dir 1) or backward (-1); expanded in the vertex shader
function lamps(g, x, y, z, dir, part) {
  for (const sx of [-1, 1]) {
    const base = g.pos.length / 3
    for (const [cx, cy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) { g.pos.push(sx * x, y, z); g.nrm.push(0, 0, dir); g.info.push(part, cx, cy) }
    g.idx.push(base, base + 1, base + 2, base, base + 2, base + 3)
  }
}

function vehicle(build) {
  const g = { pos: [], nrm: [], info: [], idx: [] }
  build(g)
  const geo = new THREE.InstancedBufferGeometry()
  geo.setIndex(g.idx)
  geo.setAttribute('position', new THREE.Float32BufferAttribute(g.pos, 3))
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(g.nrm, 3))
  geo.setAttribute('aInfo', new THREE.Float32BufferAttribute(g.info, 3))
  return geo
}

const { PAINT: P0, GLASS, DARK, BOX, HEAD, TAIL } = PART
const KINDS = {
  car: { cap: 5000, geo: () => vehicle(g => {
    block(g, 0.9, 0.3, 0.8, -2.2, 2.2, P0, { t0: -2.12, t1: 2.08 })
    block(g, 0.84, 0.8, 1.38, -1.7, 0.85, P0, { wt: 0.7, t0: -1.35, t1: 0.05, sides: GLASS })
    block(g, 0.93, 0, 0.62, 1.05, 1.68, DARK)
    block(g, 0.93, 0, 0.62, -1.68, -1.05, DARK)
    lamps(g, 0.62, 0.64, 2.2, 1, HEAD)
    lamps(g, 0.66, 0.68, -2.2, -1, TAIL)
  }) },
  truck: { cap: 900, geo: () => vehicle(g => {
    block(g, 1.25, 0.6, 1.9, 4.3, 6.8, P0)
    block(g, 1.25, 1.9, 3.0, 4.3, 6.75, P0, { wt: 1.18, t1: 6.55, sides: GLASS, back: P0 })
    block(g, 1.28, 1.2, 4.0, -6.9, 4.1, BOX)
    block(g, 0.95, 0.5, 1.2, -6.4, 4.3, DARK)
    for (const [z0, z1] of [[5.1, 6.1], [2.6, 3.9], [-6.3, -4.3]]) block(g, 1.27, 0, 1.0, z0, z1, DARK)
    lamps(g, 0.95, 0.95, 6.8, 1, HEAD)
    lamps(g, 1.0, 1.25, -6.9, -1, TAIL)
  }) },
  bus: { cap: 400, geo: () => vehicle(g => {
    block(g, 1.26, 0.4, 1.4, -6, 6, P0)
    block(g, 1.26, 1.4, 2.6, -6, 6, GLASS, { back: P0, top: P0 })
    block(g, 1.26, 2.6, 3.1, -6, 6, P0, { wt: 1.2, t0: -5.9, t1: 5.9 })
    for (const [z0, z1] of [[3.6, 4.6], [-4.4, -3.4]]) block(g, 1.27, 0, 1.0, z0, z1, DARK)
    lamps(g, 0.95, 0.75, 6, 1, HEAD)
    lamps(g, 1.0, 1.0, -6, -1, TAIL)
  }) },
}

// instance buffer for one vehicle kind: 18 floats per vehicle (see iA..iE in traffic.glsl.js)
const STRIDE = 18
class Fleet {
  constructor(kind, material, scene) {
    this.cap = kind.cap
    this.n = 0
    this.data = new Float32Array(this.cap * STRIDE)
    this.buf = new THREE.InstancedInterleavedBuffer(this.data, STRIDE).setUsage(THREE.DynamicDrawUsage)
    this.geo = kind.geo()
    ;['iA', 'iB', 'iC', 'iD', 'iE'].forEach((k, i) => this.geo.setAttribute(k, new THREE.InterleavedBufferAttribute(this.buf, i < 4 ? 4 : 2, i * 4)))
    this.mesh = new THREE.Mesh(this.geo, material)
    this.mesh.frustumCulled = false
    scene.add(this.mesh)
  }
  add(rel, dir, tan, lenScale, hScale, seed, col, win, deck, grade) {
    if (this.n >= this.cap) return
    const d = this.data, o = this.n++ * STRIDE
    d[o] = rel[0]; d[o + 1] = rel[1]; d[o + 2] = rel[2]; d[o + 3] = lenScale
    d[o + 4] = dir[0]; d[o + 5] = dir[1]; d[o + 6] = dir[2]; d[o + 7] = hScale
    d[o + 8] = tan[0]; d[o + 9] = tan[1]; d[o + 10] = tan[2]; d[o + 11] = seed
    d[o + 12] = col[0]; d[o + 13] = col[1]; d[o + 14] = col[2]; d[o + 15] = win
    d[o + 16] = deck ?? -1e9; d[o + 17] = grade
  }
  commit() {
    this.geo.instanceCount = this.n
    this.mesh.visible = this.n > 0
    this.buf.clearUpdateRanges()
    this.buf.addUpdateRange(0, this.n * STRIDE)
    this.buf.needsUpdate = true
  }
}

// unit direction and tangent at s metres along a route, written into P and T (no allocation)
function routeAt(r, s, P, T) {
  const cum = r.cum, X = r.pts
  let lo = 0, hi = cum.length - 1
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cum[m] <= s) lo = m; else hi = m }
  const t = Math.min(1, Math.max(0, (s - cum[lo]) / Math.max(cum[hi] - cum[lo], 1e-9))), a = lo * 3, b = hi * 3
  for (let i = 0; i < 3; i++) { T[i] = X[b + i] - X[a + i]; P[i] = X[a + i] + T[i] * t }
  normalize(P)
  const k = dot3(T, P)
  for (let i = 0; i < 3; i++) T[i] -= P[i] * k
  normalize(T)
}
function normalize(v) { const l = Math.hypot(v[0], v[1], v[2]); v[0] /= l; v[1] /= l; v[2] /= l; return v }

// ---- district layout (notes/layout.md): built blocks near the camera and closed loops round them ----
// A Loop is a closed path on the ground: unit dirs every <= ~15 m, their grid points and cumulative metres.
// Cars circle their block clockwise in the lane next to it
// (right turns only, so nothing drives through a T-junction into a building); walkers circle it on its sidewalk.
const CH = 16                                   // points per culling chunk
export class Loop {
  constructor(city, pts) {                      // pts: [[u, v], ...] grid points, clockwise, not repeated
    const n = pts.length
    this.n = n
    this.dirs = new Float64Array(n * 3); this.uv = new Float64Array(n * 2); this.cum = new Float64Array(n + 1)
    for (let i = 0; i < n; i++) {
      this.dirs.set(gridToDir(city, pts[i][0], pts[i][1]), i * 3)
      this.uv[i * 2] = pts[i][0]; this.uv[i * 2 + 1] = pts[i][1]
    }
    const D = this.dirs
    for (let i = 0; i < n; i++) { const a = i * 3, b = ((i + 1) % n) * 3; this.cum[i + 1] = this.cum[i] + Math.hypot(D[b] - D[a], D[b + 1] - D[a + 1], D[b + 2] - D[a + 2]) * R }
    this.L = this.cum[n]
    // culling chunks: centre (unit dir) and radius (m)
    this.chunks = []
    for (let c = 0; c < n; c += CH) {
      const e = Math.min(n, c + CH), m = [0, 0, 0]
      for (let i = c; i <= e; i++) { const o = (i % n) * 3; m[0] += D[o]; m[1] += D[o + 1]; m[2] += D[o + 2] }
      normalize(m)
      let rad = 0
      for (let i = c; i <= e; i++) { const o = (i % n) * 3; rad = Math.max(rad, Math.hypot(D[o] - m[0], D[o + 1] - m[1], D[o + 2] - m[2]) * R) }
      this.chunks.push({ C: m, rad, s0: this.cum[c], s1: this.cum[e] })
    }
  }
  // unit dir P and heading T at s metres along (wrapped); returns the segment index
  at(s, P, T) {
    const cum = this.cum, n = this.n, L = this.L
    s = ((s % L) + L) % L
    let lo = 0, hi = n
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cum[m] <= s) lo = m; else hi = m }
    const t = (s - cum[lo]) / Math.max(cum[lo + 1] - cum[lo], 1e-9), a = lo * 3, b = ((lo + 1) % n) * 3, X = this.dirs
    for (let i = 0; i < 3; i++) { T[i] = X[b + i] - X[a + i]; P[i] = X[a + i] + T[i] * t }
    normalize(P)
    const k = dot3(T, P)
    for (let i = 0; i < 3; i++) T[i] -= P[i] * k
    normalize(T)
    return lo
  }
  // grid point at s metres along
  uvAt(s) {
    const L = this.L, n = this.n, cum = this.cum
    s = ((s % L) + L) % L
    let lo = 0, hi = n
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cum[m] <= s) lo = m; else hi = m }
    const t = (s - cum[lo]) / Math.max(cum[lo + 1] - cum[lo], 1e-9), a = lo * 2, b = ((lo + 1) % n) * 2, U = this.uv
    return [U[a] + (U[b] - U[a]) * t, U[a + 1] + (U[b + 1] - U[a + 1]) * t]
  }
}

// metres per district unit across connector lines (x = const) and across rows (y = const) at grid (u, v), district x
function districtMetric(city, D, u, v, x) {
  const [, , a, b, c, d] = warpGrid(city, u, v), det = a * d - b * c, w = D.A * D.k * Math.cos(D.k * x)
  const m = (gx, gy) => 1 / Math.hypot((d * gx - c * gy) / det, (-b * gx + a * gy) / det)
  return [m(D.c, D.s), m(-D.s - w * D.c, D.c - w * D.s)]
}

// per district: its cell outline as <= 25 m pieces in district coords (counter-clockwise: inside on the left), their
// bounding box, and memoised blocks
const DDATA = new WeakMap()
function districtData(city, D) {
  let dd = DDATA.get(D)
  if (dd) return dd
  const pieces = [], box = [Infinity, Infinity, -Infinity, -Infinity]
  for (const e of districtEdges(city, D)) {
    const ga = fromLattice(city, ...e.a), gb = fromLattice(city, ...e.b), n = Math.max(1, Math.ceil(Math.hypot(gb[0] - ga[0], gb[1] - ga[1]) / 25))
    let p = toDistrict(D, ...ga)
    for (let i = 1; i <= n; i++) {
      const g = fromLattice(city, e.a[0] + (e.b[0] - e.a[0]) * i / n, e.a[1] + (e.b[1] - e.a[1]) * i / n), q = toDistrict(D, g[0], g[1])
      pieces.push({ a: p, b: q })
      box[0] = Math.min(box[0], q[0]); box[1] = Math.min(box[1], q[1]); box[2] = Math.max(box[2], q[0]); box[3] = Math.max(box[3], q[1])
      p = q
    }
  }
  DDATA.set(D, dd = { pieces, box, blocks: new Map() })
  return dd
}

// built blocks (centre in their own district) within radius (grid units) of grid (u, v), memoised per district, so
// loops built on a block record are kept: [{ D, kx, ky, rect, lot, lots, density, zone, park, plaza, centre, dir,
// metric [mx, my], rad (m) }]
export function cityBlocks(city, u, v, radius) {
  const out = []
  for (const D of districtsNear(city, u, v, radius)) {
    const dd = districtData(city, D), [x, y] = toDistrict(D, u, v), [bx0, by0, bx1, by1] = dd.box, pad = 2 * Math.abs(D.A)
    const ya = Math.max(y - radius - pad, by0 - pad), yb = Math.min(y + radius + pad, by1 + pad)
    const xa = Math.max(x - radius, bx0), xb = Math.min(x + radius, bx1)
    if (ya > yb || xa > xb) continue
    if (dd.blocks.size > 20000) dd.blocks.clear()
    for (let ky = Math.floor((ya - D.oy) / D.by) - 1; ky <= Math.floor((yb - D.oy) / D.by) + 1; ky++) {
      const { bx, off } = rowConnectors(D, ky)
      for (let kx = Math.floor((xa - off) / bx) - 1; kx <= Math.floor((xb - off) / bx); kx++) {
        const k = (kx + 32768) * 65536 + ky + 32768
        let b = dd.blocks.get(k)
        if (b === undefined) dd.blocks.set(k, b = makeBlock(city, D, kx, ky))
        if (b && Math.hypot(b.centre[0] - u, b.centre[1] - v) <= radius) out.push(b)
      }
    }
  }
  return out
}
function makeBlock(city, D, kx, ky) {
  const b = blockInfo(city, D, kx, ky)
  if (b.density <= 0 || districtAt(city, b.centre[0], b.centre[1]).D !== D) return null
  const r = b.rect, metric = districtMetric(city, D, b.centre[0], b.centre[1], (r.x0 + r.x1) / 2)
  return { ...b, D, kx, ky, metric, dir: gridToDir(city, ...b.centre), rad: Math.hypot((r.x1 - r.x0) * metric[0], (r.y1 - r.y0) * metric[1]) / 2 }
}

// Clockwise loop of block b: off m inside the street centres round it, cut off eoff m inside the district's edge
// streets (Sutherland-Hodgman against the outline pieces), corners rounded over rc m. Null if nothing is left.
export function blockLoop(city, b, off, eoff, rc) {
  const { D, rect: r, metric: [mx, my] } = b, ix = off / mx, iy = off / my, me = (mx + my) / 2
  if (r.x1 - r.x0 < 2 * ix + 4 || r.y1 - r.y0 < 2 * iy + 4) return null
  let poly = [[r.x0 + ix, r.y0 + iy], [r.x0 + ix, r.y1 - iy], [r.x1 - ix, r.y1 - iy], [r.x1 - ix, r.y0 + iy]]
  const cx = (r.x0 + r.x1) / 2, cy = (r.y0 + r.y1) / 2, reach = Math.hypot(r.x1 - r.x0, r.y1 - r.y0) / 2 + 30, e = eoff / me
  for (const pc of districtData(city, D).pieces) {
    const [ax, ay] = pc.a, ex = pc.b[0] - ax, ey = pc.b[1] - ay, l = Math.hypot(ex, ey)
    if (Math.hypot(ax + ex / 2 - cx, ay + ey / 2 - cy) > reach + l) continue
    const f = p => (ex * (p[1] - ay) - ey * (p[0] - ax)) / l - e
    if (poly.every(p => f(p) >= 0)) continue
    const next = []
    poly.forEach((p, i) => {
      const q = poly[(i + 1) % poly.length], fp = f(p), fq = f(q), cut = () => { const t = fp / (fp - fq); return [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t] }
      if (fp >= 0) { next.push(p); if (fq < 0) next.push(cut()) }
      else if (fq >= 0) next.push(cut())
    })
    poly = next
    if (poly.length < 3) return null
  }
  const pts = [], n = poly.length, rcd = rc / me, trim = (len) => Math.min(0.45, rcd / Math.max(len, 1e-6))
  for (let i = 0; i < n; i++) {
    const p = poly[i], q = poly[(i + 1) % n], o = poly[(i + 2) % n], len = Math.hypot(q[0] - p[0], q[1] - p[1])
    if (len < 0.5) continue
    const t0 = trim(len), m = Math.max(1, Math.ceil(len * (1 - 2 * t0) / 15))
    for (let j = 0; j <= m; j++) { const t = t0 + (1 - 2 * t0) * j / m; pts.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]) }
    const t2 = trim(Math.hypot(o[0] - q[0], o[1] - q[1]))       // rounded corner at q: midpoint of the quadratic
    pts.push([0.25 * (q[0] - (q[0] - p[0]) * t0) + 0.5 * q[0] + 0.25 * (q[0] + (o[0] - q[0]) * t2),
      0.25 * (q[1] - (q[1] - p[1]) * t0) + 0.5 * q[1] + 0.25 * (q[1] + (o[1] - q[1]) * t2)])
  }
  if (pts.length < 4) return null
  const loop = new Loop(city, pts.map(([x, y]) => fromDistrict(D, x, y)))
  return loop.L > 30 ? loop : null
}

// clockwise loop round district D's cell, off m inside its edge streets (the inner lane), corners cut
function districtLoop(city, D, off) {
  const E = districtEdges(city, D), pts = []
  for (let i = E.length - 1; i >= 0; i--) {
    const e = E[i], A = e.b, B = e.a, ga = fromLattice(city, ...A), gb = fromLattice(city, ...B)
    const len = Math.hypot(gb[0] - ga[0], gb[1] - ga[1]), tx = (gb[0] - ga[0]) / len, ty = (gb[1] - ga[1]) / len
    const t0 = Math.min(0.45, (off + 6) / len), m = Math.max(1, Math.ceil(len * (1 - 2 * t0) / 20))
    for (let j = 0; j <= m; j++) {
      const t = t0 + (1 - 2 * t0) * j / m, g = fromLattice(city, A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t)
      const [, , a, b, c, d] = warpGrid(city, g[0], g[1]), mm = Math.hypot(a * ty - b * tx, c * ty - d * tx)
      pts.push([g[0] + ty * off / mm, g[1] - tx * off / mm])
    }
  }
  return pts.length > 3 ? new Loop(city, pts) : null
}

// distance (m) from unit direction p to route segment k, measured on the ground
function segDistance(S, k, p) {
  const o = k * 8, ab = [S[o + 4] - S[o], S[o + 5] - S[o + 1], S[o + 6] - S[o + 2]], pa = [p[0] - S[o], p[1] - S[o + 1], p[2] - S[o + 2]]
  const u = Math.min(1, Math.max(0, dot3(pa, ab) / Math.max(dot3(ab, ab), 1e-20)))
  const v = pa.map((x, i) => x - ab[i] * u), kk = dot3(v, p)
  return Math.hypot(v[0] - p[0] * kk, v[1] - p[1] * kk, v[2] - p[2] * kk) * R
}

// ---- junctions (notes/traffic.md "Junctions") ----
// Every city lane path (block loop, district loop, trunk lane) is closed, length L. Where two paths come close enough for footprints to touch
// of each other (crossings, shared pavement) both get a conflict zone, found once per path from a per-city spatial
// hash of path segments. A vehicle stays on its free slot position until something holds it up; then it carries a
// lag (m behind its slot) until it has caught up again, so only held-up vehicles have state. Each frame a vehicle
// keeps a gap to its leader (the next slot) and, before a zone, asks for it: granted when no vehicle of a conflicting
// path holds it or asked first (first come first served; local streets ask LOCAL_LATE s "later", so they give way to
// trunks), otherwise it stops short of the zone. WAIT_MAX breaks any deadlock.
// Two segments at angle th (sin sn, |cos| cs) conflict when their footprints could touch side by side: closer than
// h sn + w cs + w' (h, w: the largest half length / width on either path); the zone round the conflict point on a path
// is (w' + w cs) / sn long either side, so a vehicle short of it cannot touch any vehicle on the other path.
const THR_MAX = 9.5       // m, the largest such distance (trucks at right angles)
const AIR = 0.15          // m of air in both
const MERGE = 9           // m, conflict points this close along a path share a zone (no room to wait between)
const CELL = 24           // m, spatial hash cell
const GAP = 2.5           // m, bumper gap in a queue
const BRAKE = 4, ACCEL = 2.5, CATCH = 1.25   // m/s^2; speed factor while catching up with the slot
const WAIT_MAX = 12       // s at a stop line, then a vehicle goes anyway
const LOCAL_LATE = 3      // s
const PIECE = 160         // m of trunk lane registered at a time
const CHAIN = 8           // m: zones closer than a vehicle length + GAP + this are asked for together (no stopping between)
const HALF = { car: 2.3, truck: 7, bus: 6.1 }   // half lengths (m) with a little air
const mod = (a, L) => ((a % L) + L) % L
let TAGS = 0

// slots of a path: the non-skipped ones with their offset, kind hashes and half length (same hashes as the
// stateless model, so free vehicles are where they always were)
function pathInit(p, flow, id, k0, k1) {
  const N = Math.max(1, Math.floor(p.L / flow.gap)), ii = [], fr = [], hk = [], hc = [], hl = [], at = new Int32Array(N).fill(-1)
  for (let i = 0; i < N; i++) {
    const h = hash2i(i, id, 17)
    if (h < flow.skip) continue
    const a = hash2i(i, id, 23), kind = a < flow.bus ? 'bus' : a < flow.bus + flow.truck ? 'truck' : 'car'
    at[i] = ii.length
    ii.push(i); fr.push(0.3 * (h - flow.skip) / (1 - flow.skip)); hk.push(a); hc.push(hash2i(i, id, 29))
    hl.push(HALF[kind] * (kind === 'car' ? 0.92 + 0.2 * (a * 13 % 1) : 1))
  }
  const n = ii.length
  p.flow = flow; p.id = id; p.v0 = flow.speed * (k0 + k1 * hash2i(id, 5, 1))
  p.hmax = Math.max(HALF.car, ...hl); p.wmax = hl.some(h => h > 5) ? 1.3 : 0.92    // largest vehicle on the path
  p.sl = { N, g: p.L / N, n, ii: Int32Array.from(ii), fr: Float64Array.from(fr), hk: Float64Array.from(hk), hc: Float64Array.from(hc), hl: Float64Array.from(hl), at }
  p.xs = new Float64Array(n); p.vs = new Float64Array(n); p.xf = new Int32Array(n).fill(-1)
  p.veh = new Map(); p.zones = []; p.zk = -1; p.ml = 0
  return p
}

let PS_D = 0
function ptSeg(px, py, ax, ay, bx, by) {       // parameter of the point of segment ab closest to p (distance in PS_D)
  const ex = bx - ax, ey = by - ay, l2 = ex * ex + ey * ey
  const t = l2 > 1e-12 ? Math.min(1, Math.max(0, ((px - ax) * ex + (py - ay) * ey) / l2)) : 0
  PS_D = Math.hypot(ax + ex * t - px, ay + ey * t - py)
  return t
}
const SS = [0, 0, 0]
function segSeg(ax, ay, bx, by, cx, cy, dx, dy) {   // [distance, t on ab, u on cd] of the closest points
  const ux = bx - ax, uy = by - ay, vx = dx - cx, vy = dy - cy, wx = ax - cx, wy = ay - cy, den = ux * vy - uy * vx
  if (Math.abs(den) > 1e-9) {
    const t = (vx * wy - vy * wx) / den, u = (ux * wy - uy * wx) / den
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) { SS[0] = 0; SS[1] = t; SS[2] = u; return SS }
  }
  let t = ptSeg(ax, ay, cx, cy, dx, dy); SS[0] = PS_D; SS[1] = 0; SS[2] = t
  t = ptSeg(bx, by, cx, cy, dx, dy); if (PS_D < SS[0]) { SS[0] = PS_D; SS[1] = 1; SS[2] = t }
  t = ptSeg(cx, cy, ax, ay, bx, by); if (PS_D < SS[0]) { SS[0] = PS_D; SS[1] = t; SS[2] = 0 }
  t = ptSeg(dx, dy, ax, ay, bx, by); if (PS_D < SS[0]) { SS[0] = PS_D; SS[1] = t; SS[2] = 1 }
  return SS
}

// add polyline (local metres X, Y at path positions S, m points) of path p to the city's segment hash; every segment
// of another path close enough (THR) makes a zone on both
function register(J, p, X, Y, S, m) {
  for (let k = 0; k + 1 < m; k++) {
    const ax = X[k], ay = Y[k], bx = X[k + 1], by = Y[k + 1], rec = { p, ax, ay, bx, by, s0: S[k], s1: S[k + 1], mark: 0 }, tag = ++J.pairs
    const i0 = Math.floor((Math.min(ax, bx) - THR_MAX) / CELL), i1 = Math.floor((Math.max(ax, bx) + THR_MAX) / CELL)
    const j0 = Math.floor((Math.min(ay, by) - THR_MAX) / CELL), j1 = Math.floor((Math.max(ay, by) + THR_MAX) / CELL)
    const ex = bx - ax, ey = by - ay, el = Math.hypot(ex, ey) || 1
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
      const key = (i + 32768) * 65536 + j + 32768
      let cell = J.cells.get(key)
      if (!cell) J.cells.set(key, cell = [])
      for (const o of cell) {
        if (o.p === p || o.mark === tag) continue
        o.mark = tag
        const [d, t, u] = segSeg(ax, ay, bx, by, o.ax, o.ay, o.bx, o.by)
        if (d >= THR_MAX) continue
        const fx = o.bx - o.ax, fy = o.by - o.ay, fl = el * (Math.hypot(fx, fy) || 1), q = o.p
        const sn = Math.abs(ex * fy - ey * fx) / fl, cs = Math.abs(ex * fx + ey * fy) / fl
        if (sn < 0.2) continue          // near parallel: side by side lanes or a shared stretch, not a crossing
        if (d >= Math.max(p.hmax * sn + p.wmax * cs + q.wmax, q.hmax * sn + q.wmax * cs + p.wmax) + AIR) continue
        const za = Math.min(25, (q.wmax + p.wmax * cs + AIR) / Math.max(sn, 0.05)), zb = Math.min(25, (p.wmax + q.wmax * cs + AIR) / Math.max(sn, 0.05))
        const a = zoneOn(p, rec.s0 + (rec.s1 - rec.s0) * t, za), b = zoneOn(q, o.s0 + (o.s1 - o.s0) * u, zb)
        if (!a.conf.includes(b)) { a.conf.push(b); b.conf.push(a) }
      }
      cell.push(rec); J.n++
    }
  }
}

// signed circular difference a - b in [-L/2, L/2)
const cdiff = (a, b, L) => mod(a - b + L / 2, L) - L / 2
// the zone of path p round position s: a new one, or an existing one grown (and merged with any it now touches)
function zoneOn(p, s, zh) {
  const L = p.L, Z = p.zones
  let z = null
  for (const w of Z) {
    const m = (w.s0 + w.s1) / 2, su = m + cdiff(s, m, L)
    if (su + zh >= w.s0 - MERGE && su - zh <= w.s1 + MERGE) { z = w; w.s0 = Math.min(w.s0, su - zh); w.s1 = Math.max(w.s1, su + zh); break }
  }
  if (!z) Z.push(z = { p, s0: s - zh, s1: s + zh, conf: [], hold: new Map(), req: new Map() })
  for (let i = Z.length - 1; i >= 0; i--) {
    const w = Z[i], zm = (z.s0 + z.s1) / 2, wm = (w.s0 + w.s1) / 2, sh = zm + cdiff(wm, zm, L) - wm
    if (w === z || w.s0 + sh > z.s1 + MERGE || w.s1 + sh < z.s0 - MERGE) continue
    z.s0 = Math.min(z.s0, w.s0 + sh); z.s1 = Math.max(z.s1, w.s1 + sh)
    for (const c of w.conf) {
      c.conf = c.conf.filter(x => x !== w && x !== z); c.conf.push(z)
      if (!z.conf.includes(c)) z.conf.push(c)
    }
    for (const [k, o] of w.hold) z.hold.set(k, o)
    for (const [k, o] of w.req) z.req.set(k, o)
    w.dead = z
    Z.splice(i, 1)
  }
  if (z.s1 - z.s0 > L - 1) { z.s0 = 0; z.s1 = L }
  const sh = mod(z.s0, L) - z.s0
  z.s0 += sh; z.s1 += sh
  Z.sort((a, b) => a.s0 - b.s0)
  return z
}

export class Traffic {
  constructor(ctx) {
    this.world = ctx.world
    this.uniforms = { ...objUniforms(ctx.shared, ctx.FAR), uPixAngle: ctx.shared.uPixAngle }
    this.material = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: TRAFFIC_VERT, fragmentShader: TRAFFIC_FRAG, uniforms: this.uniforms })
    this.fleets = Object.fromEntries(Object.entries(KINDS).map(([k, kind]) => [k, new Fleet(kind, this.material, ctx.scene)]))
    this.spans = []          // { route, ri, s0, s1 } stretches of road near the camera
    this.cities = []         // { city, C, ex, ny, water, at, blocks, districts } near the camera
    this.last = null         // where spans were last gathered
    this.P = new Float64Array(3); this.T = new Float64Array(3); this.rel = new Float64Array(3)
    this.fwd = new THREE.Vector3()
    this.lakes = new LakeTest(ctx.sampler)
    this.deck = null         // per vehicle: bridge deck height (m) or null on the ground, and its grade along the heading
    this.grade = 0
    this.frame = 0; this.lastT = null; this.dt = 0
    this.ox = new Float64Array(512)   // drive() output: per vehicle (position, slot index, braking)
    this.need = []; this.needE = []   // step(): the chain of zones asked for, tail distances to their ends
    this.stats = { car: 0, truck: 0, bus: 0, built: 0, held: 0, waiting: 0, forced: 0 }
  }

  update(frame) {
    this.frame++
    this.dt = this.lastT == null ? 0 : Math.max(0, Math.min(5, frame.time - this.lastT))
    this.lastT = frame.time
    this.stats.held = 0; this.stats.waiting = 0
    const cam = frame.camPos, L = cam.length()
    this.up = [cam.x / L, cam.y / L, cam.z / L]
    this.cam = cam
    this.agl = frame.geo.agl
    this.br = globalThis.terra?.civ?.parts.bridges     // deck heights on bridges (bridges.js routeDeck / streetDeckAt)
    this.ground = L - this.agl              // radius of the ground under the camera
    frame.camera.getWorldDirection(this.fwd)
    this.cosView = Math.cos(Math.atan(Math.tan(frame.camera.fov * Math.PI / 360) * Math.hypot(1, frame.camera.aspect)))
    for (const f of Object.values(this.fleets)) f.n = 0
    if (this.agl < ROUTE_RANGE) {
      this.gather()
      this.onRoutes(frame.time)
      if (this.agl < STREET_RANGE) this.onStreets(frame.time)
    }
    for (const [k, f] of Object.entries(this.fleets)) { f.commit(); this.stats[k] = f.n }
  }

  // re-list nearby road spans and cities after the camera moves 150 m
  gather() {
    const up = this.up, w = this.world
    if (this.last && Math.hypot(up[0] - this.last[0], up[1] - this.last[1], up[2] - this.last[2]) * R < 150) return
    this.last = up
    const near = w.near(up, ROUTE_RANGE + 400), spans = new Map()
    for (const k of near.segs) {
      const type = w.seg[k * 8 + 3]
      if (type === TYPE.RAIL || segDistance(w.seg, k, up) > ROUTE_RANGE + 300) continue
      const ri = w.segRoute[k], r = w.routes[ri], i = w.segIndex[k]
      const s0 = r.cum[i], s1 = r.cum[i + 1], sp = spans.get(ri)
      if (sp) { sp.s0 = Math.min(sp.s0, s0); sp.s1 = Math.max(sp.s1, s1) } else spans.set(ri, { route: r, ri, s0, s1 })
    }
    this.spans = [...spans.values()]
    const old = new Map(this.cities.map(sc => [sc.city, sc]))
    this.cities = near.cities
      .filter(c => Math.hypot(c.dir[0] - up[0], c.dir[1] - up[1], c.dir[2] - up[2]) * R < c.radius * CITY_REACH + STREET_RANGE)
      .map(city => {
        if (old.has(city)) return old.get(city)
        const [ex, ny] = cityFrame(city.dir, city.angle)
        return { city, C: city.dir, ex, ny, water: cityWater(w, city, city.radius * CITY_REACH + 400), at: null, blocks: [], districts: [], trunk: new Map() }
      })
  }

  // vehicle slots on every lane of the nearby road spans, both directions
  onRoutes(time) {
    const { P, T } = this
    for (const { route: r, ri, s0, s1 } of this.spans) {
      const Len = r.length
      LANES[r.type].forEach((off, li) => {
        const flow = FLOW[r.type][li]
        const N = Math.max(1, Math.floor(Len / flow.gap)), g = Len / N
        for (const dir of [1, -1]) {
          const id = (ri * 4 + li) * 2 + (dir > 0 ? 1 : 0)
          const shift = (flow.speed * (0.94 + 0.12 * hash2i(ri, li, 71)) * time) % Len
          const q0 = dir > 0 ? s0 : Len - s1, q1 = dir > 0 ? s1 : Len - s0   // travel coordinate
          for (let i = Math.floor((q0 - shift) / g); i <= Math.floor((q1 - shift) / g); i++) {
            const ii = ((i % N) + N) % N, h = hash2i(ii, id, 17)
            if (h < flow.skip) continue
            const q = (i + 0.35 * (h - flow.skip) / (1 - flow.skip)) * g + shift
            if (q < q0 || q > q1) continue
            const sr = dir > 0 ? q : Len - q
            routeAt(r, sr, P, T)
            if (dir < 0) { T[0] = -T[0]; T[1] = -T[1]; T[2] = -T[2] }
            this.offsetRight(off)
            this.onDeck(s => this.br?.routeDeck(ri, s), sr, dir, 3)
            if (this.dbg) this.src = `route ${ri}/${li}/${dir}`
            this.place(flow, hash2i(ii, id, 23), hash2i(ii, id, 29), ROUTE_RANGE, false)
          }
        }
      })
    }
  }

  // City streets (district layout, spec.js): block lane loops, district edge loops, boulevards and the ring road.
  // Blocks and districts are re-listed after the camera moves 100 m; loops are built lazily, nearest first, at most
  // BUILD_BUDGET loop points per frame.
  onStreets(time) {
    const rc = Math.sqrt(Math.max(0, STREET_RANGE ** 2 - this.agl ** 2))
    let budget = BUILD_BUDGET
    for (const sc of this.cities) {
      const city = sc.city, [cu, cv] = dirToGrid(city, this.up)
      if (!sc.at || Math.hypot(cu - sc.at[0], cv - sc.at[1]) > 100) {
        sc.at = [cu, cv]
        sc.blocks = cityBlocks(city, cu, cv, rc * GRID_SLACK + 250)
        sc.blocks.sort((a, b) => Math.hypot(a.centre[0] - cu, a.centre[1] - cv) - Math.hypot(b.centre[0] - cu, b.centre[1] - cv))
        sc.districts = districtsNear(city, cu, cv, rc * GRID_SLACK + 250)
      }
      sc.J ??= { cells: new Map(), n: 0, pairs: 0, tag: ++TAGS }
      if (sc.J.n > 300000) sc.J = { cells: new Map(), n: 0, pairs: 0, tag: ++TAGS }     // paths re-register lazily
      for (const b of sc.blocks) {
        if (!this.seen(b.dir, b.rad)) continue
        if (!b.lanes) {
          if (budget <= 0) continue
          const lanes = [...new Set(profileOf(b.D.prof).lanes)], eo = PROFILES[SKIND.EDGE].lanes
          const flow = b.zone ? IND_FLOW : LOCAL_FLOWS[Math.min(b.D.prof, 2)]
          b.lanes = lanes.map((off, li) => {
            const loop = this.wetLoop(sc, blockLoop(city, b, off, eo[lanes.length === 1 ? 1 : li], 6))
            return loop && pathInit(loop, flow, hash2i(b.kx * 4 + li, b.ky, b.D.hseed) * 2147483647 | 0, 0.85, 0.3)
          })
          budget -= b.lanes.reduce((n, l) => n + (l ? l.n : 0), 1)
        }
        for (const loop of b.lanes) if (loop) budget -= this.onLoop(sc, loop, time, budget)
      }
      for (const D of sc.districts) {
        let dl = DLOOP.get(D)
        if (dl === undefined) {
          if (budget <= 0) continue
          dl = this.wetLoop(sc, districtLoop(city, D, PROFILES[SKIND.EDGE].lanes[0]))
          if (dl) {                                     // built-up stretches only (the paint covers built blocks)
            dl.built = new Uint8Array(dl.n)
            for (let i = 0; i < dl.n; i++) dl.built[i] = cityCell(city, dl.uv[i * 2], dl.uv[i * 2 + 1]).density > 0 ? 1 : 0
            pathInit(dl, EDGE_FLOW, hash2i(D.i, D.j, city.seed + 5) * 2147483647 | 0, 0.85, 0.3)
          }
          DLOOP.set(D, dl)
          budget -= dl ? 3 * dl.n : 10
        }
        if (dl && dl.chunks.some(ch => this.chunkSeen(ch))) budget -= this.onLoop(sc, dl, time, budget)
      }
      budget -= this.onTrunks(sc, cu, cv, rc * GRID_SLACK, time, budget)
    }
    this.stats.built = BUILD_BUDGET - budget
  }

  chunkSeen(ch) {
    if (ch.sf !== this.frame) { ch.sf = this.frame; ch.sv = this.seen(ch.C, ch.rad) }
    return ch.sv
  }

  // register loop / trunk-piece geometry with the city's segment hash (at most `budget` points); returns points used
  zonesOf(sc, p, budget) {
    if (p.zk === sc.J.tag || budget <= 0) return 0
    p.zk = sc.J.tag; p.zones = []; p.pieces = new Set()
    if (p.trunk) return 0
    const n = p.n, X = new Float64Array(n + 1), Y = new Float64Array(n + 1)
    for (let i = 0; i <= n; i++) this.toLocal(sc, p.dirs, (i % n) * 3, X, Y, i)
    register(sc.J, p, X, Y, p.cum, n + 1)
    return n
  }

  // local metres (city frame) of unit dir D[o..o+2] into X[i], Y[i]
  toLocal({ C, ex, ny }, D, o, X, Y, i) {
    const dx = D[o] - C[0], dy = D[o + 1] - C[1], dz = D[o + 2] - C[2]
    X[i] = (dx * ex[0] + dy * ex[1] + dz * ex[2]) * R; Y[i] = (dx * ny[0] + dy * ny[1] + dz * ny[2]) * R
  }

  // Advance the vehicles of lane path p one frame: every slot (wins null) or those within windows [[q0, q1], ..] of
  // the path's position (trunks), plus trunk vehicles with state anywhere. Writes (position, slot index, braking) of
  // the ones to show into this.ox; returns their count.
  drive(p, time, wins) {
    const sl = p.sl, n = sl.n, L = p.L, F = this.frame, base = p.v0 * time
    let cnt = 0
    const out = (x, j) => {
      if (cnt * 3 + 3 > this.ox.length) { const o = new Float64Array(this.ox.length * 2); o.set(this.ox); this.ox = o }
      this.ox[cnt * 3] = x; this.ox[cnt * 3 + 1] = j; this.ox[cnt * 3 + 2] = this.brk ? 1 : 0; cnt++
    }
    if (!wins) {
      for (let j = n - 1; j >= 0; j--) out(this.step(p, j, base, time), j)
      return cnt
    }
    const ml = p.ml + 20, N = sl.N, g = sl.g, bw = mod(base, L)
    p.ml = 0
    for (const [q0, q1] of wins) {
      for (let i = Math.floor((q1 + ml - bw) / g); i >= Math.floor((q0 - bw) / g) - 1; i--) {
        const j = sl.at[mod(i, N)]
        if (j < 0) continue
        if (p.xf[j] !== F) this.step(p, j, base, time)
        const lg = p.veh.get(j)?.lag ?? 0, q = (i + sl.fr[j]) * g + bw - lg
        p.ml = Math.max(p.ml, lg)
        if (q >= q0 && q <= q1) { this.brk = p.vs[j] < p.v0 * 0.55; out(mod(q, L), j) }
      }
    }
    for (const [j, s] of p.veh) if (p.xf[j] !== F) { this.step(p, j, base, time); p.ml = Math.max(p.ml, s.lag) }
    return cnt
  }

  // one vehicle (slot index j of path p): keep a gap to the leader, ask for the next zone, move; returns its position
  step(p, j, base, time) {
    const sl = p.sl, L = p.L, F = this.frame, dt = Math.max(this.dt, 1e-4), v0 = p.v0, hl = sl.hl[j]
    let s = p.veh.get(j)
    const lag = s ? s.lag : 0, v = s ? s.v : v0
    const u = (sl.ii[j] + sl.fr[j]) * sl.g + base - lag - v0 * this.dt       // travelled (unwrapped) at the last frame
    const x = mod(u, L)
    let room = Infinity, vL = v0
    if (sl.n > 1) {                               // leader: the next slot along
      const jl = j + 1 < sl.n ? j + 1 : 0
      let xl
      if (p.xf[jl] === F) { xl = p.xs[jl]; vL = p.vs[jl] }
      else { const o = p.veh.get(jl); vL = o ? o.v : v0; xl = (sl.ii[jl] + sl.fr[jl]) * sl.g + base - (o ? o.lag : 0) - v0 * this.dt + vL * this.dt }
      room = mod(xl - x, L) - hl - sl.hl[jl] - GAP
    }
    let vm = Math.min(lag > 0 ? Math.min(v0 * CATCH, v0 + lag / dt) : v0, v + ACCEL * dt), cap = room, denied = false
    if (room < Infinity) vm = Math.min(vm, room > 0 ? vL + Math.sqrt(2 * BRAKE * room) : 0)
    const Z = p.zones
    if (s?.hz.length) {                           // let go of zones my tail has left
      for (let k = s.hz.length - 1; k >= 0; k--) {
        let z = s.hz[k]
        while (z.dead) z = s.hz[k] = z.dead      // merged into another zone meanwhile
        if (u - hl >= s.hp[k]) { z.hold.delete(j); s.hz.splice(k, 1); s.hp.splice(k, 1) }
      }
    }
    if (Z.length) {
      // walk the zones from the first one my tail has not left: take any I am already in, then collect the chain
      // ahead (zones too close together to stop between) if it starts within braking distance
      const xt = mod(x - hl, L), last = Z[Z.length - 1], look = v * v / (2 * BRAKE) + 3 + v * dt, need = this.need
      let i = 0, sh = 0, nn = 0, e0 = 0, end = 0
      if (last.s1 > L && last.s1 - L > xt) { i = Z.length - 1; sh = -L }
      else {
        let hi = Z.length
        while (i < hi) { const m = (i + hi) >> 1; if (Z[m].s1 > xt) hi = m; else i = m + 1 }
        if (i === Z.length) { i = 0; sh = L }
      }
      for (let c = 0; c < Z.length; c++) {
        const z = Z[i], a0 = z.s0 + sh - xt, a1 = z.s1 + sh - xt          // from my tail to its start / end
        if (a0 < 2 * hl) {                                               // my front is in it already
          if (!s?.hz.includes(z)) { s ??= this.state(p, j, lag, v); this.take(s, z, j, u - hl + a1) }
        } else if (!nn) {
          if (a0 - 2 * hl > look) break
          this.needE[nn] = a1; need[nn++] = z; e0 = a0; end = a1
        } else if (a0 - end < 2 * hl + GAP || (a0 - end < 2 * hl + GAP + CHAIN && a1 - e0 < 60)) { this.needE[nn] = a1; need[nn++] = z; end = a1 }
        else break
        if (++i === Z.length) { i = 0; sh += L }
      }
      let have = !!s
      for (let k = 0; k < nn && have; k++) have = s.hz.includes(need[k])
      if (nn && !have) {
        s ??= this.state(p, j, lag, v)
        if (s.zr !== need[0]) { s.zr = need[0]; s.rq = time + (p.trunk ? 0 : LOCAL_LATE) }
        // room behind the leader to get past the chain (or 20 m into a long one: shared stretches)
        const forced = s.w > WAIT_MAX, fits = room + vL * 1.5 >= Math.min(end, e0 + 20)
        let ok = forced
        if (!ok && fits) { ok = true; for (let k = 0; k < nn && ok; k++) ok = s.hz.includes(need[k]) || this.clear(need[k], s.rq) }
        if (ok) {
          for (let k = 0; k < nn; k++) if (!s.hz.includes(need[k])) this.take(s, need[k], j, u - hl + this.needE[k])
          s.zr = null
          if (forced) { this.stats.forced++; if (this.dbg) (this.forcedAt ??= []).push([p.trunk ? p.key : p.built ? 'district' : 'block', ...(p.trunk ? p.at(x) : p.uvAt(x)).map(Math.round), Math.round(end), nn]) }
        } else {
          if (fits) for (let k = 0; k < nn; k++) { need[k].req.set(j, s); s.rf = F }
          denied = true; this.stats.waiting++
          const stop = Math.max(0, e0 - 2 * hl - 1)
          vm = Math.min(vm, Math.sqrt(2 * BRAKE * stop)); cap = Math.min(cap, stop)
        }
      }
    }
    const dx = Math.max(0, Math.min(vm * dt, cap)), vn = dx / dt, lagN = lag + v0 * this.dt - dx
    if (s || lagN > 1e-3) {
      s ??= this.state(p, j, lag, v)
      s.lag = Math.max(0, lagN); s.v = vn; s.f = F
      if (denied) s.w += this.dt; else if (vn > 2) s.w = 0
      if (lagN <= 0 && !s.hz.length && !denied) p.veh.delete(j)
      else this.stats.held++
    }
    const xn = mod(u + dx, L)
    p.xs[j] = xn; p.vs[j] = vn; p.xf[j] = F
    this.brk = vn < v0 * 0.55
    return xn
  }

  // state of a held-up vehicle: lag (m), speed, frame, held zones and the tail positions that free them, the zone
  // asked for (first of a chain), when (effective s) and in which frame, seconds waited at a stop line
  state(p, j, lag, v) {
    const s = { lag, v, f: this.frame, hz: [], hp: [], zr: null, rq: 0, rf: -1, w: 0 }
    p.veh.set(j, s)
    return s
  }
  take(s, z, j, pass) { s.hz.push(z); s.hp.push(pass); z.hold.set(j, s); z.req.delete(j) }
  // may a vehicle that asked at (effective) time t enter zone z? not while a conflicting zone is held or was asked first
  clear(z, t) {
    const F = this.frame
    for (const w of z.conf) {
      for (const [k, o] of w.hold) { if (o.f < F - 1 || !o.hz.includes(w)) w.hold.delete(k); else return false }
      for (const [k, o] of w.req) { if (o.f < F - 1 || o.rf < F - 1) w.req.delete(k); else if (o.rq < t) return false }
    }
    return true
  }

  // is a circle (unit dir C, radius m) in range and in front of the camera?
  seen(C, rad, range = STREET_RANGE) {
    const rx = C[0] * this.ground - this.cam.x, ry = C[1] * this.ground - this.cam.y, rz = C[2] * this.ground - this.cam.z
    const dist = Math.hypot(rx, ry, rz)
    return dist - rad < range && rx * this.fwd.x + ry * this.fwd.y + rz * this.fwd.z > dist * this.cosView - rad - 20
  }

  // river points of a new loop (cities with water near only), chunks near a river (bridge decks are looked up there)
  // and a lake test of the whole loop
  wetLoop(sc, loop) {
    if (!loop || !sc.water) return loop
    let lake = -Infinity, wet = null
    for (const ch of loop.chunks) ch.near = waterAt(this.world, ch.C, ch.rad + 300) === 'river'
    for (let i = 0; i < loop.n; i++) {
      const w = waterAt(this.world, loop.dirs.subarray(i * 3, i * 3 + 3), 5)
      if (w === 'river') (wet ??= new Uint8Array(loop.n))[i] = 1
      else lake = Math.max(lake, w)
    }
    loop.wet = wet
    if (lake > -Infinity) {
      loop.lake = { wet: true }
      const dirs = [], k = Math.max(1, Math.floor(loop.n / 16))
      for (let i = 0; i < loop.n; i += k) dirs.push(...loop.dirs.subarray(i * 3, i * 3 + 3))
      this.lakes.add(loop.lake, dirs, lake)
    }
    return loop
  }

  // vehicles round a loop (whole loop simulated, those in chunks in view shown); returns hash points registered
  onLoop(sc, loop, time, budget) {
    if (loop.lake?.wet) return 0
    const { P, T } = this, city = sc.city, used = this.zonesOf(sc, loop, budget)
    const n = this.drive(loop, time, null), ox = this.ox
    for (let c = 0; c < n; c++) {
      const s = ox[c * 3], j = ox[c * 3 + 1], k = loop.at(s, P, T), ch = loop.chunks[(k / CH) | 0]
      if (!this.chunkSeen(ch) || (loop.built && !loop.built[k])) continue
      this.deck = null; this.grade = 0
      if (ch.near) this.onDeck(x => this.br?.streetDeckAt?.(city.id, ...loop.uvAt(x)), s, 1, 3)
      if (this.deck == null && (loop.wet?.[k] || loop.wet?.[(k + 1) % loop.n])) continue     // over a river, no bridge
      if (this.dbg) this.src = `${loop.built ? 'district' : 'block'} ${loop.id} ${j} z${loop.zones.length} h${loop.veh.get(j)?.hz.length ?? '-'}`
      this.place(loop.flow, loop.sl.hk[j], loop.sl.hc[j], STREET_RANGE, ox[c * 3 + 2] > 0)
    }
    return used
  }

  // boulevards (out on the right, back, U-turn round the centre) and the ring road, evaluated per vehicle; returns
  // hash points registered
  onTrunks(sc, cu, cv, rc, time, budget) {
    const city = sc.city, lanes = PROFILES[SKIND.BOULEVARD].lanes
    let used = 0
    sc.tp ??= new Map()
    const B = boulevards(city)
    B.forEach((a, bi) => {
      // U-turn across the median Tu out, where this carriageway is clear of its neighbours' (all six crossing at the
      // centre would be one long conflict)
      let gap = Math.PI
      for (const b of B) if (b !== a) gap = Math.min(gap, Math.abs(cdiff(b, a, 2 * Math.PI)))
      const Tu = Math.max(20, 17 / Math.sin(gap / 2)), Tn = BOULEVARD_END * city.radius, D = Tn - Tu
      const ex = Math.cos(a), ey = Math.sin(a), tc = cu * ex + cv * ey, cr = -cu * ey + cv * ex
      if (Math.abs(cr) > rc) return
      const half = Math.sqrt(rc * rc - cr * cr), t0 = Math.max(Tu, tc - half), t1 = Math.min(Tn, tc + half)
      if (t1 < Tu - lanes[1] || t0 > Tn) return
      lanes.forEach((off, li) => {
        const Lq = 2 * D + Math.PI * off, wins = []
        if (t0 <= t1) wins.push([t0 - Tu, t1 - Tu], [2 * D - (t1 - Tu), 2 * D - (t0 - Tu)])
        if (tc - half <= Tu + off) wins.push([2 * D, Lq])
        const p = this.trunkPath(sc, `b:${bi}`, bi * 2 + li, Lq, q => {
          if (q < D) { const t = Tu + q; return this.trunkPoint(sc, t * ex, t * ey, ex, ey, off) }
          if (q < 2 * D) { const t = Tu + 2 * D - q; return this.trunkPoint(sc, t * ex, t * ey, -ex, -ey, off) }
          const f = (q - 2 * D) / off, c = Math.cos(f), s = Math.sin(f)          // inbound lane -> outbound
          return this.trunkPoint(sc, Tu * ex + off * (-ey * c - ex * s), Tu * ey + off * (ex * c - ey * s), ey * s - ex * c, -ex * s - ey * c, 0)
        })
        used += this.onLine(sc, p, wins, time, budget - used)
      })
    })
    if (city.tier < 2) return used
    const th = Math.atan2(cv, cu), rr = ringRadius(city, th), d = Math.abs(Math.hypot(cu, cv) - rr[0]), R0 = rr[2]
    if (d > rc) return used
    const dth = Math.sqrt(rc * rc - d * d) / rr[0], Lq = 2 * Math.PI * R0
    for (const dir of [1, -1]) lanes.forEach((off, li) => {
      const w = dir > 0 ? [R0 * (th - dth), R0 * (th + dth)] : [-R0 * (th + dth), -R0 * (th - dth)]
      const p = this.trunkPath(sc, 'ring', 100 + li * 2 + (dir > 0 ? 1 : 0), Lq, q => {
        const t = dir * q / R0, [r, dR] = ringRadius(city, t), c = Math.cos(t), s = Math.sin(t)
        const hx = dR * c - r * s, hy = dR * s + r * c, l = Math.hypot(hx, hy)
        return this.trunkPoint(sc, r * c, r * s, dir * hx / l, dir * hy / l, off)
      })
      used += this.onLine(sc, p, [w], time, budget - used)
    })
    return used
  }

  // a trunk lane as a lane path: closed, length Lq, at(q) sets P, T and returns the grid point
  trunkPath(sc, key, lid, Lq, at) {
    const k = key + '/' + lid
    let p = sc.tp.get(k)
    if (!p) sc.tp.set(k, p = pathInit({ L: Lq, key, trunk: true, at }, TRUNK_FLOW, hash2i(lid, 3, sc.city.seed) * 2147483647 | 0, 0.9, 0.2))
    return p
  }

  // vehicles of trunk lane p within windows [q0, q1] (registering PIECE m pieces of it round them); returns hash
  // points registered
  onLine(sc, p, wins, time, budget) {
    const city = sc.city, L = p.L, at = p.at, np = Math.ceil(L / PIECE)
    let used = this.zonesOf(sc, p, budget)
    for (const [q0, q1] of wins) for (let k = Math.floor((q0 - 60) / PIECE); k <= Math.floor((q1 + 60) / PIECE) && used < budget; k++) {
      const kk = mod(k, np)
      if (p.pieces.has(kk)) continue
      p.pieces.add(kk)
      const a = kk * PIECE, b = Math.min(L, a + PIECE), m = Math.max(2, Math.ceil((b - a) / 8) + 1)
      const X = new Float64Array(m), Y = new Float64Array(m), S = new Float64Array(m)
      for (let i = 0; i < m; i++) { S[i] = a + (b - a) * i / (m - 1); at(S[i]); this.toLocal(sc, this.P, 0, X, Y, i) }
      register(sc.J, p, X, Y, S, m)
      used += m
    }
    const n = this.drive(p, time, wins), ox = this.ox
    for (let c = 0; c < n; c++) {
      const qq = ox[c * 3], j = ox[c * 3 + 1], uv = at(qq)
      const piece = this.trunkPiece(sc, p.key, uv)
      if (!piece.built) continue
      this.deck = null; this.grade = 0
      if (piece.near) {                               // bridge deck: height here, grade from 3 m either side
        const z = this.br?.streetDeckAt?.(city.id, uv[0], uv[1]) ?? null
        if (z != null) {
          const zf = this.br.streetDeckAt(city.id, ...at(qq + 3)) ?? z, zb = this.br.streetDeckAt(city.id, ...at(qq - 3)) ?? z
          at(qq)
          this.deck = z; this.grade = (zf - zb) / 6
        }
      }
      if (this.deck == null && piece.wet) continue    // over a river, no bridge
      if (this.dbg) this.src = `trunk ${p.key}/${p.id} ${j} z${p.zones.length} h${p.veh.get(j)?.hz.length ?? '-'}`
      this.place(p.flow, p.sl.hk[j], p.sl.hc[j], STREET_RANGE, ox[c * 3 + 2] > 0)
    }
    return used
  }

  // built-up / river (wet: on it, near: within 300 m, bridge decks looked up) flags of a trunk near grid point uv,
  // cached per 40 m cell
  trunkPiece(sc, key, uv) {
    const k = key + ':' + Math.floor(uv[0] / 40) + ':' + Math.floor(uv[1] / 40)
    let p = sc.trunk.get(k)
    if (!p) {
      if (sc.trunk.size > 20000) sc.trunk.clear()
      const d = gridToDir(sc.city, uv[0], uv[1]), near = sc.water && waterAt(this.world, d, 300) === 'river'
      p = { built: cityCell(sc.city, uv[0], uv[1]).density > 0, near, wet: near && waterAt(this.world, d, 12) === 'river' }
      sc.trunk.set(k, p)
    }
    return p
  }

  // P, T of a vehicle at grid (u, v) heading (hx, hy) (grid, unit), off m to the right of its heading; returns [u, v]
  trunkPoint({ city, C, ex, ny }, u, v, hx, hy, off) {
    const { P, T } = this, [x, y, a, b, c, d] = warpGrid(city, u, v)
    const nx = hy, nyy = -hx, jx = a * nx + b * nyy, jy = c * nx + d * nyy, k = off / Math.hypot(jx, jy)
    const lx = x + jx * k, ly = y + jy * k, tx = a * hx + b * hy, ty = c * hx + d * hy
    for (let j = 0; j < 3; j++) { P[j] = C[j] + (lx * ex[j] + ly * ny[j]) / R; T[j] = tx * ex[j] + ty * ny[j] }
    normalize(P)
    const kk = dot3(T, P)
    for (let j = 0; j < 3; j++) T[j] -= P[j] * kk
    normalize(T)
    return [u, v]
  }

  // deck height and grade at x along a route / street loop (deckAt(x) = height or null), travelling dir (+-1)
  onDeck(deckAt, x, dir, dx) {
    const z = deckAt(x) ?? null
    this.deck = z
    this.grade = z == null ? 0 : dir * ((deckAt(x + dx) ?? z) - (deckAt(x - dx) ?? z)) / (2 * dx)
  }

  // move P sideways by off metres to the right of the heading T
  offsetRight(off) {
    const { P, T } = this, r = [T[1] * P[2] - T[2] * P[1], T[2] * P[0] - T[0] * P[2], T[0] * P[1] - T[1] * P[0]]
    for (let j = 0; j < 3; j++) P[j] += r[j] * off / R
    normalize(P)
  }

  // cull by distance and view cone, pick kind and paint from the slot hashes hk, hc, and emit (brake: slow / stopped)
  place(flow, hk, hc, range, brake) {
    const { P, T, rel, cam, up } = this
    const hd = Math.hypot(P[0] - up[0], P[1] - up[1], P[2] - up[2]) * R
    if (hd * hd + this.agl * this.agl > range * range) return
    rel[0] = P[0] * this.ground - cam.x; rel[1] = P[1] * this.ground - cam.y; rel[2] = P[2] * this.ground - cam.z
    const dist = Math.hypot(rel[0], rel[1], rel[2])
    if (rel[0] * this.fwd.x + rel[1] * this.fwd.y + rel[2] * this.fwd.z < dist * this.cosView - 20) return
    const kind = hk < flow.bus ? 'bus' : hk < flow.bus + flow.truck ? 'truck' : 'car'
    const col = kind === 'bus' ? PAINT[hc < 0.5 ? 12 : 8] : PAINT[Math.floor(hc * PAINT.length)]
    // anchor at sea level: the shader adds the ground height
    rel[0] = P[0] * R - cam.x; rel[1] = P[1] * R - cam.y; rel[2] = P[2] * R - cam.z
    const suv = kind === 'car' && hc * 7 % 1 < 0.3, fl = this.fleets[kind]
    if (this.dbg && fl.n < fl.cap) (fl.src ??= [])[fl.n] = this.src     // (debug: which path each instance is on)
    fl.add(rel, P, T, kind === 'car' ? 0.92 + 0.2 * (hk * 13 % 1) : 1, suv ? 1.18 : 1, hc, col, (kind === 'bus' ? 1 : 0) + (brake ? 2 : 0), this.deck, this.grade)
  }
}
