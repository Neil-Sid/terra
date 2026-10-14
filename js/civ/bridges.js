// Bridges where highways, country roads and railways cross a river channel near the camera, and where city
// trunk streets (district edges, boulevards, the ring) cross rivers. Crossings are found lazily per camera area and cached. A bridge's deck follows the
// levelled road/rail bed (sampled once on the GPU; streets: the ground at both ends plus a hump over the water),
// its piers stand on the river bed, and its type fits the site: concrete beams for short spans, girders on piers,
// steel trusses (railways, some big rivers), tied arches over big channels, stone arches in towns. Every piece
// is a (sheared) box over a chord of the deck centreline, drawn by one instanced mesh (bridges.glsl.js).
import * as THREE from 'three'
import { R } from '../config.js'
import { TYPE, HALF, BRIDGE, bridgeClear, hash2i, riverDepth, riverPlain, riverMeander, CITY_REACH, dirToGrid, gridToDir, cityShape, dirToLocal,
  SKIND, profileOf, cityCell, districtAt, districtEdges, fromLattice, boulevards, BOULEVARD_END, ringRadius } from './spec.js'
import { objUniforms } from './objects.glsl.js'
import { BRIDGE_VERT, bridgeFrag, KIND } from './bridges.glsl.js'

const RANGE = 9000                   // bridges are drawn within this distance (m); the shader dithers them out from 0.8 of it
const AREA = 12000                   // crossings are listed within this radius of a scan centre ...
const RESCAN = 2500                  // ... and listed again once the camera is this far from it
const REFILL = 400                   // instance data is rebuilt around a new anchor after this much camera travel (m)
const MAX = 30000                    // box instances
const DETAIL = [Infinity, 3500, 1300]   // pieces of detail level 0 / 1 / 2 are drawn while their bridge is this close
const T = 0.6                        // deck slab depth below the road surface (m)
const STEP = 10                      // route bridges: terrain samples along the road (m)
const BED_F = [R / 1500, R / 1500, R / 4000]   // band limit of road / rail beds in the country (civ.glsl.js terrainCiv)

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const norm = a => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l] }
const flat = (v, p) => { const k = dot(v, p); return [v[0] - p[0] * k, v[1] - p[1] * k, v[2] - p[2] * k] }   // tangential part at p
const lerp3 = (a, b, t) => norm([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t])
const move = (p, t, m) => norm([p[0] + t[0] * m / R, p[1] + t[1] * m / R, p[2] + t[2] * m / R])
const dist = (a, b) => Math.hypot(...flat(sub(a, b), a)) * R
const at3 = (F, o) => [F[o], F[o + 1], F[o + 2]]
const smooth01 = t => { t = Math.min(1, Math.max(0, t)); return t * t * (3 - 2 * t) }
const cross2 = (a, b) => a[0] * b[1] - a[1] * b[0]
// river segment nearest to d among segment ids ks as civRiver (civ.glsl.js) picks it: { d (m from the meandering
// centre), hw, level } or null
function channelAt(S, ks, d) {
  let best = null
  for (const k of ks) {
    const o = k * 12, A = at3(S, o), ab = sub(at3(S, o + 4), A), L2 = Math.max(dot(ab, ab), 1e-20), pa = sub(d, A)
    const t = Math.min(1, Math.max(0, dot(pa, ab) / L2)), v = flat(pa.map((x, i) => x - ab[i] * t), d)
    const hw = S[o + 3] + (S[o + 7] - S[o + 3]) * t, s = S[o + 10] + t * Math.sqrt(L2) * R
    const sd = norm(cross(d, ab)), off = riverMeander(s, hw, S[o + 11] - s) / R, e = Math.hypot(v[0] - sd[0] * off, v[1] - sd[1] * off, v[2] - sd[2] * off) * R
    if (!best || e - hw < best.d - best.hw) best = { d: e, hw, level: S[o + 8] + (S[o + 9] - S[o + 8]) * t }
  }
  return best
}
// a bed h (m) descending towards the channel ch at grade, as civBed lowers it
function descend(h, ch, grade) {
  if (!ch) return h
  const fp = ch.hw + riverPlain(ch.hw)
  return Math.min(h, Math.max(ch.level + bridgeClear(ch.hw) + grade * Math.max(0, ch.d - fp), h - grade * Math.max(0, fp + BRIDGE.ramp - Math.max(ch.d, fp))))
}
// city ground smoothing weight at dir (c.flatW in civ.glsl.js)
function flatAt(cities, d) {
  let f = 0
  for (const c of cities) { const [x, y] = dirToLocal(c, d); if (Math.hypot(x, y) < c.radius * CITY_REACH) f = Math.max(f, smooth01((cityShape(c, x, y) + 0.2) / 0.22)) }
  return f
}
// 2D segment intersection: [t along p0->p1, u along q0->q1] or null
function intersect(p0, p1, q0, q1) {
  const d1 = [p1[0] - p0[0], p1[1] - p0[1]], d2 = [q1[0] - q0[0], q1[1] - q0[1]], den = cross2(d1, d2)
  if (Math.abs(den) < 1e-9) return null
  const w = [q0[0] - p0[0], q0[1] - p0[1]], t = cross2(w, d2) / den, u = cross2(w, d1) / den
  return t >= 0 && t <= 1 && u >= 0 && u <= 1 ? [t, u] : null
}
const lerp2 = (a, b, f) => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]

// Trunk streets of a city that may cross the grid points gpts (a river's course): the edges of the districts under
// them, the boulevards and the ring, sampled exactly as spec.js streetsNear samples them (so street bridges sit on the
// painted centre lines and on traffic's polylines): [{ id, kind (SKIND), pts: [[u, v], ...] (grid, <= 25 m apart) }].
// Local rows / connectors (lanes, streets, SUPER rows) get no bridges: they end at the river bank.
function trunksAlong(city, gpts) {
  const out = [], seen = new Set()
  const poly = (id, f, len, kind) => { const n = Math.max(1, Math.ceil(len / 25)); out.push({ id, kind, pts: Array.from({ length: n + 1 }, (_, i) => f(i / n)) }) }
  for (const g of gpts) for (const e of districtEdges(city, districtAt(city, g[0], g[1]).D)) {
    if (seen.has(e.id)) continue
    seen.add(e.id)
    poly(e.id, t => fromLattice(city, e.a[0] + (e.b[0] - e.a[0]) * t, e.a[1] + (e.b[1] - e.a[1]) * t), Math.hypot(e.b[0] - e.a[0], e.b[1] - e.a[1]) * 3, SKIND.EDGE)
  }
  const L = BOULEVARD_END * city.radius
  boulevards(city).forEach((a, i) => poly(`b${i}`, t => [Math.cos(a) * t * L, Math.sin(a) * t * L], L, SKIND.BOULEVARD))
  if (city.tier >= 2) poly('ring', t => { const th = t * 2 * Math.PI, r = ringRadius(city, th)[0]; return [Math.cos(th) * r, Math.sin(th) * r] }, 2 * Math.PI * city.radius * 0.5, SKIND.RING)
  return out
}
// the street polyline pts (grid) from `half` m before to `half` m after the point at segment j, fraction t (continued
// straight past the polyline's ends): { gp (grid points), xs (m along, 0 .. 2 half) }
function pathAround(city, pts, j, t, half) {
  const c = lerp2(pts[j], pts[j + 1], t), cd = gridToDir(city, ...c)
  const side = s => {
    const out = []
    let cur = c, curD = cd, prev = s > 0 ? pts[j] : pts[j + 1], k = s > 0 ? j + 1 : j, m = 0
    for (let guard = 0; guard < 500; guard++) {
      let nx = pts[k]
      if (!nx) { const dx = cur[0] - prev[0], dy = cur[1] - prev[1], l = Math.hypot(dx, dy) || 1; nx = [cur[0] + dx / l * 20, cur[1] + dy / l * 20] }
      const nd = gridToDir(city, ...nx), len = dist(curD, nd)
      k += s
      if (len < 1e-3) continue
      if (m + len >= half) { out.push([lerp2(cur, nx, (half - m) / len), half]); break }
      m += len; out.push([nx, m]); prev = cur; cur = nx; curD = nd
    }
    return out
  }
  const back = side(-1).reverse(), fwd = side(1)
  return { gp: [...back.map(p => p[0]), c, ...fwd.map(p => p[0])], xs: [...back.map(p => half - p[1]), half, ...fwd.map(p => half + p[1])] }
}
// nearest point of a street bridge's grid path to (u, v): { x (m along the deck), y (m across), i (segment) } or null
// beyond its ends
function projectPath(b, u, v) {
  const { gp, xs } = b
  let best = null
  for (let i = 0; i < gp.length - 1; i++) {
    const a = gp[i], e = [gp[i + 1][0] - a[0], gp[i + 1][1] - a[1]], L2 = e[0] * e[0] + e[1] * e[1] || 1e-9
    const f = ((u - a[0]) * e[0] + (v - a[1]) * e[1]) / L2
    if (f < -0.02 || f > 1.02) continue
    const ft = Math.min(1, Math.max(0, f)), mpg = (xs[i + 1] - xs[i]) / Math.sqrt(L2)   // metres per grid unit
    const y = ((u - a[0]) * e[1] - (v - a[1]) * e[0]) / Math.sqrt(L2) * mpg
    if (!best || Math.abs(y) < Math.abs(best.y)) best = { x: xs[i] + ft * (xs[i + 1] - xs[i]), y, i }
  }
  return best
}

// unit box: x along 0..1, y across 0..1, z up 0..1
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
  const g = new THREE.InstancedBufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3))
  g.setIndex(idx)
  return g
}

export class Bridges {
  constructor(ctx) {
    this.world = ctx.world
    this.sampler = ctx.sampler
    this.bridges = new Map()         // key -> bridge (crossing + built pieces)
    this.visible = []
    this.scanAt = null
    this.anchor = new THREE.Vector3()
    this.time = 0
    this.stats = { crossings: 0, drawn: 0, instances: 0 }
    this.uniforms = { ...objUniforms(ctx.shared), uAnchor: { value: new THREE.Vector3() }, uAnchorRel: { value: new THREE.Vector3() } }
    const geo = boxGeometry()
    const attr = (name, size) => { const a = new THREE.InstancedBufferAttribute(new Float32Array(MAX * size), size); geo.setAttribute(name, a); return a }
    this.attrs = { A: attr('iA', 3), B: attr('iB', 3), X: attr('iBox', 4), M: attr('iMisc', 4), C: attr('iArch', 4) }
    geo.instanceCount = 0
    const material = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: BRIDGE_VERT, fragmentShader: bridgeFrag(RANGE), uniforms: this.uniforms })
    this.mesh = new THREE.Mesh(geo, material)
    this.mesh.frustumCulled = false
    this.mesh.visible = false
    ctx.scene.add(this.mesh)
  }

  // ---- finding crossings ----
  scan(C) {
    this.scanAt = C
    const w = this.world, near = w.near(C, AREA)
    for (const [k, b] of this.bridges) if (dist(C, b.dir) > 2.5 * AREA) this.bridges.delete(k)
    const e = norm([C[2], 0, -C[0]]), nn = cross(C, e), xy = p => { const v = sub(p, C); return [dot(v, e) * R, dot(v, nn) * R] }
    const rivers = near.rivers.map(k => {
      const o = k * 12, A = at3(w.riv, o), B = at3(w.riv, o + 4)
      return { o, A, B, a: xy(A), b: xy(B), hw: Math.max(w.riv[o + 3], w.riv[o + 7]), len: Math.hypot(...sub(B, A)) * R }
    }).filter(r => Math.min(Math.hypot(...r.a), Math.hypot(...r.b)) < AREA + r.len)
    if (!rivers.length) return
    for (const j of near.segs) {
      const A = at3(w.seg, j * 8), B = at3(w.seg, j * 8 + 4), a = xy(A), b = xy(B)
      for (const r of rivers) {
        const m = 2 * r.hw
        if (Math.max(a[0], b[0]) < Math.min(r.a[0], r.b[0]) - m || Math.min(a[0], b[0]) > Math.max(r.a[0], r.b[0]) + m) continue
        if (Math.max(a[1], b[1]) < Math.min(r.a[1], r.b[1]) - m || Math.min(a[1], b[1]) > Math.max(r.a[1], r.b[1]) + m) continue
        const hit = intersect(a, b, r.a, r.b)
        if (hit) this.routeCrossing(j, A, B, hit[0], r, hit[1])
      }
    }
    for (const city of near.cities) if (dist(C, city.dir) < AREA + city.radius * CITY_REACH) this.streetCrossings(city, rivers)
    this.stats.crossings = this.bridges.size
  }

  // river r at chord fraction u: polyline point, tangent, side (as the shader's meander uses), half width, level, s
  riverAt(r, u, P) {
    const S = this.world.riv, o = r.o, tn = norm(flat(sub(r.B, r.A), P))
    return { tn, sd: norm(cross(P, tn)), hw: S[o + 3] + (S[o + 7] - S[o + 3]) * u, level: S[o + 8] + (S[o + 9] - S[o + 8]) * u, s: S[o + 10] + u * r.len, total: S[o + 11] }
  }

  // road/rail segment j (A->B) crosses river chord r at t / u: solve for the meandering channel centre along the road.
  // The terrain shader opens the bed there when |chord sine| >= BRIDGE.sin; how far is found by sampling (finishRoute)
  routeCrossing(j, A, B, t, r, u) {
    const w = this.world, ri = w.segRoute[j], route = w.routes[ri], type = w.seg[j * 8 + 3]
    const P = lerp3(A, B, t), rt = norm(flat(sub(B, A), P)), rv = this.riverAt(r, u, P)
    const a = dot(rt, rv.sd), b = dot(rt, rv.tn), off = s => riverMeander(s, rv.hw, rv.total - s)
    if (Math.abs(a) < BRIDGE.sin) return                          // running along the river: causeway, as the shader keeps it
    const g = X => X * a - off(rv.s + X * b)
    let lo = -(1.6 * rv.hw + 5) / Math.abs(a), hi = -lo
    if (g(lo) * g(hi) > 0) return
    for (let i = 0; i < 30; i++) { const m = (lo + hi) / 2; if (g(lo) * g(m) <= 0) hi = m; else lo = m }
    const X = (lo + hi) / 2, sc = rv.s + X * b, dOff = (off(sc + 1) - off(sc - 1)) / 2
    const Tc = norm(rv.tn.map((v, k) => v + rv.sd[k] * dOff)), sin = Math.max(0.2, Math.abs(dot(rt, norm(cross(P, Tc))))), gate = smooth01((Math.abs(a) - BRIDGE.sin) / 0.05)
    const sC = Math.min(route.length, Math.max(0, w.seg[j * 8 + 7] + t * Math.hypot(...sub(B, A)) * R + X))
    for (const o of this.bridges.values()) if (o.ri === ri && Math.abs(o.sC - sC) < 30) return   // found via the next chord
    const reach = (rv.hw + riverPlain(rv.hw) + BRIDGE.reach) / Math.abs(a) + 100
    const win0 = Math.max(0, sC - reach), win1 = Math.min(route.length, sC + reach)
    const key = `${ri}:${Math.round(sC)}`, seed = Math.floor(hash2i(ri, Math.round(sC / 10), 77) * 4096)
    const pt = w.routePoint(route, sC), city = w.nearestCity(pt.dir, 40000)
    this.bridges.set(key, {
      key, ri, route, type, sC, win0, win1, s0: win0, S: win1 - win0, W: HALF[type] + (type === TYPE.RAIL ? 1.4 : 1.0), dir: pt.dir, axis: pt.tan,
      hw: rv.hw, level: rv.level, D: riverDepth(rv.hw), plain: riverPlain(rv.hw), sin, wet: rv.hw / sin, seed, gate,
      lit: !!city && city.distance < city.city.radius * 1.6 + 1500,
    })
  }

  // trunk streets of a city (district edges, boulevards, ring) crossing the meandering river centres near it. Rivers
  // are walked in steps of <= 100 m and mapped to grid space; trunks win over each other by kind (ring / boulevard first)
  streetCrossings(city, rivers) {
    const reach = city.radius * CITY_REACH, found = []
    for (const r of rivers) {
      const mid = lerp3(r.A, r.B, 0.5)
      if (dist(mid, city.dir) > reach + r.len / 2 + 2 * r.hw) continue
      const n = Math.max(1, Math.ceil(r.len / Math.min(100, (28 * r.hw + 300) / 10))), pts = []
      for (let i = 0; i <= n; i++) {
        const u = i / n, P = lerp3(r.A, r.B, u), rv = this.riverAt(r, u, P)
        const M = move(P, rv.sd, riverMeander(rv.s, rv.hw, rv.total - rv.s))
        pts.push(dist(M, city.dir) < reach ? { g: dirToGrid(city, M), hw: rv.hw, level: rv.level } : null)
      }
      const on = pts.filter(p => p)
      if (on.length < 2) continue
      const m = 60, box = [Infinity, Infinity, -Infinity, -Infinity]
      for (const p of on) { box[0] = Math.min(box[0], p.g[0] - m); box[1] = Math.min(box[1], p.g[1] - m); box[2] = Math.max(box[2], p.g[0] + m); box[3] = Math.max(box[3], p.g[1] + m) }
      for (const tr of trunksAlong(city, on.map(p => p.g))) {
        const S = tr.pts
        for (let j = 0; j < S.length - 1; j++) {
          const a = S[j], b = S[j + 1]
          if (Math.max(a[0], b[0]) < box[0] || Math.min(a[0], b[0]) > box[2] || Math.max(a[1], b[1]) < box[1] || Math.min(a[1], b[1]) > box[3]) continue
          for (let i = 0; i < n; i++) {
            const p = pts[i], q = pts[i + 1]
            if (!p || !q) continue
            const hit = intersect(a, b, p.g, q.g)
            if (!hit) continue
            const ts = [b[0] - a[0], b[1] - a[1]], tr2 = [q.g[0] - p.g[0], q.g[1] - p.g[1]]
            const sin = Math.abs(cross2(ts, tr2)) / (Math.hypot(...ts) * Math.hypot(...tr2) || 1), f = hit[1]
            // big rivers only where the street crosses near square (no km-long skewed decks)
            if (sin >= 0.3 + 0.3 * smooth01(((p.hw + q.hw) / 2 - 60) / 200)) found.push({ tr, j, t: hit[0], sin, hw: p.hw + (q.hw - p.hw) * f, level: p.level + (q.level - p.level) * f })
          }
        }
      }
    }
    found.sort((x, y) => y.tr.kind - x.tr.kind || y.hw - x.hw)
    for (const c of found) this.streetCrossing(city, c)
  }

  // a bridge along trunk tr over the river at its segment j, fraction t: deck half = the channel and half the
  // floodplain beyond the banks (along the street), if the street is built up at both ends and no other bridge is there
  streetCrossing(city, { tr, j, t, sin, hw, level }) {
    const plain = riverPlain(hw), half = (hw + 0.5 * plain + 12) / sin, prof = profileOf(tr.kind), W = prof.walk[1] + 0.5
    const { gp, xs } = pathAround(city, tr.pts, j, t, half), S = xs[xs.length - 1]
    if (gp.length < 2 || S < 20) return
    // no overlap with another street bridge of this city (the same crossing found from the next river chord, or two
    // trunks meeting at the river)
    for (const o of this.bridges.values()) {
      if (!o.street || o.city !== city) continue
      const lim = ((o.W + W) * 0.8) ** 2
      if (gp.some(p => o.gp.some(q => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < lim))) return
    }
    // built up (painted) at both ends: a block with density > 0 on either side of the street
    const built = (p, q) => {
      const e = [q[0] - p[0], q[1] - p[1]], l = Math.hypot(...e) || 1, o = prof.half + 15
      return [-1, 1].some(s => cityCell(city, p[0] - s * e[1] / l * o, p[1] + s * e[0] / l * o).density > 0)
    }
    const k = gp.length - 1
    if (!built(gp[0], gp[1]) || !built(gp[k], gp[k - 1])) return
    const c = lerp2(tr.pts[j], tr.pts[j + 1], t), dir = gridToDir(city, ...c)
    const key = `c${city.id}:${tr.id}:${Math.round(c[0] / 10)}:${Math.round(c[1] / 10)}`, seed = Math.floor(hash2i(city.id, Math.round(c[0] / 10) * 7 + Math.round(c[1] / 10), 78) * 4096)
    const br = {
      key, street: true, city, sid: tr.id, kind: tr.kind, prof: Math.min(tr.kind, 3), yellow: hash2i(city.seed, 7, 3) < 0.4, gp, xs, type: -1, S, W, dir,
      hw, level, D: riverDepth(hw), plain, sin, wet: hw / sin, seed, lit: true,
    }
    br.axis = norm(flat(sub(this.dirAt(br, S / 2 + 1), this.dirAt(br, S / 2 - 1)), dir))
    br.style = this.styleOf(br)
    this.bridges.set(key, br)
  }

  styleOf(b) {
    const h = hash2i(b.seed, 1, 5), S = b.S
    if (S < 50) return 'beam'
    if (b.street) return S < 450 && h < 0.45 ? 'stone' : b.hw > 60 && h < 0.75 ? 'arch' : 'girder'
    if (b.type === TYPE.RAIL) return S < 300 && b.tall < 25 && h < 0.2 ? 'stone' : 'truss'
    if (b.hw > 110 && h < (b.type === TYPE.HIGHWAY ? 0.35 : 0.3)) return 'arch'
    return b.type === TYPE.ROAD && b.hw > 70 && h < 0.55 ? 'truss' : 'girder'
  }

  // ---- geometry ----
  dirAt(b, x) {
    if (!b.street) return this.world.routePoint(b.route, b.s0 + x).dir
    return gridToDir(b.city, ...this.gridAt(b, x))
  }
  // street bridges: grid point x m along the deck (on the street's polyline, straight beyond its sampled ends)
  gridAt(b, x) {
    const { gp, xs } = b, n = xs.length - 1
    let i = 0
    while (i < n - 1 && xs[i + 1] < x) i++
    return lerp2(gp[i], gp[i + 1], (x - xs[i]) / Math.max(1e-6, xs[i + 1] - xs[i]))
  }
  // deck height at x (m along the bridge), from heights at even steps over 0..S
  deck(b, x) { return this.lerpAt(b.hs, x / b.S) }
  lerpAt(hs, u) {
    const n = hs.length - 1, f = Math.min(n, Math.max(0, u * n)), i = Math.min(n - 1, Math.floor(f))
    return hs[i] + (hs[i + 1] - hs[i]) * (f - i)
  }
  // ground under x: sampled along route bridges; streets: the analytic channel and floodplain, as civRiverBed shapes them
  ground(b, x) {
    if (b.gs) return this.lerpAt(b.gs, x / b.S)
    const d = Math.abs(x - b.S / 2) * b.sin, hw = b.hw
    return d < hw ? b.level - b.D * (1 - (d / hw) ** 2) : b.level + (1 + 0.004 * hw) * smooth01((d - hw) / b.plain)
  }
  // streets run from the ground at one end to the ground at the other with a hump for clearance
  streetHeights(b, [hA, hB]) {
    const n = Math.max(3, Math.ceil(b.S / 20) + 1), clear = b.level + 3 + 0.02 * b.hw + (b.S > 200 ? 3 : 0)
    return Array.from({ length: n }, (_, i) => {
      const u = i / (n - 1), base = hA + (hB - hA) * u
      return base + Math.max(0, clear - base) * smooth01(u / 0.3) * smooth01((1 - u) / 0.3)
    })
  }
  // Route bridges span the stretch around the crossing where terrain() lies below the bed (the shader opened it
  // there), bridging gaps of up to GAP samples. g = terrain along the window, bedLow = the bed's band-limited terrain,
  // descended towards the river as civBed does (only valid out of towns; in towns just the channel opens). Deck = the
  // bed fitted to the ground at both ends (in towns a straight line between them), climbing at <= 4% to a clearance.
  finishRoute(b, g, bedLow, dirs) {
    const n = g.length, GAP = 15, near = this.world.near(b.dir, (b.win1 - b.win0) / 2 + 1500), grade = BRIDGE.grade[b.type]
    const fl = g.map((_, i) => flatAt(near.cities, at3(dirs, i * 3))), town = fl.map(f => f >= BRIDGE.town)
    const bed = bedLow.map((h, i) => {
      h = Math.max(h, 2)
      return h + (descend(h, channelAt(this.world.riv, near.rivers, at3(dirs, i * 3)), grade) - h) * b.gate * (1 - smooth01(fl[i] / BRIDGE.town))
    })
    const open = i => i >= 0 && i < n && (town[i] ? Math.abs(i - ic) * STEP * b.sin < b.hw + 6 : bed[i] - g[i] > BRIDGE.open)
    const ic = Math.min(n - 1, Math.max(0, Math.round((b.sC - b.win0) / STEP)))
    let c = -1
    for (let k = 0; k <= Math.ceil((b.wet + 20) / STEP) && c < 0; k++) c = open(ic + k) ? ic + k : open(ic - k) ? ic - k : -1
    let lo = c, hi = c
    if (c < 0) {                                           // not opened (city or lake bed): the old span over the causeway
      const half = Math.ceil((b.wet * 1.12 + 8 + (b.type === TYPE.RAIL ? 4 : 0)) / STEP)
      lo = Math.max(1, ic - half); hi = Math.min(n - 2, ic + half)
    } else {
      for (let i = c; i >= 0; i--) { if (open(i)) lo = i; else if (lo - i > GAP) break }
      for (let i = c; i < n; i++) { if (open(i)) hi = i; else if (i - hi > GAP) break }
    }
    const clear = b.level + 3 + 0.02 * b.hw + (b.type === TYPE.RAIL ? 2 : 0), hump = i => clear - 0.04 * Math.max(0, Math.abs(i - ic) * STEP - b.wet)
    const ref = i => town[i] ? g[i] : bed[i]
    let i0 = Math.max(0, lo - 1), i1 = Math.min(n - 1, hi + 1)
    while (i0 > 0 && hump(i0) > ref(i0)) i0--
    while (i1 < n - 1 && hump(i1) > ref(i1)) i1++
    const fit = i => { const e = g[i] - bed[i]; return Math.abs(e) < 3 ? e : 0 }, f0 = fit(i0), f1 = fit(i1), inTown = town.slice(i0, i1 + 1).some(t => t)
    const hs = [], gs = []
    for (let i = i0; i <= i1; i++) {
      const u = (i - i0) / Math.max(1, i1 - i0)
      hs.push(Math.max(inTown ? g[i0] + (g[i1] - g[i0]) * u : bed[i] + f0 + (f1 - f0) * u, hump(i)))
      gs.push(g[i])
    }
    b.s0 = b.win0 + i0 * STEP; b.S = (i1 - i0) * STEP; b.xc = (ic - i0) * STEP; b.gs = gs
    for (const o of this.bridges.values()) if (o !== b && o.ri === b.ri && o.hs && o.s0 < b.s0 + b.S && o.s0 + o.S > b.s0) { this.bridges.delete(b.key); return }
    const mid = this.world.routePoint(b.route, b.s0 + b.S / 2)
    b.dir = mid.dir; b.axis = mid.tan
    b.tall = Math.max(...hs.map((h, i) => h - gs[i]))
    b.style = this.styleOf(b)
    this.build(b, hs)
  }
  build(b, hs) {
    b.hs = hs
    b.cl = hs.map((_, i) => this.dirAt(b, b.S * i / (hs.length - 1)))     // centreline at the height samples
    b.parts = this.design(b).map(p => ({ ...p, a: this.dirAt(b, p.x0).map(v => v * R), b: this.dirAt(b, p.x1).map(v => v * R) }))
    b.born = this.time
  }

  spansOf(b) {
    const S = b.S, h = hash2i(b.seed, 3, 5), tall = b.tall ?? 0
    const gl = Math.max(b.type === TYPE.HIGHWAY ? 45 : 34 + 10 * h, Math.min(110, 0.5 * tall))   // tall bridges: fewer, longer spans
    const even = (a, c, len, kind) => {
      const n = Math.max(1, Math.round((c - a) / len))
      return Array.from({ length: n }, (_, i) => ({ a: a + (c - a) * i / n, b: a + (c - a) * (i + 1) / n, kind }))
    }
    switch (b.style) {
      case 'beam': return [{ a: 0, b: S, kind: 'girder' }]
      case 'girder': return even(0, S, gl, 'girder')
      case 'truss': return even(0, S, Math.max(70 + 30 * h, Math.min(160, 0.6 * tall)), 'truss')
      case 'stone': return even(0, S, 22 + 14 * h, 'stone')
    }
    // tied arch over the channel, girder approaches
    const M = Math.min(S, Math.max(2 * b.wet * 0.8, 120 + 140 * h), 320), a = Math.min(S - M, Math.max(0, (b.xc ?? S / 2) - M / 2))
    const x0 = a < 20 ? 0 : a, x1 = S - a - M < 20 ? S : a + M
    return [...(x0 > 0 ? even(0, x0, gl, 'girder') : []), { a: x0, b: x1, kind: 'arch' }, ...(x1 < S ? even(x1, S, gl, 'girder') : [])]
  }

  // pieces in bridge coordinates: x along the deck 0..S (m), y left of the direction of travel, z absolute (m)
  // at x0, sheared by dz up to x1. lod 0 always, 1 mid range, 2 close up (see DETAIL)
  design(b) {
    const out = [], { S, W } = b, Z = x => this.deck(b, x), rail = b.type === TYPE.RAIL
    const box = (x0, x1, y0, y1, z0, z1, kind, lod = 0, dz = 0, arch = null) => out.push({ x0, x1, y0: Math.min(y0, y1), y1: Math.max(y0, y1), z0, z1, dz, kind, lod, arch })
    // a stretch following the deck in pieces of <= 20 m; zb / zt relative to the road surface
    const along = (x0, x1, y0, y1, zb, zt, kind, lod = 0, arch = null) => {
      const n = Math.max(1, Math.ceil((x1 - x0) / 20))
      for (let i = 0; i < n; i++) {
        const a = x0 + (x1 - x0) * i / n, c = x0 + (x1 - x0) * (i + 1) / n
        box(a, c, y0, y1, Z(a) + zb, Z(a) + zt, kind, lod, Z(c) - Z(a), arch)
      }
    }
    // straight member from (xa, za) to (xb, zb), t thick across its axis
    const member = (xa, xb, y0, y1, za, zb, t, kind, lod = 0) => { const k = t * Math.hypot(1, (zb - za) / (xb - xa)); box(xa, xb, y0, y1, za - k / 2, za + k / 2, kind, lod, zb - za) }
    const steelBeams = hash2i(b.seed, 4, 5) < 0.45, hwy = b.type === TYPE.HIGHWAY
    // street decks carry the painted profile (iArch.y: 2 edge street, 3 boulevard / ring; iArch.z: yellow centre line)
    along(0, S, -W, W, -T, 0.05, b.street ? KIND.STREET : [KIND.HW, KIND.ROAD, KIND.RAIL][b.type], 0, b.street ? [b.prof, b.yellow ? 1 : 0, 0] : null)
    const spans = this.spansOf(b), depth = []
    for (const sp of spans) {
      const L = sp.b - sp.a
      if (sp.kind === 'girder') {
        const g = Math.min(6, Math.max(0.9, L / 20))
        depth.push(g)
        for (const f of W > 8 ? [-0.72, -0.24, 0.24, 0.72] : [-0.5, 0.5]) along(sp.a, sp.b, f * W - 0.4, f * W + 0.4, -T - g, -T, steelBeams ? KIND.STEEL : KIND.CONCRETE, Math.abs(f) < 0.5 ? 1 : 0)
      } else if (sp.kind === 'truss') {
        depth.push(1.0)
        const H = Math.min(22, Math.max(6.5, L / 7)) + (rail ? 1 : 0), np = Math.max(4, Math.round(L / (H * 0.95))), p = L / np, yc = W + 0.3, xa = sp.a, xb = sp.b
        for (const s of [-1, 1]) {
          const y0 = s * yc - 0.28, y1 = s * yc + 0.28
          along(xa, xb, y0, y1, -T - 0.5, 0.45, KIND.STEEL)                                    // bottom chord
          along(xa + p, xb - p, y0, y1, H - 0.6, H, KIND.STEEL)                                // top chord
          member(xa, xa + p, y0, y1, Z(xa) + 0.2, Z(xa + p) + H - 0.3, 0.55, KIND.STEEL)       // end posts
          member(xb - p, xb, y0, y1, Z(xb - p) + H - 0.3, Z(xb) + 0.2, 0.55, KIND.STEEL)
          for (let i = 1; i < np; i++) { const x = xa + i * p; box(x - 0.18, x + 0.18, y0 + 0.05, y1 - 0.05, Z(x) + 0.4, Z(x) + H - 0.6, KIND.STEEL, 1) }
          for (let i = 1; i < np - 1; i++) {                                                     // Pratt diagonals, falling towards mid-span
            const x = xa + i * p, down = x + p / 2 < (xa + xb) / 2
            member(x, x + p, y0 + 0.08, y1 - 0.08, Z(x) + (down ? H - 0.35 : 0.45), Z(x + p) + (down ? 0.45 : H - 0.35), 0.4, KIND.STEEL, 1)
          }
        }
        for (let i = 1; i < np; i++) { const x = xa + i * p; box(x - 0.2, x + 0.2, -yc, yc, Z(x) + H - 0.5, Z(x) + H - 0.1, KIND.STEEL, 1) }   // top struts
        for (const x of [xa + p, xb - p]) box(x - 0.25, x + 0.25, -yc, yc, Z(x) + H - 1.8, Z(x) + H - 0.1, KIND.STEEL)                        // portals
      } else if (sp.kind === 'arch') {
        depth.push(1.2)
        const xa = sp.a, Ha = Math.min(55, Math.max(12, L * 0.19)), yc = W + 0.45, N = 16
        const za = x => Z(x) + 0.4 + Ha * 4 * ((x - xa) / L) * (1 - (x - xa) / L)
        for (const s of [-1, 1]) {
          const y0 = s * yc - 0.45, y1 = s * yc + 0.45
          along(xa, sp.b, y0, y1, -T - 1.1, 0.5, KIND.STEEL)                                     // tie girder
          for (let i = 0; i < N; i++) member(xa + L * i / N, xa + L * (i + 1) / N, y0, y1, za(xa + L * i / N), za(xa + L * (i + 1) / N), 1.3, KIND.STEEL)
          const nh = Math.max(6, Math.round(L / 9))
          for (let i = 1; i < nh; i++) { const x = xa + L * i / nh; box(x - 0.07, x + 0.07, s * yc - 0.07, s * yc + 0.07, Z(x) + 0.5, za(x), KIND.STEEL, 2) }   // hangers
        }
        for (let i = 1; i < N; i++) { const x = xa + L * i / N; if (za(x) - Z(x) > 7.5) box(x - 0.35, x + 0.35, -yc, yc, za(x) - 0.4, za(x) + 0.4, KIND.STEEL, 1) }   // wind bracing
      } else {
        // stone arch: a masonry block pier to pier with the opening cut out in the shader, and the barrel's soffit
        depth.push(0)
        const pw = Math.max(2.5, L * 0.16), ho = (L - pw) / 2, zs = b.level + 0.8, xm = (sp.a + sp.b) / 2
        const rise = Math.max(1.5, Math.min(ho * 0.95, Z(xm) - T - 1.2 - zs)), foot = Math.min(b.level - b.D, this.ground(b, sp.a), this.ground(b, sp.b)) - 1
        box(sp.a, sp.b, -W + 0.3, W - 0.3, foot, Z(sp.a) - T + 0.02, KIND.STONE_ARCH, 0, Z(sp.b) - Z(sp.a), [ho, zs, rise])
        const zc = u => zs + rise * Math.sqrt(Math.max(0, 1 - (u / ho) ** 2))
        for (let i = 0; i < 12; i++) { const u0 = -ho + 2 * ho * i / 12, u1 = u0 + 2 * ho / 12; box(xm + u0, xm + u1, -W + 0.3, W - 0.3, zc(u0), zc(u0) + 0.8, KIND.STONE, 0, zc(u1) - zc(u0)) }
      }
    }
    // piers between spans, abutments at the ends
    for (let i = 1; i < spans.length; i++) {
      if (spans[i].kind === 'stone') continue
      const x = spans[i].a, top = Z(x) - T - Math.max(depth[i - 1], depth[i]), foot = this.ground(b, x) - 1
      if (top - 1.3 - foot < 0.5) continue
      const k = 1 + 0.012 * (top - foot)                  // tall viaduct piers are thicker
      box(x - 1.2 * k, x + 1.2 * k, -W * 0.92, W * 0.92, top - 1.3, top, KIND.CONCRETE)
      if (W > 9) for (const s of [-1, 1]) box(x - 0.9 * k, x + 0.9 * k, s * W * 0.55 - 0.9 * k, s * W * 0.55 + 0.9 * k, foot, top - 1.3, KIND.CONCRETE)
      else box(x - 1.1 * k, x + 1.1 * k, -W * 0.6, W * 0.6, foot, top - 1.3, KIND.CONCRETE)
    }
    for (const [x0, x1, x] of [[-4, 0.5, 0], [S - 0.5, S + 4, S]]) box(x0, x1, -W - 0.8, W + 0.8, Math.min(this.ground(b, x), Z(x) - 6) - 2, Z(x) - T, b.style === 'stone' ? KIND.STONE : KIND.CONCRETE)
    // parapets and railings, highway median barrier, lamps
    for (const s of [-1, 1]) {
      if (b.style === 'stone') along(0, S, s * (W - 0.45), s * W, 0.0, 1.0, KIND.STONE)
      else if (b.street || rail) along(0, S, s * (W - 0.12), s * W, 0.05, 1.15, KIND.RAILING, 1)
      else { along(0, S, s * (W - 0.35), s * W, 0.05, 0.85, KIND.CONCRETE); along(0, S, s * (W - 0.24), s * (W - 0.14), 0.85, 1.25, KIND.RAILING, 2) }
    }
    if (hwy) along(0, S, -0.35, 0.35, 0.05, 0.85, KIND.CONCRETE, 1)
    if (b.lit && !rail) for (let x = 16; x < S - 8; x += 32) for (const s of [-1, 1]) {
      const y = s * (W - (b.style === 'stone' ? 0.25 : 0.55)), z = Z(x)
      box(x - 0.1, x + 0.1, y - 0.1, y + 0.1, z + 0.05, z + 8.2, KIND.POST, 1)
      box(x - 0.3, x + 0.3, y, y - s * 1.8, z + 8.0, z + 8.25, KIND.LAMP, 1)
    }
    return out
  }

  // ---- per frame ----
  update(frame) {
    this.time = frame.time
    const cam = frame.camPos, L = cam.length(), up = [cam.x / L, cam.y / L, cam.z / L], agl = frame.geo.agl
    if (agl > RANGE) { this.mesh.visible = false; return }
    if (!this.scanAt || dist(up, this.scanAt) > RESCAN) { this.scan(up); this.dirty = true }
    if (this.dirty || this.anchor.distanceTo(cam) > REFILL) this.fill(cam, up, agl)
    this.uniforms.uAnchorRel.value.copy(this.anchor).sub(cam)
    this.mesh.visible = this.stats.instances > 0
  }

  // instance data relative to a fresh anchor at the camera; bridges without heights are sampled first
  fill(cam, up, agl) {
    this.dirty = false
    this.anchor.copy(cam)
    this.uniforms.uAnchor.value.copy(cam)
    const vis = []
    for (const b of this.bridges.values()) {
      b.near = Math.hypot(dist(up, b.dir), agl)
      if (b.near < RANGE + b.S / 2) vis.push(b)
    }
    vis.sort((a, b) => a.near - b.near)
    this.request(vis.filter(b => !b.hs))
    this.visible = vis.filter(b => b.parts)
    this.streets = new Map()                              // built bridges by street line and by route (vehicle queries)
    this.routes = new Map()
    const file = (m, key, b) => { if (!m.has(key)) m.set(key, []); m.get(key).push(b) }
    for (const b of this.bridges.values()) if (b.hs) {
      if (b.street) file(this.streets, b.city.id, b)
      else file(this.routes, b.ri, b)
    }
    const { A, B, X, M, C } = this.attrs
    let n = 0
    for (const b of this.visible) {
      const seed = b.seed + (b.lit ? 4096 : 0), far = b.near - b.S / 2
      for (const p of b.parts) {
        if (n >= MAX) break
        if (far > DETAIL[p.lod]) continue
        A.array.set([p.a[0] - cam.x, p.a[1] - cam.y, p.a[2] - cam.z], n * 3)
        B.array.set([p.b[0] - cam.x, p.b[1] - cam.y, p.b[2] - cam.z], n * 3)
        X.array.set([p.y0, p.y1, p.z0, p.z1], n * 4)
        M.array.set([p.dz, p.kind, seed, b.born], n * 4)
        C.array.set([p.x0, ...(p.arch ?? [0, 0, 0])], n * 4)
        n++
      }
    }
    for (const a of Object.values(this.attrs)) { a.clearUpdateRanges(); a.addUpdateRange(0, n * a.itemSize); a.needsUpdate = true }
    this.mesh.geometry.instanceCount = n
    this.stats.drawn = this.visible.length
    this.stats.instances = n
  }

  // one batch: street ends; route windows every STEP m at full detail and at the bed's band limit (civ off)
  request(list) {
    if (this.pending || !list.length) return
    const dirs = [], beds = [[], []], jobs = []
    for (const b of list.slice(0, 40)) {
      if (b.street) { jobs.push([b, dirs.length / 3, 2]); for (const x of [0, b.S]) dirs.push(...this.dirAt(b, x)); continue }
      const n = Math.floor((b.win1 - b.win0) / STEP) + 1, bd = beds[b.type === TYPE.RAIL ? 1 : 0]
      jobs.push([b, dirs.length / 3, n, bd.length / 3])
      for (let i = 0; i < n; i++) { const d = this.world.routePoint(b.route, b.win0 + i * STEP).dir; dirs.push(...d); bd.push(...d) }
    }
    this.pending = true
    const S = this.sampler, get = (s, o, n) => Array.from({ length: n }, (_, i) => s[(o + i) * 8])
    Promise.all([S.sampleAsync(new Float32Array(dirs), 1.2e6, true), ...beds.map((d, i) => d.length ? S.sampleAsync(new Float32Array(d), BED_F[i], true) : null)]).then(([s, ...bs]) => {
      for (const [b, o, n, ob] of jobs) {
        if (b.street) this.build(b, this.streetHeights(b, get(s, o, n)))
        else this.finishRoute(b, get(s, o, n), get(bs[b.type === TYPE.RAIL ? 1 : 0], ob, n), dirs.slice(o * 3, (o + n) * 3))
      }
      this.pending = false
      this.dirty = true
    })
  }

  // ---- queries ----
  // deck height (m above sea level) if dir is on a built bridge deck, else null (camera collision). With h (the
  // asker's height, m), decks more than 8 m above it are skipped, so a camera can pass under a tall bridge.
  onBridge(dir, h = Infinity) {
    const d = dir.isVector3 ? [dir.x, dir.y, dir.z] : dir
    for (const b of this.visible) {
      if (dist(d, b.dir) > b.S / 2 + 60) continue
      const cl = b.cl, n = cl.length - 1
      let j = 0, best = Infinity
      for (let k = 0; k <= n; k++) { const e = dist(d, cl[k]); if (e < best) { best = e; j = k } }
      j = Math.min(j, n - 1)
      const t = norm(flat(sub(cl[j + 1], cl[j]), cl[j])), v = sub(d, cl[j])
      const x = b.S * j / n + dot(v, t) * R, y = dot(v, cross(cl[j], t)) * R
      if (x < -1 || x > b.S + 1 || Math.abs(y) > b.W) continue
      const z = this.deck(b, x) + 0.05
      if (z < h + 8) return z
    }
    return null
  }

  // vehicles: deck height at s m along route ri, or null off its bridges; up to `margin` m beyond a deck end the
  // end height is returned (train cars reaching onto a bridge)
  routeDeck(ri, s, margin = 0) {
    const list = this.routes?.get(ri)
    if (list) for (const b of list) if (s > b.s0 - margin && s < b.s0 + b.S + margin) return this.deck(b, Math.min(b.S, Math.max(0, s - b.s0)))
    return null
  }
  // ---- street bridges for vehicles (contract with traffic.js; notes/bridges.md, notes/layout.md). All in the city's
  // grid space (spec.js), city = city.id; only built bridges (deck heights known) count.
  // Deck height (m above sea level) where a street bridge covers grid (u, v): within W + margin m of its centre line
  // and between its ends; else null. Grade along a heading: ask at two points.
  streetDeckAt(cityId, u, v, margin = 0) {
    const list = this.streets?.get(cityId)
    if (list) for (const b of list) {
      const p = this.onStreetBridge(b, u, v, b.W + margin)
      if (p) return this.deck(b, p.x)
    }
    return null
  }
  // true when a built street bridge runs along the polyline pts (grid, e.g. a streetsNear span): a point of pts lies on
  // its deck and the polyline runs with the bridge there (streets ending on the bridge in a T-junction don't count)
  bridgedAt(cityId, pts) {
    const list = this.streets?.get(cityId)
    if (!list) return false
    for (const b of list) for (let i = 0; i < pts.length; i++) {
      const p = this.onStreetBridge(b, pts[i][0], pts[i][1], b.W)
      if (!p) continue
      const q = pts[i + 1] ?? pts[i - 1], e = [q[0] - pts[i][0], q[1] - pts[i][1]], a = b.gp[p.i], c = b.gp[p.i + 1], t = [c[0] - a[0], c[1] - a[1]]
      if (Math.abs(e[0] * t[0] + e[1] * t[1]) > 0.8 * Math.hypot(...e) * Math.hypot(...t)) return true
    }
    return false
  }
  // built street bridges of a city: [{ kind (SKIND), W (deck half width, m), pts (grid centre line), xs (m along at
  // each point), S (length, m), deckAt(x) (m above sea level at x m along) }]
  streetBridges(cityId) {
    return (this.streets?.get(cityId) ?? []).map(b => ({ kind: b.kind, W: b.W, pts: b.gp, xs: b.xs, S: b.S, deckAt: x => this.deck(b, Math.min(b.S, Math.max(0, x))) }))
  }
  // (u, v) on bridge b's deck (|across| <= w m, between the ends): { x, y (m), i (path segment) } or null
  onStreetBridge(b, u, v, w) {
    const bb = b.gbox ??= b.gp.reduce((m, p) => [Math.min(m[0], p[0]), Math.min(m[1], p[1]), Math.max(m[2], p[0]), Math.max(m[3], p[1])], [Infinity, Infinity, -Infinity, -Infinity])
    const pad = w * 1.4 + 2
    if (u < bb[0] - pad || u > bb[2] + pad || v < bb[1] - pad || v > bb[3] + pad) return null
    const p = projectPath(b, u, v)
    return p && Math.abs(p.y) <= w && p.x >= -0.5 && p.x <= b.S + 0.5 ? p : null
  }
  // deprecated round-4 hooks (grid lines u / v = k*B): no street bridges sit on those lines any more
  streetDeck() { return null }
  bridgedSpan() { return false }

  // nearest crossing to dir (lists the area around dir first if needed); for tests and tools
  nearest(dir, filter = null) {
    const d = dir.isVector3 ? [dir.x, dir.y, dir.z] : dir
    if (!this.scanAt || dist(d, this.scanAt) > RESCAN) this.scan(d)
    let best = null, bd = Infinity
    for (const b of this.bridges.values()) { const e = dist(d, b.dir); if (e < bd && (!filter || filter(b))) { bd = e; best = b } }
    if (!best) return null
    const p = best.dir, east = norm([p[2], 0, -p[0]]), north = cross(p, east)
    return {
      lat: Math.asin(p[1]) * 180 / Math.PI, lon: Math.atan2(p[0], p[2]) * 180 / Math.PI, heading: Math.atan2(dot(best.axis, east), dot(best.axis, north)),
      kind: best.street ? 'street' : ['highway', 'road', 'rail'][best.type], style: best.style, span: Math.round(best.S), river: Math.round(2 * best.hw),
      level: +best.level.toFixed(1), deck: best.hs ? +this.deck(best, best.S / 2).toFixed(1) : null, distance: Math.round(bd), key: best.key,
    }
  }
}
