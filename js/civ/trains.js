// Trains: a timetable for every railway, drawn near the camera. Contracts in notes/CIV.md, log in notes/trains.md.
// Main lines get stations at both end cities and at towns they pass within 1.5 km of; a line is split into
// 1-3 sections (split at a station, or a halt added there) with one train each, so trains never meet on the
// single track. Passenger trains stop at every station of their section, freight runs end to end, and each
// freight spur has a shunter going between the city yard and its industrial site. Positions are a pure function
// of time. Cars near the camera are instances, one draw call per car shape; stations.js draws the stations.
import * as THREE from 'three'
import { R } from '../config.js'
import { TYPE, TIER, hash2i } from './spec.js'
import { objUniforms } from './objects.glsl.js'
import { TRAIN_VERT, TRAIN_FRAG } from './trains.glsl.js'
import { Stations } from './stations.js'

const RANGE = 7000             // draw cars within this distance of the camera (m)
const MAX_CARS = 700           // per car shape
const GAP = 1.0                // coupler gap between cars (m)
const PLAT = 240               // platform length (m): fits the longest passenger train
const END = 90                 // platforms of end stations start this far from the city centre (m)
const TOWN_R = 1500            // a line stops at towns whose centre is this close (m)
const APART = 2 * PLAT + 900   // minimum spacing of stations along a line (m)
const HALTS = 38e3             // open country gets a halt about this often (m)
const PSTEP = 15               // rail height profile spacing for riding (m)
const CIV_FLAT = R / 300       // terrain() detail at which rail beds are levelled (CIV_FLAT_F in GLSL)
const KIND = { LOCO: 0, COACH: 1, BOX: 2, TANK: 3, HOPPER: 4, FLAT: 5 }
const HALT_NAMES = ['Road', 'Halt', 'Junction', 'Parkway', 'Crossing', 'Siding', 'Bridge', 'Heath', 'Green', 'Moor']
const COMPASS = ['North', 'North East', 'East', 'South East', 'South', 'South West', 'West', 'North West']
const mod = (x, m) => ((x % m) + m) % m

// car shapes in car-local metres (x forward, y left, z up from the rail top); b = bogie offset from the centre
const rect = (y, z0, z1) => [[-y, z0], [y, z0], [y, z1], [-y, z1]]
const octagon = (r, zc) => Array.from({ length: 8 }, (_, i) => [r * Math.cos((i + 0.5) * Math.PI / 4), zc + r * Math.sin((i + 0.5) * Math.PI / 4)])
const SHAPES = [
  { len: 21, b: 7.2, build: g => {            // locomotive: hood unit with a raked cab nose at +x
    loft(g, -10.4, 8.0, rect(1.5, 1.2, 4.1), 3)
    loft(g, 8.0, 10.5, rect(1.5, 1.2, 4.1), 3, rect(1.35, 1.2, 3.2))
    loft(g, -8.5, 5.5, rect(0.8, 4.1, 4.45), 1)
  } },
  { len: 26, b: 9.5, build: g => loft(g, -12.95, 12.95, [[-1.45, 1.2], [1.45, 1.2], [1.45, 3.6], [1.05, 4.1], [-1.05, 4.1], [-1.45, 3.6]], 2) },
  { len: 16, b: 5.6, build: g => loft(g, -7.95, 7.95, rect(1.5, 1.2, 4.3), 0) },
  { len: 17, b: 6.0, build: g => { loft(g, -8.3, 8.3, octagon(1.45, 2.75), 0); loft(g, -0.6, 0.6, rect(0.5, 4.0, 4.5), 0) } },
  { len: 15, b: 5.2, build: g => loft(g, -7.45, 7.45, [[-0.9, 1.2], [0.9, 1.2], [1.5, 2.2], [1.5, 4.0], [-1.5, 4.0], [-1.5, 2.2]], 0) },
  { len: 18.6, b: 6.8, build: g => {          // container flat: deck with a 40 ft and a 20 ft box
    loft(g, -9.3, 9.3, rect(1.3, 1.05, 1.4), 1)
    loft(g, -9.2, 2.9, rect(1.22, 1.4, 4.0), 4)
    loft(g, 3.1, 9.2, rect(1.22, 1.4, 4.0), 4)
  } },
]
const LIVERY = {
  passenger: [0xb3202a, 0x1d4f91, 0x2f6b3a, 0xd8d8d0, 0x6c7a89, 0x5b2a6e],
  freight: [0xe07b1a, 0xf2c200, 0x1d4f91, 0x2a2a2a, 0x2f6b3a, 0xa01818],
  [KIND.BOX]: [0x6b2d1c, 0x8a3b22, 0x2e4a6b, 0x3c5a3a, 0x7a5a2a],
  [KIND.TANK]: [0x1c1c1c, 0xb8bcc0, 0xe8e8e0, 0x1c1c1c],
  [KIND.HOPPER]: [0x5a5550, 0x6b4a33, 0x7a7a70],
  [KIND.FLAT]: [0x444444],
}
const colour = (list, u) => new THREE.Color(list[Math.floor(u * list.length) % list.length])

// geometry: triangle soup with flat outward normals
function tri(g, a, b, c, part) {
  const e1 = a.map((v, i) => b[i] - v), e2 = a.map((v, i) => c[i] - v)
  let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]
  const out = [0, 1, 2].map(i => (a[i] + b[i] + c[i]) / 3 - g.centre[i])
  if (n[0] * out[0] + n[1] * out[1] + n[2] * out[2] < 0) { [b, c] = [c, b]; n = n.map(v => -v) }
  const l = Math.hypot(...n) || 1
  for (const p of [a, b, c]) { g.pos.push(...p); g.nrm.push(n[0] / l, n[1] / l, n[2] / l); g.part.push(part) }
}
// extrude a convex cross-section [[y, z]...] from x0 to x1 (sec1 != sec0 tapers it), with end caps;
// axis 'y' extrudes across the car instead ([[x, z]...] from y0 to y1), for wheels
function loft(g, x0, x1, sec0, part, sec1 = sec0, axis = 'x') {
  const at = (x, [u, z]) => axis === 'x' ? [x, u, z] : [u, x, z]
  const a = sec0.map(p => at(x0, p)), b = sec1.map(p => at(x1, p)), n = a.length
  g.centre = [0, 1, 2].map(i => [...a, ...b].reduce((s, p) => s + p[i], 0) / (2 * n))
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    tri(g, a[i], a[j], b[j], part); tri(g, a[i], b[j], b[i], part)
  }
  for (let i = 1; i < n - 1; i++) { tri(g, a[0], a[i], a[i + 1], part); tri(g, b[0], b[i], b[i + 1], part) }
}
function buildShape(shape) {
  const g = { pos: [], nrm: [], part: [] }
  for (const x of [-shape.b, shape.b]) {
    loft(g, x - 1.4, x + 1.4, rect(1.0, 0.45, 1.0), 1)                                        // bogie frame
    for (const w of [x - 1.05, x + 1.05]) {                                                    // wheelsets on the rails
      const disc = Array.from({ length: 6 }, (_, i) => [w + 0.46 * Math.cos(i * Math.PI / 3), 0.46 + 0.46 * Math.sin(i * Math.PI / 3)])
      loft(g, -0.84, 0.84, disc, 5, disc, 'y')
    }
  }
  loft(g, -shape.len / 2 + 0.4, shape.len / 2 - 0.4, rect(1.3, 0.95, 1.25), 1)                 // underframe
  shape.build(g)
  const geo = new THREE.InstancedBufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(g.pos, 3))
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(g.nrm, 3))
  geo.setAttribute('aPart', new THREE.Float32BufferAttribute(g.part, 1))
  return geo
}

// A run through stops x[0] = 0 < x[1] < ... (distance of the train's low end from the first stop); each leg
// accelerates, cruises and brakes, then the train dwells dwell[i] at stop i (the last one before turning back).
function makeRun(x, dwell, vmax, acc) {
  const legs = []
  let t = 0
  for (let i = 0; i + 1 < x.length; i++) {
    const D = x[i + 1] - x[i], v = Math.min(vmax, Math.sqrt(acc * D)), ta = v / acc, T = D / v + ta
    legs.push({ x: x[i], D, v, ta, T, t0: t, acc })
    t += T + dwell[i + 1]
  }
  return { legs, T: t, end: x[x.length - 1], moving: t - dwell[x.length - 1] }
}
const legPos = (g, t) => t <= g.ta ? 0.5 * g.acc * t * t : t < g.T - g.ta ? g.v * t - 0.5 * g.v * g.ta : g.D - 0.5 * g.acc * (g.T - t) ** 2
function runAt(run, t) {
  for (const g of run.legs) if (t < g.t0 + g.T) return g.x + legPos(g, Math.max(0, t - g.t0))
  return run.end
}
// inverse: seconds into the run when the low end first reaches x
function runTime(run, x) {
  const g = run.legs.find(g => x <= g.x + g.D) ?? run.legs[run.legs.length - 1], d = Math.min(Math.max(x - g.x, 0), g.D), da = 0.5 * g.v * g.ta
  return g.t0 + (d < da ? Math.sqrt(2 * d / g.acc) : d < g.D - da ? (d + da) / g.v : g.T - Math.sqrt(2 * (g.D - d) / g.acc))
}

// distance (m) from dir to segment k of the packed segment table, and s along its route there
function segDist(S, k, p) {
  const A = [S[k * 8], S[k * 8 + 1], S[k * 8 + 2]], ab = [S[k * 8 + 4] - A[0], S[k * 8 + 5] - A[1], S[k * 8 + 6] - A[2]]
  const pa = [p[0] - A[0], p[1] - A[1], p[2] - A[2]], L2 = Math.max(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2, 1e-20)
  const u = Math.min(1, Math.max(0, (pa[0] * ab[0] + pa[1] * ab[1] + pa[2] * ab[2]) / L2))
  return { d: Math.hypot(pa[0] - ab[0] * u, pa[1] - ab[1] * u, pa[2] - ab[2] * u) * R, s: S[k * 8 + 7] + u * Math.sqrt(L2) * R }
}

export class Trains {
  constructor(ctx) {
    this.ctx = ctx
    this.world = ctx.world
    this.now = 0
    this.enabled = true
    this.trains = []
    this.stations = []
    this.byRoute = new Map()
    this.profiles = new Map()
    this.names = new Set()
    const towns = this.townStops()
    this.world.routes.forEach((r, ri) => {
      if (r.type !== TYPE.RAIL) return
      if (r.spur) this.addSpur(r, ri)
      else this.addLine(r, ri, towns.get(ri) ?? [])
    })
    this.nearDir = null
    this.nearRoutes = []
    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: TRAIN_VERT, fragmentShader: TRAIN_FRAG,
      uniforms: { ...objUniforms(ctx.shared, ctx.FAR), uCamPos: { value: new THREE.Vector3() } },
    })
    this.parts = SHAPES.map(shape => {
      const geo = buildShape(shape), n = MAX_CARS
      const attr = (name, size) => {
        const a = new THREE.InstancedBufferAttribute(new Float32Array(n * size), size)
        a.setUsage(THREE.DynamicDrawUsage)
        geo.setAttribute(name, a)
        return a
      }
      const part = { A: attr('iA', 3), B: attr('iB', 3), C: attr('iCol', 3), M: attr('iMisc', 2), H: attr('iH', 2), n: 0, mesh: new THREE.Mesh(geo, this.material) }
      part.mesh.frustumCulled = false
      part.mesh.visible = false
      geo.instanceCount = 0
      ctx.scene.add(part.mesh)
      return part
    })
    // headlight beams: a strip on the track ahead of each locomotive, drawn additively with the same program
    const loco = this.parts[KIND.LOCO].mesh.geometry, beam = new THREE.InstancedBufferGeometry(), pos = []
    for (let i = 0; i < 16; i++) {
      const x0 = 10.6 + i * 5, x1 = x0 + 5, w0 = 0.9 + 0.05 * (x0 - 10.5), w1 = 0.9 + 0.05 * (x1 - 10.5)
      pos.push(x0, -w0, 0.12, x1, w1, 0.12, x0, w0, 0.12, x0, -w0, 0.12, x1, -w1, 0.12, x1, w1, 0.12)
    }
    beam.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    beam.setAttribute('normal', new THREE.Float32BufferAttribute(pos.map((_, i) => i % 3 === 2 ? 1 : 0), 3))
    beam.setAttribute('aPart', new THREE.Float32BufferAttribute(pos.filter((_, i) => i % 3 === 0).map(() => 6), 1))
    for (const n of ['iA', 'iB', 'iCol', 'iMisc', 'iH']) beam.setAttribute(n, loco.attributes[n])
    this.beam = new THREE.Mesh(beam, new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: TRAIN_VERT, fragmentShader: TRAIN_FRAG, uniforms: this.material.uniforms,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    }))
    this.beam.frustumCulled = false
    ctx.scene.add(this.beam)
    this.stationView = new Stations(ctx, this.stations)
    this.fwd = new THREE.Vector3()
    const count = k => this.trains.filter(t => t.kind === k).length
    console.info(`trains: ${this.trains.length} trains (${count('passenger')} passenger, ${count('freight')} freight, ${count('shunt')} shunters), ` +
      `${this.trains.reduce((n, t) => n + t.cars.length, 0)} cars, ${this.stations.length} stations`)
  }

  // towns within TOWN_R of a main line that doesn't end there: route index -> [{ s, city }]
  townStops() {
    const w = this.world, out = new Map()
    for (const c of w.cities) {
      if (c.tier < TIER.TOWN) continue
      const best = new Map()
      for (const k of w.near(c.dir, TOWN_R).segs) {
        const ri = w.segRoute[k], r = w.routes[ri]
        if (w.seg[k * 8 + 3] !== TYPE.RAIL || r.spur || r.from === c.id || r.to === c.id) continue
        const hit = segDist(w.seg, k, c.dir)
        if (hit.d < TOWN_R && !(best.get(ri)?.d < hit.d)) best.set(ri, hit)
      }
      for (const [ri, hit] of best) { if (!out.has(ri)) out.set(ri, []); out.get(ri).push({ s: hit.s, city: c, d: hit.d }) }
    }
    return out
  }

  station(r, ri, s, city, kind, len = PLAT) {
    const seed = Math.floor(hash2i(ri, Math.round(s), 4247) * 1e6)
    let side = hash2i(seed, 1, 4248) < 0.5 ? -1 : 1
    if (city && kind === 'town') {             // building on the town side of the line
      const { dir: p, tan } = this.world.routePoint(r, s), c = city.dir
      const left = [p[1] * tan[2] - p[2] * tan[1], p[2] * tan[0] - p[0] * tan[2], p[0] * tan[1] - p[1] * tan[0]]
      side = (c[0] - p[0]) * left[0] + (c[1] - p[1]) * left[1] + (c[2] - p[2]) * left[2] >= 0 ? 1 : -1
    }
    return { route: r, ri, s, len, city, kind, side, seed, name: this.name(r, s, city, seed), big: kind === 'terminus' && city.tier >= TIER.CITY }
  }
  // a unique name: the town's own, then by compass point (termini of a city with several lines); halts are named
  // after the nearest settlement plus Road, Halt, Junction...
  name(r, s, city, seed) {
    const w = this.world, p = w.routePoint(r, s).dir, c = city ?? w.nearestCity(p, 60000)?.city ?? w.cities[r.from]
    const e = [c.dir[2], 0, -c.dir[0]], n = [c.dir[1] * e[2], c.dir[2] * e[0] - c.dir[0] * e[2], -c.dir[1] * e[0]], d = p.map((v, i) => v - c.dir[i])
    const octant = mod(Math.round(Math.atan2(d[0] * e[0] + d[2] * e[2], d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) / (Math.PI / 4)), 8)
    const list = city ? ['', ...COMPASS.map((_, i) => ' ' + COMPASS[(octant + i) % 8])] : HALT_NAMES.map((_, i) => ' ' + HALT_NAMES[(seed + i) % HALT_NAMES.length])
    const name = list.map(x => c.name + x).find(x => !this.names.has(x)) ?? c.name + list[0]
    this.names.add(name)
    return name
  }

  addLine(r, ri, towns) {
    const L = r.length, cities = this.world.cities
    if (L < 4 * APART) return
    const first = this.station(r, ri, END + PLAT / 2, cities[r.from], 'terminus'), last = this.station(r, ri, L - END - PLAT / 2, cities[r.to], 'terminus')
    const st = [first]
    for (const t of towns.sort((a, b) => a.s - b.s)) {
      if (t.s - st[st.length - 1].s > APART && last.s - t.s > APART) st.push(this.station(r, ri, t.s, t.city, 'town'))
    }
    st.push(last)
    // country halts in the long gaps (towns hardly ever lie right on a line)
    for (let i = st.length - 1; i > 0; i--) {
      const a = st[i - 1].s, gap = st[i].s - a, n = Math.floor(gap / HALTS * (0.8 + 0.5 * hash2i(ri, i, 4245)))
      for (let k = n - 1; k >= 1; k--) st.splice(i, 0, this.station(r, ri, a + gap * (k + 0.3 * hash2i(ri, i * 8 + k, 4250) - 0.15) / n, null, 'halt'))
    }
    // sections: split at the station nearest each cut, or add a halt there
    const n = Math.max(1, Math.min(3, Math.floor(L / 70e3 * (0.6 + hash2i(ri, 0, 4241))))), cuts = [0]
    for (let k = 1; k < n; k++) {
      const target = L * k / n
      let i = st.reduce((b, x, j) => Math.abs(x.s - target) < Math.abs(st[b].s - target) ? j : b, 0)
      if (Math.abs(st[i].s - target) > 0.3 * L / n) {
        i = st.findIndex(x => x.s > target)
        st.splice(i, 0, this.station(r, ri, target, null, 'halt'))
      }
      if (i > cuts[cuts.length - 1] && i < st.length - 1) cuts.push(i)
    }
    cuts.push(st.length - 1)
    for (let k = 0; k + 1 < cuts.length; k++) {
      if (k) Object.assign(st[cuts[k]], { boundary: true, len: 2 * PLAT + 30 })
      const rnd = i => hash2i(ri, k * 16 + i, 4242), kind = rnd(0) < 0.62 ? 'passenger' : 'freight'
      const cars = this.makeCars(rnd, kind), len = this.couple(cars)
      // stops of the train's low end: centred on platforms, and clear of the neighbour at section boundaries
      const sec = st.slice(cuts[k], cuts[k + 1] + 1), use = kind === 'passenger' ? sec : [sec[0], sec[sec.length - 1]]
      if (kind === 'passenger') for (const x of sec) x.served = true
      const stop = (x, i) => x.boundary && (i === 0 || i === use.length - 1) ? (i === 0 ? x.s + 8 : x.s - 8 - len) : Math.min(Math.max(x.s - len / 2, 30), L - 30 - len)
      const stops = [], dwell = [], at = []
      use.forEach((x, i) => {
        const s = stop(x, i)
        if (stops.length && s - stops[stops.length - 1] < 300 && i < use.length - 1) return
        const end = i === 0 || i === use.length - 1
        stops.push(s); at.push(x)
        dwell.push(end ? (kind === 'passenger' ? 110 : 160) + 80 * hash2i(x.seed, k, 4249) : 40 + 20 * hash2i(x.seed, 2, 4249))
      })
      const vmax = kind === 'passenger' ? 34 + 10 * rnd(4) : 22 + 8 * rnd(4)
      this.addTrain({ route: r, ri, cars, len, kind, stations: at }, stops, dwell, vmax, kind === 'passenger' ? 0.5 : 0.28, rnd(5))
    }
    this.stations.push(...st.filter(x => x.served || x.kind === 'terminus'))   // no halts where only freight runs
  }

  // freight spur: a shunter with a few wagons between the city yard and the industrial site, loading there
  addSpur(r, ri) {
    const rnd = i => hash2i(ri, i, 4244), cars = this.makeCars(rnd, 'shunt'), len = this.couple(cars)
    const stops = [40, r.length - 30 - len]
    if (stops[1] - stops[0] < 400) return
    this.addTrain({ route: r, ri, cars, len, kind: 'shunt', stations: [] }, stops, [80 + 60 * rnd(5), 220 + 200 * rnd(6)], 11 + 5 * rnd(4), 0.25, rnd(7))
  }

  addTrain(tr, stops, dwell, vmax, acc, u) {
    const s0 = stops[0], s1 = stops[stops.length - 1]
    tr.up = makeRun(stops.map(s => s - s0), dwell, vmax, acc)
    tr.down = makeRun(stops.map(s => s1 - s).reverse(), [...dwell].reverse(), vmax, acc)
    Object.assign(tr, { s0, s1, cycle: tr.up.T + tr.down.T, v: vmax })
    tr.phase = u * tr.cycle
    this.trains.push(tr)
    if (!this.byRoute.has(tr.ri)) this.byRoute.set(tr.ri, [])
    this.byRoute.get(tr.ri).push(tr)
  }

  // lay the cars out from the low-s end: offsets of their centres, seeds; returns the train length
  couple(cars) {
    let off = 0
    const seed = Math.floor(hash2i(cars.length, cars[0].col.r * 997, 4246) * 9973)
    cars.forEach((c, i) => { c.off = off + SHAPES[c.kind].len / 2; c.seed = seed * 64 + i; off += SHAPES[c.kind].len + GAP })
    return off - GAP
  }

  // cars from the low-s end to the high-s end; locomotives at both ends of main-line trains, the low one flipped
  makeCars(rnd, kind) {
    const cars = [], add = (k, col, flip = false) => cars.push({ kind: k, col, flip })
    cars.passenger = kind === 'passenger'
    if (cars.passenger) {
      const col = colour(LIVERY.passenger, rnd(1))
      add(KIND.LOCO, col, true)
      for (let i = 3 + Math.floor(rnd(2) * 5); i > 0; i--) add(KIND.COACH, col)
      add(KIND.LOCO, col)
      return cars
    }
    const col = colour(LIVERY.freight, rnd(1)), shunt = kind === 'shunt'
    if (!shunt) add(KIND.LOCO, col, true)
    let left = shunt ? 4 + Math.floor(rnd(2) * 9) : 15 + Math.floor(rnd(2) * 26), block = 0
    while (left > 0) {                          // wagons come in blocks of one type
      const bs = Math.floor(rnd(3) * 1e6) + block, k = shunt ? KIND.BOX + Math.floor(rnd(8) * 3) : KIND.BOX + Math.floor(hash2i(block, 1, bs) * 4)
      const m = shunt ? left : Math.min(left, 3 + Math.floor(hash2i(block, 2, bs) * 9)), bc = hash2i(block, 3, bs)
      for (let i = 0; i < m; i++) add(k, colour(LIVERY[k], hash2i(block, i, bs + 1) < 0.8 ? bc : hash2i(block, i, bs + 2)))
      left -= m; block++
    }
    add(KIND.LOCO, col)
    if (!shunt && rnd(7) < 0.5) add(KIND.LOCO, col)
    return cars
  }

  // where the train is: s of its low end, travel direction (+1 up the route, -1 down) and the direction it faces
  // (the next move's, so a train standing at the end of its run already faces back: the driver has changed ends)
  state(tr, time = this.now) {
    const c = mod(time + tr.phase, tr.cycle), up = c < tr.up.T, t = up ? c : c - tr.up.T, sign = up ? 1 : -1
    const low = up ? tr.s0 + runAt(tr.up, t) : tr.s1 - runAt(tr.down, t)
    return { low, sign, facing: t > (up ? tr.up : tr.down).moving ? -sign : sign }
  }

  // rail routes near the camera and the stretch [s0, s1] of each that lies near it (refreshed every km)
  refreshNear(dir) {
    if (this.nearDir && this.nearDir.dot(dir) > Math.cos(1000 / R)) return
    this.nearDir = dir.clone()
    const w = this.world, spans = new Map()
    for (const k of w.near([dir.x, dir.y, dir.z], RANGE + 2000).segs) {
      if (w.seg[k * 8 + 3] !== TYPE.RAIL) continue
      const ri = w.segRoute[k], i = w.segIndex[k], cum = w.routes[ri].cum, sp = spans.get(ri)
      if (sp) { sp[0] = Math.min(sp[0], cum[i]); sp[1] = Math.max(sp[1], cum[i + 1]) }
      else spans.set(ri, [cum[i], cum[i + 1]])
    }
    this.nearRoutes = [...spans].filter(([ri]) => this.byRoute.has(ri))
  }

  update(frame) {
    this.now = frame.time
    this.geo = frame.geo
    this.bridges = globalThis.terra?.civ?.parts.bridges
    for (const p of this.parts) p.n = 0
    const on = this.enabled && frame.geo.agl < 25000
    if (on) {
      this.refreshNear(frame.camDir)
      frame.camera.getWorldDirection(this.fwd)
      this.material.uniforms.uCamPos.value.copy(frame.camPos)
      for (const [ri, span] of this.nearRoutes) for (const tr of this.byRoute.get(ri)) this.place(tr, span, frame.camPos)
    }
    for (const p of this.parts) {
      p.mesh.visible = p.n > 0
      p.mesh.geometry.instanceCount = p.n
      if (p.n) for (const a of [p.A, p.B, p.C, p.M, p.H]) { a.clearUpdateRanges(); a.addUpdateRange(0, p.n * a.itemSize); a.needsUpdate = true }
    }
    const locos = this.parts[KIND.LOCO].n
    this.beam.visible = locos > 0 && frame.camDir.dot(frame.sun) < 0.15
    this.beam.geometry.instanceCount = locos
    this.stationView.update(frame, on ? this.nearRoutes : [], this.fwd)
  }

  // write the train's cars that are within range and not behind the camera
  place(tr, [a, b], cam) {
    const { low, facing } = this.state(tr), r = tr.route, w = this.world
    if (low > b + 50 || low + tr.len < a - 50) return
    const rel = s => { const d = w.routePoint(r, s).dir; return [d[0] * R - cam.x, d[1] * R - cam.y, d[2] * R - cam.z] }
    const deck = s => this.bridges?.routeDeck(tr.ri, s, 30) ?? null   // cars on or reaching onto a bridge ride its deck
    for (const car of tr.cars) {
      const s = low + car.off, part = this.parts[car.kind]
      if (s < a - 30 || s > b + 30 || part.n >= MAX_CARS) continue
      const m = rel(s), d2 = m[0] * m[0] + m[1] * m[1] + m[2] * m[2]
      if (d2 > RANGE * RANGE || m[0] * this.fwd.x + m[1] * this.fwd.y + m[2] * this.fwd.z < -60) continue
      const bo = SHAPES[car.kind].b * (car.flip ? -1 : 1), i = part.n++
      part.A.array.set(rel(s - bo), i * 3)
      part.B.array.set(rel(s + bo), i * 3)
      part.C.array.set([car.col.r, car.col.g, car.col.b], i * 3)
      part.M.array.set([(car.flip ? -1 : 1) === facing ? 1 : 0, car.seed], i * 2)
      const ha = deck(s - bo), hb = deck(s + bo)
      part.H.array.set(ha == null && hb == null ? [-1e9, -1e9] : [ha ?? hb, hb ?? ha], i * 2)
    }
  }

  // --- riding ---
  // rail-bed heights along a route, sampled once on the GPU (the same terrain() detail the cars use)
  profile(ri) {
    if (!this.profiles.has(ri)) {
      const r = this.world.routes[ri], n = Math.ceil(r.length / PSTEP) + 1, dirs = new Float32Array(n * 3), p = {}
      for (let i = 0; i < n; i++) dirs.set(this.world.routePoint(r, i * PSTEP).dir, i * 3)
      this.profiles.set(ri, p)
      this.ctx.sampler.sampleAsync(dirs, CIV_FLAT * 1.001, true).then(s => { p.h = Float32Array.from({ length: n }, (_, i) => Math.max(s[i * 8], 0)) })
    }
    return this.profiles.get(ri).h
  }
  heightAt(h, s, ri) {
    const z = ri == null ? null : this.bridges?.routeDeck(ri, s)
    if (z != null) return z
    const x = Math.min(Math.max(s / PSTEP, 0), h.length - 1), i = Math.min(Math.floor(x), h.length - 2)
    return h[i] + (h[i + 1] - h[i]) * (x - i)
  }

  // pose of the leading cab, all in planet-frame doubles: eye, unit axis along the facing direction (tilted with
  // the grade), local up, a focus point for the chase view, and which end leads; null until the profile is in
  pose(tr, t) {
    const h = this.profile(tr.ri)
    if (!h) return null
    const { low, facing: f } = this.state(tr, t), car = f > 0 ? tr.cars[tr.cars.length - 1] : tr.cars[0], sh = SHAPES[car.kind]
    const s = low + car.off, at = s => new THREE.Vector3(...this.world.routePoint(tr.route, s).dir).multiplyScalar(R)
    const a = at(s - f * sh.b), b = at(s + f * sh.b), mid = a.clone().add(b).multiplyScalar(0.5), up = mid.clone().normalize()
    const fwd = b.sub(a), xe = sh.len / 2 - 1.5         // eye inside the cab nose (its rear cap would block the view)
    const H = x => this.heightAt(h, x, tr.ri), grade = (H(s + f * (xe + 15)) - H(s + f * (xe - 15))) / 30
    fwd.addScaledVector(up, -fwd.dot(up)).normalize()
    const eye = mid.clone().addScaledVector(fwd, xe).addScaledVector(up, H(s + f * xe) + 3.4)
    const focus = mid.clone().addScaledVector(fwd, -12).addScaledVector(up, H(s - f * 12) + 2.5)
    return { eye, fwd: fwd.addScaledVector(up, grade).normalize(), up, focus, cab: f }
  }

  // HUD button: fly to the train nearest the camera (main-line passenger trains preferred) and ride it; again to leave
  ride(geo) {
    if (geo.follow) { geo.unfollow(); return null }
    const u = geo.up, dir = [u.x, u.y, u.z]
    let best = null
    for (const tr of this.trains) {
      const { low, sign } = this.state(tr), p = this.world.routePoint(tr.route, sign > 0 ? low + tr.len : low).dir
      const d = Math.acos(Math.min(1, p[0] * dir[0] + p[1] * dir[1] + p[2] * dir[2])) * R
      const score = (d + 2000) * (tr.kind === 'passenger' ? 1 : tr.kind === 'freight' ? 1.6 : 4)
      if (!best || score < best.score) best = { tr, score }
    }
    const tr = best.tr
    this.riding = tr
    this.profile(tr.ri)
    // aim the flight at where the train will be when it lands: behind it and above, looking along the track
    const aim = t => {
      const { low, facing } = this.state(tr, t), s = facing > 0 ? low + tr.len : low, { dir: p, tan } = this.world.routePoint(tr.route, s)
      const f = tan.map(v => v * facing), c = p.map((v, i) => v - f[i] * 260 / R), l = Math.hypot(...c), cd = c.map(v => v / l)
      const east = [cd[2], 0, -cd[0]].map(v => v / Math.hypot(cd[2], cd[0])), north = [cd[1] * east[2], cd[2] * east[0] - cd[0] * east[2], -cd[1] * east[0]]
      return { lat: Math.asin(cd[1]), lon: Math.atan2(cd[0], cd[2]), agl: 120, pitch: -0.35, heading: Math.atan2(f[0] * east[0] + f[2] * east[2], f.reduce((s, v, i) => s + v * north[i], 0)) }
    }
    geo.travel(aim(this.now + 3))
    geo.travel(aim(this.now + geo.flight.dur))
    geo.follow = { pose: dt => this.riding === tr ? this.pose(tr, this.now + dt) : null, end: () => { if (this.riding === tr) this.riding = null } }
    geo.followView = 'chase'
    geo.fYaw = 0.5
    return this.rideInfo()
  }

  // one line for the HUD while riding
  rideInfo() {
    const tr = this.riding
    if (!tr) return null
    const a = this.state(tr), b = this.state(tr, this.now + 1), kmh = Math.round(Math.abs(b.low - a.low) * 3.6), f = a.facing
    const head = f > 0 ? a.low + tr.len : a.low
    const next = tr.stations.filter(x => (x.s - head) * f > -tr.len - 20).sort((x, y) => (x.s - y.s) * f)[0]
    const what = tr.kind === 'shunt' ? `Freight shunter · ${this.world.cities[tr.route.from].name} yard` : tr.kind === 'freight' ? 'Freight' : 'Passenger'
    const line = tr.stations.length ? ` · ${tr.stations[0].name} – ${tr.stations[tr.stations.length - 1].name}` : ''
    const stop = next && tr.kind === 'passenger' ? (kmh < 1 && Math.abs(next.s - (a.low + tr.len / 2)) < next.len ? ` · at ${next.name}` : ` · next ${next.name}`) : ''
    return `${what}${line} · ${kmh} km/h${stop}`
  }

  // --- test helpers ---
  // head of the train nearest to dir: { lat, lon, distance (m), train }
  nearest(dir) {
    let best = null
    for (const tr of this.trains) {
      const { low, sign } = this.state(tr), p = this.world.routePoint(tr.route, sign > 0 ? low + tr.len : low).dir
      const d = Math.acos(Math.min(1, p[0] * dir[0] + p[1] * dir[1] + p[2] * dir[2])) * R
      if (!best || d < best.distance) best = { lat: Math.asin(p[1]), lon: Math.atan2(p[0], p[2]), distance: d, train: tr }
    }
    return best
  }
  // closest rail route vertex (with a train) to dir, however far
  nearestRail(dir, ok) {
    let best = null, bd = -2
    for (const tr of this.trains.filter(ok)) {
      const P = tr.route.pts
      for (let i = 0; i < P.length; i += 3) {
        const d = P[i] * dir[0] + P[i + 1] * dir[1] + P[i + 2] * dir[2]
        if (d > bd) { bd = d; best = [P[i], P[i + 1], P[i + 2]] }
      }
    }
    return best
  }
  inTown(r, s) {
    const c = this.world.nearestCity(this.world.routePoint(r, s).dir, 40000)
    return c && c.distance < c.city.radius * 1.3 + 400
  }
  // time a main-line train nearest to dir so its head passes a point `sec` seconds from now on its upward run,
  // and fly the camera `side` metres to the right of the track at `agl`, looking at it
  stage(dir, { sec = 4, side = 150, agl = 40, passenger } = {}) {
    const ok = t => t.kind !== 'shunt' && (passenger === undefined || t.kind === (passenger ? 'passenger' : 'freight'))
    const hit = this.world.nearestSegment(this.nearestRail(dir, ok), 3000, t => t === TYPE.RAIL)
    const list = hit && this.byRoute.get(hit.route)?.filter(ok)
    if (!list?.length) return null
    const tr = list.reduce((b, t) => Math.abs((t.s0 + t.s1) / 2 - hit.s) < Math.abs((b.s0 + b.s1) / 2 - hit.s) ? t : b)
    const clamp = s => Math.min(Math.max(s, tr.s0 + tr.len + 600), tr.s1 + tr.len - 600)
    let s = clamp(hit.s)
    for (let k = 0; k < 50 && this.inTown(tr.route, s); k++) s = clamp(s + (s < (tr.s0 + tr.s1) / 2 + tr.len ? 1500 : -1500))   // out into the country
    tr.phase = runTime(tr.up, s - tr.len - tr.s0) - sec - this.now
    this.view(tr.route, s, { side, agl })
    return { route: hit.route, s: Math.round(s), train: this.trains.indexOf(tr), cars: tr.cars.length, kind: tr.kind }
  }
  // put the camera `side` m right of route point s and `back` m behind it along the route, at agl, looking at it
  view(r, s, { side = 150, back = 0, agl = 40, pitch = -0.12 } = {}) {
    const { dir: p, tan } = this.world.routePoint(r, s)
    const right = [tan[1] * p[2] - tan[2] * p[1], tan[2] * p[0] - tan[0] * p[2], tan[0] * p[1] - tan[1] * p[0]]
    const c = p.map((v, i) => v + (right[i] * side - tan[i] * back) / R), l = Math.hypot(...c), cd = c.map(v => v / l)
    const east = [cd[2], 0, -cd[0]].map(v => v / Math.hypot(cd[2], cd[0])), north = [cd[1] * east[2], cd[2] * east[0] - cd[0] * east[2], -cd[1] * east[0]]
    const look = p.map((v, i) => v - cd[i]), heading = Math.atan2(look[0] * east[0] + look[2] * east[2], look.reduce((s, v, i) => s + v * north[i], 0))
    this.geo?.unfollow()
    this.geo?.flyTo({ lat: Math.asin(cd[1]), lon: Math.atan2(cd[0], cd[2]), agl, pitch, heading })
    if (this.geo) this.geo.agl = agl
  }
  // time the train calling at station i to pull in `sec` s from now (on its up run) and look at the platform
  arrive(i, sec = 5, opts) {
    const st = this.stations[i], tr = this.byRoute.get(st.ri)?.find(t => t.stations.includes(st))
    const k = tr?.stations.indexOf(st), g = tr?.up.legs[k - 1]
    if (g) tr.phase = g.t0 + g.T - sec - this.now
    this.view(st.route, st.s, { side: st.side * 60, back: 40, agl: 12, ...opts })
    return { kind: st.kind, name: st.name, big: st.big, train: tr ? `${tr.kind} ${tr.cars.length} cars, stop ${k}` : 'none' }
  }
}
