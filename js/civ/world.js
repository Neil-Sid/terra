// Procedural civilisation. Settlements go where the land suits them (flat, temperate, wet, near the
// sea); highways, country roads and railways join them along routes chosen by a small dynamic
// program over the real terrain (avoiding water and climbs). Everything is deterministic and built
// once at start-up, then packed into lookup textures for the shaders (see civ.glsl.js).
import * as THREE from 'three'
import { R } from '../config.js'
import { TYPE, TIER, BLOCK, CITY_BIT, RIVER_BIT, RIVER_REACH, riverMeander, GRID_W, GRID_H, ROW, CITY_REACH, HYDRO, hash2i, industrySites, cityShape, dirToLocal } from './spec.js'
import { placeName } from './names.js'
import { buildHydro } from './hydro.js'
import { RMAP, rmapAlpha } from '../glsl/water.glsl.js'

const CANDIDATES = 120000
const MARGIN = 160                 // metres around items when filing them into lookup cells
const SPLIT_AT = 28                // cells with more items get a 4x4 subgrid
const MAX_ITEMS = 160              // per cell, matching the shader loop

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t) }
const clamp01 = x => Math.min(1, Math.max(0, x))
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const norm = a => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l] }
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const arc = (a, b) => Math.acos(Math.min(1, Math.max(-1, dot(a, b)))) * R
const lerpDir = (a, b, t) => norm([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t])
export const latLon = d => [Math.asin(Math.max(-1, Math.min(1, d[1]))), Math.atan2(d[0], d[2])]
const cellXY = d => { const [la, lo] = latLon(d); return [(lo / (2 * Math.PI) + 0.5) * GRID_W, (la / Math.PI + 0.5) * GRID_H] }

// per type: DP stage length, lateral spacing and half-count, climb penalty (m of detour per m climbed), longest link
const ROUTE = {
  [TYPE.HIGHWAY]: { step: 6000, lat: 2500, J: 3, climb: 45, maxLen: 700e3 },
  [TYPE.ROAD]: { step: 4000, lat: 1500, J: 2, climb: 20, maxLen: 140e3 },
  [TYPE.RAIL]: { step: 6000, lat: 3000, J: 3, climb: 140, maxLen: 500e3 },
}
const BRIDGE = 60000               // cost of one DP stage over water (m of detour); routes may bridge 2 stages
const MAX_WATER = 2

// stage(name) is awaited between steps so the loading overlay can paint; it doesn't change the result
export async function buildWorld(sampler, stage = async () => {}) {
  const t0 = performance.now(), lap = label => console.info(`civ: ${label} at ${Math.round(performance.now() - t0)} ms`)
  const hydro = await buildHydro(sampler, stage)
  lap('rivers and lakes')
  await stage('Founding towns')
  const cities = placeSettlements(sampler)
  lap(`${cities.length} settlements`)
  await stage('Routing roads and railways')
  const routes = buildRoutes(sampler, cities)
  lap(`${routes.length} routes`)
  await stage('Mapping the network')
  const world = new World(cities, routes, hydro)
  lap('index and textures')
  world.buildTime = Math.round(performance.now() - t0)
  return world
}

function placeSettlements(sampler) {
  const dirs = new Float32Array(CANDIDATES * 3), ga = Math.PI * (3 - Math.sqrt(5))
  for (let i = 0; i < CANDIDATES; i++) {
    const y = 1 - 2 * (i + 0.5) / CANDIDATES, r = Math.sqrt(1 - y * y), th = ga * i
    const j = k => (hash2i(i, k, 1) - 0.5) * 0.008          // ~25 km jitter so the pattern isn't regular
    dirs.set(norm([Math.cos(th) * r + j(1), y + j(2), Math.sin(th) * r + j(3)]), i * 3)
  }
  const s = sampler.sample(dirs, R / 3000)
  const found = []
  for (let i = 0; i < CANDIDATES; i++) {
    const [h, land, , , slope, T, M] = s.subarray(i * 8, i * 8 + 8)
    if (h < 3 || h > 2400 || land < 0.003 || slope > 0.08 || T < -3 || T > 32) continue
    const coast = 1 - smooth(0, 0.07, land)
    const score = (0.3 + 0.7 * coast) * clamp01(1 - Math.abs(T - 15) / 17) * (0.25 + 0.75 * smooth(0.1, 0.45, M))
      * (1 - slope / 0.08) * (1 - h / 3000)
    if (hash2i(i, 1, 3) > score * 1.2) continue
    const pop = Math.round(1500 * Math.pow(1 / (1 - 0.9995 * hash2i(i, 2, 5)), 1.35) * (0.6 + score))
    found.push({ i, h, pop, dir: [dirs[i * 3], dirs[i * 3 + 1], dirs[i * 3 + 2]] })
  }
  // biggest first; smaller places don't sit on top of bigger ones
  found.sort((a, b) => b.pop - a.pop)
  const cities = [], grid = new Map()
  const key = (la, lo) => `${Math.floor(la / 0.02)},${Math.floor(lo / 0.02)}`
  for (const f of found) {
    const radius = Math.min(30000, 40 * Math.pow(f.pop, 0.45))
    const [lat, lon] = latLon(f.dir)
    let clash = false
    for (let a = -2; a <= 2 && !clash; a++) for (let b = -2; b <= 2 && !clash; b++) {
      for (const c of grid.get(key(lat + a * 0.02, lon + b * 0.02)) ?? []) {
        if (arc(c.dir, f.dir) < (c.radius + radius) * 0.9) { clash = true; break }
      }
    }
    if (clash) continue
    const tier = f.pop > 1.5e6 ? TIER.METRO : f.pop > 1.5e5 ? TIER.CITY : f.pop > 1.5e4 ? TIER.TOWN : TIER.VILLAGE
    const seed = (f.i * 7 + 13) | 0
    const city = {
      id: cities.length, dir: f.dir, lat, lon, h: f.h, pop: f.pop, tier, radius, seed,
      block: BLOCK[tier], angle: hash2i(f.i, 4, 9) * Math.PI / 2, name: placeName(seed),
    }
    cities.push(city)
    const k = key(lat, lon)
    if (!grid.has(k)) grid.set(k, [])
    grid.get(k).push(city)
  }
  return cities
}

// neighbours within maxLen via lat/lon buckets
function neighbours(nodes, maxLen) {
  const B = 2 * Math.PI / 180, buckets = new Map(), k = (a, b) => a * 4096 + b
  nodes.forEach((n, i) => {
    const kk = k(Math.floor(n.lat / B), Math.floor(n.lon / B))
    if (!buckets.has(kk)) buckets.set(kk, [])
    buckets.get(kk).push(i)
  })
  return nodes.map(n => {
    const out = [], dLat = Math.ceil(maxLen / R / B), dLon = Math.ceil(maxLen / R / Math.max(Math.cos(n.lat), 0.1) / B)
    const la0 = Math.floor(n.lat / B), lo0 = Math.floor(n.lon / B), nLon = Math.round(2 * Math.PI / B)
    for (let a = -dLat; a <= dLat; a++) for (let b = -dLon; b <= dLon; b++) {
      let lo = lo0 + b
      if (lo < -nLon / 2) lo += nLon; else if (lo >= nLon / 2) lo -= nLon
      for (const j of buckets.get(k(la0 + a, lo)) ?? []) {
        if (nodes[j] === n) continue
        const d = arc(n.dir, nodes[j].dir)
        if (d <= maxLen) out.push([j, d])
      }
    }
    return out
  })
}

// relative neighbourhood graph: keep a-b unless some c is closer to both
function rngEdges(nodes, maxLen) {
  const nb = neighbours(nodes, maxLen), edges = []
  nb.forEach((list, i) => {
    for (const [j, d] of list) {
      if (j <= i) continue
      const blocked = list.some(([k, dik]) => k !== j && dik < d && arc(nodes[j].dir, nodes[k].dir) < d)
      if (!blocked) edges.push([nodes[i], nodes[j], d])
    }
  })
  return edges
}

function buildRoutes(sampler, cities) {
  // highways join cities and large towns; railways join cities and most bigger towns
  const hubs = cities.filter(c => c.tier >= TIER.CITY || (c.tier === TIER.TOWN && c.pop >= 40000))
  const stations = cities.filter(c => c.tier >= TIER.CITY || (c.tier === TIER.TOWN && c.pop >= 25000))
  const plans = []
  const hw = rngEdges(hubs, ROUTE[TYPE.HIGHWAY].maxLen)
  const hwPairs = new Set(hw.map(([a, b]) => `${a.id}-${b.id}`))
  for (const [a, b] of hw) plans.push(planEdge(a, b, TYPE.HIGHWAY))
  for (const [a, b] of rngEdges(cities, ROUTE[TYPE.ROAD].maxLen)) {
    if (!hwPairs.has(`${a.id}-${b.id}`) && !hwPairs.has(`${b.id}-${a.id}`)) plans.push(planEdge(a, b, TYPE.ROAD))
  }
  for (const [a, b] of rngEdges(stations, ROUTE[TYPE.RAIL].maxLen)) {
    if (Math.max(a.tier, b.tier) >= TIER.CITY || hash2i(a.id, b.id, 77) < 0.6) plans.push(planEdge(a, b, TYPE.RAIL))
  }
  // one GPU pass for every candidate point of every link
  const total = plans.reduce((n, p) => n + p.pts.length / 3, 0)
  const dirs = new Float32Array(total * 3)
  let o = 0
  for (const p of plans) { p.offset = o; dirs.set(p.pts, o * 3); o += p.pts.length / 3 }
  const s = sampler.sample(dirs, R / 2000)
  const routes = []
  for (const p of plans) {
    const pts = solveEdge(p, s)
    if (pts) routes.push(makeRoute(p.type, chaikin(pts), p.a, p.b))
  }
  // freight spurs: from each rail-served city centre out to its industrial districts
  const served = new Set(routes.filter(r => r.type === TYPE.RAIL).flatMap(r => [r.from, r.to]))
  for (const c of cities) {
    if (!served.has(c.id)) continue
    for (const site of industrySites(c)) {
      const n = Math.max(2, Math.ceil(arc(c.dir, site.dir) / 400))
      const r = makeRoute(TYPE.RAIL, Array.from({ length: n + 1 }, (_, i) => lerpDir(c.dir, site.dir, i / n)), c, c)
      r.spur = true
      routes.push(r)
    }
  }
  return routes
}

// candidate grid for one link: K-1 stages across the great circle, 2J+1 lateral offsets each
function planEdge(a, b, type) {
  const spec = ROUTE[type], D = arc(a.dir, b.dir), K = Math.max(2, Math.ceil(D / spec.step)), W = 2 * spec.J + 1
  const side = norm(cross(a.dir, b.dir)), pts = new Float32Array((K - 1) * W * 3)
  for (let k = 1; k < K; k++) {
    const base = lerpDir(a.dir, b.dir, k / K), taper = Math.min(1, 3 * Math.min(k, K - k) / K)
    for (let j = -spec.J; j <= spec.J; j++) {
      const off = j * spec.lat * taper / R
      pts.set(norm([base[0] + side[0] * off, base[1] + side[1] * off, base[2] + side[2] * off]), ((k - 1) * W + j + spec.J) * 3)
    }
  }
  return { a, b, type, spec, K, W, pts }
}

// cheapest path through the candidate grid (distance + climbing, short bridges allowed); null if blocked
function solveEdge(p, s) {
  const { K, W, spec, pts, a, b } = p
  const dirAt = (k, j) => [pts[((k - 1) * W + j) * 3], pts[((k - 1) * W + j) * 3 + 1], pts[((k - 1) * W + j) * 3 + 2]]
  const hAt = (k, j) => s[(p.offset + (k - 1) * W + j) * 8]
  const step = (d0, h0, d1, h1) => arc(d0, d1) + spec.climb * Math.abs(Math.max(h1, 0) - Math.max(h0, 0)) + (h1 < 1 ? BRIDGE : 0)
  let cost = Array.from({ length: W }, (_, j) => step(a.dir, a.h, dirAt(1, j), hAt(1, j)))
  const back = []
  for (let k = 2; k < K; k++) {
    const next = new Array(W).fill(Infinity), from = new Array(W).fill(0)
    for (let j = 0; j < W; j++) {
      const dj = dirAt(k, j), hj = hAt(k, j)
      for (let i = Math.max(0, j - 2); i <= Math.min(W - 1, j + 2); i++) {
        const c = cost[i] + step(dirAt(k - 1, i), hAt(k - 1, i), dj, hj)
        if (c < next[j]) { next[j] = c; from[j] = i }
      }
    }
    back.push(from)
    cost = next
  }
  let best = 0, bestC = Infinity
  for (let j = 0; j < W; j++) {
    const c = cost[j] + step(dirAt(K - 1, j), hAt(K - 1, j), b.dir, b.h)
    if (c < bestC) { bestC = c; best = j }
  }
  const path = [b.dir]
  let wet = 0
  for (let k = K - 1, j = best; k >= 1; k--) {
    path.push(dirAt(k, j))
    if (hAt(k, j) < 1) wet++
    if (k >= 2) j = back[k - 2][j]
  }
  if (wet > MAX_WATER) return null
  path.push(a.dir)
  return path.reverse()
}

// one round of corner cutting so routes curve instead of zig-zagging
function chaikin(pts) {
  const out = [pts[0]]
  for (let i = 0; i < pts.length - 1; i++) {
    out.push(lerpDir(pts[i], pts[i + 1], 0.25), lerpDir(pts[i], pts[i + 1], 0.75))
  }
  out.push(pts[pts.length - 1])
  return out
}

function makeRoute(type, pts, a, b) {
  const n = pts.length, P = new Float64Array(n * 3), cum = new Float64Array(n)
  pts.forEach((d, i) => { P.set(d, i * 3); if (i) cum[i] = cum[i - 1] + arc(pts[i - 1], d) })
  return { type, pts: P, cum, length: cum[n - 1], from: a.id, to: b.id }
}

export class World {
  constructor(cities, routes, hydro) {
    this.cities = cities
    this.routes = routes
    this.hydro = hydro
    this.packSegments()
    this.packRivers()
    this.buildIndex()
    this.textures = this.makeTextures()
  }

  // flat segment table: A, type, B, distance along the route at A
  packSegments() {
    const n = this.routes.reduce((k, r) => k + r.cum.length - 1, 0)
    this.seg = new Float32Array(n * 8)
    this.segRoute = new Uint32Array(n)
    this.segIndex = new Uint32Array(n)
    let k = 0
    this.routes.forEach((r, ri) => {
      r.firstSeg = k
      for (let i = 0; i < r.cum.length - 1; i++, k++) {
        this.seg.set([r.pts[i * 3], r.pts[i * 3 + 1], r.pts[i * 3 + 2], r.type, r.pts[i * 3 + 3], r.pts[i * 3 + 4], r.pts[i * 3 + 5], r.cum[i]], k * 8)
        this.segRoute[k] = ri; this.segIndex[k] = i
      }
    })
    this.segCount = n
  }

  // river segment table, 12 floats each: A, half width at A, B, half width at B, water level at A and B, distance from the source at A
  packRivers() {
    const rs = this.hydro.rivers, n = rs.reduce((k, r) => k + r.pts.length - 1, 0)
    this.riv = new Float32Array(n * 12)
    let k = 0
    for (const r of rs) {
      const cum = [0]
      for (let i = 1; i < r.pts.length; i++) cum.push(cum[i - 1] + arc(r.pts[i - 1].d, r.pts[i].d))
      r.length = cum[cum.length - 1]
      r.firstSeg = k
      for (let i = 0; i < r.pts.length - 1; i++, k++) {
        const a = r.pts[i], b = r.pts[i + 1]
        this.riv.set([...a.d, a.hw, ...b.d, b.hw, a.level, b.level, cum[i], r.length], k * 12)
      }
    }
    this.rivCount = n
  }

  // lookup cells: Map cell -> [ids]; boxes in cell coordinates for the subgrid split
  buildIndex() {
    this.cells = new Map()
    this.boxes = new Map()
    const file = (id, x0, y0, x1, y1) => {
      this.boxes.set(id, [x0, y0, x1, y1])
      for (let y = Math.max(0, Math.floor(y0)); y <= Math.min(GRID_H - 1, Math.floor(y1)); y++) {
        for (let x = Math.floor(x0); x <= Math.floor(x1); x++) {
          const c = y * GRID_W + ((x % GRID_W) + GRID_W) % GRID_W
          if (!this.cells.has(c)) this.cells.set(c, [])
          this.cells.get(c).push(id)
        }
      }
    }
    const pad = (lat, m) => [m / R * GRID_W / (2 * Math.PI) / Math.max(Math.cos(lat), 0.02), m / R * GRID_H / Math.PI]
    this.cities.forEach((c, i) => {
      const [x, y] = cellXY(c.dir), [px, py] = pad(c.lat, c.radius * CITY_REACH + MARGIN)
      file((CITY_BIT | i) >>> 0, x - px, y - py, x + px, y + py)
    })
    for (let k = 0; k < this.segCount; k++) {
      const s = this.seg, A = [s[k * 8], s[k * 8 + 1], s[k * 8 + 2]], B = [s[k * 8 + 4], s[k * 8 + 5], s[k * 8 + 6]]
      let [xa, ya] = cellXY(A), [xb, yb] = cellXY(B)
      if (xb - xa > GRID_W / 2) xb -= GRID_W; else if (xa - xb > GRID_W / 2) xb += GRID_W
      const [px, py] = pad(latLon(A)[0], MARGIN)
      file(k, Math.min(xa, xb) - px, Math.min(ya, yb) - py, Math.max(xa, xb) + px, Math.max(ya, yb) + py)
    }
    for (let k = 0; k < this.rivCount; k++) {
      const s = this.riv, o = k * 12, A = [s[o], s[o + 1], s[o + 2]], B = [s[o + 4], s[o + 5], s[o + 6]]
      let [xa, ya] = cellXY(A), [xb, yb] = cellXY(B)
      if (xb - xa > GRID_W / 2) xb -= GRID_W; else if (xa - xb > GRID_W / 2) xb += GRID_W
      const [px, py] = pad(latLon(A)[0], RIVER_REACH(Math.max(s[o + 3], s[o + 7])))
      file((RIVER_BIT | k) >>> 0, Math.min(xa, xb) - px, Math.min(ya, yb) - py, Math.max(xa, xb) + px, Math.max(ya, yb) + py)
    }
  }

  makeTextures() {
    const header = new Uint32Array(GRID_W * GRID_H), items = []
    // rivers first (the far water lines stop at the first non-river item), then cities, so a truncated list keeps them
    const rank = id => id >>> 31 ? 1 : id & RIVER_BIT ? 2 : 0
    const order = ids => ids.sort((a, b) => rank(b) - rank(a)).slice(0, MAX_ITEMS)
    const overlaps = (id, cx, cy, sx, sy) => {
      const [x0, y0, x1, y1] = this.boxes.get(id), X0 = cx + sx / 4, Y0 = cy + sy / 4
      if (y1 < Y0 || y0 > Y0 + 0.25) return false
      for (const w of [0, GRID_W, -GRID_W]) if (x1 + w >= X0 && x0 + w <= X0 + 0.25) return true
      return false
    }
    for (const [c, ids] of this.cells) {
      if (ids.length <= SPLIT_AT) {
        header[c] = ((items.length << 8) | Math.min(255, ids.length)) >>> 0
        items.push(...order(ids))
        continue
      }
      const base = items.length, cx = c % GRID_W, cy = Math.floor(c / GRID_W)
      header[c] = (0x80000000 | (base << 8)) >>> 0
      for (let i = 0; i < 16; i++) items.push(0)
      for (let sy = 0; sy < 4; sy++) for (let sx = 0; sx < 4; sx++) {
        const sub = order(ids.filter(id => overlaps(id, cx, cy, sx, sy)))
        items[base + sy * 4 + sx] = ((items.length << 8) | sub.length) >>> 0
        items.push(...sub)
      }
    }
    const rows = n => Math.max(1, Math.ceil(n / ROW))
    const itemData = new Uint32Array(rows(items.length) * ROW)
    itemData.set(items)
    const segData = new Float32Array(rows(this.segCount * 2) * ROW * 4)
    segData.set(this.seg)
    const cityData = new Float32Array(rows(this.cities.length * 2) * ROW * 4)
    this.cities.forEach((c, i) => cityData.set([...c.dir, c.radius, c.angle, c.block, c.seed, c.tier], i * 8))
    this.itemCount = items.length
    const rivData = new Float32Array(rows(this.rivCount * 3) * ROW * 4)
    rivData.set(this.riv)
    const lake = new THREE.DataTexture(this.hydro.lakeMap, this.hydro.NX, this.hydro.NY, THREE.RedFormat, THREE.FloatType)
    lake.minFilter = lake.magFilter = THREE.NearestFilter
    lake.unpackAlignment = 1
    lake.needsUpdate = true
    const tex = {
      river: floatTex(rivData, ROW, rows(this.rivCount * 3)), lake,
      grid: uintTex(header, GRID_W, GRID_H), items: uintTex(itemData, ROW, rows(items.length)),
      seg: floatTex(segData, ROW, rows(this.segCount * 2)), city: floatTex(cityData, ROW, rows(this.cities.length * 2)),
      urban: this.urbanMap(), farm: this.farmMap(), rivers: this.riverMap(),
    }
    // main.js wires the civ textures it knows; the water ones go straight into the shared uniforms
    if (U) { U.uCivRiver.value = tex.river; U.uLakeMap.value = tex.lake; U.uRiverMap.value = tex.rivers }
    return tex
  }

  // coarse global map for distant views: r = urban, g = highway, b = rail, a = road
  urbanMap() {
    const W = 4096, H = 2048, data = new Uint8Array(W * H * 4)
    const put = (lat, lon, ch, v) => {
      const x = Math.floor((lon / (2 * Math.PI) + 0.5) * W), y = Math.floor((lat / Math.PI + 0.5) * H)
      if (y < 0 || y >= H) return
      const o = (y * W + ((x % W) + W) % W) * 4 + ch
      data[o] = Math.max(data[o], v)
    }
    for (const c of this.cities) {
      // full inside the city's lobed outline, then a suburban glow fading over about a radius outside it;
      // brighter for bigger places; places smaller than a texel light only part of it
      const rr = c.radius * 3.2 / R, n = Math.ceil(rr / (Math.PI / H)) + 1
      const bright = clamp01(0.3 + 0.18 * Math.log10(c.pop / 1000)), cover = Math.min(1, (c.radius * 2.2 / 9800) ** 2)
      for (let a = -n; a <= n; a++) for (let b = -n; b <= n; b++) {
        const la = c.lat + a * Math.PI / H, lo = c.lon + b * 2 * Math.PI / W
        const [x, y] = dirToLocal(c, [Math.cos(la) * Math.sin(lo), Math.sin(la), Math.cos(la) * Math.cos(lo)])
        const shape = a === 0 && b === 0 ? Math.max(cityShape(c, x, y), 0.3) : cityShape(c, x, y)
        const glow = shape > 0 ? 0.55 + 0.45 * clamp01(shape * 2.5) : 0.55 * Math.exp(shape * 3.5)
        const v = bright * glow * (a === 0 && b === 0 ? Math.max(cover, 0.08) : cover)
        if (v > 0.004) put(la, lo, 0, v * 255)
      }
    }
    // roads stay uniform here (they also darken the ground by day); the shader lights them mostly near towns
    const chan = [1, 3, 2]
    for (const r of this.routes) {
      for (let s = 0; s < r.length; s += 2500) {
        const [la, lo] = latLon(this.routePoint(r, s).dir)
        put(la, lo, chan[r.type], 255)
      }
    }
    const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.UnsignedByteType)
    tex.wrapS = THREE.RepeatWrapping
    tex.minFilter = THREE.LinearMipmapLinearFilter
    tex.magFilter = THREE.LinearFilter
    tex.generateMipmaps = true
    tex.needsUpdate = true
    return tex
  }

  // coarse global river map for distant views (the exact lookup only lists a river within ~2 km of it).
  // Per texel, from the nearest river segment in a local east/north metre frame: r = orbit opacity x a tent
  // RMAP.tent wide (mipmapped, so the shader reads line density), g = signed distance to the centreline
  // (bilinear at LOD 0 puts the line between texels exactly), b = log2(Q/100)/10 of that river.
  riverMap() {
    const t0 = performance.now(), { W, H, tent, range } = RMAP, TX = Math.PI / H, N = W * H
    const data = new Uint8Array(N * 4), best = new Uint16Array(N).fill(65535)       // best: metres to the nearest line
    new Uint32Array(data.buffer).fill(0x0000ff00)                                    // r 0, g 1 = far from any river, b 0
    for (const r of this.hydro.rivers) {
      const P = r.pts
      for (let i = 0; i < P.length - 1; i++) {
        const lqa = Math.log10(P[i].Q), lqb = Math.log10(P[i + 1].Q), aa = rmapAlpha(lqa), ab = rmapAlpha(lqb)
        if (aa <= 0 && ab <= 0) continue
        const [la0, lo0] = latLon(P[i].d), [la1, l1] = latLon(P[i + 1].d)
        const lo1 = l1 - lo0 > Math.PI ? l1 - 2 * Math.PI : lo0 - l1 > Math.PI ? l1 + 2 * Math.PI : l1
        const cl = Math.max(Math.cos((la0 + la1) / 2), 0.02), bx = (lo1 - lo0) * cl * R, by = (la1 - la0) * R, L2 = Math.max(bx * bx + by * by, 1)
        const pad = range / R, y0 = Math.max(0, Math.floor((Math.min(la0, la1) - pad) / TX + H / 2)), y1 = Math.min(H - 1, Math.floor((Math.max(la0, la1) + pad) / TX + H / 2))
        const x0 = Math.floor((Math.min(lo0, lo1) - pad / cl) / TX + W / 2), x1 = Math.floor((Math.max(lo0, lo1) + pad / cl) / TX + W / 2)
        for (let y = y0; y <= y1; y++) {
          const py = ((y + 0.5 - H / 2) * TX - la0) * R
          for (let x = x0; x <= x1; x++) {
            const px = ((x + 0.5 - W / 2) * TX - lo0) * cl * R
            const t = (px * bx + py * by) / L2
            if ((t < 0 && i === 0) || (t > 1 && i === P.length - 2)) continue            // flat ends: no sign flip around the tips
            const u = Math.min(1, Math.max(0, t)), ex = px - bx * u, ey = py - by * u, d2 = ex * ex + ey * ey
            if (d2 >= range * range) continue
            const d = Math.sqrt(d2), k = y * W + ((x % W) + W) % W, o = k * 4
            if (d < tent) data[o] = Math.max(data[o], 255 * (aa + (ab - aa) * u) * (1 - d / tent) + 0.5)
            if (d < best[k]) {
              best[k] = d
              data[o + 1] = 127.5 + 127.5 * (bx * py - by * px < 0 ? d : -d) / range + 0.5
              data[o + 2] = 255 * clamp01((lqa + (lqb - lqa) * u - 2) / 3.0103) + 0.5          // log2(Q/100)/10
            }
          }
        }
      }
    }
    const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.UnsignedByteType)
    tex.wrapS = THREE.RepeatWrapping
    tex.minFilter = THREE.LinearMipmapLinearFilter
    tex.magFilter = THREE.LinearFilter
    tex.generateMipmaps = true
    tex.needsUpdate = true
    console.info(`civ: river map ${Math.round(performance.now() - t0)} ms`)
    return tex
  }

  // farming intensity around settlements (fields fade out ~5 km + 3 radii from each place)
  // Regional settlement density (not a disc per town): every place adds a broad Gaussian (~110 km) weighted by
  // sqrt(population), and the sum saturates, so whole populated regions read as farming country. The terrain
  // shader then decides where fields actually go (climate, slope, soil noise).
  farmMap() {
    const W = 1024, H = 512, dens = new Float32Array(W * H), data = new Uint8Array(W * H)
    const sigma = 110e3 / R, reach = 3 * sigma, n = Math.ceil(reach / (Math.PI / H)) + 1
    for (const c of this.cities) {
      const w = Math.sqrt(c.pop) / 300
      for (let a = -n; a <= n; a++) {
        const la = c.lat + a * Math.PI / H, y = Math.floor((la / Math.PI + 0.5) * H)
        if (y < 0 || y >= H) continue
        const m = Math.ceil(n / Math.max(Math.cos(la), 0.05))
        for (let b = -m; b <= m; b++) {
          const lo = c.lon + b * 2 * Math.PI / W, d2 = ((la - c.lat) ** 2 + ((lo - c.lon) * Math.cos(c.lat)) ** 2) / (sigma * sigma)
          if (d2 >= 9) continue
          const x = ((Math.floor((lo / (2 * Math.PI) + 0.5) * W) % W) + W) % W
          dens[y * W + x] += w * Math.exp(-0.5 * d2)
        }
      }
    }
    for (let i = 0; i < W * H; i++) data[i] = 255 * (1 - Math.exp(-dens[i] / 1.5))
    const tex = new THREE.DataTexture(data, W, H, THREE.RedFormat, THREE.UnsignedByteType)
    tex.wrapS = THREE.RepeatWrapping
    tex.minFilter = THREE.LinearMipmapLinearFilter
    tex.magFilter = THREE.LinearFilter
    tex.generateMipmaps = true
    tex.unpackAlignment = 1
    tex.needsUpdate = true
    return tex
  }

  // position and heading along a route at distance s (m)
  routePoint(r, s) {
    const cum = r.cum
    s = Math.min(Math.max(s, 0), r.length)
    let lo = 0, hi = cum.length - 1
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cum[m] <= s) lo = m; else hi = m }
    const t = (s - cum[lo]) / Math.max(cum[hi] - cum[lo], 1e-9), P = r.pts
    const a = [P[lo * 3], P[lo * 3 + 1], P[lo * 3 + 2]], b = [P[hi * 3], P[hi * 3 + 1], P[hi * 3 + 2]]
    const dir = lerpDir(a, b, t), d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], k = dot(d, dir)
    return { dir, tan: norm([d[0] - dir[0] * k, d[1] - dir[1] * k, d[2] - dir[2] * k]), seg: r.firstSeg + lo }
  }

  // ids of segments and cities filed near a direction (radius in metres)
  near(dir, radius) {
    const [x, y] = cellXY(dir), [la] = latLon(dir)
    const px = Math.ceil(radius / R * GRID_W / (2 * Math.PI) / Math.max(Math.cos(la), 0.02)), py = Math.ceil(radius / R * GRID_H / Math.PI)
    const segs = new Set(), cities = new Set(), rivers = new Set()
    for (let cy = Math.floor(y) - py; cy <= Math.floor(y) + py; cy++) {
      if (cy < 0 || cy >= GRID_H) continue
      for (let cx = Math.floor(x) - px; cx <= Math.floor(x) + px; cx++) {
        for (const id of this.cells.get(cy * GRID_W + ((cx % GRID_W) + GRID_W) % GRID_W) ?? []) {
          if (id & CITY_BIT) cities.add((id & 0x7fffffff) >>> 0); else if (id & RIVER_BIT) rivers.add((id & 0x3fffffff) >>> 0); else segs.add(id)
        }
      }
    }
    return { segs: [...segs], cities: [...cities].map(i => this.cities[i]), rivers: [...rivers] }
  }

  // nearest road/rail centreline to dir within radius: { d (m), seg, type, s (m along route), route }
  nearestSegment(dir, radius = 200, filter = null) {
    let best = null
    for (const k of this.near(dir, radius).segs) {
      const S = this.seg, t = S[k * 8 + 3]
      if (filter && !filter(t)) continue
      const A = [S[k * 8], S[k * 8 + 1], S[k * 8 + 2]], B = [S[k * 8 + 4], S[k * 8 + 5], S[k * 8 + 6]]
      const ab = [B[0] - A[0], B[1] - A[1], B[2] - A[2]], pa = [dir[0] - A[0], dir[1] - A[1], dir[2] - A[2]]
      const L2 = Math.max(dot(ab, ab), 1e-20), u = Math.min(1, Math.max(0, dot(pa, ab) / L2))
      const v = [pa[0] - ab[0] * u, pa[1] - ab[1] * u, pa[2] - ab[2] * u], k2 = dot(v, dir)
      const d = Math.hypot(v[0] - dir[0] * k2, v[1] - dir[1] * k2, v[2] - dir[2] * k2) * R
      if (d < radius && (!best || d < best.d)) best = { d, seg: k, type: t, s: S[k * 8 + 7] + u * Math.sqrt(L2) * R, route: this.segRoute[k] }
    }
    return best
  }

  // nearest river or lake within radius: { d (m to the water's edge, 0 inside), type 'river' | 'lake', width (m), level (m) } or null.
  // Rivers are exact (channel edge); lakes use the lake map cells (the real shoreline follows the terrain contour at 'level').
  nearestWater(dir, radius = 500) {
    let best = null
    for (const k of this.near(dir, radius + 800).rivers) {
      const S = this.riv, o = k * 12, A = [S[o], S[o + 1], S[o + 2]], B = [S[o + 4], S[o + 5], S[o + 6]]
      const ab = [B[0] - A[0], B[1] - A[1], B[2] - A[2]], pa = [dir[0] - A[0], dir[1] - A[1], dir[2] - A[2]]
      const u = Math.min(1, Math.max(0, dot(pa, ab) / Math.max(dot(ab, ab), 1e-20)))
      const v = [pa[0] - ab[0] * u, pa[1] - ab[1] * u, pa[2] - ab[2] * u], k2 = dot(v, dir)
      const hw = S[o + 3] + (S[o + 7] - S[o + 3]) * u, s = S[o + 10] + u * Math.sqrt(dot(ab, ab)) * R
      const side = norm(cross(dir, ab)), off = riverMeander(s, hw, S[o + 11] - s)       // same meander as the shader
      const w = [v[0] - dir[0] * k2 - side[0] * off / R, v[1] - dir[1] * k2 - side[1] * off / R, v[2] - dir[2] * k2 - side[2] * off / R]
      const d = Math.max(0, Math.hypot(w[0], w[1], w[2]) * R - hw)
      if (d < radius && (!best || d < best.d)) best = { d, type: 'river', width: 2 * hw, level: S[o + 8] + (S[o + 9] - S[o + 8]) * u }
    }
    const { NX, NY, lakeMap } = this.hydro, [la, lo] = latLon(dir), cell = HYDRO.cell * Math.PI / 180
    const fx = (lo / (2 * Math.PI) + 0.5) * NX - 0.5, fy = (la / Math.PI + 0.5) * NY - 0.5
    const ry = Math.ceil(radius / R / cell) + 1, rx = Math.ceil(ry / Math.max(Math.cos(la), 0.05))
    for (let y = Math.round(fy) - ry; y <= Math.round(fy) + ry; y++) {
      if (y < 0 || y >= NY) continue
      for (let x = Math.round(fx) - rx; x <= Math.round(fx) + rx; x++) {
        const level = lakeMap[y * NX + ((x % NX) + NX) % NX]
        if (level < -1e4) continue
        const dy = (y - fy) * cell * R, dx = (x - fx) * cell * R * Math.cos(la)
        const d = Math.max(0, Math.max(Math.abs(dx) - cell * R * Math.cos(la) / 2, Math.abs(dy) - cell * R / 2))
        if (d < radius && (!best || d < best.d)) best = { d, type: 'lake', width: cell * R, level }
      }
    }
    return best
  }

  nearestCity(dir, radius = 60000) {
    let best = null, bd = Infinity
    for (const c of this.near(dir, radius).cities) {
      const d = arc(c.dir, dir)
      if (d < bd) { bd = d; best = c }
    }
    return best && { city: best, distance: bd }
  }
}

function uintTex(data, w, h) {
  const t = new THREE.DataTexture(data, w, h, THREE.RedIntegerFormat, THREE.UnsignedIntType)
  t.internalFormat = 'R32UI'
  t.minFilter = t.magFilter = THREE.NearestFilter
  t.needsUpdate = true
  return t
}
function floatTex(data, w, h) {
  const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.FloatType)
  t.minFilter = t.magFilter = THREE.NearestFilter
  t.needsUpdate = true
  return t
}

// placeholder textures so every shader has valid bindings before the world exists (a 1x1 lake map = no lakes)
let U = null
export function civUniforms() {
  const u = uintTex(new Uint32Array(1), 1, 1), f = floatTex(new Float32Array(4), 1, 1)
  const map = new THREE.DataTexture(new Uint8Array(4), 1, 1)
  map.needsUpdate = true
  const lake = new THREE.DataTexture(new Float32Array([-1e5]), 1, 1, THREE.RedFormat, THREE.FloatType)
  lake.needsUpdate = true
  const rivers = new THREE.DataTexture(new Uint8Array([0, 255, 0, 0]), 1, 1)     // no rivers anywhere
  rivers.needsUpdate = true
  U = { uCivGrid: { value: u }, uCivItems: { value: u }, uCivSeg: { value: f }, uCivCity: { value: f }, uCivRiver: { value: f }, uLakeMap: { value: lake }, uRiverMap: { value: rivers }, uUrbanMap: { value: map }, uFarmMap: { value: map }, uCamDir: { value: new THREE.Vector3(0, 0, 1) }, uCamR: { value: 6371000 } }
  U.uSeason = { value: new THREE.Vector2() }   // (cos, sin) of the year's phase from the June solstice, set per frame; (0, 0) = annual mean
  return U
}
