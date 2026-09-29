// Rivers and lakes from the real height field. The land is sampled on a global lat/lon grid; a priority
// flood from the sea fills every depression (a lake at its spill level, or lower where evaporation beats
// the inflow, then it has no outlet), runoff from the climate's moisture is accumulated down the flow
// tree, and cells above a discharge threshold become rivers. The coarse paths are refined to follow the
// valleys and get a width and a water level that never rises downstream. world.js files the segments
// into the civ lookup; civ.glsl.js carves the beds and reads the lake map.
import { R } from '../config.js'
import { HYDRO, riverDepth } from './spec.js'

const DEG = Math.PI / 180
const NX = Math.round(360 / HYDRO.cell), NY = Math.round(180 / HYDRO.cell)
const YEAR = 3.156e7
const QMIN = 250                   // m3/s: coarse cells carrying more than this are rivers
const STEP = 5000                  // refined point spacing along small rivers (m); big ones get 2.5 km
const KEEP = 0.2                   // share of a basin's depth left as lake once its outlet has cut down
const J = 3                        // lateral candidates each side when fitting rivers into valleys
const CLIMB = 4                    // DP cost per metre the path climbs downstream (each metre is a metre of trench)
const NONE = -1e5                  // lake map value for dry land
const PLAYA = 1.5                  // closed basins whose equilibrium lake is smaller than this many cells stay dry

const cellDir = c => {
  const i = c % NX, j = (c - i) / NX, la = (-90 + (j + 0.5) * HYDRO.cell) * DEG, lo = (-180 + (i + 0.5) * HYDRO.cell) * DEG
  return [Math.cos(la) * Math.sin(lo), Math.sin(la), Math.cos(la) * Math.cos(lo)]
}
const norm = a => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l] }
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const arc = (a, b) => Math.acos(Math.min(1, Math.max(-1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * R
const lerp3 = (a, b, t) => norm([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t])
const halfWidth = Q => Math.min(750, Math.max(1.5, Math.pow(Q, 0.62)))    // w = 2 Q^0.62: ~10 m creeks .. ~1 km near big mouths

// binary min-heap of cell ids keyed by height
class Heap {
  constructor(n) { this.k = new Float32Array(n); this.v = new Int32Array(n); this.n = 0 }
  push(v, k) {
    let i = this.n++
    while (i > 0) { const p = (i - 1) >> 1; if (this.k[p] <= k) break; this.k[i] = this.k[p]; this.v[i] = this.v[p]; i = p }
    this.k[i] = k; this.v[i] = v
  }
  pop() {
    const top = this.v[0], k = this.k[--this.n], v = this.v[this.n]
    let i = 0
    for (;;) {
      let c = 2 * i + 1
      if (c >= this.n) break
      if (c + 1 < this.n && this.k[c + 1] < this.k[c]) c++
      if (this.k[c] >= k) break
      this.k[i] = this.k[c]; this.v[i] = this.v[c]; i = c
    }
    this.k[i] = k; this.v[i] = v
    return top
  }
}

// 8 neighbours with longitude wrap; returns the count written to out
function neighbours(c, out) {
  const x = c % NX, y = (c - x) / NX
  let n = 0
  for (let dy = -1; dy <= 1; dy++) {
    const yy = y + dy
    if (yy < 0 || yy >= NY) continue
    for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue
      const xx = x + dx < 0 ? NX - 1 : x + dx >= NX ? 0 : x + dx
      out[n++] = yy * NX + xx
    }
  }
  return n
}

export async function buildHydro(sampler, stage) {
  await stage('Raising continents')
  const t0 = performance.now(), N = NX * NY, NB = new Int32Array(8)
  // height, temperature and moisture per cell, band-limited to the grid
  const h = new Float32Array(N), T = new Float32Array(N), M = new Float32Array(N)
  for (let j0 = 0; j0 < NY; j0 += 300) {
    const rows = Math.min(300, NY - j0), dirs = new Float32Array(rows * NX * 3)
    for (let k = 0; k < rows * NX; k++) dirs.set(cellDir(j0 * NX + k), k * 3)
    const s = sampler.sample(dirs, HYDRO.lakeF)
    for (let k = 0; k < rows * NX; k++) { h[j0 * NX + k] = s[k * 8]; T[j0 * NX + k] = s[k * 8 + 5]; M[j0 * NX + k] = s[k * 8 + 6] }
  }

  await stage('Filling lakes')
  // priority flood from the sea (Barnes 2014, with a plain queue for cells inside depressions)
  const fill = new Float32Array(N), rec = new Int32Array(N).fill(-1), order = new Int32Array(N), seen = new Uint8Array(N)
  const heap = new Heap(N), pit = new Int32Array(N)
  for (let c = 0; c < N; c++) if (h[c] < 0) seen[c] = 1
  for (let c = 0; c < N; c++) {
    if (h[c] >= 0) continue
    const k = neighbours(c, NB)
    for (let e = 0; e < k; e++) if (h[NB[e]] >= 0) { heap.push(c, 0); break }
  }
  let no = 0, q0 = 0, q1 = 0
  while (q1 > q0 || heap.n) {
    const c = q1 > q0 ? pit[q0++] : heap.pop()
    if (h[c] >= 0) order[no++] = c
    const k = neighbours(c, NB)
    for (let e = 0; e < k; e++) {
      const n = NB[e]
      if (seen[n]) continue
      seen[n] = 1; rec[n] = c
      if (h[n] <= fill[c]) { fill[n] = fill[c]; pit[q1++] = n } else { fill[n] = h[n]; heap.push(n, h[n]) }
    }
  }

  // runoff (m3/s per cell) from moisture; frozen ground gives little
  const cellArea = new Float32Array(NY), Q = new Float32Array(N)
  for (let j = 0; j < NY; j++) cellArea[j] = (HYDRO.cell * DEG * R) ** 2 * Math.cos((-90 + (j + 0.5) * HYDRO.cell) * DEG)
  for (let c = 0; c < N; c++) {
    if (h[c] < 0) continue
    const r = 1.5 * Math.pow(Math.max(0, M[c] - 0.15), 1.5) * (0.1 + 0.9 * Math.min(1, Math.max(0, (T[c] + 12) / 10)))
    Q[c] = r * cellArea[(c / NX) | 0] / YEAR
  }
  const acc = Float32Array.from(Q)
  for (let k = no - 1; k >= 0; k--) { const c = order[k]; if (rec[c] >= 0) acc[rec[c]] += acc[c] }

  // lakes: connected flooded cells at one level; evaporation decides whether they reach the spill level
  const comp = new Int32Array(N).fill(-1), lakes = []
  for (let c0 = 0; c0 < N; c0++) {
    if (comp[c0] >= 0 || h[c0] < 0 || !(fill[c0] > h[c0])) continue
    const id = lakes.length, cells = [c0], lv = fill[c0]
    comp[c0] = id
    for (let i = 0; i < cells.length; i++) {
      const k = neighbours(cells[i], NB)
      for (let e = 0; e < k; e++) {
        const n = NB[e]
        if (comp[n] < 0 && fill[n] === lv && fill[n] > h[n]) { comp[n] = id; cells.push(n) }
      }
    }
    lakes.push({ id, cells, spill: lv })
  }
  const water = new Int32Array(N).fill(-1)        // lake id of cells under lake water
  const budget = new Float32Array(lakes.length)    // evaporation left to take from the outflow
  const kept = []
  for (const L of lakes) {
    let inflow = 0, lo = L.cells[0]
    for (const c of L.cells) { if (comp[rec[c]] !== L.id) inflow += acc[c]; if (h[c] < h[lo]) lo = c }
    const evap = Math.min(1.8, Math.max(0.1, 0.2 + 0.055 * T[lo])) * (1.3 - M[lo]) / YEAR   // m/s over the lake
    L.cells.sort((a, b) => h[a] - h[b])
    // equilibrium area (inflow = evaporation): fill cells from the bottom until it is reached; the level sits part way
    // up the last cell by the share of it needed, so the shore is the terrain contour inside the basin, not a cell edge
    const need = inflow / evap
    let area = 0, k = 0, part = 1
    for (; k < L.cells.length; k++) {
      const a = cellArea[(L.cells[k] / NX) | 0]
      if (area + a >= need) { part = (need - area) / a; break }
      area += a
    }
    L.open = k === L.cells.length                   // evaporation over the whole basin can't take the inflow: it spills
    const hk = h[L.cells[k]], hn = k + 1 < L.cells.length ? h[L.cells[k + 1]] : L.spill
    // the outlet river cuts down through the rim: only the deepest part of a basin stays a lake
    L.level = Math.min(L.open ? L.spill : hk + (hn - hk) * 0.5 * part, h[lo] + KEEP * (L.spill - h[lo]))
    // endorheic basins whose inflow holds less than PLAYA cells of lake stay dry: a playa / salt pan, no water
    const dry = !L.open && need < PLAYA * cellArea[(lo / NX) | 0]
    L.depth = L.level - h[lo]
    const wet = L.cells.filter(c => h[c] < L.level)
    L.area = wet.reduce((a, c) => a + cellArea[(c / NX) | 0], 0)
    budget[L.id] = L.open ? evap * L.area : Infinity
    if (!L.open) {
      // closed basin: the lake is a sink, the dry rest of the basin drains into it (flood inward from the water)
      for (const c of wet) rec[c] = -1
      const inner = new Heap(L.cells.length), done = new Set(wet)
      for (const c of wet) inner.push(c, h[c])
      while (inner.n) {
        const c = inner.pop(), k2 = neighbours(c, NB)
        for (let e = 0; e < k2; e++) {
          const n = NB[e]
          if (comp[n] !== L.id || done.has(n)) continue
          done.add(n); rec[n] = c; inner.push(n, h[n])
        }
      }
    }
    // too shallow (a marsh), too dry (a salt pan), or a one-cell pit in a dry climate (boxy, and mostly noise)
    if (L.depth < 12 || dry || (wet.length < 2 && M[lo] < 0.3)) continue
    for (const c of wet) water[c] = L.id
    kept.push(L)
  }

  // final accumulation in topological order (Kahn), open lakes lose their evaporation at the outlet
  const indeg = new Uint8Array(N), flow = Float32Array.from(Q), queue = new Int32Array(N)
  let qn = 0
  for (let c = 0; c < N; c++) if (h[c] >= 0 && rec[c] >= 0) indeg[rec[c]]++
  for (let c = 0; c < N; c++) if (h[c] >= 0 && !indeg[c]) queue[qn++] = c
  for (let qi = 0; qi < qn; qi++) {
    const c = queue[qi], r = rec[c]
    if (r < 0) continue
    let out = flow[c]
    const L = comp[c]
    if (L >= 0 && comp[r] !== L) { const take = Math.min(out, budget[L]); budget[L] -= take; out -= take }
    flow[r] += out
    if (--indeg[r] === 0 && h[r] >= 0) queue[qn++] = r
  }

  await stage('Tracing rivers')
  // river cells and their upstream donors
  const river = new Uint8Array(N), riverCells = []
  for (let c = 0; c < N; c++) if (h[c] >= 0 && flow[c] >= QMIN && water[c] < 0) { river[c] = 1; riverCells.push(c) }
  const kids = new Map()
  const addKid = (r, c) => { if (!kids.has(r)) kids.set(r, []); kids.get(r).push(c) }
  for (const c of riverCells) if (rec[c] >= 0 && river[rec[c]]) addKid(rec[c], c)
  for (let c = 0; c < N; c++) if (water[c] >= 0 && rec[c] >= 0 && river[rec[c]]) addKid(rec[c], c)   // lake outlets
  // trunks follow the biggest donor upstream; other donors start tributaries ending at the junction
  const lines = [], todo = []
  for (const c of riverCells) if (rec[c] < 0 || !river[rec[c]]) todo.push({ c, end: rec[c], parent: -1 })
  for (let ti = 0; ti < todo.length; ti++) {
    const { c: start, end, parent } = todo[ti], path = [], id = lines.length
    let c = start
    for (;;) {
      path.push(c)
      if (water[c] >= 0) break                       // reached the lake this river drains
      const ks = kids.get(c)
      if (!ks) break
      ks.sort((a, b) => flow[b] - flow[a])
      for (let i = 1; i < ks.length; i++) if (river[ks[i]]) todo.push({ c: ks[i], end: c, parent: id })
      c = ks[0]
    }
    path.reverse()
    lines.push({ id, cells: path, end, parent, lakeStart: water[path[0]] >= 0 ? kept.find(L => L.id === water[path[0]]) : null })
  }
  const t1 = performance.now()
  const rivers = refine(sampler, lines, { h, flow, water, lakes, river })
  const lakeMap = new Float32Array(N).fill(NONE)
  for (const L of kept) for (const c of L.cells) if (water[c] === L.id) lakeMap[c] = L.level
  for (const L of kept) { const d = cellDir(L.cells[0]); L.lat = Math.asin(d[1]) / DEG; L.lon = Math.atan2(d[0], d[2]) / DEG; delete L.cells }
  console.info(`hydro: ${kept.length} lakes (${kept.filter(L => !L.open).length} closed), ${riverCells.length} river cells, ${rivers.length} rivers, flood ${Math.round(t1 - t0)} ms, refine ${Math.round(performance.now() - t1)} ms`)
  const ll = d => `${(Math.asin(d[1]) / DEG).toFixed(2)},${(Math.atan2(d[0], d[2]) / DEG).toFixed(2)}`
  for (const r of [...rivers].sort((a, b) => b.Q - a.Q).slice(0, 5)) {
    const m = r.pts[r.pts.length - 2], mid = r.pts[r.pts.length >> 1]
    console.info(`hydro: river ${r.id} Q ${Math.round(r.Q)} m3/s, ${Math.round(2 * m.hw)} m wide, mouth ${ll(m.d)} (level ${m.level.toFixed(1)}), middle ${ll(mid.d)} (${Math.round(2 * mid.hw)} m, level ${mid.level.toFixed(1)})`)
  }
  for (const L of [...kept].sort((a, b) => b.area - a.area).slice(0, 3)) console.info(`hydro: lake at ${L.lat.toFixed(2)},${L.lon.toFixed(2)} level ${L.level.toFixed(1)} area ${Math.round(L.area / 1e6)} km2 ${L.open ? 'open' : 'closed'}`)
  return { NX, NY, lakeMap, lakes: kept, rivers }
}

// coarse cell paths -> valley-following polylines with half widths and monotone water levels
function refine(sampler, lines, { h, flow, water, lakes }) {
  // candidate stages along each line: STEP apart, 2J+1 lateral offsets (zero at the ends)
  const plans = [], dirs = []
  for (const L of lines) {
    const ctrl = L.cells.map(cellDir), q = L.cells.map(c => flow[c])
    q[0] *= L.lakeStart ? 1 : 0.15                   // sources start as creeks
    const endDir = L.end >= 0 ? cellDir(L.end) : null
    if (endDir) { ctrl.push(endDir); q.push(q[q.length - 1]) }
    if (ctrl.length < 2) continue
    const st = []                                    // stages: { d, Q, coarse index }
    for (let k = 0; k < ctrl.length - 1; k++) {
      const n = Math.max(1, Math.ceil(arc(ctrl[k], ctrl[k + 1]) / (q[k] > 3000 ? 2500 : STEP)))
      for (let i = 0; i < n; i++) st.push({ d: lerp3(ctrl[k], ctrl[k + 1], i / n), Q: q[k] + (q[k + 1] - q[k]) * i / n, k })
    }
    st.push({ d: ctrl[ctrl.length - 1], Q: q[q.length - 1], k: ctrl.length - 1 })
    const K = st.length, off = dirs.length / 3
    for (let s = 0; s < K; s++) {
      const a = st[Math.max(0, s - 1)].d, b = st[Math.min(K - 1, s + 1)].d
      const side = norm(cross(st[s].d, [b[0] - a[0], b[1] - a[1], b[2] - a[2]]))
      const lat = (500 + 2.5 * halfWidth(st[s].Q)) * Math.min(1, s / 2, (K - 1 - s) / 2) / R
      for (let j = -J; j <= J; j++) dirs.push(...norm([st[s].d[0] + side[0] * j * lat, st[s].d[1] + side[1] * j * lat, st[s].d[2] + side[2] * j * lat]))
    }
    plans.push({ L, st, K, off })
  }
  const W = 2 * J + 1, S = sampler.sample(new Float32Array(dirs), HYDRO.bedF)
  const hAt = (p, s, j) => S[(p.off + s * W + j) * 8]
  const dirAt = (p, s, j) => { const o = (p.off + s * W + j) * 3; return [dirs[o], dirs[o + 1], dirs[o + 2]] }
  const byLine = new Map(), out = []
  for (const p of plans) {
    // cheapest path: low ground, small lateral jumps, and above all no climbing downstream: the level can't rise, so
    // every metre of ridge crossed would be cut as a trench
    let cost = Array.from({ length: W }, (_, j) => hAt(p, 0, j))
    const back = []
    for (let s = 1; s < p.K; s++) {
      const next = new Array(W), from = new Array(W)
      for (let j = 0; j < W; j++) {
        let best = Infinity, bi = j
        const hj = hAt(p, s, j)
        for (let i = Math.max(0, j - 1); i <= Math.min(W - 1, j + 1); i++) {
          const c = cost[i] + 25 * Math.abs(i - j) + CLIMB * Math.max(0, hj - hAt(p, s - 1, i))
          if (c < best) { best = c; bi = i }
        }
        next[j] = best + hAt(p, s, j); from[j] = bi
      }
      back.push(from); cost = next
    }
    let j = J
    const pick = new Array(p.K)
    for (let s = p.K - 1; s >= 0; s--) { pick[s] = j; if (s) j = back[s - 1][j] }
    let pts = pick.map((jj, s) => ({ d: dirAt(p, s, jj), h: hAt(p, s, jj), Q: p.st[s].Q, k: p.st[s].k }))
    // tributaries join the trunk's refined point at the junction
    const L = p.L
    let endLevel = L.end >= 0 && h[L.end] < 0 ? 0 : -Infinity
    if (L.end >= 0 && water[L.end] >= 0) endLevel = lakes[water[L.end]].level
    const par = L.parent >= 0 ? byLine.get(L.parent) : null
    if (par) {
      let bi = 0, bd = Infinity
      const e = pts[pts.length - 1].d
      for (let i = 0; i < par.pts.length; i++) { const d = arc(par.pts[i].d, e); if (d < bd) { bd = d; bi = i } }
      pts[pts.length - 1] = { ...pts[pts.length - 1], d: par.pts[bi].d }
      endLevel = par.pts[bi].level
    }
    pts = chaikin(pts)
    // water level: running minimum of (ground - channel depth) downstream, never below the receiving water. The
    // ground is a 3-point median, so one sample in a dip (a side gully) doesn't pin the whole river below it
    let lv = Infinity
    for (let i = 0; i < pts.length; i++) {
      const q = pts[i], hw = halfWidth(q.Q), a = pts[Math.max(0, i - 1)].h, b = pts[Math.min(pts.length - 1, i + 1)].h
      const hm = Math.max(Math.min(a, b), Math.min(Math.max(a, b), q.h))
      lv = L.lakeStart && q.k === 0 ? L.lakeStart.level : Math.min(lv, hm - riverDepth(hw))
      q.level = Math.max(lv, endLevel); q.hw = hw
    }
    const r = { id: L.id, parent: L.parent, pts, Q: flow[L.cells[L.cells.length - 1]] }
    byLine.set(L.id, r)
    out.push(r)
  }
  return out
}

function chaikin(pts) {
  const out = [pts[0]]
  const mix = (a, b, t) => ({ d: lerp3(a.d, b.d, t), h: a.h + (b.h - a.h) * t, Q: a.Q + (b.Q - a.Q) * t, k: t < 0.5 ? a.k : b.k })
  for (let i = 0; i < pts.length - 1; i++) out.push(mix(pts[i], pts[i + 1], 0.25), mix(pts[i], pts[i + 1], 0.75))
  out.push(pts[pts.length - 1])
  return out
}
