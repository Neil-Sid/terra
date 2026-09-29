// Shared civilisation constants. JS (buildings, traffic, people) and GLSL (terrain painting)
// must agree on these, so both read them from here.

export const TYPE = { HIGHWAY: 0, ROAD: 1, RAIL: 2 }
export const HALF = [12.0, 4.5, 3.2]          // half width of the paved/ballast bed (m) per type
// lane centres measured from the centreline, positive side = right-hand traffic in the route's direction
export const LANES = [[4.3, 7.9], [1.8], []]
export const RAIL_GAUGE = 1.435
export const SPEED = [30, 20, 38]              // typical speeds (m/s) per type

// city street grid: streets run along block boundaries (multiples of the city's block size)
export const STREET = {
  half: 7, avenueHalf: 13, avenueEvery: 4,     // half widths (m) incl. sidewalks
  lanes: [1.8], avenueLanes: [3.8, 7.4],       // lane centres from the street centreline
  walk: [4.8, 6.6], avenueWalk: [10.8, 12.6],  // sidewalk band (m from centreline)
}
export const TIER = { VILLAGE: 0, TOWN: 1, CITY: 2, METRO: 3 }
export const BLOCK = [70, 90, 110, 130]        // block size per tier (m)
export const CITY_BIT = 0x80000000             // item ids with this bit set are cities

export const CELLS_PER_DEG = 8                 // lookup grid resolution (cells per degree)
export const GRID_W = 360 * CELLS_PER_DEG, GRID_H = 180 * CELLS_PER_DEG
export const ROW = 4096                        // texels per row in the data textures

// integer hash identical in JS and GLSL, returns [0, 1)
export function hash2i(x, y, seed) {
  let h = Math.imul(x, 0x27d4eb2d) ^ Math.imul(y, 0x165667b1) ^ Math.imul(seed, 0x9e3779b1)
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b)
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

// local street frame of a city centre C (unit vector) rotated by angle: [ex, ny] tangent axes
export function cityFrame(C, angle) {
  const el = Math.hypot(C[2], C[0]) || 1
  const e = [C[2] / el, 0, -C[0] / el]
  const n = [C[1] * e[2] - C[2] * e[1], C[2] * e[0] - C[0] * e[2], C[0] * e[1] - C[1] * e[0]]
  const c = Math.cos(angle), s = Math.sin(angle)
  return [e.map((v, i) => v * c + n[i] * s), e.map((v, i) => -v * s + n[i] * c)]
}

// ---- organic city layout ----
// Streets are a grid in "grid space" (u, v). A smooth warp made of shear waves bends that grid into the
// city's local tangent plane (x, y metres, axes from cityFrame): three district-scale waves turn
// neighbourhoods against each other and two finer waves make the old core irregular; both ease off
// towards the outskirts, where a gentler grid remains. Shear waves preserve area and never fold, so the
// warp inverts exactly (Newton). The outline is lobed and lumpy per city with ragged sprawl at the edge.
// Everything here is mirrored in SPEC_GLSL below; keep them identical.
const R_PLANET = 6371000
const smooth01 = t => { t = Math.min(1, Math.max(0, t)); return t * t * (3 - 2 * t) }
const WARP_TERMS = 5, WARP_CORE = 0.07, WARP_DIST = 0.1   // wave slopes (round 4: 0.22 / 0.12; districts now carry the irregularity)

function warpTerms(city) {
  if (city._warp) return city._warp
  const h = (i, s) => hash2i(city.seed, i, s), scale = Math.min(1, Math.max(0.3, city.radius / 12000)), t = []
  for (let i = 0; i < WARP_TERMS; i++) {
    const core = i >= 3, ang = h(i, 91) * 2 * Math.PI
    const k = 2 * Math.PI / (core ? 380 + 420 * h(i, 92) : (2200 + 4500 * h(i, 92)) * scale)
    t.push({ kx: Math.cos(ang) * k, ky: Math.sin(ang) * k, nx: -Math.sin(ang), ny: Math.cos(ang), A: (core ? WARP_CORE : WARP_DIST) / k, ph: h(i, 93) * 2 * Math.PI, core })
  }
  return (city._warp = t)
}

// grid point (u, v) -> [x, y, dx/du, dx/dv, dy/du, dy/dv] in local metres
export function warpGrid(city, u, v) {
  const r = Math.hypot(u, v) / city.radius
  const wCore = Math.exp(-((r / 0.55) ** 2)), wDist = 1 - 0.65 * smooth01((r - 0.5) / 0.6)
  let x = u, y = v, a = 1, b = 0, c = 0, d = 1
  for (const t of warpTerms(city)) {
    const w = t.core ? wCore : wDist, ph = t.kx * u + t.ky * v + t.ph
    const sn = t.A * w * Math.sin(ph), cs = t.A * w * Math.cos(ph)
    x += sn * t.nx; y += sn * t.ny
    a += cs * t.nx * t.kx; b += cs * t.nx * t.ky; c += cs * t.ny * t.kx; d += cs * t.ny * t.ky
  }
  return [x, y, a, b, c, d]
}
// local metres -> grid coordinates
export function localToGrid(city, x, y) {
  let u = x, v = y
  for (let i = 0; i < 4; i++) {
    const [gx, gy, a, b, c, d] = warpGrid(city, u, v), det = a * d - b * c, ex = gx - x, ey = gy - y
    u -= (d * ex - b * ey) / det
    v -= (a * ey - c * ex) / det
  }
  return [u, v]
}
// city outline in local metres: > 0 inside (1 at the centre), 0 on the edge, < 0 outside
export function cityShape(city, x, y) {
  const h = k => hash2i(city.seed, k, 97) * 2 * Math.PI, R0 = city.radius
  const th = Math.atan2(y, x), r = Math.hypot(x, y) / R0
  let s = 1 + 0.30 * Math.sin(2 * th + h(0)) + 0.22 * Math.sin(3 * th + h(1)) + 0.12 * Math.sin(5 * th + h(2))
  s *= 1 + 0.18 * Math.sin((x * Math.cos(h(3)) + y * Math.sin(h(3))) / R0 * 3.1 + h(4))
         + 0.12 * Math.sin((x * Math.cos(h(5)) + y * Math.sin(h(5))) / R0 * 5.3 + h(6))
  return 1 - r / (0.8 * s)
}
export const CITY_REACH = 1.8          // the outline never extends beyond this many radii

// local frame helpers: unit direction <-> local metres <-> grid coordinates
const frameOf = city => city._frame ??= cityFrame(city.dir, city.angle)
export function localToDir(city, x, y) {
  const [ex, ny] = frameOf(city), C = city.dir
  const d = [C[0] + (x * ex[0] + y * ny[0]) / R_PLANET, C[1] + (x * ex[1] + y * ny[1]) / R_PLANET, C[2] + (x * ex[2] + y * ny[2]) / R_PLANET]
  const l = Math.hypot(d[0], d[1], d[2])
  return [d[0] / l, d[1] / l, d[2] / l]
}
export function dirToLocal(city, dir) {
  const [ex, ny] = frameOf(city), C = city.dir, v = [dir[0] - C[0], dir[1] - C[1], dir[2] - C[2]]
  return [(v[0] * ex[0] + v[1] * ex[1] + v[2] * ex[2]) * R_PLANET, (v[0] * ny[0] + v[1] * ny[1] + v[2] * ny[2]) * R_PLANET]
}
export const gridToDir = (city, u, v) => { const [x, y] = warpGrid(city, u, v); return localToDir(city, x, y) }
export const dirToGrid = (city, dir) => { const [x, y] = dirToLocal(city, dir); return localToGrid(city, x, y) }
// world tangents of the grid's +u / +v directions at (u, v) and metres per grid unit along each
export function gridAxes(city, u, v) {
  const [, , a, b, c, d] = warpGrid(city, u, v), [ex, ny] = frameOf(city)
  const su = Math.hypot(a, c), sv = Math.hypot(b, d)
  return {
    eu: ex.map((e, i) => (e * a + ny[i] * c) / su), ev: ex.map((e, i) => (e * b + ny[i] * d) / sv), su, sv,
    angle: Math.atan2(c, a),     // rotation of the local grid against the city frame
  }
}

// is block (kx, ky) of a city built up? returns density 0..1 (0 = countryside)
export function blockDensity(city, kx, ky) {
  const [x, y] = warpGrid(city, (kx + 0.5) * city.block, (ky + 0.5) * city.block)
  const d = cityShape(city, x, y)
  if (d >= 0.2) return d
  if (d < -0.15) return 0
  // ragged edge: clusters of blocks thinning out into the countryside
  const f = smooth01((d + 0.15) / 0.35)
  return hash2i(Math.floor(kx / 3), Math.floor(ky / 3), city.seed + 5) < f * 1.15 && hash2i(kx, ky, city.seed + 6) < 0.3 + f ? Math.max(d, 0.02) : 0
}
export const isPark = (city, kx, ky) => hash2i(kx, ky, city.seed + 7) < 0.07
// industrial districts: 1 (towns) to 3 (metros) sectors on a city's outer ring; returns 1 = industrial, 0 = not
const SECTOR = 0.2
export function blockZone(city, kx, ky) {
  if (city.tier < 1) return 0
  const [x, y] = warpGrid(city, (kx + 0.5) * city.block, (ky + 0.5) * city.block), d = cityShape(city, x, y)
  if (d > 0.42 || d < 0.02 || hash2i(Math.floor(kx / 2), Math.floor(ky / 2), city.seed + 17) > 0.8) return 0
  const a = Math.atan2(y, x)
  for (let i = 0; i < city.tier; i++) {
    const da = Math.abs(a - (hash2i(i, 3, city.seed) * 2 - 1) * Math.PI)
    if (Math.min(da, 2 * Math.PI - da) < SECTOR) return 1
  }
  return 0
}
// centres of a city's industrial districts: [{ dir, x, y, angle, D }] (x, y local metres of the district seed), one per
// sector (tier of them): the INDUSTRY district nearest to where the sector's bearing meets outline value 0.22
export function industrySites(city) {
  if (city.tier < 1) return []
  const P = districtPitch(city), n = Math.ceil(toLattice(city, city.radius * CITY_REACH, 0)[0] / P), ind = []
  for (let j = -n; j <= n; j++) for (let i = -n; i <= n; i++) { const D = district(city, i, j); if (D.kind === DKIND.INDUSTRY) ind.push(D) }
  const out = []
  for (let i = 0; i < city.tier; i++) {
    const angle = (hash2i(i, 3, city.seed) * 2 - 1) * Math.PI, cx = Math.cos(angle), cy = Math.sin(angle)
    let r = 0
    while (r < city.radius * CITY_REACH && cityShape(city, cx * r, cy * r) > 0.22) r += 50
    let best = null, bd = Infinity
    for (const D of ind) { const [x, y] = warpGrid(city, D.ou, D.ov), d = Math.hypot(x - cx * r, y - cy * r); if (d < bd) { bd = d; best = { x, y, D } } }
    if (best && !out.some(o => o.D === best.D)) out.push({ dir: localToDir(city, best.x, best.y), x: best.x, y: best.y, angle, D: best.D })
  }
  return out
}
// ---- deprecated (round 4 single lattice; kept until buildings/factories/traffic/people/bridges are ported) ----
export const isAvenue = k => ((k % STREET.avenueEvery) + STREET.avenueEvery) % STREET.avenueEvery === 0
export const streetHalf = k => isAvenue(k) ? STREET.avenueHalf : STREET.half

// Buildable rectangle of block (kx, ky) in city-local metres (streets run along multiples of the
// block size), split into nx * ny lots. Lot (i, j) spans x0 + (x1 - x0) * i / nx .. (i + 1) / nx.
export function blockLots(city, kx, ky) {
  const B = city.block
  return {
    x0: kx * B + streetHalf(kx) + 1, x1: (kx + 1) * B - streetHalf(kx + 1) - 1,
    y0: ky * B + streetHalf(ky) + 1, y1: (ky + 1) * B - streetHalf(ky + 1) - 1,
    nx: 1 + Math.floor(hash2i(kx, ky, city.seed + 11) * 3), ny: 1 + Math.floor(hash2i(kx, ky, city.seed + 13) * 3),
  }
}

// ---- districts (round 5): the layout every consumer should use ----
// See notes/layout.md. Grid space (u, v) = the warped space above. It is cut into districts: a jittered Voronoi on
// a lattice in log-radial space (toLattice), so districts are ~250-750 m across at the centre and grow outwards.
// Each district has its own frame: district coords (x, y) = rotate (g - origin) by angle, then y -= A sin(k x)
// (a shear: rows curve into crescents in suburbs). Rows (y = const lines) are continuous local streets; connectors
// (x = const) run only within a row, with a per-row offset and spacing, so they end in T-junctions. Block (kx, ky):
// row ky between row lines y0..y1, connector cell kx between x0..x1. Trunk streets cut across: district edges
// (the Voronoi boundaries), radial boulevards and a ring road.
export const DKIND = { CORE: 0, GRID: 1, SUPER: 2, SUBURB: 3, INDUSTRY: 4 }
export const SKIND = { LANE: 0, STREET: 1, EDGE: 2, BOULEVARD: 3, RING: 4 }
// cross-section per profile (0 lane, 1 street, 2 edge, 3 boulevard/ring): half width incl. sidewalks, lane centres per
// side, sidewalk band [kerb, back] (m from the centreline), planted median half width
export const PROFILES = [
  { half: 5.0, lanes: [1.5, 1.5], walk: [3.2, 4.6], median: 0 },
  { half: 7.0, lanes: [1.8, 1.8], walk: [4.8, 6.6], median: 0 },
  { half: 10.0, lanes: [1.8, 5.2], walk: [7.2, 9.2], median: 0 },
  { half: 15.0, lanes: [5.0, 8.6], walk: [10.6, 13.4], median: 3.2 },
]
export const profileOf = kind => PROFILES[Math.min(kind, 3)]
const LAT_R1 = 0.8                                   // log-radial scale (radii)
export const districtPitch = city => Math.min(750, Math.max(250, 200 + 0.05 * city.radius))
const latScale = (r, R1, inv) => r < 1e-3 ? 1 : (inv ? R1 * Math.expm1(r / R1) : R1 * Math.log1p(r / R1)) / r
export function toLattice(city, u, v) { const s = latScale(Math.hypot(u, v), LAT_R1 * city.radius, false); return [u * s, v * s] }
export function fromLattice(city, a, b) { const s = latScale(Math.hypot(a, b), LAT_R1 * city.radius, true); return [a * s, b * s] }
const latticeSeed = (city, i, j, P) => [(i + 0.2 + 0.6 * hash2i(i, j, city.seed + 61)) * P, (j + 0.2 + 0.6 * hash2i(i, j, city.seed + 62)) * P]
const dKey = (i, j) => i * 1024 + j

// inverse warp Jacobian at grid (u, v): metres across the zero line of a function with grid gradient (gx, gy) = 1 / |Ji^T g|
function metresPer(city, u, v) {
  const [, , a, b, c, d] = warpGrid(city, u, v), det = a * d - b * c
  return (gx, gy) => 1 / Math.hypot((d * gx - c * gy) / det, (-b * gx + a * gy) / det)
}

// district of lattice cell (i, j): { i, j, key, kind, ou, ov (origin, grid), angle, c, s, A, k (shear), bx, by (connector
// and row spacing), ox, oy, stagger, jit (row spacing jitter), varx (per-row connector spacing variation), prof, hseed }
export function district(city, i, j) {
  const cache = city._districts ??= new Map(), key = dKey(i, j)
  if (cache.has(key)) return cache.get(key)
  const P = districtPitch(city), s = city.seed, h = n => hash2i(i, j, s + 63 + n)
  const [la, lb] = latticeSeed(city, i, j, P), [ou, ov] = fromLattice(city, la, lb)
  const [x, y] = warpGrid(city, ou, ov), shape = cityShape(city, x, y), r = Math.hypot(ou, ov)
  const kind = districtKind(city, x, y, shape, r, P, h(0), h(9))
  const D = { i, j, key, kind, ou, ov, hseed: s + key * 131, angle: h(6) * Math.PI, A: 0, k: 0, stagger: 0, jit: 0, varx: 0, prof: SKIND.STREET }
  if (kind === DKIND.CORE) Object.assign(D, { by: 45 + 30 * h(1), bx: 55 + 40 * h(2), stagger: 1, jit: 0.35, varx: 0.5, prof: SKIND.LANE, A: 6 + 10 * h(3), k: 2 * Math.PI / (180 + 200 * h(4)) })
  else if (kind === DKIND.GRID) {
    const B = city.block * (0.85 + 0.3 * h(1))
    Object.assign(D, { by: B, bx: B * (1 + 1.8 * h(2)), stagger: h(3) < 0.35 ? 0.5 : 0 })
    if (h(4) < 0.3) Object.assign(D, { A: B * 0.3, k: 2 * Math.PI / (1200 + 1500 * h(5)) })
  } else if (kind === DKIND.SUPER) Object.assign(D, { by: 200 + 120 * h(1), bx: 240 + 140 * h(2), jit: 0.1, prof: SKIND.EDGE })
  else if (kind === DKIND.SUBURB) {
    const lam = 350 + 450 * h(4)
    Object.assign(D, { by: 70 + 25 * h(1), bx: 170 + 150 * h(2), stagger: 1, jit: 0.15, varx: 0.4, A: lam * (0.04 + 0.07 * h(3)), k: 2 * Math.PI / lam })
  } else Object.assign(D, { by: 130 + 90 * h(1), bx: 170 + 120 * h(2), stagger: 0.5, jit: 0.1 })
  D.ox = h(7) * D.bx; D.oy = h(8) * D.by; D.c = Math.cos(D.angle); D.s = Math.sin(D.angle)
  cache.set(key, D)
  return D
}
// district kind from its seed point (local metres x, y; outline value; grid radius r), h, h2 = hashes 0..1.
// Industry: whole districts on the outer ring, clustered loosely (IND_SPREAD rad, 60 %) round 1..3 bearings per city
const IND_SPREAD = 0.6, CORE_R = 0.05
function districtKind(city, x, y, shape, r, P, h, h2) {
  if (r < Math.max(CORE_R * city.radius, 1.1 * P)) return DKIND.CORE
  if (city.tier >= 1 && shape > 0.0 && shape < 0.4 && h2 < 0.6) {
    const a = Math.atan2(y, x)
    for (let i = 0; i < city.tier; i++) {
      const da = Math.abs(a - (hash2i(i, 3, city.seed) * 2 - 1) * Math.PI)
      if (Math.min(da, 2 * Math.PI - da) < IND_SPREAD) return DKIND.INDUSTRY
    }
  }
  if (r < 0.15 * city.radius && h < 0.3) return DKIND.CORE      // absorbed old villages
  if (shape < 0.3) return h < 0.75 ? DKIND.SUBURB : DKIND.GRID
  return city.tier >= 2 && h < 0.22 ? DKIND.SUPER : h > 0.8 ? DKIND.SUBURB : DKIND.GRID
}

// grid <-> district street coordinates
export function toDistrict(D, u, v) {
  const du = u - D.ou, dv = v - D.ov, p = du * D.c + dv * D.s, q = -du * D.s + dv * D.c
  return [p, q - D.A * Math.sin(D.k * p)]
}
export function fromDistrict(D, x, y) {
  const q = y + D.A * Math.sin(D.k * x)
  return [D.ou + x * D.c - q * D.s, D.ov + x * D.s + q * D.c]
}
// grid-space gradients of the district coords x and y at district x (for metres across rows / connectors)
const districtGrads = (D, x) => { const w = D.A * D.k * Math.cos(D.k * x); return [[D.c, D.s], [-D.s - w * D.c, D.c - w * D.s]] }

// Voronoi lookup: the district under grid (u, v) and its nearest edge: { D, nb (neighbour [i, j]), edge (lattice units),
// n (lattice unit normal towards the neighbour), mid (lattice point on the edge) }. 3x3 lattice cells, same order as GLSL
export function districtAt(city, u, v) {
  const P = districtPitch(city), [a, b] = toLattice(city, u, v), fi = Math.floor(a / P), fj = Math.floor(b / P), pts = []
  let best = 0, bd = Infinity
  for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
    const s = latticeSeed(city, fi + di, fj + dj, P), d = (s[0] - a) ** 2 + (s[1] - b) ** 2
    pts.push([fi + di, fj + dj, s[0], s[1]])
    if (d < bd) { bd = d; best = pts.length - 1 }
  }
  const [ci, cj, sa, sb] = pts[best]
  let edge = Infinity, nb = null, n = [1, 0], mid = [a, b]
  for (const [i, j, pa, pb] of pts) {
    if (i === ci && j === cj) continue
    const dx = pa - sa, dy = pb - sb, l = Math.hypot(dx, dy), mx = (pa + sa) / 2, my = (pb + sb) / 2
    const e = ((mx - a) * dx + (my - b) * dy) / l
    if (e < edge) { edge = e; nb = [i, j]; n = [dx / l, dy / l]; mid = [mx, my] }
  }
  return { D: district(city, ci, cj), nb, edge, n, mid, a, b }
}

// row line k of a district (district y), jittered by jit; row ky containing y: { ky, y0, y1 }
export const rowLine = (D, k) => (k + D.jit * (hash2i(k, D.key, D.hseed + 71) - 0.5)) * D.by + D.oy
export function rowAt(D, y) {
  const k0 = Math.floor((y - D.oy) / D.by), y0 = rowLine(D, k0), y1 = rowLine(D, k0 + 1)
  if (y < y0) return { ky: k0 - 1, y0: rowLine(D, k0 - 1), y1: y0 }
  if (y >= y1) return { ky: k0 + 1, y0: y1, y1: rowLine(D, k0 + 2) }
  return { ky: k0, y0, y1 }
}
// connectors of row ky: spacing and offset (connector kx at x = kx * bx + off)
export function rowConnectors(D, ky) {
  const bx = D.bx * (1 + D.varx * (hash2i(ky, D.key, D.hseed + 72) - 0.5))
  return { bx, off: D.ox + D.stagger * D.bx * hash2i(ky, D.key, D.hseed + 73) }
}
// block (kx, ky) of a district: street-centre rect in district coords { x0, x1, y0, y1 }
export function blockRectD(D, kx, ky) {
  const { bx, off } = rowConnectors(D, ky)
  return { x0: kx * bx + off, x1: (kx + 1) * bx + off, y0: rowLine(D, ky), y1: rowLine(D, ky + 1) }
}

// trunk streets: radial boulevards (grid-space rays from the centre) and the ring road
export function boulevards(city) {
  const n = [0, 2, 4, 6][city.tier], a0 = hash2i(city.seed, 1, 81) * 2 * Math.PI
  return Array.from({ length: n }, (_, i) => a0 + 2 * Math.PI * (i + 0.35 * (hash2i(i, 2, city.seed + 81) - 0.5)) / n)
}
export const BOULEVARD_END = 1.15                  // boulevards end this many radii out
// ring radius (grid units) at bearing th, and d(radius)/d(th); ring only for tier >= 2
export function ringRadius(city, th) {
  const h = n => hash2i(city.seed, n, 83) * 2 * Math.PI, R0 = city.radius * (0.4 + 0.1 * hash2i(city.seed, 0, 84))
  return [R0 * (1 + 0.1 * Math.sin(2 * th + h(1)) + 0.05 * Math.sin(3 * th + h(2))), R0 * (0.2 * Math.cos(2 * th + h(1)) + 0.15 * Math.cos(3 * th + h(2))), R0]
}

// Everything about the city layout at grid point (u, v) (see notes/layout.md):
// { D (district), x, y (district coords), kx, ky, rect (street-centre rect), lot (buildable rect), lots {nx, ny},
//   density, zone (1 industrial), park, plaza,
//   row { d (signed m), half, k }, con { d (signed m, to the connector segment; sign of the x offset), half, k, ky }, trunk { kind, d (signed m), half, s (m along) },
//   trunk2 { kind, d, half } (runner-up, for junctions) }. Street kinds SKIND, cross-sections profileOf(kind).
export function cityCell(city, u, v) {
  const at = districtAt(city, u, v), D = at.D, [x, y] = toDistrict(D, u, v), m = metresPer(city, u, v)
  const [gx, gy] = districtGrads(D, x), mx = m(gx[0], gx[1]), my = m(gy[0], gy[1])
  const row = rowAt(D, y), { bx, off } = rowConnectors(D, row.ky), kx = Math.floor((x - off) / bx)
  const half = profileOf(D.prof).half
  const rowNear = y - row.y0 < row.y1 - y ? { d: (y - row.y0) * my, k: row.ky } : { d: (y - row.y1) * my, k: row.ky + 1 }
  // connectors: this row's, and the adjacent row's across the nearest row line (they end at the row line: capsules)
  const con = (ky, y0, y1) => {
    const c = rowConnectors(D, ky), k = Math.floor((x - c.off) / c.bx + 0.5), dx = (x - (k * c.bx + c.off)) * mx
    const d = Math.hypot(dx, Math.max(0, y0 - y, y - y1) * my)
    return { d: dx < 0 ? -d : d, k, ky }
  }
  let cn = con(row.ky, row.y0, row.y1)
  const ca = rowNear.k === row.ky ? con(row.ky - 1, rowLine(D, row.ky - 1), row.y0) : con(row.ky + 1, row.y1, rowLine(D, row.ky + 2))
  if (Math.abs(ca.d) < Math.abs(cn.d)) cn = ca
  const trunks = trunkStreets(city, u, v, at, m).sort((p, q) => (Math.abs(p.d) - p.half) - (Math.abs(q.d) - q.half))
  const b = blockInfo(city, D, kx, row.ky, { x0: kx * bx + off, x1: (kx + 1) * bx + off, y0: row.y0, y1: row.y1 })
  return { D, x, y, kx, ky: row.ky, ...b, metric: [mx, my], row: { ...rowNear, half }, con: { d: cn.d, k: cn.k, ky: cn.ky, half }, trunk: trunks[0], trunk2: trunks[1] }
}
// trunk candidates at grid (u, v): the district edge, each boulevard, the ring
function trunkStreets(city, u, v, at, m) {
  const R1 = LAT_R1 * city.radius, r = Math.hypot(u, v), gu = r > 1e-3 ? u / r : 1, gv = r > 1e-3 ? v / r : 0
  // edge: gradient of the lattice-space distance, pulled back to grid space (J of the log-radial map is symmetric)
  const s = latScale(r, R1, false), s1 = 1 / (1 + r / R1), gn = (s1 - s) * (gu * at.n[0] + gv * at.n[1])
  const eg = [s * at.n[0] + gn * gu, s * at.n[1] + gn * gv]
  // signed and along consistently from both sides: the normal points from the lower district key to the higher
  const [mu, mv] = fromLattice(city, ...at.mid), tl = Math.hypot(eg[0], eg[1]), sg = at.D.key < dKey(...at.nb) ? 1 : -1
  const out = [{ kind: SKIND.EDGE, d: -sg * at.edge * m(eg[0], eg[1]), half: PROFILES[2].half, s: sg * ((u - mu) * -eg[1] + (v - mv) * eg[0]) / tl }]
  for (const a of boulevards(city)) {
    const ex = Math.cos(a), ey = Math.sin(a), t = u * ex + v * ey, cr = -u * ey + v * ex
    if (t > 0 && t < BOULEVARD_END * city.radius) out.push({ kind: SKIND.BOULEVARD, d: cr * m(-ey, ex), half: PROFILES[3].half, s: t })
  }
  if (city.tier >= 2) {
    const th = Math.atan2(v, u), [Rr, dR, R0] = ringRadius(city, th), f = r - Rr
    const g = [gu + dR / Math.max(r, 1) * gv, gv - dR / Math.max(r, 1) * gu]
    out.push({ kind: SKIND.RING, d: f * m(g[0], g[1]), half: PROFILES[3].half, s: th * R0 })
  }
  while (out.length < 2) out.push({ kind: SKIND.EDGE, d: 1e9, half: 0, s: 0 })
  return out
}

// block of a district: { centre [u, v], density, zone, park, plaza, lot (buildable rect, district coords), lots { nx, ny } }
// Lot (i, j) spans lot.x0 + (lot.x1 - lot.x0) * i / nx ..; lot hashes: hash2i(kx * 8 + i, ky * 8 + j, D.hseed + n)
export function blockInfo(city, D, kx, ky, rect = blockRectD(D, kx, ky)) {
  const centre = fromDistrict(D, (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
  const [x, y] = warpGrid(city, centre[0], centre[1]), d = cityShape(city, x, y)
  let density = d >= 0.2 ? d : 0
  if (d < 0.2 && d >= -0.15) {
    const f = smooth01((d + 0.15) / 0.35)   // ragged edge: clusters of blocks thinning out into the countryside
    density = hash2i(Math.floor(kx / 3), Math.floor(ky / 2), D.hseed + 5) < f * 1.15 && hash2i(kx, ky, D.hseed + 6) < 0.3 + f ? Math.max(d, 0.02) : 0
  }
  const zone = D.kind === DKIND.INDUSTRY && density > 0 ? 1 : 0
  const park = zone === 0 && hash2i(kx, ky, D.hseed + 7) < (D.kind === DKIND.SUPER ? 0.12 : D.kind === DKIND.CORE ? 0.03 : 0.07)
  const hw = profileOf(D.prof).half + 1
  const lot = { x0: rect.x0 + hw, x1: rect.x1 - hw, y0: rect.y0 + hw, y1: rect.y1 - hw }
  return { centre, density, zone, park, plaza: park && D.kind === DKIND.CORE, rect, lot, lots: blockLotsD(D, kx, ky, lot) }
}
const LOT_W = [22, 34, 55, 24, 60]                 // typical lot width per district kind (m)
export function blockLotsD(D, kx, ky, lot) {
  const w = LOT_W[D.kind] * (0.8 + 0.5 * hash2i(kx, ky, D.hseed + 11)), dep = D.kind === DKIND.SUBURB ? 1e9 : w * (1 + 0.6 * hash2i(kx, ky, D.hseed + 13))
  return {
    nx: Math.min(6, Math.max(1, Math.floor((lot.x1 - lot.x0) / w))),
    ny: D.kind === DKIND.SUBURB ? 2 : Math.min(4, Math.max(1, Math.floor((lot.y1 - lot.y0) / dep))),
  }
}

// ---- enumeration for JS consumers (grid coords; gridToDir / gridAxes / districtAxes map them to the world) ----
// districts whose cells may reach within radius of grid (u, v)
export function districtsNear(city, u, v, radius) {
  const P = districtPitch(city), r = Math.hypot(u, v), R1 = LAT_R1 * city.radius
  const [a, b] = toLattice(city, u, v), rl = radius / (1 + Math.max(0, r - radius) / R1) + 1.2 * P
  const out = []
  for (let j = Math.floor((b - rl) / P); j <= Math.floor((b + rl) / P); j++)
    for (let i = Math.floor((a - rl) / P); i <= Math.floor((a + rl) / P); i++) out.push(district(city, i, j))
  return out
}
// blocks with their centre within radius of grid (u, v) that are built up (density > 0) or all (all = true):
// [{ D, kx, ky, key, ...blockInfo, dir, angle (district x axis against the grid u axis) }]
export function blocksNear(city, u, v, radius, all = false) {
  const out = []
  for (const D of districtsNear(city, u, v, radius)) {
    const [x, y] = toDistrict(D, u, v), m = Math.abs(D.A) + D.by
    for (let ky = Math.floor((y - radius - m - D.oy) / D.by); ky <= Math.floor((y + radius + m - D.oy) / D.by); ky++) {
      const { bx, off } = rowConnectors(D, ky)
      for (let kx = Math.floor((x - radius - off) / bx) - 1; kx <= Math.floor((x + radius - off) / bx) + 1; kx++) {
        const b = blockInfo(city, D, kx, ky)
        if (Math.hypot(b.centre[0] - u, b.centre[1] - v) > radius || (!all && b.density <= 0)) continue
        if (districtAt(city, b.centre[0], b.centre[1]).D !== D) continue      // centre outside this district
        out.push({ D, kx, ky, key: `${D.key}:${kx}:${ky}`, ...b, dir: gridToDir(city, ...b.centre), angle: D.angle })
      }
    }
  }
  return out
}
// world tangents of district +x / +y at district (x, y), metres per unit along each (like gridAxes)
export function districtAxes(city, D, x, y) {
  const [u, v] = fromDistrict(D, x, y), q = D.A * D.k * Math.cos(D.k * x)
  const G = gridAxes(city, u, v), [ex, ny] = frameOf(city), [, , a, b, c, d] = warpGrid(city, u, v)
  const tx = [D.c - q * D.s, D.s + q * D.c], ty = [-D.s, D.c]           // grid-space images of +x / +y
  const map = t => { const lx = a * t[0] + b * t[1], ly = c * t[0] + d * t[1], l = Math.hypot(lx, ly); return [ex.map((e, i) => (e * lx + ny[i] * ly) / l), l] }
  const [ex2, sx] = map(tx), [ey2, sy] = map(ty)
  return { ex: ex2, ey: ey2, sx, sy, grid: G }
}
// street spans near grid (u, v): [{ kind (SKIND), half, pts: [[u, v], ...] (centre line, <= 25 m apart), D?, k? }]
// local rows / connectors per block side (the row below and the connector left of each block), district edges
// (each once), boulevards and the ring, clipped to radius (spans end at T-junctions, so they overlap trunks a little)
export function streetsNear(city, u, v, radius) {
  const out = [], seen = new Set(), inR = p => Math.hypot(p[0] - u, p[1] - v) <= radius
  const poly = (f, len, kind, extra) => {
    const n = Math.max(1, Math.ceil(len / 25)), pts = []
    for (let i = 0; i <= n; i++) pts.push(f(i / n))
    if (pts.some(inR)) out.push({ kind, half: profileOf(kind).half, pts, ...extra })
  }
  for (const b of blocksNear(city, u, v, radius + 400, true)) {
    const { D, rect: r } = b
    if (b.density <= 0) continue
    poly(t => fromDistrict(D, r.x0 + (r.x1 - r.x0) * t, r.y0), r.x1 - r.x0, D.prof, { D, axis: 0, k: b.ky, kx: b.kx })
    poly(t => fromDistrict(D, r.x0, r.y0 + (r.y1 - r.y0) * t), r.y1 - r.y0, D.prof, { D, axis: 1, k: b.kx, ky: b.ky })
  }
  for (const D of districtsNear(city, u, v, radius)) for (const e of districtEdges(city, D)) {
    if (seen.has(e.id)) continue
    seen.add(e.id)
    const len = Math.hypot(...[0, 1].map(i => e.b[i] - e.a[i])) * 3
    poly(t => fromLattice(city, e.a[0] + (e.b[0] - e.a[0]) * t, e.a[1] + (e.b[1] - e.a[1]) * t), len, SKIND.EDGE, { D, nb: e.nb })
  }
  for (const a of boulevards(city)) poly(t => [Math.cos(a) * t * BOULEVARD_END * city.radius, Math.sin(a) * t * BOULEVARD_END * city.radius], BOULEVARD_END * city.radius, SKIND.BOULEVARD, {})
  if (city.tier >= 2) poly(t => { const th = t * 2 * Math.PI, r = ringRadius(city, th)[0]; return [Math.cos(th) * r, Math.sin(th) * r] }, 2 * Math.PI * city.radius * 0.5, SKIND.RING, {})
  return out
}
// Voronoi cell polygon of a district in lattice space, as edges [{ id, a, b (lattice points), nb (neighbour [i, j]) }]
// (Sutherland-Hodgman against the bisectors of the 5x5 neighbours; each vertex carries the neighbour of its outgoing edge)
export function districtEdges(city, D) {
  if (D._edges) return D._edges
  const P = districtPitch(city), s = latticeSeed(city, D.i, D.j, P)
  let poly = [[-3, -3], [3, -3], [3, 3], [-3, 3]].map(([x, y]) => [s[0] + x * P, s[1] + y * P, null])
  for (let dj = -2; dj <= 2; dj++) for (let di = -2; di <= 2; di++) {
    if (!di && !dj) continue
    const q = latticeSeed(city, D.i + di, D.j + dj, P), n = [q[0] - s[0], q[1] - s[1]], c = ((q[0] + s[0]) * n[0] + (q[1] + s[1]) * n[1]) / 2
    const f = p => p[0] * n[0] + p[1] * n[1] - c, nb = [D.i + di, D.j + dj], next = []
    poly.forEach((p, i) => {
      const qn = poly[(i + 1) % poly.length], fp = f(p), fq = f(qn), cut = () => { const t = fp / (fp - fq); return [p[0] + (qn[0] - p[0]) * t, p[1] + (qn[1] - p[1]) * t] }
      if (fp <= 0) { next.push(p); if (fq > 0) next.push([...cut(), nb]) }
      else if (fq <= 0) next.push([...cut(), p[2]])
    })
    poly = next
  }
  return (D._edges = poly.map((p, i) => {
    const q = poly[(i + 1) % poly.length], nb = p[2], k = nb ? dKey(nb[0], nb[1]) : 0
    return { id: `${Math.min(D.key, k)}|${Math.max(D.key, k)}`, a: [p[0], p[1]], b: [q[0], q[1]], nb }
  }).filter(e => e.nb))
}

export const SPEC_GLSL = /* glsl */`
#define CIV_HIGHWAY 0
#define CIV_ROAD 1
#define CIV_RAIL 2
const float CIV_HALF[3] = float[3](${HALF.map(v => v.toFixed(2)).join(', ')});
#define RAIL_GAUGE ${RAIL_GAUGE.toFixed(4)}
#define STREET_HALF ${STREET.half.toFixed(2)}
#define AVENUE_HALF ${STREET.avenueHalf.toFixed(2)}
#define AVENUE_EVERY ${STREET.avenueEvery}
#define GRID_W ${GRID_W}
#define GRID_H ${GRID_H}
#define CIV_ROW ${ROW}u
#define CITY_BIT ${CITY_BIT}u
float hash2i(int x, int y, int seed){
  uint h = (uint(x)*0x27d4eb2du) ^ (uint(y)*0x165667b1u) ^ (uint(seed)*0x9e3779b1u);
  h = (h ^ (h >> 15u))*0x85ebca6bu;
  h = (h ^ (h >> 13u))*0xc2b2ae35u;
  return float(h ^ (h >> 16u))/4294967296.0;
}
// organic layout, identical to spec.js (warpTerms / warpGrid / localToGrid / cityShape / blockDensity / blockZone).
// Compile time: ANGLE's D3D compiler inlines every call and unrolls constant-bound loops, so the per-city hashes and
// the warp terms run in loops bounded by n = N + int(min(rad, 0.0)) (rad > 0: n = N, but fxc can't know), which it
// keeps as real loops; each hash / warp term / warpGrid then has one copy per lookup instead of 5-18.
// city outline (cityShape): its per-city hashes h0..h6, computed once per city
struct CityOutline { vec3 h; vec4 dir; vec2 ph; float rad; };   // h0..h2; (cos h3, sin h3, cos h5, sin h5); (h4, h6)
CityOutline cityOutline(int seed, float rad){
  float h[7];
  for (int i = 0; i < 7 + int(min(rad, 0.0)); i++) h[i] = hash2i(seed, i, 97)*2.0*PI;
  CityOutline o;
  o.h = vec3(h[0], h[1], h[2]); o.dir = vec4(cos(h[3]), sin(h[3]), cos(h[5]), sin(h[5])); o.ph = vec2(h[4], h[6]); o.rad = rad;
  return o;
}
float cityShape(CityOutline o, vec2 x){
  float th = atan(x.y, x.x), r = length(x)/o.rad;
  float s = 1.0 + 0.30*sin(2.0*th + o.h.x) + 0.22*sin(3.0*th + o.h.y) + 0.12*sin(5.0*th + o.h.z);
  s *= 1.0 + 0.18*sin(dot(x, o.dir.xy)/o.rad*3.1 + o.ph.x) + 0.12*sin(dot(x, o.dir.zw)/o.rad*5.3 + o.ph.y);
  return 1.0 - r/(0.8*s);
}
float cityShape(vec2 x, float rad, int seed){ return cityShape(cityOutline(seed, rad), x); }
struct CityWarp { vec2 k[5]; vec2 n[5]; vec3 ap[5]; float rad; int nt; CityOutline o; };   // ap = (amplitude, phase, core); nt = 5 (loop bound)
CityWarp cityWarp(int seed, float rad, CityOutline o){
  CityWarp w; w.rad = rad; w.o = o;
  w.nt = 5 + int(min(rad, 0.0));
  float scale = clamp(rad/12000.0, 0.3, 1.0);
  vec2 kv[5], nv[5]; vec3 ap[5];   // local arrays: fxc can't index struct arrays as l-values in a real loop
  for (int i = 0; i < w.nt; i++){
    bool core = i >= 3;
    float ang = hash2i(seed, i, 91)*2.0*PI;
    float k = 2.0*PI/(core ? 380.0 + 420.0*hash2i(seed, i, 92) : (2200.0 + 4500.0*hash2i(seed, i, 92))*scale);
    kv[i] = vec2(cos(ang), sin(ang))*k; nv[i] = vec2(-sin(ang), cos(ang));
    ap[i] = vec3((core ? ${WARP_CORE.toFixed(3)} : ${WARP_DIST.toFixed(3)})/k, hash2i(seed, i, 93)*2.0*PI, core ? 1.0 : 0.0);
  }
  w.k = kv; w.n = nv; w.ap = ap;
  return w;
}
// grid point q -> local metres; J = d(local)/d(grid), columns = images of the grid axes
vec2 warpGrid(CityWarp w, vec2 q, out mat2 J){
  float r = length(q)/w.rad;
  float wCore = exp(-(r/0.55)*(r/0.55)), wDist = 1.0 - 0.65*smoothstep(0.0, 1.0, (r - 0.5)/0.6);
  vec2 x = q; J = mat2(1.0);
  for (int i = 0; i < w.nt; i++){
    float a = w.ap[i].x*(w.ap[i].z > 0.5 ? wCore : wDist), ph = dot(w.k[i], q) + w.ap[i].y;
    x += a*sin(ph)*w.n[i];
    J += a*cos(ph)*outerProduct(w.n[i], w.k[i]);
  }
  return x;
}
// 4 Newton steps, then J at the result: 5 warpGrid evaluations through one call site
vec2 localToGrid(CityWarp w, vec2 x, out mat2 J){
  vec2 q = x;
  J = mat2(1.0);
  for (int i = 0; i < w.nt; i++){
    vec2 g = warpGrid(w, q, J);
    if (i < 4) q -= inverse(J)*(g - x);
  }
  return q;
}
float blockDensityG(CityWarp w, int kx, int ky, float B, int seed, out vec2 centre){
  mat2 J;
  centre = warpGrid(w, (vec2(kx, ky) + 0.5)*B, J);
  float d = cityShape(w.o, centre);
  if (d >= 0.2) return d;
  if (d < -0.15) return 0.0;
  float f = smoothstep(0.0, 1.0, (d + 0.15)/0.35);
  bool keep = hash2i(int(floor(float(kx)/3.0)), int(floor(float(ky)/3.0)), seed + 5) < f*1.15 && hash2i(kx, ky, seed + 6) < 0.3 + f;
  return keep ? max(d, 0.02) : 0.0;
}
int blockZoneG(vec2 centre, int kx, int ky, float rad, int seed, int tier){
  if (tier < 1) return 0;
  float d = cityShape(centre, rad, seed);
  if (d > 0.42 || d < 0.02 || hash2i(int(floor(float(kx)/2.0)), int(floor(float(ky)/2.0)), seed + 17) > 0.8) return 0;
  float a = atan(centre.y, centre.x);
  for (int i = 0; i < tier; i++){
    float da = abs(a - (hash2i(i, 3, seed)*2.0 - 1.0)*PI);
    if (min(da, 2.0*PI - da) < ${SECTOR.toFixed(3)}) return 1;
  }
  return 0;
}
bool isAvenue(int k){ return (k - AVENUE_EVERY*int(floor(float(k)/float(AVENUE_EVERY)))) == 0; }
float streetHalf(int k){ return isAvenue(k) ? AVENUE_HALF : STREET_HALF; }
// same as blockLots() in spec.js: rect (x0, y0, x1, y1) and lot counts
vec4 blockRect(int kx, int ky, float B){
  return vec4(float(kx)*B + streetHalf(kx) + 1.0, float(ky)*B + streetHalf(ky) + 1.0,
              float(kx + 1)*B - streetHalf(kx + 1) - 1.0, float(ky + 1)*B - streetHalf(ky + 1) - 1.0);
}
ivec2 blockLotCount(int kx, int ky, int seed){
  return ivec2(1 + int(floor(hash2i(kx, ky, seed + 11)*3.0)), 1 + int(floor(hash2i(kx, ky, seed + 13)*3.0)));
}
// ---- districts (round 5), identical to spec.js district / districtAt / rowAt / rowConnectors / blockInfo / trunkStreets ----
#define DK_CORE 0
#define DK_GRID 1
#define DK_SUPER 2
#define DK_SUBURB 3
#define DK_INDUSTRY 4
#define SK_LANE 0
#define SK_STREET 1
#define SK_EDGE 2
#define SK_BOULEVARD 3
#define SK_RING 4
#define LAT_R1 ${LAT_R1.toFixed(3)}
#define BOULEVARD_END ${BOULEVARD_END.toFixed(3)}
const float PROF_HALF[4] = float[4](${PROFILES.map(p => p.half.toFixed(2)).join(', ')});
const vec2 PROF_LANES[4] = vec2[4](${PROFILES.map(p => `vec2(${p.lanes.map(v => v.toFixed(2)).join(', ')})`).join(', ')});
const vec2 PROF_WALK[4] = vec2[4](${PROFILES.map(p => `vec2(${p.walk.map(v => v.toFixed(2)).join(', ')})`).join(', ')});
const float PROF_MEDIAN[4] = float[4](${PROFILES.map(p => p.median.toFixed(2)).join(', ')});
const float LOT_W[5] = float[5](${LOT_W.map(v => v.toFixed(1)).join(', ')});
int profileOf(int kind){ return min(kind, 3); }
float districtPitch(float rad){ return clamp(200.0 + 0.05*rad, 250.0, 750.0); }
float latScale(float r, float R1, bool inv){ return r < 1e-3 ? 1.0 : (inv ? R1*(exp(r/R1) - 1.0) : R1*log(1.0 + r/R1))/r; }
vec2 latticeSeed(ivec2 c, int seed, float P){ return (vec2(c) + 0.2 + 0.6*vec2(hash2i(c.x, c.y, seed + 61), hash2i(c.x, c.y, seed + 62)))*P; }
struct District { ivec2 id; int key, kind, hseed, prof; vec2 o; float c, s, A, k, bx, by, ox, oy, stagger, jit, varx; };
int districtKind(vec2 x, float shape, float r, float P, float h, float h2, float rad, int seed, int tier){
  if (r < max(${CORE_R.toFixed(3)}*rad, 1.1*P)) return DK_CORE;
  if (tier >= 1 && shape > 0.0 && shape < 0.4 && h2 < 0.6){
    float a = atan(x.y, x.x);
    for (int i = 0; i < tier; i++){
      float da = abs(a - (hash2i(i, 3, seed)*2.0 - 1.0)*PI);
      if (min(da, 2.0*PI - da) < ${IND_SPREAD.toFixed(3)}) return DK_INDUSTRY;
    }
  }
  if (r < 0.15*rad && h < 0.3) return DK_CORE;
  if (shape < 0.3) return h < 0.75 ? DK_SUBURB : DK_GRID;
  return tier >= 2 && h < 0.22 ? DK_SUPER : h > 0.8 ? DK_SUBURB : DK_GRID;
}
District district(ivec2 id, CityWarp w, int seed, int tier, float B){
  District D;
  float P = districtPitch(w.rad);
  D.id = id; D.key = id.x*1024 + id.y; D.hseed = seed + D.key*131;
  float h[12];                                   // hash2i(id, seed + 61 + n): 61, 62 latticeSeed, 63.. below
  for (int n = 0; n < 12 + int(min(w.rad, 0.0)); n++) h[n] = hash2i(id.x, id.y, seed + 61 + n);
  vec2 lat = (vec2(id) + 0.2 + 0.6*vec2(h[0], h[1]))*P;   // latticeSeed(id, seed, P)
  D.o = lat*latScale(length(lat), LAT_R1*w.rad, true);
  mat2 J;
  vec2 x = warpGrid(w, D.o, J);
  float h0 = h[2], h1 = h[3], h2 = h[4];
  float h3 = h[5], h4 = h[6], h5 = h[7];
  D.kind = districtKind(x, cityShape(w.o, x), length(D.o), P, h0, h[11], w.rad, seed, tier);
  float ang = h[8]*PI;
  D.c = cos(ang); D.s = sin(ang);
  D.A = 0.0; D.k = 0.0; D.stagger = 0.0; D.jit = 0.0; D.varx = 0.0; D.prof = SK_STREET;
  if (D.kind == DK_CORE){ D.by = 45.0 + 30.0*h1; D.bx = 55.0 + 40.0*h2; D.stagger = 1.0; D.jit = 0.35; D.varx = 0.5; D.prof = SK_LANE; D.A = 6.0 + 10.0*h3; D.k = 2.0*PI/(180.0 + 200.0*h4); }
  else if (D.kind == DK_GRID){
    float Bd = B*(0.85 + 0.3*h1);
    D.by = Bd; D.bx = Bd*(1.0 + 1.8*h2); D.stagger = h3 < 0.35 ? 0.5 : 0.0;
    if (h4 < 0.3){ D.A = Bd*0.3; D.k = 2.0*PI/(1200.0 + 1500.0*h5); }
  } else if (D.kind == DK_SUPER){ D.by = 200.0 + 120.0*h1; D.bx = 240.0 + 140.0*h2; D.jit = 0.1; D.prof = SK_EDGE; }
  else if (D.kind == DK_SUBURB){
    float lam = 350.0 + 450.0*h4;
    D.by = 70.0 + 25.0*h1; D.bx = 170.0 + 150.0*h2; D.stagger = 1.0; D.jit = 0.15; D.varx = 0.4; D.A = lam*(0.04 + 0.07*h3); D.k = 2.0*PI/lam;
  } else { D.by = 130.0 + 90.0*h1; D.bx = 170.0 + 120.0*h2; D.stagger = 0.5; D.jit = 0.1; }
  D.ox = h[9]*D.bx; D.oy = h[10]*D.by;
  return D;
}
// Voronoi district under grid point g (3x3 lattice cells): id; edge = lattice distance to the nearest bisector,
// en its normal (towards neighbour nb), em its midpoint
ivec2 districtAt(vec2 g, float rad, int seed, out float edge, out vec2 en, out vec2 em, out ivec2 nb){
  float P = districtPitch(rad);
  vec2 q = g*latScale(length(g), LAT_R1*rad, false);
  ivec2 f = ivec2(floor(q/P)), best = f;
  vec2 sa = vec2(0.0), S[9];
  float bd = 1e30;
  int n9 = 9 + int(min(rad, 0.0));               // real loops (see cityWarp), seeds kept for the second pass
  for (int k = 0; k < n9; k++){
    ivec2 id = f + ivec2(k%3 - 1, k/3 - 1);
    vec2 s = latticeSeed(id, seed, P);
    S[k] = s;
    float d = dot(s - q, s - q);
    if (d < bd){ bd = d; best = id; sa = s; }
  }
  edge = 1e30; en = vec2(1.0, 0.0); em = q; nb = best;
  for (int k = 0; k < n9; k++){
    ivec2 id = f + ivec2(k%3 - 1, k/3 - 1);
    if (id == best) continue;
    vec2 s = S[k], n = normalize(s - sa), m = 0.5*(s + sa);
    float e = dot(m - q, n);
    if (e < edge){ edge = e; nb = id; en = n; em = m; }
  }
  return best;
}
vec2 toDistrict(District D, vec2 g){
  vec2 d = g - D.o;
  float p = d.x*D.c + d.y*D.s, q = -d.x*D.s + d.y*D.c;
  return vec2(p, q - D.A*sin(D.k*p));
}
vec2 fromDistrict(District D, vec2 l){
  float q = l.y + D.A*sin(D.k*l.x);
  return D.o + vec2(l.x*D.c - q*D.s, l.x*D.s + q*D.c);
}
float rowLine(District D, int k){ return (float(k) + D.jit*(hash2i(k, D.key, D.hseed + 71) - 0.5))*D.by + D.oy; }
int rowAt(District D, float y, out float y0, out float y1){
  int k0 = int(floor((y - D.oy)/D.by));
  y0 = rowLine(D, k0); y1 = rowLine(D, k0 + 1);
  if (y < y0){ y1 = y0; y0 = rowLine(D, k0 - 1); return k0 - 1; }
  if (y >= y1){ y0 = y1; y1 = rowLine(D, k0 + 2); return k0 + 1; }
  return k0;
}
vec2 rowConnectors(District D, int ky){   // (spacing, offset)
  return vec2(D.bx*(1.0 + D.varx*(hash2i(ky, D.key, D.hseed + 72) - 0.5)), D.ox + D.stagger*D.bx*hash2i(ky, D.key, D.hseed + 73));
}
// block density from its centre (grid coords); same as blockInfo() in spec.js
float blockDensityD(CityWarp w, vec2 centre, int kx, int ky, int hseed, int seed){
  mat2 J;
  float d = cityShape(w.o, warpGrid(w, centre, J));
  if (d >= 0.2) return d;
  if (d < -0.15) return 0.0;
  float f = smoothstep(0.0, 1.0, (d + 0.15)/0.35);
  bool keep = hash2i(int(floor(float(kx)/3.0)), int(floor(float(ky)/2.0)), hseed + 5) < f*1.15 && hash2i(kx, ky, hseed + 6) < 0.3 + f;
  return keep ? max(d, 0.02) : 0.0;
}
ivec2 blockLotsD(int kind, int kx, int ky, int hseed, vec4 lot){
  float w = LOT_W[kind]*(0.8 + 0.5*hash2i(kx, ky, hseed + 11)), dep = w*(1.0 + 0.6*hash2i(kx, ky, hseed + 13));
  return ivec2(clamp(int(floor((lot.z - lot.x)/w)), 1, 6), kind == DK_SUBURB ? 2 : clamp(int(floor((lot.w - lot.y)/dep)), 1, 4));
}
`

// ---- rivers and lakes (hydro.js builds them, civ.glsl.js carves and looks them up) ----
export const RIVER_BIT = 0x40000000            // item ids with this bit (and not CITY_BIT) are river segments
export const HYDRO = {
  cell: 0.15,                                  // flood grid / lake map cell (degrees)
  lakeF: 400,                                  // band limit of the flood grid samples; lake beds are smoothed to it
  bedF: R_PLANET / 1500,                       // band limit of the terrain the river levels follow (same as road beds)
}
// river cross-section from the half width hw (m): channel depth, floodplain half width beyond the banks
export const riverDepth = hw => 0.4 + 0.25 * Math.pow(2 * hw, 0.6)
export const riverPlain = hw => hw * 1.5 + 30
export const RIVER_REACH = hw => 1.9 * hw + riverPlain(hw) + 360   // farthest a (meandering) river bed changes the terrain

export const HYDRO_GLSL = /* glsl */`
#define RIVER_BIT ${RIVER_BIT}u
#define LAKE_F ${HYDRO.lakeF.toFixed(1)}
#define RIVER_BED_F ${HYDRO.bedF.toFixed(1)}
float riverDepth(float hw){ return 0.4 + 0.25*pow(2.0*hw, 0.6); }
float riverPlain(float hw){ return hw*1.5 + 30.0; }
`
// meanders: the channel centre swings off the stored polyline by up to 0.9 half widths (so tributaries,
// which end on the trunk's polyline, still meet its water), easing to 0 over the last 3 km before a river ends
export function riverMeander(s, hw, left) {
  const lam = 14 * 2 * hw + 300
  return 0.6 * hw * Math.min(1, left / 3000) * (Math.sin(2 * Math.PI * s / lam) + 0.5 * Math.sin(2 * Math.PI * s / (0.37 * lam) + 1.7))
}
export const RIVER_MEANDER_GLSL = /* glsl */`
float riverMeander(float s, float hw, float left){
  float lam = 28.0*hw + 300.0;
  return 0.6*hw*min(1.0, left/3000.0)*(sin(2.0*PI*s/lam) + 0.5*sin(2.0*PI*s/(0.37*lam) + 1.7));
}
`

// Bridges (civ.glsl.js civBed <-> bridges.js). Where a road or railway crosses a river chord with |sine| >= sin, its bed
// descends towards the river at grade[type] (m per m off the channel, from floodplain + ramp m out) to bridgeClear(hw)
// over the floodplain (or as far as it gets by the floodplain edge, level from there), never rising; then it gives way to the natural (river-carved) ground over the channel and
// wherever that ground lies more than e0 m below the bed, out to reach m beyond the floodplain. bridges.js lists the
// same crossings, samples terrain() along the road to find where the bed stopped (open: bed - ground > open m) and
// spans it. In towns (city ground smoothing flatW >= town) beds don't descend and only the channel opens: beds there
// are band-limited differently, so bridges.js can't sample them. ramp is bounded by the river lookup reach (RIVER_REACH).
export const BRIDGE = { sin: 0.2, e0: 3.0, reach: 400, open: 0.3, town: 0.02, ramp: 250, grade: [0.06, 0.08, 0.035] }
export const bridgeClear = hw => Math.min(30, Math.max(10, 8 + 0.08 * hw))    // descended bed over the water (m)
export const BRIDGE_GLSL = /* glsl */`
#define BRIDGE_SIN ${BRIDGE.sin.toFixed(3)}
#define BRIDGE_E0 ${BRIDGE.e0.toFixed(2)}
#define BRIDGE_REACH ${BRIDGE.reach.toFixed(1)}
#define BRIDGE_TOWN ${BRIDGE.town.toFixed(3)}
#define BRIDGE_RAMP ${BRIDGE.ramp.toFixed(1)}
const float BRIDGE_GRADE[3] = float[3](${BRIDGE.grade.map(g => g.toFixed(3)).join(', ')});
float bridgeClear(float hw){ return clamp(8.0 + 0.08*hw, 10.0, 30.0); }
`
