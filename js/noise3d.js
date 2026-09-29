// Tileable 3D noise volume for cloud shapes, built once on the CPU.
// R: Perlin-Worley (billowy base), G: Worley fbm (medium erosion), B: Worley fbm (fine), A: gradient (Perlin) fbm.
import * as THREE from 'three'

function hash(i, j, k, s) {
  let h = Math.imul(i, 374761393) ^ Math.imul(j, 668265263) ^ Math.imul(k, 2147483647) ^ Math.imul(s, 144665)
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}
const wrap = (i, n) => ((i % n) + n) % n

function worleyPoints(F, seed) {
  const pts = new Float32Array(F * F * F * 3)
  for (let k = 0; k < F; k++) for (let j = 0; j < F; j++) for (let i = 0; i < F; i++) {
    const o = ((k * F + j) * F + i) * 3
    pts[o] = hash(i, j, k, seed); pts[o + 1] = hash(i, j, k, seed + 1); pts[o + 2] = hash(i, j, k, seed + 2)
  }
  return pts
}

// 1 at feature points, falling to 0 a cell away
function worley(x, y, z, F, pts) {
  const px = x * F, py = y * F, pz = z * F
  const ci = Math.floor(px), cj = Math.floor(py), ck = Math.floor(pz)
  let best = 9
  for (let dk = -1; dk <= 1; dk++) for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
    const i = ci + di, j = cj + dj, k = ck + dk
    const o = ((wrap(k, F) * F + wrap(j, F)) * F + wrap(i, F)) * 3
    const ex = i + pts[o] - px, ey = j + pts[o + 1] - py, ez = k + pts[o + 2] - pz
    const d = ex * ex + ey * ey + ez * ez
    if (d < best) best = d
  }
  return 1 - Math.min(1, Math.sqrt(best))
}

// periodic Perlin gradient noise (about -0.45..0.45): unlike value noise it has no lattice-aligned blocks,
// which read as square cloud edges once thresholded
function gradNoise(x, y, z, F, seed) {
  const px = x * F, py = y * F, pz = z * F
  const i = Math.floor(px), j = Math.floor(py), k = Math.floor(pz)
  const fx = px - i, fy = py - j, fz = pz - k
  const g = (a, b, c) => {
    const A = wrap(i + a, F), B = wrap(j + b, F), C = wrap(k + c, F)
    const th = 2 * Math.PI * hash(A, B, C, seed), gz = 2 * hash(A, B, C, seed + 1) - 1, r = Math.sqrt(1 - gz * gz)
    return r * Math.cos(th) * (fx - a) + r * Math.sin(th) * (fy - b) + gz * (fz - c)
  }
  const s = t => t * t * t * (t * (t * 6 - 15) + 10)
  const l = (a, b, t) => a + (b - a) * t
  const u = s(fx), v = s(fy), w = s(fz)
  return l(l(l(g(0, 0, 0), g(1, 0, 0), u), l(g(0, 1, 0), g(1, 1, 0), u), v),
    l(l(g(0, 0, 1), g(1, 0, 1), u), l(g(0, 1, 1), g(1, 1, 1), u), v), w)
}

export function makeCloudNoise(S = 64) {
  const data = new Uint8Array(S * S * S * 4)
  const W = [4, 8, 16, 32].map((F, n) => [F, worleyPoints(F, 11 + n * 7)])
  for (let z = 0; z < S; z++) for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const u = x / S, v = y / S, w = z / S
    const [w4, w8, w16, w32] = W.map(([F, p]) => worley(u, v, w, F, p))
    // same mean (0.5) and spread (sd 0.12) as the value-noise fbm it replaced, so the shader thresholds still hold
    const perlin = 0.5 + gradNoise(u, v, w, 4, 1) * 0.5 + gradNoise(u, v, w, 8, 2) * 0.3 + gradNoise(u, v, w, 16, 3) * 0.2
    const wf = w4 * 0.625 + w8 * 0.25 + w16 * 0.125
    const pw = Math.min(1, Math.max(0, (perlin - (1 - wf)) / Math.max(wf, 0.05) * 0.6 + wf * 0.55))
    const o = ((z * S + y) * S + x) * 4
    data[o] = pw * 255
    data[o + 1] = (w8 * 0.625 + w16 * 0.25 + w32 * 0.125) * 255
    data[o + 2] = (w16 * 0.7 + w32 * 0.3) * 255
    data[o + 3] = perlin * 255
  }
  const tex = new THREE.Data3DTexture(data, S, S, S)
  tex.format = THREE.RGBAFormat
  tex.type = THREE.UnsignedByteType
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.RepeatWrapping
  tex.minFilter = THREE.LinearMipmapLinearFilter
  tex.magFilter = THREE.LinearFilter
  tex.generateMipmaps = true
  tex.needsUpdate = true
  return tex
}
