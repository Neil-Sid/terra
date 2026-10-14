// 3D trees: instanced low-poly conifers, broadleaf trees (also shrubs and acacias) and palms within ~2 km of the
// camera. Candidates sit on a stable jittered grid (an equiangular cube-sphere quadtree); a GPU placement pass decides
// which grow (trees.glsl.js), so no readback. Also feeds the canopy painting its precision and 3D-range uniforms.
// Notes: notes/forests.md; contracts in notes/CIV.md.
import * as THREE from 'three'
import { R } from '../config.js'
import { hash2i } from './spec.js'
import { objUniforms } from './objects.glsl.js'
import { FOREST_CELL } from '../glsl/forest.glsl.js'
import { PLACE_W, PLACE_FRAG, treeVert, TREE_FRAG } from './trees.glsl.js'

const R0 = 250                 // every tree drawn within R0; beyond, a tree of rank r shows while d < R0/sqrt(r)
const DMAX = 2500              // no 3D trees past this (all dithered out over its second half)
const LMAX = 3                 // quadtree levels of thinning (R0 * 2^(LMAX+1) >= DMAX)
const DEPTH = 20               // level-0 cells: 2^20 per cube face edge (~7-9.5 m)
const MARGIN = 120             // candidates reach this far past where they show; rebuild after moving half of it
const MAX_INST = PLACE_W * 256
const SEED = 4711

// cube faces: normal F and in-face axes U, V (as terrain.js)
const FACES = [
  [[1, 0, 0], [0, 0, -1], [0, 1, 0]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
  [[0, 1, 0], [1, 0, 0], [0, 0, -1]], [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
  [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
]
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
// equiangular face coordinates (a, b in -1..1) <-> unit direction
function faceDir(f, a, b) {
  const [F, U, V] = FACES[f], u = Math.tan(a * Math.PI / 4), v = Math.tan(b * Math.PI / 4)
  const x = F[0] + u * U[0] + v * V[0], y = F[1] + u * U[1] + v * V[1], z = F[2] + u * U[2] + v * V[2], l = Math.hypot(x, y, z)
  return [x / l, y / l, z / l]
}
function faceCoords(f, d) {
  const [F, U, V] = FACES[f], k = dot(d, F)
  return [Math.atan(dot(d, U) / k) * 4 / Math.PI, Math.atan(dot(d, V) / k) * 4 / Math.PI]
}
// level-0 cell of the tree a level-L cell promotes (one child per level, by hash)
function promoted(f, i, j, L) {
  for (let l = L; l > 0; l--) {
    const q = Math.floor(hash2i(i, j, SEED + f * 32 + l) * 4)
    i = 2 * i + (q & 1); j = 2 * j + (q >> 1)
  }
  return [i, j]
}
const isPromoted = (f, i, j, L) => { const q = Math.floor(hash2i(i >> 1, j >> 1, SEED + f * 32 + L + 1) * 4); return (i & 1) === (q & 1) && (j & 1) === (q >> 1) }

// ---- meshes: unit trees (height 1, z up); part = (0 trunk / 1 foliage, height 0..1) ----
function builder() {
  const pos = [], nor = [], part = [], idx = []
  const v = (p, n, foliage) => { pos.push(...p); nor.push(...n); part.push(foliage, Math.max(0, Math.min(1, p[2]))); return pos.length / 3 - 1 }
  // triangle wound so its face normal agrees with the vertex normals (outward)
  const tri = (a, b, c) => {
    const P = k => pos.slice(k * 3, k * 3 + 3), A = P(a), B = P(b), C = P(c)
    const e1 = [B[0] - A[0], B[1] - A[1], B[2] - A[2]], e2 = [C[0] - A[0], C[1] - A[1], C[2] - A[2]]
    const fn = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]
    const n = [0, 1, 2].map(i => nor[a * 3 + i] + nor[b * 3 + i] + nor[c * 3 + i])
    dot(fn, n) < 0 ? idx.push(a, c, b) : idx.push(a, b, c)
  }
  // ring-to-ring tube: rings of s points at (x, y, z, radius)
  const tube = (rings, s, foliage) => {
    const ids = rings.map(([x, y, z, r]) => Array.from({ length: s }, (_, k) => {
      const t = k / s * 2 * Math.PI
      return v([x + Math.cos(t) * r, y + Math.sin(t) * r, z], [Math.cos(t), Math.sin(t), 0], foliage)
    }))
    for (let q = 0; q + 1 < ids.length; q++) for (let k = 0; k < s; k++) {
      const a = ids[q][k], b = ids[q][(k + 1) % s], c = ids[q + 1][k], d = ids[q + 1][(k + 1) % s]
      tri(a, b, d); tri(a, d, c)
    }
  }
  const geometry = () => {
    const g = new THREE.InstancedBufferGeometry()
    g.setIndex(idx)
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3))
    g.setAttribute('part', new THREE.Float32BufferAttribute(part, 2))
    return g
  }
  return { v, tri, tube, geometry }
}

// stacked cones on a short trunk
function conifer() {
  const b = builder(), S = 7
  b.tube([[0, 0, -0.1, 0.03], [0, 0, 0.3, 0.02]], 5, 0)
  const tiers = [[0.12, 0.5, 0.25], [0.32, 0.7, 0.2], [0.52, 0.88, 0.14], [0.7, 1.0, 0.08]]
  for (const [z0, z1, r] of tiers) {
    const slant = r / (z1 - z0), nz = slant / Math.hypot(1, slant)
    const apex = b.v([0, 0, z1], [0, 0, 1], 1), centre = b.v([0, 0, z0 + 0.02], [0, 0, -1], 1)
    const ring = Array.from({ length: S }, (_, k) => {
      const t = k / S * 2 * Math.PI, c = Math.cos(t), s = Math.sin(t), nr = Math.sqrt(1 - nz * nz)
      return b.v([c * r, s * r, z0], [c * nr, s * nr, nz], 1)
    })
    const under = ring.map((_, k) => { const t = k / S * 2 * Math.PI; return b.v([Math.cos(t) * r, Math.sin(t) * r, z0], [0, 0, -1], 1) })
    for (let k = 0; k < S; k++) { b.tri(ring[k], ring[(k + 1) % S], apex); b.tri(under[k], under[(k + 1) % S], centre) }
  }
  return b.geometry()
}
// trunk and a round crown (icosphere; the shader makes it lumpy, flattens acacias, drops shrubs to the ground)
function broadleaf() {
  const b = builder()
  b.tube([[0, 0, -0.1, 0.04], [0, 0, 0.35, 0.03], [0.02, 0, 0.55, 0.02]], 5, 0)
  const t = (1 + Math.sqrt(5)) / 2
  let verts = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
  let faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
  verts = verts.map(p => { const l = Math.hypot(...p); return p.map(c => c / l) })
  const mid = new Map(), split = (a, c) => {
    const key = a < c ? `${a},${c}` : `${c},${a}`
    if (!mid.has(key)) { const m = verts[a].map((x, i) => x + verts[c][i]), l = Math.hypot(...m); verts.push(m.map(x => x / l)); mid.set(key, verts.length - 1) }
    return mid.get(key)
  }
  faces = faces.flatMap(([a, c, d]) => { const ab = split(a, c), bc = split(c, d), ca = split(d, a); return [[a, ab, ca], [c, bc, ab], [d, ca, bc], [ab, bc, ca]] })
  const ids = verts.map(n => b.v([n[0] * 0.36, n[1] * 0.36, 0.62 + n[2] * 0.36], n, 1))
  for (const [a, c, d] of faces) b.tri(ids[a], ids[c], ids[d])
  return b.geometry()
}
// leaning ringed trunk and a crown of drooping fronds
function palm() {
  const b = builder()
  b.tube([0, 0.25, 0.5, 0.75, 0.96].map(z => [0.1 * z * z, 0, z - (z ? 0 : 0.1), 0.03 - 0.012 * z]), 5, 0)
  for (let k = 0; k < 8; k++) {
    const a = k / 8 * 2 * Math.PI + 0.3 * (k & 1), c = Math.cos(a), s = Math.sin(a), rows = []
    for (let q = 0; q <= 3; q++) {
      const r = 0.03 + 0.12 * q, z = 0.97 + 0.07 * q - 0.045 * q * q, w = [0.01, 0.05, 0.045, 0.01][q], up = [s * 0.2, -c * 0.2, 1]
      rows.push([b.v([0.1 + c * r - s * w, s * r + c * w, z], up, 1), b.v([0.1 + c * r + s * w, s * r - c * w, z], up, 1)])
    }
    for (let q = 0; q < 3; q++) { b.tri(rows[q][0], rows[q][1], rows[q + 1][1]); b.tri(rows[q][0], rows[q + 1][1], rows[q + 1][0]) }
  }
  return b.geometry()
}

export class Trees {
  constructor(ctx) {
    this.ctx = ctx
    // canopy painting (forest.glsl.js) in the terrain material: add ours before it first compiles (as farms.js does)
    this.paint = { uForestOff: { value: new THREE.Vector3() }, uTreeR0: { value: 0 }, uTreeMax: { value: 0 } }
    const terrainMat = ctx.scene.children.find(m => m.material?.fragmentShader?.includes('forestPaint('))?.material
    if (terrainMat) Object.assign(terrainMat.uniforms, this.paint)
    const farm = k => terrainMat?.uniforms[k] ?? { value: k === 'uFarmOn' ? 0 : null }

    // placement pass: candidates in, (height, kind, size, tone) out
    this.cand = new THREE.DataTexture(new Float32Array(MAX_INST * 4), PLACE_W, MAX_INST / PLACE_W, THREE.RGBAFormat, THREE.FloatType)
    this.place = new THREE.WebGLRenderTarget(PLACE_W, MAX_INST / PLACE_W, { type: THREE.FloatType, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter })
    this.placeMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: ctx.shared.quadVert, fragmentShader: PLACE_FRAG,
      uniforms: {
        uCand: { value: this.cand }, uCount: { value: 0 }, uSun: ctx.shared.uSun, uPixAngle: ctx.shared.uPixAngle, ...ctx.shared.civ, ...this.paint,
        uFarmTerr: farm('uFarmTerr'), uFarmDay: farm('uFarmDay'), uFarmOn: farm('uFarmOn'), uFarmA: farm('uFarmA'), uFarmCamA: farm('uFarmCamA'),
      },
    })
    this.placeQuad = ctx.shared.makeQuad(this.placeMat)

    // instances: sea-level anchor relative to the rebuild anchor, up, (rank, random). The full meshes draw every
    // candidate and read the placement texture; once its readback arrives the compact meshes (one kind each, the
    // placement in iPlace) take over, so no vertex work goes to empty candidates or other kinds.
    this.buf = new THREE.InstancedInterleavedBuffer(new Float32Array(MAX_INST * 8), 8).setUsage(THREE.DynamicDrawUsage)
    this.uniforms = { ...objUniforms(ctx.shared), ...this.paint, uPlace: { value: this.place.texture }, uAnchor: { value: new THREE.Vector3() } }
    const shapes = [conifer, broadleaf, palm]
    this.full = shapes.map((shape, kind) => this.makeMesh(shape(), kind, this.buf, false))
    this.compact = shapes.map((shape, kind) => this.makeMesh(shape(), kind, new THREE.InstancedInterleavedBuffer(new Float32Array(MAX_INST * 12), 12).setUsage(THREE.DynamicDrawUsage), true))
    this.anchor = new THREE.Vector3()
    this.lastDir = new THREE.Vector3(9, 0, 0)
    this.count = 0
    this.trees = 0
    this.gen = 0
    this.useCompact = false
    this.enabled = true
  }

  makeMesh(geo, kind, buf, compact) {
    const attrs = [['iPos', 3, 0], ['iUp', 3, 3], ['iMisc', 2, 6]].concat(compact ? [['iPlace', 4, 8]] : [])
    for (const [name, size, off] of attrs) geo.setAttribute(name, new THREE.InterleavedBufferAttribute(buf, size, off))
    geo.instanceCount = 0
    const mat = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: treeVert(compact), fragmentShader: TREE_FRAG, uniforms: { ...this.uniforms, uKind: { value: kind } }, side: kind === 2 ? THREE.DoubleSide : THREE.FrontSide })
    const mesh = new THREE.Mesh(geo, mat)
    mesh.frustumCulled = false
    mesh.visible = false
    mesh.userData.buf = buf
    this.ctx.scene.add(mesh)
    return mesh
  }

  update(frame) {
    const c = frame.camPos, d = c.length(), k = R / FOREST_CELL, m = x => ((x % 1024) + 1024) % 1024
    this.paint.uForestOff.value.set(m(c.x / d * k), m(c.y / d * k), m(c.z / d * k))
    const on = this.enabled && frame.geo.agl < DMAX
    this.paint.uTreeR0.value = on ? R0 : 0
    this.paint.uTreeMax.value = on ? DMAX : 0
    if (on && frame.camDir.distanceTo(this.lastDir) * R > MARGIN / 2) this.rebuild(frame)        // horizontal moves only
    this.uniforms.uAnchor.value.subVectors(this.anchor, c)
    for (const mesh of this.full) mesh.visible = on && !this.useCompact && this.count > 0
    for (const mesh of this.compact) mesh.visible = on && this.useCompact && mesh.geometry.instanceCount > 0
  }

  // enumerate the candidates around the camera, then run the placement pass for them
  rebuild(frame) {
    const cam = frame.camPos, len = cam.length(), cd = [cam.x / len, cam.y / len, cam.z / len]
    const A = this.anchor.copy(cam), D = this.buf.array, C = this.cand.image.data
    let n = 0
    for (let f = 0; f < 6 && n < MAX_INST; f++) {
      if (dot(cd, FACES[f][0]) < 0.5) continue
      const [ca, cb] = faceCoords(f, cd)
      for (let L = 0; L <= LMAX; L++) {
        const NL = 2 ** (DEPTH - L), reach = Math.min(R0 * 2 ** (L + 1), DMAX) + MARGIN
        // cells per metre near the camera (the equiangular grid is within ~30 % of uniform)
        const e = 1e-4, p0 = faceDir(f, ca, cb), pa = faceDir(f, ca + e, cb), pb = faceDir(f, ca, cb + e)
        const sa = Math.hypot(pa[0] - p0[0], pa[1] - p0[1], pa[2] - p0[2]) * R / e * 2 / NL, sb = Math.hypot(pb[0] - p0[0], pb[1] - p0[1], pb[2] - p0[2]) * R / e * 2 / NL
        const ci = Math.floor((ca + 1) / 2 * NL), cj = Math.floor((cb + 1) / 2 * NL), ri = Math.ceil(reach / sa) + 1, rj = Math.ceil(reach / sb) + 1
        for (let j = Math.max(0, cj - rj); j <= Math.min(NL - 1, cj + rj) && n < MAX_INST; j++) {
          for (let i = Math.max(0, ci - ri); i <= Math.min(NL - 1, ci + ri); i++) {
            if (L < LMAX && isPromoted(f, i, j, L)) continue
            const [i0, j0] = promoted(f, i, j, L), N0 = 2 ** DEPTH
            const dir = faceDir(f, (i0 + 0.15 + 0.7 * hash2i(i0, j0, SEED + f)) / N0 * 2 - 1, (j0 + 0.15 + 0.7 * hash2i(i0, j0, SEED + 6 + f)) / N0 * 2 - 1)
            const dx = dir[0] - cd[0], dy = dir[1] - cd[1], dz = dir[2] - cd[2]
            if ((dx * dx + dy * dy + dz * dz) * R * R > reach * reach) continue               // horizontal distance
            const hr = hash2i(i0, j0, SEED + 13 + f), rank = L < LMAX ? 4 ** -(L + 1) * (1 + 3 * hr) : 4 ** -LMAX * Math.max(hr, 1e-3)
            const o = n * 8
            D[o] = dir[0] * R - A.x; D[o + 1] = dir[1] * R - A.y; D[o + 2] = dir[2] * R - A.z
            D[o + 3] = dir[0]; D[o + 4] = dir[1]; D[o + 5] = dir[2]; D[o + 6] = rank; D[o + 7] = hash2i(i0, j0, SEED + 19 + f)
            C.set([dir[0], dir[1], dir[2], hash2i(i0, j0, SEED + 23 + f)], n * 4)
            if (++n >= MAX_INST) break
          }
        }
      }
    }
    this.count = n
    this.lastDir.copy(frame.camDir)
    this.buf.needsUpdate = true
    this.cand.needsUpdate = true
    for (const mesh of this.full) mesh.geometry.instanceCount = n
    // placement pass over the rows in use
    const renderer = this.ctx.renderer, prev = renderer.getRenderTarget(), rows = Math.max(1, Math.ceil(n / PLACE_W))
    this.placeMat.uniforms.uCount.value = n
    this.place.viewport.set(0, 0, PLACE_W, rows)
    renderer.setRenderTarget(this.place)
    renderer.render(this.placeQuad.scene, this.placeQuad.camera)
    renderer.setRenderTarget(prev)
    // until the readback arrives the full meshes draw this set
    const gen = ++this.gen, inst = D.slice(0, n * 8), out = new Float32Array(PLACE_W * rows * 4)
    this.useCompact = false
    renderer.readRenderTargetPixelsAsync(this.place, 0, 0, PLACE_W, rows, out).then(() => { if (gen === this.gen) this.fillCompact(inst, out, n) }, () => {})
  }

  // split the grown trees by mesh kind into the compact meshes
  fillCompact(inst, place, n) {
    const counts = [0, 0, 0]
    for (let i = 0; i < n; i++) {
      const y = place[i * 4 + 1]
      if (y < 0) continue
      const k = Math.min(Math.floor(y), 3) === 3 ? 1 : Math.floor(y), mesh = this.compact[k], o = counts[k]++ * 12
      mesh.userData.buf.array.set(inst.subarray(i * 8, i * 8 + 8), o)
      mesh.userData.buf.array.set(place.subarray(i * 4, i * 4 + 4), o + 8)
    }
    this.compact.forEach((mesh, k) => { mesh.geometry.instanceCount = counts[k]; mesh.userData.buf.needsUpdate = true })
    this.trees = counts[0] + counts[1] + counts[2]
    this.useCompact = true
  }
}
