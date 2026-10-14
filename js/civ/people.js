// People: pedestrians near the camera. Walkers loop round the sidewalk of every built-up block (district layout,
// notes/layout.md: the middle of the sidewalk band on the streets round it, cut off by the district's edge streets),
// some stand about, and parks / plazas get strollers and people sitting down. Everything is a function of time:
// position = seed + speed*time along a loop. Sidewalk loops are traffic.js blockLoop()s (world polylines, cached
// on the block records of cityBlocks()); park paths live in district coords. Contracts in notes/CIV.md,
// notes/people.md; shaders in people.glsl.js.
import * as THREE from 'three'
import { R } from '../config.js'
import { CITY_REACH, SKIND, PROFILES, profileOf, hash2i, dirToGrid, gridToDir, fromDistrict } from './spec.js'
import { cityBlocks, blockLoop } from './traffic.js'
import { objUniforms } from './objects.glsl.js'
import { TEX_W, TEX_H, HEIGHT_FRAG, PEOPLE_VERT, PEOPLE_FRAG } from './people.glsl.js'
import { cityWater, waterAt, LakeTest } from './shore.js'

const CAP = TEX_W * TEX_H           // instance slots (one height texel each)
const MAX_DRAW = 1500               // people drawn per frame, nearest blocks first
const RANGE = 380                   // metres from the camera
const REBUILD = 20                  // rebuild the block list after the camera moves this far
const GRID_SLACK = 1.15             // grid units per metre at most (round-5 warp)
const TIER_RATE = [0.35, 0.5, 0.8, 1.0]
const WALK = 0, STAND = 1, CIRCLE = 2, SIT = 3
const F = 8                         // floats per person record

// one low-poly humanoid, 1.75 m tall, x right, y up, z forward: [part, material, x0, y0, z0, x1, y1, z1]
// parts 0 body, 1 head, 2 thigh, 3 shin, 4 arm; materials 0 skin, 1 hair, 2 top, 3 trousers, 4 shoes
const BOXES = [
  [0, 3, -0.19, 0.86, -0.1, 0.19, 1.0, 0.1], [0, 2, -0.2, 1.0, -0.11, 0.2, 1.47, 0.11], [0, 0, -0.05, 1.45, -0.05, 0.05, 1.53, 0.05],
  [1, 0, -0.1, 1.52, -0.105, 0.1, 1.73, 0.105], [1, 1, -0.108, 1.64, -0.118, 0.108, 1.765, 0.085],
]
for (const s of [-1, 1]) {        // limbs, mirrored: [part, material, |x| inner, |x| outer, y0, z0, y1, z1]
  for (const [part, mat, a, b, y0, z0, y1, z1] of [
    [2, 3, 0.03, 0.18, 0.48, -0.08, 0.95, 0.08], [3, 3, 0.045, 0.165, 0.07, -0.06, 0.5, 0.06], [3, 4, 0.05, 0.16, 0.0, -0.06, 0.07, 0.16],
    [4, 2, 0.205, 0.295, 0.86, -0.05, 1.47, 0.05], [4, 0, 0.215, 0.285, 0.74, -0.04, 0.86, 0.04],
  ]) BOXES.push(s > 0 ? [part, mat, a, y0, z0, b, y1, z1] : [part, mat, -b, y0, z0, -a, y1, z1])
}

function humanGeometry() {
  const pos = [], nrm = [], pm = [], idx = []
  for (const [part, mat, x0, y0, z0, x1, y1, z1] of BOXES) {
    const b = new THREE.BoxGeometry(x1 - x0, y1 - y0, z1 - z0).translate((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    const base = pos.length / 3
    pos.push(...b.attributes.position.array); nrm.push(...b.attributes.normal.array)
    for (let i = 0; i < b.attributes.position.count; i++) pm.push(part * 8 + mat)
    for (const i of b.index.array) idx.push(base + i)
  }
  const g = new THREE.InstancedBufferGeometry()
  g.setIndex(idx)
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3))
  g.setAttribute('pm', new THREE.Float32BufferAttribute(pm, 1))
  return g
}

const mid = w => (w[0] + w[1]) / 2

export class People {
  constructor(ctx) {
    this.ctx = ctx
    this.records = new Float64Array(12000 * F)
    this.count = 0                                 // records in range
    this.blocks = []                               // blocks referenced by records, nearest first (see rebuild)
    this.at = null                                 // camera position of the last rebuild
    this.frustum = new THREE.Frustum()
    this.lakes = new LakeTest(ctx.sampler, () => { this.at = null })   // lake blocks tested: rebuild
    this.P = new Float64Array(3); this.T = new Float64Array(3)        // scratch: a person's unit dir and heading
    this.m = new THREE.Matrix4()

    const g = humanGeometry()
    const inst = n => new THREE.InstancedBufferAttribute(new Float32Array(CAP * n), n).setUsage(THREE.DynamicDrawUsage)
    this.attr = { iRel: inst(3), iFwd: inst(3), iMisc: inst(4) }
    for (const [k, a] of Object.entries(this.attr)) g.setAttribute(k, a)
    g.instanceCount = 0
    this.geo = g

    this.dirTex = new THREE.DataTexture(new Float32Array(CAP * 4), TEX_W, TEX_H, THREE.RGBAFormat, THREE.FloatType)
    this.heights = new THREE.WebGLRenderTarget(TEX_W, TEX_H, { type: THREE.FloatType, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter })
    this.heightMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: ctx.shared.quadVert, fragmentShader: HEIGHT_FRAG,
      uniforms: { uDirs: { value: this.dirTex }, uCount: { value: 0 }, ...ctx.shared.civ },
    })
    this.heightQuad = ctx.shared.makeQuad(this.heightMat)

    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: PEOPLE_VERT, fragmentShader: PEOPLE_FRAG,
      uniforms: { ...objUniforms(ctx.shared), uDirs: { value: this.dirTex }, uHeights: { value: this.heights.texture }, uHRef: { value: 0 } },
    })
    this.mesh = new THREE.Mesh(g, this.material)
    this.mesh.frustumCulled = false
    ctx.scene.add(this.mesh)
  }

  // collect person records for built-up blocks within range of the camera, nearest blocks first. Each block keeps
  // its sidewalk loop (built once, on the cityBlocks record) and a bounding circle for culling.
  rebuild(camDir) {
    const world = this.ctx.world, found = []
    for (const city of world.near(camDir, RANGE + 200).cities) {
      const [cu, cv] = dirToGrid(city, camDir)
      if (Math.hypot(cu, cv) > city.radius * CITY_REACH + RANGE) continue
      const water = cityWater(world, city, city.radius * CITY_REACH + RANGE)
      for (const b of cityBlocks(city, cu, cv, RANGE * GRID_SLACK + 150)) {
        const dist = Math.max(0, Math.hypot(b.dir[0] - camDir[0], b.dir[1] - camDir[1], b.dir[2] - camDir[2]) * R - b.rad)
        if (dist > RANGE || (water && this.blockWet(city, b))) continue
        const w = profileOf(b.D.prof).walk
        if (b.walk === undefined) b.walk = blockLoop(city, b, mid(w), mid(PROFILES[SKIND.EDGE].walk), 3)
        if (b.walk) found.push({ city, b, loop: b.walk, band: w[1] - w[0], dist, first: 0, last: 0 })
      }
    }
    found.sort((a, b) => a.dist - b.dist)
    this.count = 0
    this.blocks = []
    for (const blk of found) {
      blk.first = this.count
      const full = !this.addBlock(blk)
      blk.last = this.count
      this.blocks.push(blk)
      if (full) break
    }
  }

  // does block b meet water? A river within its radius, or lake ground (centre and corners) under the level; cached on
  // the block, wet while the lake heights are on their way
  blockWet(city, b) {
    if (!b.water) {
      b.water = { wet: false }
      const w = waterAt(this.ctx.world, b.dir, b.rad), r = b.rect
      if (w === 'river') b.water.wet = true
      else if (w > -Infinity) this.lakes.add(b.water, [b.dir, ...[[r.x0, r.y0], [r.x1, r.y0], [r.x0, r.y1], [r.x1, r.y1]].map(([x, y]) => gridToDir(city, ...fromDistrict(b.D, x, y)))].flat(), w)
    }
    return b.water.wet
  }

  // record: kind, 3 path numbers, start, speed, height, colour seed
  push(kind, a, b, c, s0, v, seed) {
    if (this.count * F >= this.records.length) return false
    const r = this.records, o = this.count++ * F
    r[o] = kind; r[o + 1] = a; r[o + 2] = b; r[o + 3] = c; r[o + 4] = s0; r[o + 5] = v
    r[o + 6] = 1.55 + 0.4 * hash2i(seed, 5, 3) ** 1.3; r[o + 7] = hash2i(seed, 6, 3) * 997
    return true
  }

  addBlock(blk) {
    const { city, b, loop, band } = blk, { kx, ky, D } = b, dens = b.density, seed0 = D.hseed
    const rate = (0.004 + 0.07 * dens * dens) * TIER_RATE[city.tier] * (b.zone ? 0.3 : 1), L0 = loop.L
    // the sidewalk loop (clockwise: shop fronts on the right). lat = m right of its middle line: walkers keep right
    const n = Math.floor(rate * L0 + hash2i(kx, ky, seed0 + 31))
    for (let i = 0; i < n; i++) {
      const seed = hash2i(kx * 131 + i, ky, seed0) * 4294967296 | 0, h = k => hash2i(seed, k, 17)
      const dirSign = h(1) < 0.5 ? -1 : 1, stand = h(2) < 0.12
      const lat = band * (stand ? 0.4 * h(3) - 0.2 : dirSign * (0.05 + 0.35 * h(3)))
      if (!this.push(stand ? STAND : WALK, lat, 0, 0, h(4) * L0, stand ? 0 : dirSign * (1.0 + 0.6 * h(5)), seed)) return false
    }
    if (b.park) {
      // lot in district coords (metric: m per unit): strollers walk circles, others sit on the grass / plaza
      const lot = b.lot, [mx, my] = b.metric, w = (lot.x1 - lot.x0) * mx, hgt = (lot.y1 - lot.y0) * my
      const m = Math.floor(w * hgt * 0.0012 * (0.4 + dens) * TIER_RATE[city.tier] * (b.plaza ? 1.5 : 1) + hash2i(kx, ky, seed0 + 37))
      for (let i = 0; i < m; i++) {
        const seed = hash2i(kx * 173 + i, ky, seed0 + 41) * 4294967296 | 0, h = k => hash2i(seed, k, 19)
        if (h(1) < 0.4 || Math.min(w, hgt) < 18) {
          if (!this.push(SIT, lot.x0 + (3 + (w - 6) * h(2)) / mx, lot.y0 + (3 + (hgt - 6) * h(3)) / my, h(4) * 2 * Math.PI, h(5) * 4, 0, seed)) return false
        } else {
          const r = 4 + (Math.min(w, hgt) / 2 - 7) * h(2)
          const cx = lot.x0 + (r + 2 + (w - 2 * r - 4) * h(3)) / mx, cy = lot.y0 + (r + 2 + (hgt - 2 * r - 4) * h(4)) / my
          if (!this.push(CIRCLE, cx, cy, r, h(5) * 2 * Math.PI, (h(6) < 0.5 ? -1 : 1) * (0.9 + 0.5 * h(7)), seed)) return false
        }
      }
    }
    return true
  }

  update(frame) {
    const { camPos, geo, camera, time } = frame
    const g = this.geo
    const camR = camPos.length(), camDir = [camPos.x / camR, camPos.y / camR, camPos.z / camR]
    const hRef = camR - R - geo.agl                 // ground height under the camera
    g.instanceCount = 0
    if (geo.agl > RANGE) return
    if (!this.at || Math.hypot(camDir[0] - this.at[0], camDir[1] - this.at[1], camDir[2] - this.at[2]) * R > REBUILD) {
      this.at = camDir
      this.rebuild(camDir)
    }
    if (!this.count) return

    this.m.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
    this.frustum.setFromProjectionMatrix(this.m, camera.coordinateSystem, camera.reversedDepth)
    const planes = this.frustum.planes, rec = this.records, gr = R + hRef
    const rel = this.attr.iRel.array, fwd = this.attr.iFwd.array, misc = this.attr.iMisc.array, dirs = this.dirTex.image.data
    const outside = (x, y, z, margin) => { for (const pl of planes) if (pl.normal.x * x + pl.normal.y * y + pl.normal.z * z + pl.constant < -margin) return true; return false }
    const { P, T } = this
    let n = 0
    for (const blk of this.blocks) {
      if (n >= MAX_DRAW) break
      const D = blk.b.dir, rad = blk.b.rad
      const bx = D[0] * gr - camPos.x, by = D[1] * gr - camPos.y, bz = D[2] * gr - camPos.z
      if (Math.hypot(bx, by, bz) - rad > RANGE || outside(bx, by, bz, rad + 3)) continue
      for (let i = blk.first; i < blk.last; i++) {
        const o = i * F, kind = rec[o], walked = this.locate(kind, rec, o, blk, time)
        const dx = P[0], dy = P[1], dz = P[2]
        const rx = dx * gr - camPos.x, ry = dy * gr - camPos.y, rz = dz * gr - camPos.z
        const dist = Math.hypot(rx, ry, rz)
        if (dist > RANGE || outside(rx, ry, rz, 3)) continue
        const fade = Math.min(1, (RANGE - dist) / 60)
        rel[n * 3] = rx; rel[n * 3 + 1] = ry; rel[n * 3 + 2] = rz
        fwd[n * 3] = T[0]; fwd[n * 3 + 1] = T[1]; fwd[n * 3 + 2] = T[2]
        const stride = 1.45 * rec[o + 6] / 1.75
        misc[n * 4] = kind === WALK || kind === CIRCLE ? (walked / stride % 1) * 2 * Math.PI : time * 0.7 + rec[o + 4]
        misc[n * 4 + 1] = rec[o + 6] * fade
        misc[n * 4 + 2] = rec[o + 7]
        misc[n * 4 + 3] = kind === SIT ? 2 : kind === STAND ? 1 : 0
        dirs[n * 4] = dx; dirs[n * 4 + 1] = dy; dirs[n * 4 + 2] = dz
        if (++n >= MAX_DRAW) break
      }
    }
    g.instanceCount = n
    if (!n) return
    for (const a of Object.values(this.attr)) { a.clearUpdateRanges(); a.addUpdateRange(0, n * a.itemSize); a.needsUpdate = true }
    this.dirTex.needsUpdate = true
    this.material.uniforms.uHRef.value = hRef
    // ground heights for this frame's people (one terrain() evaluation each)
    this.heightMat.uniforms.uCount.value = n
    const renderer = this.ctx.renderer, prev = renderer.getRenderTarget()
    renderer.setRenderTarget(this.heights)
    renderer.render(this.heightQuad.scene, this.heightQuad.camera)
    renderer.setRenderTarget(prev)
  }

  // unit dir (this.P) and heading (this.T) of record o (block blk) at time t; returns metres walked
  locate(kind, r, o, blk, t) {
    const { P, T } = this
    if (kind === CIRCLE || kind === SIT) {
      // park paths in district coords: a circle of radius rad m (or a seat facing r[o + 3]); heading by a step along it
      const { D, metric: [mx, my] } = blk.b, city = blk.city
      let x = r[o + 1], y = r[o + 2], hx, hy, walked = 0
      if (kind === CIRCLE) {
        const rad = r[o + 3], v = r[o + 5], a = r[o + 4] + v * t / rad, s = v > 0 ? 1 : -1
        x += rad * Math.cos(a) / mx; y += rad * Math.sin(a) / my
        hx = -Math.sin(a) * s / mx; hy = Math.cos(a) * s / my; walked = Math.abs(v) * t
      } else { hx = Math.cos(r[o + 3]) / mx; hy = Math.sin(r[o + 3]) / my }
      const d0 = gridToDir(city, ...fromDistrict(D, x, y)), d1 = gridToDir(city, ...fromDistrict(D, x + hx, y + hy))
      for (let j = 0; j < 3; j++) { P[j] = d0[j]; T[j] = d1[j] - d0[j] }
      const k = T[0] * P[0] + T[1] * P[1] + T[2] * P[2], l = Math.hypot(T[0] - P[0] * k, T[1] - P[1] * k, T[2] - P[2] * k) || 1
      for (let j = 0; j < 3; j++) T[j] = (T[j] - P[j] * k) / l
      return walked
    }
    // sidewalk loop, lat m to the right of its middle line
    const v = kind === WALK ? r[o + 5] : 0, s = r[o + 4] + v * t, lat = r[o + 1]
    blk.loop.at(s, P, T)
    const rx = T[1] * P[2] - T[2] * P[1], ry = T[2] * P[0] - T[0] * P[2], rz = T[0] * P[1] - T[1] * P[0]
    P[0] += rx * lat / R; P[1] += ry * lat / R; P[2] += rz * lat / R
    const l = Math.hypot(P[0], P[1], P[2]); P[0] /= l; P[1] /= l; P[2] /= l
    if (kind === STAND) { T[0] = -rx; T[1] = -ry; T[2] = -rz }        // face the street (left of the loop)
    else if (v < 0) { T[0] = -T[0]; T[1] = -T[1]; T[2] = -T[2] }
    return Math.abs(v) * t + r[o + 4]
  }
}
