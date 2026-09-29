// Geodetic camera: position is (lat, lon, height above ground), view is (heading, pitch).
// Left drag grabs the ground under the cursor, right drag looks around, wheel zooms toward the cursor,
// WASD flies. Horizontal moves are rotations of the whole frame, so crossing a pole just works.
// follow = { pose(dt) -> { eye, fwd, up, focus } | null, end() } rides along with something (a train): a cab
// view at eye looking along fwd, or (V) a chase view orbiting focus; right drag looks, wheel sets the chase
// distance, any move key or left drag gets off.
import * as THREE from 'three'
import { R } from './config.js'

const clamp = (x, a, b) => Math.min(b, Math.max(a, x))
const smooth = t => t * t * (3 - 2 * t)
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a))
const MIN_AGL = 1.7, MAX_AGL = 6e7
const dirOf = (lat, lon) => new THREE.Vector3(Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon))

// Straight down from space, easing towards the horizon near the ground.
export function autoPitch(agl) {
  const t = clamp((Math.log10(agl) - 2.8) / (6.6 - 2.8), 0, 1)
  return -0.12 - (Math.PI / 2 - 0.12) * t * t * (3 - 2 * t)
}

export class GeoCamera {
  constructor(dom) {
    Object.assign(this, { lat: 0.3, lon: 0.4, agl: 2.2e7, heading: 0, pitch: -Math.PI / 2 })
    this.dom = dom
    this.targetAgl = this.agl
    this.ground = 0            // terrain (or sea) height under the camera, from the GPU probe
    this.groundSmooth = 0
    this.pixAngle = 0.001      // radians per CSS pixel, set by main on resize
    this.speed = 0
    this.keys = new Set()
    this.pos = new THREE.Vector3()
    this.quat = new THREE.Quaternion()
    this.up = new THREE.Vector3(); this.east = new THREE.Vector3(); this.north = new THREE.Vector3()
    this.look = new THREE.Vector3(); this.camUp = new THREE.Vector3(); this.right = new THREE.Vector3()
    this.ahead = new THREE.Vector3()   // where the camera will be shortly, for the terrain probe
    this.m = new THREE.Matrix4()
    this.follow = null
    this.followView = 'chase'
    this.fYaw = 0; this.fPitch = 0; this.chaseDist = 70
    this.compose()
    this.bind(dom)
  }

  get altitude() { return this.groundSmooth + this.agl }

  bind(dom) {
    dom.addEventListener('contextmenu', e => e.preventDefault())
    dom.addEventListener('pointerdown', e => {
      const look = e.button !== 0 || e.shiftKey
      this.drag = { look, x: e.clientX, y: e.clientY, grab: look ? null : this.pick(e) }
      if (!look) this.unfollow()
      if (!this.follow) this.flight = this.anchor = null
      try { dom.setPointerCapture(e.pointerId) } catch {}
    })
    dom.addEventListener('pointermove', e => {
      if (!this.drag) return
      const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y
      this.drag.x = e.clientX; this.drag.y = e.clientY
      if (this.drag.look) this.lookBy(dx, dy)
      else this.grab(e, dx, dy)
    })
    dom.addEventListener('pointerup', () => { this.drag = null })
    dom.addEventListener('wheel', e => {
      e.preventDefault()
      if (!this.follow) return this.zoom(e.deltaY, e)
      this.chaseDist = clamp(this.chaseDist * Math.exp(e.deltaY * 0.0013), 12, 900)
      this.followView = 'chase'
    }, { passive: false })
    addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT') return
      if (e.code === 'KeyV' && this.follow) { this.followView = this.followView === 'cab' ? 'chase' : 'cab'; this.fYaw = this.followView === 'cab' ? 0 : 0.5; this.fPitch = 0 }
      else this.keys.add(e.code)
    })
    addEventListener('keyup', e => this.keys.delete(e.code))
    addEventListener('blur', () => this.keys.clear())
  }

  lookBy(dx, dy) {
    if (this.follow) {
      const cab = this.followView === 'cab'
      this.fYaw = cab ? clamp(this.fYaw + dx * 0.0035, -1.8, 1.8) : this.fYaw + dx * 0.0035
      this.fPitch = cab ? clamp(this.fPitch - dy * 0.0035, -0.7, 0.6) : clamp(this.fPitch + dy * 0.0035, -0.3, 1.1)
      return
    }
    this.heading += dx * 0.0035
    this.pitch = clamp(this.pitch - dy * 0.0035, -Math.PI / 2, 1.3)
  }

  // world direction of the ray through a screen point
  ray(e) {
    const r = this.dom.getBoundingClientRect(), k = this.pixAngle
    const x = (e.clientX - r.left - r.width / 2) * k, y = (r.top + r.height / 2 - e.clientY) * k
    return this.look.clone().addScaledVector(this.right, x).addScaledVector(this.camUp, y).normalize()
  }

  // unit direction (from the planet centre) of the ground under a screen point; the horizon point on a miss
  pick(e, hitOnly = false) {
    const d = this.ray(e), c = this.up.clone().multiplyScalar(R + this.altitude)
    const rg = R + Math.max(this.groundSmooth, 0)
    const b = c.dot(d), disc = b * b - (c.lengthSq() - rg * rg)
    if (disc < 0 && (hitOnly || b > 0)) return null
    const t = disc < 0 ? -b : -b - Math.sqrt(disc)
    return t > 0 ? c.addScaledVector(d, t).normalize() : null
  }

  // metres of ground under one screen pixel, roughly
  groundPerPixel() {
    const d = this.agl / Math.max(Math.sin(-this.pitch), 0.2)
    return Math.min(d, Math.sqrt(2 * R * this.agl + this.agl * this.agl) + this.agl) * this.pixAngle
  }

  // keep the grabbed ground point under the cursor by rotating the camera around the planet
  grab(e, dx, dy) {
    const p0 = this.drag.grab, p1 = p0 && this.pick(e)
    if (!p1) {
      const mpp = this.groundPerPixel() * (this.agl > 3e6 ? 0.6 : 1)
      return this.move(dy * mpp, -dx * mpp)
    }
    const maxA = Math.hypot(dx, dy) * this.agl * this.pixAngle * 40 / R   // grazing rays slip instead of flinging
    const a = p1.angleTo(p0)
    if (a < 1e-12) return
    const axis = p1.clone().cross(p0).normalize()
    this.rotateBy(new THREE.Quaternion().setFromAxisAngle(axis, Math.min(a, maxA)))
  }

  zoom(delta, e) {
    this.flight = null
    this.anchor = delta < 0 && e ? this.pick(e, true) : null
    this.targetAgl = clamp(this.targetAgl * Math.exp(delta * 0.0013), MIN_AGL, MAX_AGL)
    this.pitch += (autoPitch(this.targetAgl) - this.pitch) * 0.18
  }

  // move over the surface: forward along the heading, right perpendicular to it (metres)
  move(fwd, right) {
    const ch = Math.cos(this.heading), sh = Math.sin(this.heading)
    const d = this.north.clone().multiplyScalar(fwd * ch - right * sh).addScaledVector(this.east, fwd * sh + right * ch)
    const len = d.length()
    if (len < 1e-9) return
    this.rotateBy(new THREE.Quaternion().setFromAxisAngle(this.up.clone().cross(d).normalize(), len / R))
  }

  // rotate position and view together; pitch is unchanged, heading is re-read in the new local frame
  rotateBy(q) {
    const look = this.look.clone().applyQuaternion(q), camUp = this.camUp.clone().applyQuaternion(q)
    this.setUp(this.up.clone().applyQuaternion(q))
    const p = this.pitch
    const fwdH = look.multiplyScalar(Math.cos(p)).addScaledVector(camUp, -Math.sin(p))
    this.heading = Math.atan2(fwdH.dot(this.east), fwdH.dot(this.north))
    this.compose()
  }

  setUp(up) {
    this.lat = Math.asin(clamp(up.y, -0.999999, 0.999999))
    this.lon = Math.atan2(up.x, up.z)
    this.frame()
  }

  // instant jump (lat/lon now, height eases); used by scripts and the harness
  flyTo({ lat = this.lat, lon = this.lon, agl = this.targetAgl, pitch, heading }) {
    Object.assign(this, { lat, lon, targetAgl: agl, flight: null, anchor: null })
    if (pitch !== undefined) this.pitch = pitch
    if (heading !== undefined) this.heading = heading
    this.compose()
  }

  // animated flight: climb high enough to see both ends, travel, descend with the view easing to the horizon
  travel({ lat = this.lat, lon = this.lon, agl = this.targetAgl, pitch, heading = this.heading }) {
    const from = { up: this.up.clone(), la: Math.log(this.agl), pitch: this.pitch, heading: this.heading }
    const to = { up: dirOf(lat, lon), la: Math.log(agl), pitch: pitch ?? autoPitch(agl), heading }
    const ang = from.up.angleTo(to.up)
    const hump = Math.max(0, Math.log(clamp(ang * R * 1.2, 1, 2.2e7)) - (from.la + to.la) / 2)
    const span = Math.abs(to.la - from.la) + 2 * hump
    this.flight = { from, to, ang, hump, t: 0, dur: clamp(1.5 + span * 0.5 + ang * 1.5, 2, 14) }
    this.anchor = null
  }

  stepFlight(dt) {
    const f = this.flight
    f.t = Math.min(1, f.t + dt / f.dur)
    const t = f.t, e = smooth(t), h = smooth(clamp((t - 0.1) / 0.8, 0, 1))   // travel mostly while high
    if (f.ang > 1e-9) {
      const axis = f.from.up.clone().cross(f.to.up).normalize()
      this.setUp(f.from.up.clone().applyAxisAngle(axis, f.ang * h))
    }
    this.agl = this.targetAgl = clamp(Math.exp(f.from.la + (f.to.la - f.from.la) * e + f.hump * 16 * t * t * (1 - t) * (1 - t)), MIN_AGL, MAX_AGL)
    const base = f.from.pitch + (f.to.pitch - f.from.pitch) * e
    this.pitch = base + (autoPitch(this.agl) - base) * Math.sin(Math.PI * t)
    this.heading = f.from.heading + wrap(f.to.heading - f.from.heading) * e
    if (t >= 1) this.flight = null
  }

  update(dt) {
    const k = this.keys, key = (...c) => c.some(x => k.has(x)) ? 1 : 0
    const boost = key('ShiftLeft', 'ShiftRight') ? 6 : 1
    const v = Math.max(3, this.agl * 0.8) * boost
    const fwd = (key('KeyW') - key('KeyS')) * v * dt, right = (key('KeyD') - key('KeyA')) * v * dt
    const climb = key('KeyE', 'Space') - key('KeyQ', 'KeyC')
    const turn = key('ArrowRight') - key('ArrowLeft'), tilt = key('ArrowUp') - key('ArrowDown')
    if (fwd || right || climb || turn || tilt) { this.flight = null; this.unfollow() }
    const before = this.up.clone(), a0 = this.agl, riding = this.follow && !this.flight

    if (this.flight) this.stepFlight(dt)
    else if (!riding) {
      if (fwd || right) this.move(fwd, right)
      if (climb) { this.targetAgl = clamp(this.targetAgl * Math.exp(climb * dt * 1.6 * boost), MIN_AGL, MAX_AGL); this.anchor = null }
      this.heading += turn * dt * 1.2
      this.pitch = clamp(this.pitch + tilt * dt * 0.9, -Math.PI / 2, 1.3)
      // ease height in log space; slower for long jumps so the descent reads
      const la = Math.log(this.agl), lt = Math.log(this.targetAgl)
      const rate = Math.abs(lt - la) > 3 ? 1.6 : 7
      this.agl = clamp(Math.exp(la + (lt - la) * (1 - Math.exp(-dt * rate))), MIN_AGL, MAX_AGL)
      this.zoomToAnchor(a0)
    }

    // ground follows the probe: rises fast (never less than half the height as clearance), sinks gently
    const g = this.ground, up = g > this.groundSmooth
    this.groundSmooth += (g - this.groundSmooth) * (1 - Math.exp(-dt * (up ? 25 : 6)))
    this.groundSmooth = Math.max(this.groundSmooth, g - this.agl * 0.5)
    if (riding) this.stepFollow(dt)
    this.speed = before.angleTo(this.up) * R / Math.max(dt, 1e-3)
    this.compose()
    // look ahead ~0.4 s along the motion so the probe sees rising ground before we reach it
    const lead = this.up.clone().sub(before).multiplyScalar(0.4 / Math.max(dt, 1e-3))
    this.ahead.copy(this.up).add(lead.clampLength(0, 0.02)).normalize()
  }

  // where the camera is heading: a flight's end, or where a zoom ends up (a wheel zoom closes in on its anchor; the
  // offset left shrinks with the height). { dir, agl } or null (not going down). The civ modules stream that area in before the camera arrives.
  destination() {
    if (this.flight) return { dir: this.flight.to.up, agl: Math.exp(this.flight.to.la) }
    if (this.targetAgl > 0.5 * this.agl) return null
    return { dir: this.anchor ? this.anchor.clone().lerp(this.up, this.targetAgl / this.agl).normalize() : this.up.clone(), agl: this.targetAgl }
  }

  unfollow() {
    if (!this.follow) return
    this.follow.end?.()
    this.follow = null
    this.fYaw = this.fPitch = 0
  }

  // place the camera on the followed pose (planet-frame doubles, so no jitter); glides over 2 s after a flight
  // and when the lead end changes (a train reversing: the view moves to the other cab)
  stepFollow(dt) {
    const f = this.follow, p = f.pose(dt)
    if (!p) return
    if (f.cab !== p.cab) { f.from = f.fwd = null; f.cab = p.cab }
    f.fwd = f.fwd ? f.fwd.lerp(p.fwd, 1 - Math.exp(-dt * 4)).normalize() : p.fwd.clone()   // smooth heading
    const ahead = f.fwd.clone().applyAxisAngle(p.up, -this.fYaw)
    let pos, look
    if (this.followView === 'cab') {
      const a = this.fPitch - 0.04
      pos = p.eye.clone()
      look = ahead.multiplyScalar(Math.cos(a)).addScaledVector(p.up, Math.sin(a))
    } else {
      const el = clamp(0.3 + this.fPitch, 0.02, 1.4), d = this.chaseDist
      pos = p.focus.clone().addScaledVector(ahead, -d * Math.cos(el)).addScaledVector(p.up, d * Math.sin(el))
      look = p.focus.clone().sub(pos).normalize()
    }
    f.from ??= { pos: this.pos.clone(), heading: this.heading, pitch: this.pitch, t: 0 }
    f.from.t = Math.min(1, f.from.t + dt / 2)
    const k = smooth(f.from.t)
    pos = f.from.pos.clone().lerp(pos, k)
    this.setUp(pos.clone().normalize())
    this.agl = this.targetAgl = Math.max(pos.length() - R - this.groundSmooth, MIN_AGL)
    const heading = Math.atan2(look.dot(this.east), look.dot(this.north)), pitch = Math.asin(clamp(look.dot(this.up), -1, 1))
    this.heading = f.from.heading + wrap(heading - f.from.heading) * k   // turn the short way, never through the zenith
    this.pitch = f.from.pitch + (pitch - f.from.pitch) * k
  }

  // wheel zoom dollies toward the ground point that was under the cursor
  zoomToAnchor(a0) {
    const p = this.anchor
    if (!p) return
    if (this.agl >= a0 || Math.abs(Math.log(this.agl / this.targetAgl)) < 1e-3) { this.anchor = null; return }
    const a = this.up.angleTo(p)
    if (a < 1e-10) return
    const step = Math.min(a, 6 * a0 / R) * (1 - this.agl / a0)
    this.rotateBy(new THREE.Quaternion().setFromAxisAngle(this.up.clone().cross(p).normalize(), step))
  }

  frame() {
    const { lat, lon } = this
    this.up.set(Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon))
    this.east.set(Math.cos(lon), 0, -Math.sin(lon))
    this.north.crossVectors(this.up, this.east)
  }

  compose() {
    this.frame()
    const { heading: h, pitch: p, up, east, north } = this
    const fwdH = north.clone().multiplyScalar(Math.cos(h)).addScaledVector(east, Math.sin(h))
    this.look.copy(fwdH).multiplyScalar(Math.cos(p)).addScaledVector(up, Math.sin(p))
    this.camUp.copy(fwdH).multiplyScalar(-Math.sin(p)).addScaledVector(up, Math.cos(p))
    this.right.crossVectors(this.look, this.camUp)
    this.m.makeBasis(this.right, this.camUp, this.look.clone().negate())
    this.quat.setFromRotationMatrix(this.m)
    this.pos.copy(up).multiplyScalar(R + this.altitude)
  }
}
