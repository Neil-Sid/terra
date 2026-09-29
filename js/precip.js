// Local precipitation around the camera (rain streaks or snowflakes) and lightning flashes,
// driven by the probe's weather reading. Drawn over the composite, in a box that follows the camera.
import * as THREE from 'three'
import { CLOUD_BASE } from './config.js'

const COUNT = 30000, BOX = 36

const VERT = /* glsl */`
precision highp float;
uniform mat4 projectionMatrix, viewMatrix;
uniform mat3 uBasis;          // columns: east, up, north at the camera
uniform vec3 uShift;          // accumulated camera motion + wind, in that basis (m)
uniform float uTime, uSnow, uBox;
in vec3 position;             // random seed in [0,1)^3
in float aEnd;                // 0 = head of the streak, 1 = tail
out float vFade;
void main(){
  vec3 s = position;
  float fall = mix(8.5 + 2.0*s.x, 1.1 + 0.4*s.y, uSnow);
  vec3 lp = s*uBox;
  lp.y -= uTime*fall;
  lp += uSnow*vec3(sin(uTime*1.3 + s.z*40.0), 0.0, cos(uTime*1.1 + s.x*40.0))*0.35;
  lp = mod(lp - uShift, uBox) - uBox*0.5;
  lp.y += aEnd*mix(1.3, 0.0, uSnow);
  vec3 w = uBasis*lp;
  gl_Position = projectionMatrix*viewMatrix*vec4(w, 1.0);
  gl_PointSize = mix(1.0, 3.0*clamp(6.0/gl_Position.w, 0.6, 3.0), uSnow);
  vFade = 1.0 - smoothstep(uBox*0.3, uBox*0.5, length(lp));
}`

const FRAG = /* glsl */`
precision highp float;
uniform float uAmount, uSnow, uLight;
in float vFade;
out vec4 outColor;
void main(){
  float a = uAmount*vFade*mix(0.4, 0.85, uSnow);
  if (uSnow > 0.5){ vec2 c = gl_PointCoord - 0.5; a *= smoothstep(0.5, 0.2, length(c)); }
  outColor = vec4(mix(vec3(0.82, 0.86, 0.92), vec3(1.0), uSnow)*uLight, a);
}`

export class Precip {
  constructor() {
    const seeds = new Float32Array(COUNT * 6), ends = new Float32Array(COUNT * 2)
    for (let i = 0; i < COUNT; i++) {
      const x = Math.random(), y = Math.random(), z = Math.random()
      seeds.set([x, y, z, x, y, z], i * 6)
      ends.set([0, 1], i * 2)
    }
    const lines = new THREE.BufferGeometry()
    lines.setAttribute('position', new THREE.BufferAttribute(seeds, 3))
    lines.setAttribute('aEnd', new THREE.BufferAttribute(ends, 1))
    this.u = {
      uBasis: { value: new THREE.Matrix3() }, uShift: { value: new THREE.Vector3() },
      uTime: { value: 0 }, uSnow: { value: 0 }, uBox: { value: BOX }, uAmount: { value: 0 }, uLight: { value: 1 },
    }
    const mat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: VERT, fragmentShader: FRAG, uniforms: this.u,
      transparent: true, depthTest: false, depthWrite: false,
    })
    this.rain = new THREE.LineSegments(lines, mat)
    // snow reuses the heads only
    const pts = new THREE.BufferGeometry()
    pts.setAttribute('position', new THREE.BufferAttribute(seeds.filter((_, i) => i % 6 < 3), 3))
    pts.setAttribute('aEnd', new THREE.BufferAttribute(new Float32Array(COUNT), 1))
    this.snow = new THREE.Points(pts, mat)
    for (const o of [this.rain, this.snow]) o.frustumCulled = false
    this.scene = new THREE.Scene()
    this.scene.add(this.rain, this.snow)
    this.amount = 0; this.flash = 0; this.nextBolt = 3; this.state = 'Clear'
    this.last = null
  }

  update(dt, realTime, cam, probe) {
    const below = cam.altitude < CLOUD_BASE + 300
    const target = below ? THREE.MathUtils.smoothstep(probe.precip, 0.15, 0.7) : 0
    this.amount += (target - this.amount) * (1 - Math.exp(-dt * 1.5))
    const snow = probe.temp - cam.altitude * 0.0065 < 0.5 ? 1 : 0
    const u = this.u
    u.uTime.value = realTime
    u.uSnow.value = snow
    u.uAmount.value = this.amount
    this.rain.visible = !snow && this.amount > 0.01
    this.snow.visible = !!snow && this.amount > 0.01
    u.uBasis.value.set(cam.east.x, cam.up.x, cam.north.x, cam.east.y, cam.up.y, cam.north.y, cam.east.z, cam.up.z, cam.north.z)
    // keep drops fixed in the world as the camera moves
    if (this.last) {
      const d = cam.pos.clone().sub(this.last)
      u.uShift.value.add(new THREE.Vector3(d.dot(cam.east), d.dot(cam.up), d.dot(cam.north)))
      u.uShift.value.x = u.uShift.value.x % 1e4; u.uShift.value.y %= 1e4; u.uShift.value.z %= 1e4
    }
    this.last = cam.pos.clone()

    // lightning near storms, bright enough to see from below and inside the clouds
    this.flash *= Math.exp(-dt * 9)
    const stormy = probe.lightning > 0.15 && cam.altitude < 20000
    this.nextBolt -= dt
    if (stormy && this.nextBolt <= 0) {
      this.flash = 0.6 + Math.random() * 0.6
      this.nextBolt = (1 - probe.lightning) * 8 + Math.random() * 6
      if (Math.random() < 0.4) setTimeout(() => { this.flash = Math.max(this.flash, 0.7) }, 90 + Math.random() * 120)
    }
    u.uLight.value = 0.8 + this.flash * 3
    this.state = describe(probe, cam, snow, this.amount)
  }

  render(renderer, camera) {
    if (this.amount <= 0.01) return
    // drops are metres away: the main camera near plane grows with altitude and would clip them all
    const near = camera.near
    camera.near = 0.1
    camera.updateProjectionMatrix()
    renderer.setRenderTarget(null)
    renderer.render(this.scene, camera)
    camera.near = near
    camera.updateProjectionMatrix()
  }
}

function describe(p, cam, snow, amount) {
  if (cam.altitude > 20000) return p.storm > 0.6 && p.cover > 0.6 ? 'Storm system below' : p.cover > 0.6 ? 'Cloud deck below' : 'Clear below'
  if (amount > 0.05) {
    const kind = snow ? 'snow' : 'rain'
    if (p.lightning > 0.15 && !snow) return 'Thunderstorm'
    return (p.precip > 0.55 ? 'Heavy ' : 'Light ') + kind
  }
  if (cam.altitude > 1400 && cam.altitude < 9500 && p.cover > 0.5) return 'In cloud'
  return p.cover > 0.75 ? 'Overcast' : p.cover > 0.4 ? 'Partly cloudy' : p.cover > 0.12 ? 'Fair' : 'Clear'
}
