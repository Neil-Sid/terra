// Reads the GPU height field, weather and climate at the camera's position (async, one frame late),
// so the camera can stand on the ground (or on lake and river water) and the HUD and precipitation know the local conditions.
import * as THREE from 'three'
import { GLSL_HEAD } from './config.js'
import { NOISE } from './glsl/noise.js'
import { TERRAIN } from './glsl/terrain.glsl.js'
import { ATMO } from './glsl/atmo.glsl.js'

const FRAG = /* glsl */`${GLSL_HEAD}
uniform vec3 uDir;
uniform vec3 uAhead;
uniform sampler2D uWeather;
in vec2 vUv;
out vec4 outP;
${NOISE}
${TERRAIN}
${ATMO}
void main(){
  int px = int(vUv.x*4.0);
  vec3 p = normalize(px == 3 ? uAhead : uDir);   // one terrain() call site keeps the shader small
  vec3 info;
  Civ c;
  float h = terrainCiv(p, 1.2e6, info, c).x;
  h = max(h, c.water);                           // the camera floats on lakes and rivers (the sea is clamped in main.js)
  if (px == 0) outP = vec4(h, info);
  else if (px == 1) outP = weatherAt(uWeather, p);
  else if (px == 2) outP = vec4(climate(p, h, info.x) + vec2(seasonT(p, info.x).x, 0.0), 0.0, 0.0);   // today's temperature, not the annual mean
  else outP = vec4(h, 0.0, 0.0, 0.0);
}`

export class Probe {
  constructor(shared) {
    this.target = new THREE.WebGLRenderTarget(4, 1, { type: THREE.FloatType, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter })
    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: shared.quadVert, fragmentShader: FRAG,
      uniforms: { uDir: { value: new THREE.Vector3(0, 0, 1) }, uAhead: { value: new THREE.Vector3(0, 0, 1) }, uWeather: shared.uWeather, ...shared.civ },
    })
    this.quad = shared.makeQuad(this.material)
    this.buf = new Float32Array(16)
    this.busy = false
    // latest readings
    this.height = 0; this.land = 0; this.ahead = 0   // ahead: height where the camera is heading
    this.cover = 0; this.storm = 0; this.precip = 0; this.lightning = 0
    this.temp = 15; this.moisture = 0.5
  }

  update(renderer, dir, ahead = dir) {
    if (this.busy) return
    this.material.uniforms.uDir.value.copy(dir)
    this.material.uniforms.uAhead.value.copy(ahead)
    renderer.setRenderTarget(this.target)
    renderer.render(this.quad.scene, this.quad.camera)
    this.busy = true
    renderer.readRenderTargetPixelsAsync(this.target, 0, 0, 4, 1, this.buf).then(b => {
      const d = b || this.buf
      this.height = d[0]; this.land = d[1]
      this.cover = d[4]; this.storm = d[5]; this.precip = d[6]; this.lightning = d[7]
      this.temp = d[8]; this.moisture = d[9]; this.ahead = d[12]
    }).catch(() => {}).finally(() => { this.busy = false })
  }
}
