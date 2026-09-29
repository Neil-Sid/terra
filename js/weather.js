// Global weather: an equirectangular map (cover, storm type, precipitation, lightning) rebuilt on
// the GPU each frame from advected noise, latitude circulation bands and a set of simulated cyclones.
import * as THREE from 'three'
import { R, GLSL_HEAD } from './config.js'
import { NOISE } from './glsl/noise.js'

const DEG = Math.PI / 180
const MAX_CYC = 10

const FRAG = /* glsl */`${GLSL_HEAD}
uniform float uT, uCover, uStorm, uItcz;
uniform vec4 uCyc[${MAX_CYC}];    // centre dir, signed strength (sign = spin)
uniform vec4 uCycB[${MAX_CYC}];   // radius (rad), tropical flag
in vec2 vUv;
out vec4 outW;
${NOISE}
vec3 rotY(vec3 p, float a){ float c = cos(a), s = sin(a); return vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z); }
vec3 rotAxis(vec3 p, vec3 k, float a){ float c = cos(a), s = sin(a); return p*c + cross(k, p)*s + k*dot(k, p)*(1.0 - c); }
float fbm(vec3 p){
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 7; i++){ s += a*snoise(p); p = p*2.03 + vec3(1.7, 9.2, 3.3); a *= 0.5; }
  return s;
}
float field(vec3 p, float seed){
  vec3 q = p*2.6 + seed;
  vec3 w = vec3(snoise(q*0.7), snoise(q*0.7 + 7.7), snoise(q*0.7 + 3.1));
  return fbm(q + w*0.55);
}
float wrapPi(float a){ return a - 2.0*PI*floor(a/(2.0*PI) + 0.5); }
void main(){
  float lon = (vUv.x - 0.5)*2.0*PI, lat = (vUv.y - 0.5)*PI;
  vec3 p = vec3(cos(lat)*sin(lon), sin(lat), cos(lat)*cos(lon));
  float latD = degrees(lat), al = abs(latD);

  // cyclones: wind the background field into a spiral and add their own structure
  vec3 ps = p;
  float cycCov = 0.0, cycStorm = 0.0, dry = 0.0, eye = 1.0;
  for (int i = 0; i < ${MAX_CYC}; i++){
    vec4 c = uCyc[i];
    if (c.w == 0.0) continue;
    float rad = uCycB[i].x, s = abs(c.w), spin = sign(c.w);
    float x = acos(clamp(dot(p, c.xyz), -1.0, 1.0))/rad;
    if (x > 2.6) continue;
    float k = max(1.0 - x, 0.0);
    ps = rotAxis(ps, c.xyz, -c.w*7.0*k*k);
    vec3 ex = normalize(cross(vec3(0.0, 1.0, 0.0), c.xyz)), ny = cross(c.xyz, ex);
    float th = spin*atan(dot(p, ny), dot(p, ex));   // 0 = east, +pi/2 = poleward, in either hemisphere
    float lx = log(x + 1e-3);
    if (uCycB[i].y > 0.5){
      // hurricane: dense central overcast, clear eye, two spiral rainbands and a clear moat
      float bands = exp(-pow(wrapPi(2.0*(th + 3.0*lx) + 0.8*snoise(p*90.0))/1.1, 2.0))*smoothstep(1.3, 0.3, x);
      float cdo = exp(-pow(x/0.2, 2.0));
      cycCov += s*(2.6*cdo + 1.0*bands);
      dry += s*0.5*exp(-pow((x - 0.33)/0.12, 2.0))*(1.0 - bands);
      cycStorm = max(cycStorm, s*max(cdo, 0.75*bands));
      eye *= mix(1.0, smoothstep(0.025, 0.055, x), smoothstep(0.5, 0.8, s));
    } else {
      // mid-latitude low: comma head wrapped round the centre, cold-front tail trailing equatorward, dry slot behind
      float ph = wrapPi(th + 2.6*lx + 0.35*snoise(p*40.0));
      float front = exp(-pow(ph*(0.5 + x)/0.55, 2.0))*smoothstep(2.6, 1.5, x);
      cycCov += s*(0.95*front + 0.35*exp(-x*x*3.0));
      dry += s*0.9*exp(-pow(wrapPi(ph + 1.3)/0.6, 2.0))*smoothstep(1.9, 0.7, x)*smoothstep(0.1, 0.35, x);
      cycStorm = max(cycStorm, s*front*0.6*smoothstep(0.5, 1.4, x));
    }
  }

  // zonal winds (m/s): trade easterlies, mid-latitude westerlies, polar easterlies
  float u = -7.0*exp(-pow(latD/18.0, 2.0)) + 14.0*exp(-pow((al - 45.0)/12.0, 2.0)) - 4.0*smoothstep(62.0, 75.0, al);
  float omega = u/(PLANET_R*max(cos(lat), 0.05));
  // two cross-faded layers keep the shear from accumulating forever
  const float PER = 2.0*86400.0;
  float ph0 = fract(uT/PER), ph1 = fract(uT/PER + 0.5);
  float w0 = 1.0 - abs(2.0*ph0 - 1.0);
  float n0 = field(rotY(ps, -omega*ph0*PER), floor(uT/PER)*3.7);
  float n1 = field(rotY(ps, -omega*ph1*PER), floor(uT/PER + 0.5)*3.7 + 1.9);
  float n = mix(n1, n0, w0) + 0.05*snoise(p*60.0 + uT*2e-5);

  // ITCZ: a broken string of convective clusters that bubble up and decay
  float itcz = exp(-pow((latD - uItcz)/7.0, 2.0));
  float clus = smoothstep(0.05, 0.6, 0.65*snoise(p*vec3(6.0, 13.0, 6.0) + vec3(0.0, 0.0, uT*4e-6)) + 0.35*snoise(p*19.0 - uT*6e-6));
  // subtropical stratocumulus decks: wide, low, flat sheets
  float sc = smoothstep(0.1, 0.5, snoise(p*4.0 + 5.3))*exp(-pow((al - 23.0)/9.0, 2.0));
  float band = itcz*(0.7*clus - 0.1) - 0.3*exp(-pow((al - 25.0)/8.0, 2.0)) + 0.45*sc
             + 0.2*exp(-pow((al - 55.0)/12.0, 2.0)) + 0.05;
  float cov = smoothstep(-0.05, 0.4, n + band + uCover + cycCov*0.6 - dry)*eye;
  float conv = smoothstep(0.25, 0.65, n + 0.9*itcz*clus - 0.2*smoothstep(35.0, 60.0, al) - 0.5*sc);
  float storm = clamp(max(conv*0.8, cycStorm)*uStorm, 0.0, 1.0)*smoothstep(0.35, 0.7, cov);
  float precip = smoothstep(0.62, 0.95, cov)*(0.25 + 0.75*storm);
  float lightning = smoothstep(0.55, 0.9, storm)*precip;
  outW = vec4(cov, storm, precip, lightning);
}`

export class Weather {
  constructor(shared) {
    this.target = new THREE.WebGLRenderTarget(2048, 1024, {
      type: THREE.HalfFloatType, depthBuffer: false, wrapS: THREE.RepeatWrapping,
      // mips feed the clouds' coarse clear-air test (explicit textureLod everywhere, so no seam derivatives)
      minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: true,
    })
    shared.uWeather.value = this.target.texture
    this.cyclones = []
    this.uCyc = Array.from({ length: MAX_CYC }, () => new THREE.Vector4())
    this.uCycB = Array.from({ length: MAX_CYC }, () => new THREE.Vector4())
    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, fragmentShader: FRAG, vertexShader: shared.quadVert, depthTest: false, depthWrite: false,
      uniforms: {
        uT: { value: 0 }, uCover: { value: 0 }, uStorm: { value: 1 }, uItcz: { value: 5 },
        uCyc: { value: this.uCyc }, uCycB: { value: this.uCycB },
      },
    })
    this.quad = shared.makeQuad(this.material)
    for (let i = 0; i < 8; i++) this.spawn(i < 3, Math.random())
  }

  spawn(tropical, age01 = 0) {
    const north = Math.random() < (tropical ? 0.6 : 0.5)
    const lat = (tropical ? 9 + Math.random() * 11 : 38 + Math.random() * 20) * (north ? 1 : -1)
    const life = (tropical ? 6 + Math.random() * 4 : 3 + Math.random() * 3) * 86400
    this.cyclones.push({
      tropical, lat: lat * DEG, lon: (Math.random() * 2 - 1) * Math.PI, life, age: age01 * life,
      radius: (tropical ? 5 + Math.random() * 3 : 10 + Math.random() * 7) * DEG,
      peak: tropical ? 0.7 + Math.random() * 0.3 : 0.45 + Math.random() * 0.4, spin: north ? 1 : -1,
    })
  }

  // dt in simulated seconds
  step(dt) {
    for (const c of this.cyclones) {
      c.age += dt
      const hemi = Math.sign(c.lat), al = Math.abs(c.lat) / DEG
      // tropical storms drift west and poleward, then recurve east; mid-latitude lows ride the westerlies
      const east = c.tropical ? (al < 25 ? -5 : 6) : 11, northward = c.tropical ? 2.2 * hemi : 1.0 * hemi
      c.lat += northward * dt / R
      c.lon += east * dt / (R * Math.cos(c.lat))
    }
    this.cyclones = this.cyclones.filter(c => {
      if (c.age < c.life && Math.abs(c.lat) < 75 * DEG) return true
      this.spawn(c.tropical)
      return false
    })
  }

  render(renderer, simTime, dt, settings) {
    this.step(dt)
    this.renderer = renderer   // kept for debug readback from the console
    const u = this.material.uniforms
    u.uT.value = simTime % (86400 * 4000)
    u.uCover.value = settings.cover
    u.uStorm.value = settings.storm
    const day = simTime / 86400
    u.uItcz.value = 6 * Math.sin(2 * Math.PI * (day - 100) / 365.25)
    this.uCyc.forEach((v, i) => {
      const c = this.cyclones[i]
      if (!c) return v.set(0, 0, 0, 0)
      const env = Math.sin(Math.PI * Math.min(1, c.age / c.life))
      const s = c.peak * Math.min(1, env * 1.6) * c.spin
      v.set(Math.cos(c.lat) * Math.sin(c.lon), Math.sin(c.lat), Math.cos(c.lat) * Math.cos(c.lon), s)
      this.uCycB[i].set(c.radius, c.tropical ? 1 : 0, 0, 0)
    })
    renderer.setRenderTarget(this.target)
    renderer.render(this.quad.scene, this.quad.camera)
  }
}
