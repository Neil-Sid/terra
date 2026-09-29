// Start-up helpers. Every program is compiled in the background (KHR_parallel_shader_compile, polled by
// three's compileAsync), so the main thread never waits on a link. The full terrain shader takes 15+ s
// under ANGLE/D3D (much less on a repeat visit: Chrome caches the program); the curtain stays up until it
// and every civ program are linked, so the first zoom never meets a half-built surface. Meanwhile a cheap
// preview surface (height, biome colour, sun, sky) draws behind it, and after HOLD_MAX the curtain lifts onto
// it anyway. The overlay in index.html names each stage.
import * as THREE from 'three'
import { GLSL_HEAD } from './config.js'
import { NOISE } from './glsl/noise.js'
import { TERRAIN } from './glsl/terrain.glsl.js'
import { ATMO } from './glsl/atmo.glsl.js'
import { BIOME } from './glsl/biome.glsl.js'

// reads the terrain vertex shader's outputs vDir, vRel, vFragDepth and its uniforms
const PREVIEW_FRAG = /* glsl */`${GLSL_HEAD}
uniform vec3 uSun;
uniform float uPixAngle, uLogDepthFC;
uniform sampler2D uTransLUT;
in vec3 vDir;
in vec3 vRel;
in float vFragDepth;
out vec4 outColor;
${NOISE}
${TERRAIN}
${ATMO}
${BIOME}
void main(){
  vec3 p = normalize(vDir);
  float dist = length(vRel), foot = max(dist*uPixAngle, 1e-3);
  float maxF = min(PLANET_R/(2.0*max(foot, 0.05)), 7e5);
  civSetRel(vRel);
  vec3 info;
  Civ cv;
  vec4 hg = terrainCiv(p, maxF, info, cv);
  float h = hg.x, muS = dot(p, uSun), wl = max(0.0, cv.water);
  vec3 N = normalize(p - (hg.yzw - p*dot(hg.yzw, p))/PLANET_R), V = -vRel/dist;
  vec3 sunC = SUN_I*sunTrans(uTransLUT, max(h, 0.0), muS);
  vec3 amb = SUN_I*vec3(0.030, 0.045, 0.075)*smoothstep(-0.18, 0.35, muS);
  vec3 col;
  if (h < wl){
    // deep blue body, sandy shallows, sky fresnel and a soft sun glint
    vec3 absorb = exp(-vec3(0.38, 0.055, 0.028)*(wl - h));
    vec3 body = vec3(0.0009, 0.0042, 0.0115)*(1.0 - absorb.b) + vec3(0.002, 0.016, 0.022)*(1.0 - absorb.g) + vec3(0.40, 0.37, 0.27)*absorb;
    float fres = 0.02 + 0.98*pow(1.0 - max(dot(p, V), 0.0), 5.0);
    float glint = pow(max(dot(p, normalize(V + uSun)), 0.0), 300.0)*40.0;
    col = body*(sunC*max(muS, 0.0) + amb)*(1.0 - fres) + SUN_I*0.012*fres*smoothstep(-0.15, 0.2, muS) + sunC*glint*0.02;
  } else {
    vec3 mat;
    vec2 cl = climate(p, h, info.x);
    vec3 alb = landAlbedo(h, cl.x, cl.y, 1.0 - dot(N, p), info, p, maxF, mat);
    col = alb*(sunC*max(dot(N, uSun), 0.0) + amb*(0.6 + 0.4*dot(N, p)));
  }
  outColor = vec4(col, 1.0);
  gl_FragDepth = log2(vFragDepth)*uLogDepthFC*0.5;
}`

// same vertex shader and uniforms as the full terrain material
export const previewMaterial = full => new THREE.RawShaderMaterial({
  glslVersion: THREE.GLSL3, vertexShader: full.vertexShader, fragmentShader: PREVIEW_FRAG, uniforms: full.uniforms,
})

// a throwaway scene holding one material, for compiling it before it is drawn
export const meshScene = (material, geometry) => new THREE.Scene().add(new THREE.Mesh(geometry, material))

// start compiling every material in the scenes now; resolves once all of them are linked
export function compile(renderer, camera, scenes, label) {
  const t0 = performance.now()
  return Promise.all(scenes.map(s => renderer.compileAsync(s, camera)))
    .then(() => console.info(`boot: ${label} compiled at ${Math.round(performance.now())} ms (${Math.round(performance.now() - t0)} ms)`))
}

// loading overlay: stage names while the world is built, then a small note until full detail is in
const overlay = document.querySelector('#boot'), steps = []
export function stage(text) {
  steps.push(text)
  overlay.querySelector('.stage').textContent = text
  overlay.style.setProperty('--progress', Math.min(1, steps.length / 10))
  // let the browser paint the overlay before the next block of work (rAF alone would run it before the paint)
  return new Promise(r => { requestAnimationFrame(() => setTimeout(r)); setTimeout(r, 100) })
}
// the last stage (the full surface shader linking): the hairline creeps on while it waits; input is held back
let held = 0
export function hold(text) {
  if (held) return
  held = performance.now()
  overlay.querySelector('.stage').textContent = text
  overlay.style.pointerEvents = 'auto'
}
export const heldFor = () => held ? performance.now() - held : 0
export function creep() {
  if (held) overlay.style.setProperty('--progress', 0.8 + 0.2 * (1 - Math.exp(-heldFor() / 12000)))
}
export function live(note) {
  overlay.style.pointerEvents = 'none'
  overlay.querySelector('.later').textContent = note
  overlay.classList.add('live')
}
export function done() {
  overlay.style.pointerEvents = 'none'
  overlay.classList.add('done')
  setTimeout(() => overlay.remove(), 1200)
}
