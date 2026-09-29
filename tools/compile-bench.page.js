// Bench page for tools/compile-bench.cjs, served at /__bench/page.js on each root's server (so it imports that root's
// own modules). Builds a program's material the way the app does, compiles it through three with a nonce comment
// (no program-cache hit), and times the link until COMPLETION_STATUS_KHR.
import * as THREE from 'three'

const COMPLETION_STATUS_KHR = 0x91B1
const canvas = document.querySelector('canvas')
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' })
const gl = renderer.getContext()
const camera = new THREE.PerspectiveCamera(55, 1, 0.5, 2e8)
const khr = gl.getExtension('KHR_parallel_shader_compile')
const dbg = gl.getExtension('WEBGL_debug_shaders')
const rinfo = gl.getExtension('WEBGL_debug_renderer_info')
const sleep = ms => new Promise(r => setTimeout(r, ms))

// the app's shared uniforms and screen-pass helper (main.js), with the passes captured instead of registered
const quadVert = /* glsl */`
in vec3 position;
out vec2 vUv;
void main(){ vUv = position.xy*0.5 + 0.5; gl_Position = vec4(position.xy, 0.0, 1.0); }`
const quadGeo = new THREE.BufferGeometry()
quadGeo.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3))
let quads = []
const shared = {
  uSun: { value: new THREE.Vector3(0, 0, 1) }, uTime: { value: 0 }, uPixAngle: { value: 0.001 },
  uTransLUT: { value: null }, uWeather: { value: null }, quadVert,
  makeQuad(material) { quads.push(material); return { scene: new THREE.Scene(), camera: new THREE.OrthographicCamera(), mesh: null } },
  civ: {},
}

// program name -> () => { material, geometry }, built from the root's own modules on first use
let terrain = null
const getTerrain = async () => terrain ??= new (await import('/js/terrain.js')).Terrain(shared)
const PROGRAMS = {
  terrain: async () => { const t = await getTerrain(); return { material: t.material, geometry: t.geo } },
  preview: async () => { const t = await getTerrain(); return { material: (await import('/js/boot.js')).previewMaterial(t.material), geometry: t.geo } },
  sampler: async () => { quads = []; new (await import('/js/civ/sampler.js')).TerrainSampler(renderer, shared); return { material: quads[0], geometry: quadGeo } },
  probe: async () => { quads = []; new (await import('/js/probe.js')).Probe(shared); return { material: quads[0], geometry: quadGeo } },
}

// stand-in for the other stage, so a single stage can be timed: its inputs / outputs match the real shader's
const IO = /^\s*((?:flat|smooth|centroid|highp|mediump|lowp)\s+)*(in|out)\s+(\w+)\s+(\w+)\s*;/gm
const ioList = (src, dir) => [...src.matchAll(IO)].filter(m => m[2] === dir && m[4] !== 'outColor' && !/^(position)$/.test(m[4]))
  .map(m => ({ qual: /flat/.test(m[1] || '') ? 'flat ' : '', type: m[3], name: m[4] }))
const live = (type, name) => /^(i|u)?vec/.test(type) ? `float(${name}.x)` : `float(${name})`
function stubVS(fs) {
  const ins = ioList(fs, 'in')
  return `precision highp float;\nprecision highp int;\nin vec3 position;\n${ins.map(v => `${v.qual}out ${v.type} ${v.name};`).join('\n')}
void main(){\n${ins.map(v => `  ${v.name} = ${v.type}(position.x);`).join('\n')}\n  gl_Position = vec4(position, 1.0);\n}\n`
}
function stubFS(vs) {
  const outs = ioList(vs, 'out')
  return `precision highp float;\nprecision highp int;\n${outs.map(v => `${v.qual}in ${v.type} ${v.name};`).join('\n')}\nout vec4 outColor;
void main(){\n  float s = 0.0;\n${outs.map(v => `  s += ${live(v.type, v.name)};`).join('\n')}\n  outColor = vec4(s);\n}\n`
}

let count = 0
// compile one program; part: 'program' (both stages), 'fs' (real FS + stub VS) or 'vs' (real VS + stub FS)
// the HLSL fxc gets: getTranslatedShaderSource holds ANGLE's initial HLSL and, once linked, the compiler input
const fxcInput = h => { const m = h.match(/\/\/ COMPILER INPUT HLSL BEGIN([\s\S]*?)\/\/ COMPILER INPUT HLSL END/); return m ? m[1] : h }
// alongside: other programs compiled at the same time, untimed (like the app's start-up, where the sampler, preview,
// weather, post and probe programs link while the terrain does)
async function compile({ program = 'terrain', defines = {}, part = 'program', hlsl = false, nonce = true, alongside = [] }) {
  const { material: src, geometry } = await PROGRAMS[program]()
  let vs = src.vertexShader, fs = src.fragmentShader
  if (part === 'fs') vs = stubVS(fs)
  if (part === 'vs') fs = stubFS(vs)
  const tagOf = () => nonce ? `\n// compile-bench nonce ${Date.now()}-${++count}-${Math.random().toString(36).slice(2)}\n` : '\n// compile-bench fixed\n'
  const tag = tagOf()
  const mat = new THREE.RawShaderMaterial({
    glslVersion: src.glslVersion, vertexShader: vs + tag, fragmentShader: fs + tag,
    uniforms: src.uniforms, defines: { ...src.defines, ...defines },
  })
  const scene = new THREE.Scene().add(new THREE.Mesh(geometry, mat))
  const others = []
  for (const name of alongside) {
    const o = await PROGRAMS[name](), t = tagOf()
    others.push(new THREE.Mesh(o.geometry, new THREE.RawShaderMaterial({ glslVersion: o.material.glslVersion,
      vertexShader: o.material.vertexShader + t, fragmentShader: o.material.fragmentShader + t, uniforms: o.material.uniforms, defines: o.material.defines })))
  }
  const t0 = performance.now()
  renderer.compile(scene, camera)
  const issued = performance.now() - t0
  const oScene = new THREE.Scene()
  others.forEach(m => oScene.add(m))
  const oDone = others.length ? renderer.compileAsync(oScene, camera) : null
  const p = renderer.properties.get(mat).currentProgram
  while (!gl.getProgramParameter(p.program, COMPLETION_STATUS_KHR)) await sleep(4)
  const ms = performance.now() - t0
  if (oDone) { await oDone; others.forEach(m => m.material.dispose()) }
  const ok = gl.getProgramParameter(p.program, gl.LINK_STATUS)
  const vsH = dbg ? dbg.getTranslatedShaderSource(p.vertexShader) : '', fsH = dbg ? dbg.getTranslatedShaderSource(p.fragmentShader) : ''
  const res = { program, part, ms: Math.round(ms), issued: Math.round(issued), ok, vsHlsl: fxcInput(vsH).length, fsHlsl: fxcInput(fsH).length }
  if (!ok) res.log = [gl.getProgramInfoLog(p.program), gl.getShaderInfoLog(p.vertexShader), gl.getShaderInfoLog(p.fragmentShader)].join('\n').slice(0, 4000)
  if (hlsl) Object.assign(res, { vsText: vsH, fsText: fsH, vsGlsl: gl.getShaderSource(p.vertexShader), fsGlsl: gl.getShaderSource(p.fragmentShader) })
  mat.dispose()
  return res
}

// a tiny program first: loads the D3D compiler and wakes the worker threads before anything is timed
async function warm() {
  const mat = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: stubVS('in vec3 vA;'), fragmentShader: stubFS('out vec3 vA;') })
  const scene = new THREE.Scene().add(new THREE.Mesh(quadGeo, mat))
  const t0 = performance.now()
  renderer.compile(scene, camera)
  const p = renderer.properties.get(mat).currentProgram
  while (!gl.getProgramParameter(p.program, COMPLETION_STATUS_KHR)) await sleep(2)
  mat.dispose()
  return Math.round(performance.now() - t0)
}

try {
  shared.civ = (await import('/js/civ/world.js')).civUniforms()
} catch (e) { console.warn('[bench] civUniforms unavailable: ' + e.message) }
window.bench = {
  compile, warm,
  info: () => ({
    khr: !!khr, debugShaders: !!dbg,
    gpu: rinfo ? gl.getParameter(rinfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    ua: navigator.userAgent,
  }),
}
window.benchReady = true
