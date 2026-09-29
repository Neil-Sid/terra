// Frame pipeline: weather map -> terrain (HDR + log depth) -> sky/cloud composite -> precipitation.
// Start-up: all shaders compile in parallel in the background while the world is built; the curtain lifts once
// the full surface and every civ program are linked (a preview surface draws behind it until then; see boot.js).
import * as THREE from 'three'
import { R, FAR } from './config.js'
import { GeoCamera } from './camera.js'
import { Terrain } from './terrain.js'
import { Weather } from './weather.js'
import { Post } from './post.js'
import { Probe } from './probe.js'
import { Precip } from './precip.js'
import { Hud } from './hud.js'
import { makeCloudNoise } from './noise3d.js'
import { TerrainSampler } from './civ/sampler.js'
import { buildWorld, civUniforms } from './civ/world.js'
import { Civilisation } from './civ/index.js'
import { previewMaterial, meshScene, compile as compileScenes, stage, hold, heldFor, creep, live, done } from './boot.js'

const DEG = Math.PI / 180
const HOLD_MAX = 75000       // ms: past this the curtain lifts onto the preview surface anyway (a very slow compile)
const canvas = document.querySelector('#view')
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' })
renderer.autoClear = false
renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
const camera = new THREE.PerspectiveCamera(55, 1, 0.5, FAR)
const compile = (scenes, label) => compileScenes(renderer, camera, scenes, label)

// full-screen triangle used by every screen pass; each pass registers its scene for precompiling
const quadVert = /* glsl */`
in vec3 position;
out vec2 vUv;
void main(){ vUv = position.xy*0.5 + 0.5; gl_Position = vec4(position.xy, 0.0, 1.0); }`
const quadGeo = new THREE.BufferGeometry()
quadGeo.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3))
const quadCam = new THREE.OrthographicCamera()
const passes = []
function makeQuad(material) {
  const mesh = new THREE.Mesh(quadGeo, material)
  mesh.frustumCulled = false
  const scene = new THREE.Scene()
  scene.add(mesh)
  passes.push(scene)
  return { scene, camera: quadCam, mesh }
}

const shared = {
  uSun: { value: new THREE.Vector3(0, 0, 1) }, uTime: { value: 0 }, uPixAngle: { value: 0.001 },
  uTransLUT: { value: null }, uWeather: { value: null }, quadVert, makeQuad,
  civ: civUniforms(),       // roads, railways and towns (filled in once the world is built)
}

// compiles start longest first (the full terrain shader), then the sampler the world build waits on,
// then what the first frame draws, then the probe
const terrain = new Terrain(shared)
const fullReady = compile([meshScene(terrain.material, terrain.geo)], 'terrain')
const sampler = new TerrainSampler(renderer, shared)
const samplerReady = compile(passes.splice(0), 'sampler')
const weather = new Weather(shared)
const post = new Post(shared, makeCloudNoise(64))
const precip = new Precip()
const preview = previewMaterial(terrain.material)
const firstReady = compile([...passes.splice(0), precip.scene, meshScene(preview, terrain.geo)], 'preview, weather and post')
const probe = new Probe(shared)
const probeReady = compile(passes.splice(0), 'probe')

const sceneRT = new THREE.WebGLRenderTarget(1, 1, {
  type: THREE.HalfFloatType, depthBuffer: true,
  depthTexture: new THREE.DepthTexture(1, 1, THREE.FloatType),
})
const geo = new GeoCamera(canvas)

// Simulated clock: seconds since Jan 1. The sun moves; the planet frame stays fixed.
const settings = { timeScale: 60, cover: 0, storm: 1, exposure: 0.55 }
const START_LON = 28 * DEG
let simTime = 172 * 86400 + (10.5 - START_LON / DEG / 15) * 3600
geo.flyTo({ lat: 14 * DEG, lon: START_LON, agl: 2.2e7 })
geo.agl = geo.targetAgl

function sunDirection(t) {
  const day = t / 86400, utc = (t / 3600) % 24
  const decl = 23.44 * DEG * Math.sin(2 * Math.PI * (day - 80) / 365.25)
  const lon = -(utc - 12) * 15 * DEG
  return new THREE.Vector3(Math.cos(decl) * Math.sin(lon), Math.sin(decl), Math.cos(decl) * Math.cos(lon))
}

const wind = new THREE.Vector3()
// the HUD reads and sets the clock, and looks up storms for "Find a storm"
const clock = { get time() { return simTime }, set time(v) { simTime = v } }
const hud = new Hud(settings, geo, clock, weather)

function resize() {
  const w = innerWidth, h = innerHeight
  renderer.setSize(w, h, false)
  const pr = renderer.getPixelRatio()
  sceneRT.setSize(Math.round(w * pr), Math.round(h * pr))
  camera.aspect = w / h
  camera.updateProjectionMatrix()
  shared.uPixAngle.value = 2 * Math.tan(camera.fov * DEG / 2) / (h * pr)
  geo.pixAngle = 2 * Math.tan(camera.fov * DEG / 2) / h
}
addEventListener('resize', resize)
resize()

// what has finished compiling: until then the probe waits, civ objects stay hidden, the preview surface draws
const on = { probe: false, civ: false, all: false }
let world, civ

async function start() {
  await stage('Compiling shaders')
  await samplerReady
  // settlements and transport networks, placed on the real terrain via the GPU sampler
  world = await buildWorld(sampler, stage)
  for (const [k, tex] of Object.entries({ uCivGrid: 'grid', uCivItems: 'items', uCivSeg: 'seg', uCivCity: 'city', uUrbanMap: 'urban', uFarmMap: 'farm' })) shared.civ[k].value = world.textures[tex]
  console.info(`terra: ${world.cities.length} settlements, ${world.routes.length} routes, ${world.segCount} segments, built in ${world.buildTime} ms`)

  // civ modules find the full terrain material to add their uniforms, so it stays on the mesh until they exist
  civ = new Civilisation({ world, sampler, shared, scene: terrain.scene, renderer, FAR })
  terrain.mesh.material = preview
  const civReady = compile([...passes.splice(0), terrain.scene], 'civ objects')
  probeReady.then(() => { on.probe = true })
  fullReady.then(() => { terrain.mesh.material = terrain.material })
  civReady.then(() => { on.civ = true; civ.prewarm() })
  Promise.all([probeReady, fullReady, civReady]).then(() => { on.all = true })

  // handle for the console and the headless harness; ready = full detail compiled and drawn
  window.terra = { geo, settings, probe, terrain, weather, world, sampler, shared, civ, renderer, camera, get simTime() { return simTime }, set simTime(v) { simTime = v }, R, ready: false, boot: {} }
  await stage('Compiling the surface shader')
  await firstReady
  last = performance.now()
  requestAnimationFrame(frame)
}

let last = performance.now(), fpsAvg = 60, cpuMs = 4
function frame(now) {
  const t0 = performance.now()
  if (!window.terra.ready) geo.keys.clear()   // no flying behind the curtain
  const dt = Math.min(0.1, (now - last) / 1000)
  last = now
  fpsAvg += (1 / Math.max(dt, 1e-3) - fpsAvg) * 0.05
  const simDt = dt * settings.timeScale
  simTime += simDt
  const season = 2 * Math.PI * (simTime / 86400 - 172) / 365.25; shared.civ.uSeason.value.set(Math.cos(season), Math.sin(season))

  geo.ground = Math.max(probe.height, probe.ahead, 0)   // the probe is a frame late: also look ahead
  // don't pass through buildings: inside a footprint below its roof, the roof becomes the floor
  const roofs = [civ.parts.buildings.heightAt?.(geo.up), civ.parts.factories.heightAt?.(geo.up), civ.parts.farms.heightAt?.(geo.up),
    civ.parts.bridges?.onBridge?.(geo.up, geo.ground + geo.agl)].filter(h => h != null)   // decks far overhead don't count
  const roof = roofs.length ? Math.max(...roofs) : null
  if (roof != null && geo.ground + geo.agl < roof + 1.5) geo.ground = roof + 1.5 - geo.agl
  geo.update(dt)
  camera.position.set(0, 0, 0)
  camera.quaternion.copy(geo.quat)
  // near plane from the clearance above whatever is below: terrain, or a roof / bridge deck underneath
  const clearance = roof != null ? Math.max(geo.altitude - roof, 0.5) : geo.agl
  camera.near = THREE.MathUtils.clamp(Math.min(geo.agl, clearance) * 0.25, 0.3, 2000)
  camera.updateProjectionMatrix()
  camera.updateMatrixWorld()

  const sun = sunDirection(simTime)
  shared.uSun.value.copy(sun)
  shared.uTime.value = now / 1000
  // cloud noise drifts with a slow westerly plus evolution
  wind.set(simTime * 0.004, simTime * 0.0003, simTime * 0.002)

  weather.render(renderer, simTime, simDt, settings)
  terrain.update(geo.pos, camera, geo.groundSmooth)
  if (on.civ) civ.update({ camPos: geo.pos, camDir: geo.up, camera, sun, simTime, time: now / 1000, dt, simDt, geo, cpuMs, dest: geo.destination() })

  renderer.setRenderTarget(sceneRT)
  renderer.setClearColor(0x000000, 0)
  renderer.clear()
  // civ objects still compiling are left out of this frame (drawing them would wait for the compile)
  const hidden = on.civ ? [] : terrain.scene.children.filter(m => m !== terrain.mesh && m.visible)
  for (const m of hidden) m.visible = false
  renderer.render(terrain.scene, camera)
  for (const m of hidden) m.visible = true

  precip.update(dt, now / 1000, geo, probe)
  post.render(renderer, sceneRT, camera, geo, {
    exposure: settings.exposure, flash: precip.flash, wind,
    fog: precip.amount * 3e-4,
  })
  precip.render(renderer, camera)
  if (on.probe) probe.update(renderer, geo.up, geo.ahead)

  hud.update({ simTime, sun, probe, precip, fps: fpsAvg, patches: terrain.count })
  bootProgress()
  cpuMs = performance.now() - t0          // main-thread time of this frame (civ/index.js sizes the streaming budget by it)
  requestAnimationFrame(frame)
}

// first frame: the curtain holds (the preview draws behind it); once everything is compiled and drawn: ready
function bootProgress() {
  const b = window.terra.boot, now = performance.now()
  if (!b.firstFrame) {
    b.firstFrame = Math.round(now)
    b.longest = 0             // longest frame (ms) between the first frame and ready: compile hitches show here
    console.info(`boot: first frame at ${b.firstFrame} ms`)
    hold('Compiling the surface shader')
  } else if (!window.terra.ready) {
    b.longest = Math.max(b.longest, Math.round(now - b.last))
    creep()
    if (!b.lifted && heldFor() > HOLD_MAX) { b.lifted = Math.round(now); live('Sharpening the surface') }
    if (on.all) {
      b.ready = Math.round(now)
      window.terra.ready = true
      console.info(`boot: full detail at ${b.ready} ms, longest frame on the way ${b.longest} ms`)
      done()
    }
  }
  b.last = now
}

start()
