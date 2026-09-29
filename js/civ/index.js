// Civilisation manager: owns the per-feature modules and updates them every frame.
// ctx   = { world, sampler, shared, scene, renderer, FAR }   (scene = terrain.scene, drawn into the HDR target)
// frame = { camPos, camDir, camera, sun, simTime, time, dt, simDt, geo, cpuMs, dest }
//   cpuMs: main-thread ms of the last frame; dest: where the camera is heading ({ dir, agl } or null, camera.js)
// Streaming budget: while a new area streams in, the rebuild jobs (buildings, factories) may use frame.budget ms per
// frame: what is left of TARGET ms of main-thread work, or of 60 % of the frame interval when the GPU sets the pace
// (then the main thread would idle anyway). Routine rebuilds keep their own small BUDGET.
// Prediction: while the camera is too high to see any of them, buildings, factories and farmsteads collect, design and
// fetch heights around frame.dest instead (frame.around), so a zoom or flight lands on blocks that are ready to draw.
import { Buildings } from './buildings.js'
import { Traffic } from './traffic.js'
import { Trains } from './trains.js'
import { People } from './people.js'
import { Factories } from './factories.js'
import { Farms } from './farms.js'
import { Trees } from './trees.js'
import { Bridges } from './bridges.js'

const TARGET = 12            // ms of main-thread work per frame while streaming (~80 fps on a quiet GPU)
const MAX_BUDGET = 14        // ms per frame for the rebuild jobs at most
const PREDICT_AGL = 25000    // above this no building or factory is drawn (22 km ranges + margin)

export class Civilisation {
  constructor(ctx) {
    this.parts = { buildings: new Buildings(ctx), traffic: new Traffic(ctx), trains: new Trains(ctx), people: new People(ctx), factories: new Factories(ctx), farms: new Farms(ctx), trees: new Trees(ctx), bridges: new Bridges(ctx) }
    this.dtAvg = 16
    this.other = 4
    this.jobMs = 0             // time the rebuild jobs took in the last frame
  }
  update(frame) {
    this.dtAvg += (frame.dt * 1000 - this.dtAvg) * 0.1
    this.other += (Math.max(0, (frame.cpuMs ?? 4) - this.jobMs) - this.other) * 0.3   // main-thread ms besides the jobs
    frame.budget = Math.min(MAX_BUDGET, Math.max(0, Math.max(TARGET, 0.6 * this.dtAvg) - this.other))
    const d = frame.dest
    frame.around = d && frame.geo.agl > PREDICT_AGL && d.agl < PREDICT_AGL ? d.dir : frame.camDir   // where to collect
    let jobs = 0
    for (const [name, part] of Object.entries(this.parts)) {
      const t0 = performance.now()
      part.update(frame)
      if (name === 'buildings' || name === 'factories') jobs += performance.now() - t0
    }
    this.jobMs = jobs
    this.budget = frame.budget     // (debug)
  }
  // while the loading curtain is up: warm caches so the first zoom has less to fetch
  prewarm(frame) {
    this.parts.buildings.prewarm?.(frame)
  }
}
