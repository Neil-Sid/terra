// Keeping CPU-placed things out of rivers and lakes (buildings, factory plots, farmsteads, street traffic, people).
// world.nearestWater() is exact for rivers (distance to the channel edge, meander included) but finds lakes by their
// 0.15 deg cells: the real shore is where the ground is below the lake level, so a lake hit needs ground heights.
// Notes: notes/CIV.md (hydro), notes/buildings.md.
export const LAKE_DRY = 0.5             // ground this far above a lake's level is dry (m)

// any river or lake within r of the city centre (cached: most cities have none, so their blocks skip every test)
const CITIES = new WeakMap()
export function cityWater(world, city, r) {
  let w = CITIES.get(city)
  if (w === undefined) CITIES.set(city, w = !!world.nearestWater(city.dir, r))
  return w
}

// a footprint of radius r (m) at dir: 'river' if it touches a river channel, else the level of a lake its ground
// must stay LAKE_DRY above (-Infinity: no lake near)
export function waterAt(world, dir, r) {
  const w = world.nearestWater(dir, r)
  return !w ? -Infinity : w.type === 'river' ? 'river' : w.level
}

// lake tests for things without ground heights of their own: add(obj, dirs, level) marks obj.wet = true, samples
// the points (flat xyz) and sets obj.wet = lowest ground < level + LAKE_DRY, then calls done(). One sampler request
// per frame's batch.
export class LakeTest {
  constructor(sampler, done = () => {}) { this.sampler = sampler; this.done = done; this.queue = []; this.busy = false }
  add(obj, dirs, level) {
    obj.wet = true
    this.queue.push([obj, dirs, level])
    if (!this.busy) { this.busy = true; queueMicrotask(() => this.flush()) }
  }
  flush() {
    const q = this.queue.splice(0, 4000)
    if (!q.length) { this.busy = false; return }
    const dirs = new Float32Array(q.reduce((n, [, d]) => n + d.length, 0))
    let o = 0
    for (const [, d] of q) { dirs.set(d, o); o += d.length }
    this.sampler.sampleAsync(dirs, 1.2e6, true).then(s => {
      let i = 0
      for (const [obj, d, level] of q) {
        let lo = Infinity
        for (let k = 0; k < d.length; k += 3, i++) lo = Math.min(lo, s[i * 8])
        obj.wet = lo < level + LAKE_DRY
      }
      this.done()
      this.flush()
    }, () => this.flush())     // failed: stays wet
  }
}
