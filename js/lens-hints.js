// Tells three-lens (../three-lens) what only terra knows: every mesh here is placed by its shader, so
// geometry bounds are meaningless. Attaches userData.lens hints; changes nothing about rendering.
import { R } from './config.js'

const PEAKS = 12000   // m: headroom above sea level for the terrain's bounding sphere
const DEG = 180 / Math.PI

// Render space is the planet frame shifted so the camera sits at the origin: planet centre = -geo.pos.
export function lensHints({ geo, terrain, world, civ }) {
  const centre = () => geo.pos.clone().negate()
  const place = dir => {
    const near = world.nearestCity([dir.x, dir.y, dir.z], 60000)
    if (!near) return 'open country'
    const distance = metres(near.city.dir, dir)
    const d = distance < 1000 ? `${distance.toFixed(0)} m` : `${(distance / 1000).toFixed(1)} km`
    return distance < near.city.radius ? `in ${near.city.name}, ${d} from its centre` : `${d} from ${near.city.name}`
  }
  const latLon = dir => `${(Math.asin(dir.y) * DEG).toFixed(3)}°, ${(Math.atan2(dir.x, dir.z) * DEG).toFixed(3)}°`

  terrain.mesh.userData.lens = {
    name: 'terrain',
    bounds: () => ({ center: centre(), radius: R + PEAKS }),
    // the same ray-sphere test the camera's own picking uses, at the local ground level
    probe(ray) {
      const c = geo.pos, rg = R + Math.max(geo.groundSmooth, 0)
      const local = ray.origin.clone().add(c), b = local.dot(ray.direction)
      const disc = b * b - (local.lengthSq() - rg * rg)
      if (disc < 0) return null
      const t = -b - Math.sqrt(disc)
      if (t <= 0) return null
      const point = ray.origin.clone().addScaledVector(ray.direction, t)
      const dir = point.clone().add(c).normalize()
      const roof = civ.parts.buildings.heightAt?.(dir)
      return { point, label: `${latLon(dir)} · ${place(dir)}${roof != null ? ` · building roof ${roof.toFixed(0)} m` : ''}` }
    },
    describe: () => `camera over ${latLon(geo.up)} (${place(geo.up)}), ${geo.agl.toFixed(0)} m above ground, ground ${geo.ground.toFixed(0)} m above sea level`,
  }

  // civ parts keep their meshes in assorted fields: name each after its part and field
  for (const [part, obj] of Object.entries(civ.parts)) {
    for (const [field, value] of Object.entries(obj)) {
      for (const [i, mesh] of meshesIn(value)) mesh.userData.lens ??= { name: field === 'mesh' ? part : `${part}.${field}${i ?? ''}` }
    }
  }
}

// Great-circle metres from the chord: stable at short range, unlike acos of a dot product of
// float32 directions (world.js arc() reads 0 for anything within ~900 m).
function metres(a, b) {
  const u = Math.hypot(a[0], a[1], a[2])
  const chord = Math.hypot(a[0] / u - b.x, a[1] / u - b.y, a[2] / u - b.z)
  return 2 * R * Math.asin(Math.min(1, chord / 2))
}

function meshesIn(value) {
  if (value?.isObject3D) return [[null, value]]
  if (Array.isArray(value)) return value.flatMap((v, i) => (v?.isObject3D ? [[i, v]] : v?.mesh?.isObject3D ? [[i, v.mesh]] : []))
  if (value && typeof value === 'object' && !value.isMaterial && !value.isBufferGeometry) {
    return Object.entries(value).flatMap(([k, v]) => (v?.mesh?.isObject3D ? [[`.${k}`, v.mesh]] : []))
  }
  return []
}
