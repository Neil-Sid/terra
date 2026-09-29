// Survey-instrument HUD: position and conditions readouts, compass tape, scale bar, clock and weather controls.
import { R } from './config.js'

const DEG = 180 / Math.PI
const $ = s => document.querySelector(s)
const clamp = (x, a, b) => Math.min(b, Math.max(a, x))
const mod = (x, m) => ((x % m) + m) % m

function fmtDist(m) {
  if (m >= 1e6) return (m / 1000).toLocaleString('en', { maximumFractionDigits: 0 }) + ' km'
  if (m >= 1e4) return (m / 1000).toFixed(1) + ' km'
  return Math.round(m).toLocaleString('en') + ' m'
}
const fmtAngle = (a, pos, neg) => `${Math.abs(a * DEG).toFixed(3)}° ${a >= 0 ? pos : neg}`
const fmtClock = h => `${String(Math.floor(h)).padStart(2, '0')}:${String(Math.floor((h % 1) * 60)).padStart(2, '0')}`
const signed = x => `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(0)}`
const TIERS = ['village', 'town', 'city', 'metropolis']
const fmtPop = n => n >= 1e6 ? `${(n / 1e6).toFixed(1)} M` : n >= 1e3 ? `${Math.round(n / 1e3)} k` : String(Math.round(n))

// standard-atmosphere profile above the local surface temperature; null above the stratosphere
function airTemp(surface, h) {
  if (h > 30000) return null
  const z = Math.max(h, 0)
  return surface - 0.0065 * Math.min(z, 11000) + 0.001 * Math.max(z - 20000, 0)
}

// current strength of a cyclone, matching the envelope in weather.js
const stormStrength = c => c.peak * Math.min(1, Math.sin(Math.PI * Math.min(1, c.age / c.life)) * 1.6)

// round scale-bar length: 1, 2 or 5 times a power of ten
function niceLength(m) {
  const p = 10 ** Math.floor(Math.log10(m))
  return [5, 2, 1].find(n => n * p <= m) * p
}

const COMPASS_K = 2.2    // tape pixels per degree
function buildTape(tape) {
  const names = { 0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW' }
  let html = ''
  for (let d = -180; d <= 540; d += 5) {
    const a = mod(d, 360), name = names[a], major = a % 15 === 0
    const cls = name ? 'tk name' : major ? 'tk major' : 'tk'
    const label = name ?? (a % 30 === 0 ? a : '')
    html += `<i class="${cls}" style="left:${(d + 180) * COMPASS_K}px">${label === '' ? '' : `<em>${label}</em>`}</i>`
  }
  tape.innerHTML = html
}

export class Hud {
  constructor(settings, geo, clock, weather) {
    Object.assign(this, { settings, geo, clock, weather })
    document.querySelectorAll('[data-warp]').forEach(b => b.addEventListener('click', () => {
      settings.timeScale = +b.dataset.warp
      this.markWarp()
    }))
    this.markWarp()
    for (const key of ['cover', 'storm', 'exposure']) {
      const input = $(`#${key}`)
      input.value = settings[key]
      input.addEventListener('input', () => { settings[key] = +input.value })
    }
    // time of day: set the local hour at the camera longitude, keep the day
    this.tod = $('#tod')
    this.tod.addEventListener('input', () => {
      const day = Math.floor(clock.time / 86400)
      clock.time = day * 86400 + mod(+this.tod.value - geo.lon * DEG / 15, 24) * 3600
    })
    $('#go-orbit').onclick = () => geo.travel({ agl: 2.2e7, pitch: -Math.PI / 2 })
    $('#go-surface').onclick = () => geo.travel({ agl: 2.5, pitch: -0.08 })
    $('#go-random').onclick = () => geo.travel({ lat: Math.asin(Math.random() * 1.6 - 0.8), lon: (Math.random() * 2 - 1) * Math.PI })
    $('#go-storm').onclick = () => this.findStorm()
    $('#go-city').onclick = () => this.visitCity()
    $('#go-train').onclick = () => this.rideTrain()
    addEventListener('keydown', e => {
      if (e.code === 'KeyH' && e.target.tagName !== 'INPUT') document.body.classList.toggle('hud-off')
    })
    if (innerWidth < 900 || innerHeight < 640) $('#panel').open = false
    buildTape($('#tape'))
    this.el = Object.fromEntries(['alt', 'agl', 'lat', 'lon', 'pitch', 'spd', 'clock', 'sunel', 'cond', 'temp', 'stats', 'note', 'place', 'hdg', 'scale', 'scalebar', 'tape', 'go-train']
      .map(k => [k, $(`#r-${k}`) || $(`#${k}`)]))
    this.tick = 0
  }

  markWarp() {
    document.querySelectorAll('[data-warp]').forEach(b => b.classList.toggle('on', +b.dataset.warp === this.settings.timeScale))
  }

  note(text) {
    this.el.note.textContent = text
    clearTimeout(this.noteTimer)
    this.noteTimer = setTimeout(() => { this.el.note.textContent = '' }, 9000)
  }

  // fly to the strongest active cyclone (preferring daylight), far enough up to frame it, looking north across it
  findStorm() {
    const s = this.sun, lit = c => !s ? 1 : 0.3 + 0.7 * clamp(3 * (s.x * Math.cos(c.lat) * Math.sin(c.lon) + s.y * Math.sin(c.lat) + s.z * Math.cos(c.lat) * Math.cos(c.lon)) + 0.3, 0, 1)
    const score = c => stormStrength(c) * lit(c)
    const c = [...this.weather.cyclones].sort((a, b) => score(b) - score(a))[0]
    if (!c) return this.note('No storms on the map right now')
    const agl = clamp(c.radius * R * 2.4, 3e5, 4e6), pitch = -1.2
    const lat = clamp(c.lat - agl / Math.tan(-pitch) / R, -1.5, 1.5)
    this.geo.travel({ lat, lon: c.lon, agl, pitch, heading: 0 })
    const kind = c.tropical ? 'tropical cyclone' : 'mid-latitude low'
    const deg = (a, pos, neg) => `${Math.abs(a * DEG).toFixed(0)}° ${a >= 0 ? pos : neg}`
    this.note(`Flying to a ${kind} near ${deg(c.lat, 'N', 'S')}, ${deg(c.lon, 'E', 'W')}`)
  }

  // fly to a random town or city in daylight, 2.5 km up and a few km out, looking at its centre
  visitCity() {
    const world = window.terra?.world
    const towns = world?.cities.filter(c => c.tier >= 1) ?? []
    if (!towns.length) return this.note('No towns on the map yet')
    const c = towns[Math.floor(Math.random() * towns.length)], lonH = c.lon * DEG / 15
    const local = mod(this.clock.time / 3600 + lonH, 24)
    if (local < 9 || local > 16) {
      const day = Math.floor(this.clock.time / 86400)
      this.clock.time = day * 86400 + mod(10 + Math.random() * 5 - lonH, 24) * 3600
    }
    const agl = 2500, back = clamp(c.radius * 1.2, 3000, 8000), pitch = -Math.atan(agl / back), heading = Math.random() * 2 * Math.PI
    const lat = c.lat - Math.cos(heading) * back / R, lon = c.lon - Math.sin(heading) * back / (R * Math.cos(c.lat))
    this.geo.travel({ lat, lon, agl, pitch, heading })
    this.note(`Flying to ${c.name} (${TIERS[c.tier]}, ${fmtPop(c.pop)})`)
  }

  // fly to the train nearest the camera and ride it (the button again, a move key or a left drag gets off)
  rideTrain() {
    const trains = window.terra?.civ?.parts.trains
    if (!trains?.trains.length) return this.note('No trains running yet')
    if (!trains.ride(this.geo)) return this.note('Off the train')
    this.riding = true
  }

  // settlement under the camera: inside its radius, near it, or none within 60 km
  place() {
    const u = this.geo.up, hit = window.terra?.world?.nearestCity([u.x, u.y, u.z], 60000)
    if (!hit) return 'open country'
    const { city: c, distance: d } = hit
    return d < c.radius ? `${c.name} · ${TIERS[c.tier]} · ${fmtPop(c.pop)}` : `near ${c.name} · ${(d / 1000).toFixed(1)} km`
  }

  // heading tape every frame (cheap transform); the text readouts every few frames
  update({ simTime, sun, probe, precip, fps, patches }) {
    const g = this.geo, e = this.el
    this.sun = sun
    const hdg = mod(g.heading * DEG, 360)
    e.tape.style.transform = `translateX(${-(hdg + 180) * COMPASS_K}px)`
    if (this.tick++ % 6) return
    e.hdg.textContent = String(Math.round(hdg) % 360).padStart(3, '0')
    e.alt.textContent = fmtDist(g.altitude)
    e.agl.textContent = fmtDist(g.agl)
    e.lat.textContent = fmtAngle(g.lat, 'N', 'S')
    e.lon.textContent = fmtAngle(g.lon, 'E', 'W')
    e.pitch.textContent = `${signed(g.pitch * DEG)}°`
    e.spd.textContent = g.speed > 1 ? fmtDist(g.speed) + '/s' : '—'
    const local = mod(simTime / 3600 + g.lon * DEG / 15, 24)
    const day = Math.floor(simTime / 86400) % 365
    e.clock.textContent = `${fmtClock(local)} · day ${day + 1}`
    if (!this.tod.matches(':active')) this.tod.value = local
    const el = Math.asin(clamp(sun.dot(g.up), -1, 1)) * DEG
    e.sunel.textContent = `${el >= 0 ? '+' : '−'}${Math.abs(el).toFixed(1)}°`
    e.cond.textContent = precip.state
    const t = airTemp(probe.temp, g.altitude - Math.max(probe.height, 0))
    e.temp.textContent = t === null ? '—' : `${t.toFixed(0)} °C` + (probe.height < 0 ? ' · over ocean' : '')
    // lateral scale at the view centre
    const mpp = g.groundPerPixel(), len = niceLength(mpp * 110)
    e.scalebar.style.width = `${(len / mpp).toFixed(0)}px`
    e.scale.textContent = len >= 1000 ? `${(len / 1000).toLocaleString('en')} km` : `${len} m`
    if (this.tick % 30 === 1) e.place.textContent = this.place()
    e.stats.textContent = `${fps.toFixed(0)} fps · ${patches} patches`
    // while riding, the note line is the train's status
    const ride = g.follow && window.terra?.civ?.parts.trains?.rideInfo()
    e['go-train'].classList.toggle('on', !!ride)
    e['go-train'].textContent = ride ? 'Leave the train' : 'Ride a train'
    if (ride) {
      clearTimeout(this.noteTimer)
      e.note.textContent = `${ride} · V ${g.followView === 'cab' ? 'chase view' : 'cab view'}`
    } else if (this.riding) { this.riding = false; e.note.textContent = '' }
  }
}
