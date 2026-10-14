// Cube-sphere quadtree terrain. Six faces, each split by camera distance; every leaf is one
// instance of a shared GRID x GRID patch. Heights come from the GPU height field, positions are
// camera-relative so the planet stays precise down to the ground.
import * as THREE from 'three'
import { R, GRID, MAX_LEVEL, GLSL_HEAD } from './config.js'
import { NOISE } from './glsl/noise.js'
import { TERRAIN } from './glsl/terrain.glsl.js'
import { ATMO } from './glsl/atmo.glsl.js'
import { CIV_PAINT } from './civ/paint.glsl.js'
import { FARM_PAINT } from './civ/farm.glsl.js'
import { BIOME } from './glsl/biome.glsl.js'
import { FOREST } from './glsl/forest.glsl.js'
import { WATER } from './glsl/water.glsl.js'

const SPLIT = 1.6           // split while closer than SPLIT patch edges: leaf vertex spacing then stays under camDist/100,
                            // the vertex shader's height band limit, so a finer mesh would add no detail
const MAX_INSTANCES = 4096
const MOUNTAIN = 9000       // tallest terrain, for culling bounds
const DETAIL_F = 0.125      // detail noise cells per metre (8 m cells, octaves down to ~3 cm)
const TAU = Math.PI * 2

// ocean wave spectrum: wavelengths 160 m .. 0.7 m, spread around the wind (long swell tighter)
const WAVES = Array.from({ length: 16 }, (_, i) => {
  const k = TAU / (160 * Math.pow(0.7 / 160, i / 15))
  const rnd = Math.sin(i * 12.9898 + 4.1) * 43758.5453 % 1
  return { k, w: Math.sqrt(9.81 * k), spread: rnd * (0.25 + 0.06 * i), phi: (rnd * 7.13 % 1) * TAU }
})

// face normal F and in-face axes U, V with U x V = F (outward winding)
const FACES = [
  [[1, 0, 0], [0, 0, -1], [0, 1, 0]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
  [[0, 1, 0], [1, 0, 0], [0, 0, -1]], [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
  [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
]

class Node {
  constructor(face, u, v, size, level) {
    const [F, U, V] = FACES[face]
    this.face = face; this.u = u; this.v = v; this.size = size; this.level = level
    this.cube = F.map((f, i) => f + u * U[i] + v * V[i])            // patch centre on the cube
    const len = Math.hypot(...this.cube)
    this.dir = this.cube.map(c => c / len)                            // centre on the unit sphere
    this.edge = size * R / len                                        // approximate edge length (m)
    this.children = null
  }
  split() {
    const h = this.size / 4, s = this.size / 2
    return this.children ??= [[-h, -h], [h, -h], [-h, h], [h, h]]
      .map(([du, dv]) => new Node(this.face, this.u + du, this.v + dv, s, this.level + 1))
  }
}

function patchGeometry() {
  const n = GRID + 1, pos = [], idx = []
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) pos.push(i / GRID, j / GRID, 0)
  for (let j = 0; j < GRID; j++) for (let i = 0; i < GRID; i++) {
    const a = j * n + i, b = a + 1, c = a + n, d = c + 1
    idx.push(a, b, c, b, d, c)
  }
  // skirts hide cracks between neighbouring levels
  const edges = [[0, 1, 0], [n * (n - 1), 1, 0], [0, n, 0], [n - 1, n, 0]]
  for (const [start, step] of edges) {
    for (let k = 0; k < GRID; k++) {
      const a = start + k * step, b = a + step
      const base = pos.length / 3
      pos.push(pos[a * 3], pos[a * 3 + 1], 1, pos[b * 3], pos[b * 3 + 1], 1)
      idx.push(a, b, base, b, base + 1, base, a, base, b, b, base, base + 1)
    }
  }
  const geo = new THREE.InstancedBufferGeometry()
  geo.setIndex(idx)
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  return geo
}

const VERT = /* glsl */`${GLSL_HEAD}
uniform mat4 projectionMatrix, viewMatrix;
in vec3 position;     // (u, v, skirt)
in vec4 iCube;        // patch centre on the cube, patch size in face units
in vec4 iRel;         // patch centre (sea level) minus camera, face index in w
out vec3 vDir;
out vec3 vRel;
const vec3 FU[6] = vec3[6](vec3(0,0,-1), vec3(0,0,1), vec3(1,0,0), vec3(1,0,0), vec3(1,0,0), vec3(-1,0,0));
const vec3 FV[6] = vec3[6](vec3(0,1,0), vec3(0,1,0), vec3(0,0,-1), vec3(0,0,1), vec3(0,1,0), vec3(0,1,0));
${NOISE}
${TERRAIN}
void main(){
  int face = int(iRel.w + 0.5);
  vec3 cC = iCube.xyz;
  vec2 d = (position.xy - 0.5)*iCube.w;
  vec3 delta = d.x*FU[face] + d.y*FV[face];
  // normalize(cC + delta) - normalize(cC), written so no large numbers cancel
  float lenC = length(cC);
  float k = 2.0*dot(cC, delta) + dot(delta, delta);
  float lenP = sqrt(lenC*lenC + k);
  vec3 diff = delta/lenP - cC*(k/(lenP*lenC*(lenP + lenC)));
  vec3 dir = cC/lenC + diff;
  float spacing = iCube.w*PLANET_R/(lenC*GRID);
  // band-limit by distance to the camera, not by this patch's spacing: a patch and its children then agree on
  // every height, so splits and merges don't pop (finer detail comes from the fragment shader's normals)
  float camDist = length(diff*PLANET_R + iRel.xyz);
  vec3 info;
  Civ cv;
  float h = terrainCiv(normalize(dir), min(PLANET_R/(2.0*max(spacing, camDist/100.0)), 1.2e6), info, cv).x;
  h = max(h, max(cv.water, 0.0));      // water surfaces: sea level, lake and river levels
  if (position.z > 0.5) h -= spacing*GRID*0.03 + 2.0;
  vec3 rel = diff*PLANET_R + dir*h + iRel.xyz;
  vDir = dir;
  vRel = rel;
  gl_Position = projectionMatrix*viewMatrix*vec4(rel, 1.0);
}`

// G-buffer pass: the patches only store what each pixel's ground is; FRAG below then shades every visible pixel once
// in a screen pass (no shading of ground that objects or hills hide, none of the 2x2-quad overshading small triangles cost)
const GBUF_FRAG = /* glsl */`${GLSL_HEAD}
in vec3 vDir;
in vec3 vRel;
layout(location = 0) out vec4 outRel;
layout(location = 1) out vec4 outDir;
void main(){ outRel = vec4(vRel, 1.0); outDir = vec4(vDir, 0.0); }`

// Debug switches (material defines; shots/perf-views.js, notes/perf.md): NO_WATER, NO_BIOME, NO_FOREST, NO_FARM,
// NO_CIVPAINT, NO_DETAIL, NO_RIVERS, and NO_CIV (no civ lookup in terrainCiv, vertex shader too)
const FRAG = /* glsl */`${GLSL_HEAD}
#define NW 16
#define DETAIL_F ${DETAIL_F.toFixed(4)}
uniform vec3 uSun, uDetailOff;
uniform float uTime, uPixAngle;
uniform vec4 uWave[NW];       // travel direction (world, tangent near the camera), wavenumber (rad/m)
uniform float uWavePh[NW];    // wave phase at the camera
uniform sampler2D uTransLUT, uWeather;
uniform highp sampler2D uGRel, uGDir;    // G-buffer: camera-relative position (w = 1 on terrain), sea-level direction
vec3 vDir, vRel;                          // the patch varyings, read from the G-buffer in main
vec2 gDuvX, gDuvY;                        // screen derivatives of the map uv, for UV_DX / UV_DY in paint and farm
#define UV_DX(uv) gDuvX
#define UV_DY(uv) gDuvY
out vec4 outColor;
${NOISE}
${TERRAIN}
${ATMO}
${CIV_PAINT}
${FARM_PAINT}

vec3 skyAmbient(float muS){
  return SUN_I*vec3(0.030, 0.045, 0.075)*smoothstep(-0.18, 0.35, muS);
}
// the sky post.js draws, seen from the ground along r, in closed form: optical depths to space with a curved-Earth
// airmass (horizon ~36 Rayleigh, ~90 aerosol), one sun transmittance per ray, the same phases and 0.035 multiple
// scattering, and the in-scatter saturating as (1 - exp(-tau))/tau: blue zenith, white hazy horizon, sun-side glow, warm dusk
vec3 skyReflect(vec3 r, vec3 up, float muS){
  float e = dot(r, up), el = max(e, 0.0), nu = dot(r, uSun);
  vec3 tR = BETA_R*(8000.0/(el + 0.028));
  float tM = BETA_M*1200.0/(el + 0.011);
  vec3 tau = tR + tM*(BETA_M_EXT/BETA_M);
  vec3 Ts = sunTrans(uTransLUT, 1500.0 + 3000.0*el, muS), Tms = sunTrans(uTransLUT, 1500.0, min(muS + 0.1, 1.0));
  vec3 L = ((tR*phaseR(nu) + tM*phaseCS(nu, 0.8))*Ts + (tR + tM)*0.035*Tms)*(1.0 - exp(-tau))/tau;
  return SUN_I*PI*L*mix(1.0, 0.35, smoothstep(0.0, -0.2, e));   // (SKY_E) rays aimed down see the next wave, not sky
}
// the deck ~3 km up along the sun ray, from the cubic-filtered map (bilinear texels cast straight-edged polygons at low
// sun) with its edge frayed at ~60 km; only dense cover shades fully, as the rendered cloud is broken below that
float cloudShadow(vec3 p, float muS){
  if (muS <= 0.0) return 1.0;
  float hc = 3000.0;
  vec3 q = normalize(p*(PLANET_R + hc) + uSun*(hc/max(muS, 0.08)));
  float c = weatherSmooth(uWeather, q).r + 0.12*snoise(q*(PLANET_R/60000.0));
  return 1.0 - 0.8*smoothstep(0.35, 0.95, c);
}
// GGX specular (already times n.l) with Schlick Fresnel on the half vector; F0 = 0.02 for water
float specGGX(vec3 N, vec3 V, vec3 L, float alpha){
  vec3 H = normalize(V + L);
  float a2 = alpha*alpha;
  float nh = max(dot(N, H), 0.0), nv = max(dot(N, V), 1e-3), nl = max(dot(N, L), 0.0);
  float dd = nh*nh*(a2 - 1.0) + 1.0;
  float D = a2/(PI*dd*dd);
  float k = alpha*0.5;
  float G = nv/(nv*(1.0 - k) + k)*nl/(nl*(1.0 - k) + k);
  float F = 0.02 + 0.98*pow(1.0 - clamp(dot(V, H), 0.0, 1.0), 5.0);
  return D*G*F/(4.0*nv);
}

// value noise with gradient, periodic every 256 cells on each axis
float hashW(vec3 i){ return hash13(mod(i, 256.0) + 0.5)*2.0 - 1.0; }
vec4 vnoised(vec3 x){
  vec3 i = floor(x), f = x - i;
  vec3 u = f*f*f*(f*(f*6.0 - 15.0) + 10.0), du = 30.0*f*f*(f*(f - 2.0) + 1.0);
  float a = hashW(i), b = hashW(i + vec3(1, 0, 0)), c = hashW(i + vec3(0, 1, 0)), d = hashW(i + vec3(1, 1, 0));
  float e = hashW(i + vec3(0, 0, 1)), g = hashW(i + vec3(1, 0, 1)), m = hashW(i + vec3(0, 1, 1)), n = hashW(i + vec3(1, 1, 1));
  float k1 = b - a, k2 = c - a, k3 = e - a, k4 = a - b - c + d, k5 = a - c - e + m, k6 = a - b - e + g, k7 = -a + b + c - d + e - g - m + n;
  float v = a + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z;
  return vec4(v, du*vec3(k1 + k4*u.y + k6*u.z + k7*u.y*u.z, k2 + k5*u.z + k4*u.x + k7*u.z*u.x, k3 + k6*u.x + k5*u.y + k7*u.x*u.y));
}
// sub-metre detail in camera-relative coordinates (no float cancellation): (colour value, height gradient per metre of amplitude)
vec4 detail(float foot, out float resolved){
  vec3 x = uDetailOff + vRel*DETAIL_F;
  vec4 s = vec4(0.0);
  float a = 1.0, f = 1.0, ac = 0.6;
  resolved = 0.0;
  for (int i = 0; i < 9; i++){
    float bw = bandw(DETAIL_F*f, 0.5/foot);
    if (bw <= 0.0) break;
    vec4 n = vnoised(x*f + float(i)*37.0);
    s += vec4(n.x*ac, n.yzw*(a*f*DETAIL_F))*bw;
    resolved += 0.25*bw;
    f *= 2.0; a *= 0.5; ac *= 0.8;
  }
  resolved = min(resolved, 1.0);
  return s;
}

// directional wind-sea spectrum; returns height gradient (m/m) and crest height / rms in .w. nL, nS: the wave-group
// noises (main's water noise batch)
vec4 waves(vec3 p, float foot, float s0, float longW, float nL, float nS, out float unres){
  vec3 g = vec3(0.0);
  float ht = 0.0, vs = 0.0;
  unres = 0.0;
  // wave groups: long and short waves swell and calm in patches
  float grpL = 0.65 + 0.45*nL;
  float grpS = 0.6 + 0.5*nS;
  for (int i = 0; i < NW; i++){
    vec4 w = uWave[i];
    float s = s0*(i < 6 ? grpL*longW : grpS);
    float bw = bandw(w.w*foot, 0.8);
    float ph = uWavePh[i] + w.w*dot(vRel, w.xyz);
    // sharpened crests: slope from a slightly trochoidal profile
    float sn = sin(ph), cs = cos(ph);
    g += w.xyz*(s*cs*(1.0 + 0.35*sn)*bw);
    ht += s/w.w*sn*bw;
    vs += s*s/(w.w*w.w)*0.5*bw;
    unres += s*s*0.5*(1.0 - bw);
  }
  return vec4(g, ht/sqrt(max(vs, 1e-6)));
}
${WATER}

${BIOME}
${FOREST}

// map uv (as civFarUV) and its screen derivative from the G-buffer neighbours: the smaller one-sided difference, so a
// pixel on a silhouette takes its own surface's side as the rasterizer's helper pixels did
vec2 eqUV(vec3 p){ return vec2(atan(p.x, p.z)*(0.5/PI) + 0.5, asin(clamp(p.y, -1.0, 1.0))/PI + 0.5); }
vec2 gbufDuv(ivec2 px, ivec2 o, vec2 uv0){
  ivec2 size = textureSize(uGRel, 0);
  vec2 d = vec2(0.0);
  float best = 1e30;
  for (int s = -1; s <= 1; s += 2){
    ivec2 q = px + o*s;
    if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size)) || texelFetch(uGRel, q, 0).w == 0.0) continue;
    vec2 dq = (eqUV(normalize(texelFetch(uGDir, q, 0).xyz)) - uv0)*float(s);
    dq.x -= floor(dq.x + 0.5);
    if (dot(dq, dq) < best){ best = dot(dq, dq); d = dq; }
  }
  return d;
}

void main(){
  ivec2 px = ivec2(gl_FragCoord.xy);
  vec4 gr = texelFetch(uGRel, px, 0);
  if (gr.w == 0.0) discard;                     // sky, or an object in front
  vRel = gr.xyz;
  vDir = texelFetch(uGDir, px, 0).xyz;
  vec3 p = normalize(vDir);
  vec2 uv0 = eqUV(p);
  gDuvX = gbufDuv(px, ivec2(1, 0), uv0);
  gDuvY = gbufDuv(px, ivec2(0, 1), uv0);
  float dist = length(vRel);
  float foot = max(dist*uPixAngle, 1e-3);          // pixel footprint (m)
  float maxF = min(PLANET_R/(2.0*max(foot, 0.05)), 7e5);   // finer than ~9 m comes from detail()
  vec4 um = civFarSample(p);
  float farmD = farmMapSample(p);
  civSetRel(vRel);
  // the far painting's lookups (civLand for the town painting up to CIV_FAB1, civRivers below 20 km) ride on
  // terrainCiv's walk (civ.glsl.js civFarMode)
#ifndef NO_CIVPAINT
  civFarMode = foot < CIV_FAB1 ? 2 : 0;
#endif
#ifndef NO_RIVERS
  if (foot < 20000.0) civFarMode = max(civFarMode, 1);
#endif
  vec3 info;
  Civ cv;
  vec4 hg = terrainCiv(p, maxF, info, cv);
  float h = hg.x;
  // water surface: the sea at 0, lakes and rivers at their own level; a river narrower than a pixel stays a line
  float wl = max(0.0, cv.waterKind == 2 && 2.0*cv.riverHalf < foot ? cv.lakeLevel : cv.water);
  vec3 g = hg.yzw - p*dot(hg.yzw, p);
  vec3 N = normalize(p - g/PLANET_R);
  vec3 V = -vRel/dist;
  vec3 L = uSun;
  float muS = dot(p, L);
  vec3 sunC = SUN_I*sunTrans(uTransLUT, max(h, 0.0), muS)*cloudShadow(p, muS);
  vec2 cl = climate(p, h, info.x);
  float T = cl.x, M = cl.y;
  vec3 amb = skyAmbient(muS);
  vec3 col;

#ifdef NO_WATER
  if (false){
#else
  if (h < wl){
#endif
    float depth = wl - h;
    float inland = clamp(wl*0.5, 0.0, 1.0);                      // lakes and rivers: calm fresh water, no surf
    // river water fades out across its flooded floodplain, so a river mouth hands over to the sea (or a backwater) smoothly
    float onRiver = cv.waterKind == 2 && wl == cv.water ? 1.0 - smoothstep(cv.riverHalf, cv.riverHalf + riverPlain(cv.riverHalf), cv.dRiver) : 0.0;
    // river water by region: muddy in warm and dry lowlands, clear (tea-dark) in the cold north, milky glacial flour on
    // cold steep reaches; rapids (rough) where the level drops fast
    float rough = onRiver*smoothstep(0.003, 0.015, cv.riverSlope)*(1.0 - smoothstep(40.0, 150.0, cv.riverHalf));
    // the water's noises through one snoisedR call site (D3D inlines every call site; noise.js loopN): 0 mud, 1-2 wind,
    // 3 surf, 4 frost, 5-6 wave groups (drifting with time), then up to 3 octaves of chop, a little noise on top of the
    // waves so the sum of sines never looks tiled
    const float WS[7] = float[7](50.0, 3.0, 11.0, PLANET_R/45.0, 40.0, PLANET_R/900.0, PLANET_R/140.0);
    const vec3 WO[7] = vec3[7](vec3(0.0), vec3(2.0, 8.0, 1.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
    const float WT[7] = float[7](0.0, 0.0, 0.0, 0.0, 0.0, 0.01, -0.03);
    float wn[7];
    vec3 chop = vec3(0.0);
    float cf = PLANET_R/60.0;
    for (int i = 0; i < loopN(10); i++){
      int k = min(i, 6);
      float bw = i < 7 ? 1.0 : bandw(cf, maxF);
      if (bw <= 0.0) break;
      vec4 n = snoisedR(i < 7 ? p*WS[k] + (WO[k] + vec3(uTime*WT[k])) : p*cf + vec3(uTime*0.4, 0.0, float(i - 7)*3.1));
      if (i < 7){ wn[k] = 105.0*n.x; continue; }
      chop += (105.0*n).yzw*0.012*bw;
      cf *= 2.3;
    }
    float mud = clamp(0.3 + 0.5*smoothstep(14.0, 26.0, T) + 0.4*(1.0 - smoothstep(0.2, 0.45, M)) - 0.6*(1.0 - smoothstep(2.0, 10.0, T)) + 0.4*wn[0], 0.0, 1.0);
    float silt = onRiver*mud*smoothstep(3.0, 150.0, cv.riverHalf)*(1.0 - rough);
    float glacial = onRiver*(1.0 - smoothstep(3.0, 9.0, T))*smoothstep(0.0005, 0.004, cv.riverSlope);
    // wind speed (m/s): slow pattern, stronger under storms; Cox-Munk mean square slope
    vec4 wx = weatherAt(uWeather, p);
    float U = clamp(6.5 + 4.0*wn[1] + 2.0*wn[2] + 8.0*wx.g, 1.0, 22.0)*(1.0 - 0.6*inland);
    float mss = 0.003 + 0.00512*U;
    float shelter = smoothstep(0.0, 25.0, depth);    // lagoons and shallows are calmer
    float unres;
    vec4 wv = waves(p, foot, sqrt(2.0*(mss - 0.003)/float(NW))*mix(0.35, 1.0, shelter)*(1.0 - 0.5*inland), 1.0 - 0.9*inland, wn[5], wn[6], unres);
    vec3 gw = wv.xyz + chop;
    float rfoam = 0.0, rock = 0.0;
    if (onRiver > 0.0){ gw += onRiver*riverFlow(cv, p, foot, uTime, rough, rfoam, rock); rfoam *= onRiver; rock *= onRiver; }
    vec3 Nw = normalize(p - (gw - p*dot(gw, p)));
    float alpha = sqrt(0.003 + unres);
    float nv = max(dot(Nw, V), 0.0);
    float fres = 0.02 + 0.98*pow(1.0 - nv, 5.0);
    vec3 refl = skyReflect(reflect(-V, Nw), p, muS);
    vec3 spec = sunC*specGGX(Nw, V, L, alpha);
    // body: deep blue scattering, sand bed showing through the shallows as turquoise. Lakes: plankton and humus make
    // them dark and opaque within a few metres (boreal lakes tea-black), over a dark organic bed
    float lakeW = inland*(1.0 - onRiver);
    float humic = lakeW*(1.0 - smoothstep(4.0, 14.0, T))*smoothstep(0.25, 0.55, M);
    vec3 kw = mix(vec3(0.42, 0.13, 0.10), mix(vec3(0.55, 0.30, 0.50), vec3(0.8, 1.2, 2.6), humic), lakeW);   // fresh water: darker, greener
    vec3 absorb = exp(-mix(vec3(0.38, 0.055, 0.028), kw, inland)*depth);
    vec3 bed = mix(vec3(0.50, 0.46, 0.34), mix(vec3(0.27, 0.24, 0.17), vec3(0.13, 0.11, 0.075), lakeW), inland)*absorb;
    vec3 body = vec3(0.0009, 0.0042, 0.0115)*(1.0 - absorb.b) + vec3(0.002, 0.016, 0.022)*(1.0 - absorb.g)*exp(-depth*0.03) + bed*0.8;
    body = mix(body, mix(vec3(0.0035, 0.0075, 0.0065), vec3(0.0045, 0.0035, 0.0018), humic)*(1.0 - absorb.g) + bed*0.8, lakeW);
    vec3 E = sunC*max(muS, 0.0) + amb;
    body = mix(body, vec3(0.062, 0.049, 0.028)*(1.0 - 0.5*absorb.r), silt*0.85);
    body = mix(body, vec3(0.030, 0.060, 0.058), glacial*0.8);
    col = body*E*(1.0 - fres) + refl*fres + spec;
    // whitecaps on crests in strong wind, averaged into a haze far away
    float cover = clamp(3.8e-6*pow(U, 3.4), 0.0, 0.3);
    float caps = smoothstep(2.4 - 6.0*cover, 3.2, wv.w)*smoothstep(5.0, 12.0, U)*shelter;
    float capsFar = cover*2.0*(1.0 - bandw(PLANET_R/20.0, maxF));
    // surf: breaker lines marching in over the shallows, swash at the waterline
    float surfN = wn[3];
    float lines = smoothstep(0.55, 1.0, sin(depth*2.4 - uTime*1.1 + 2.5*surfN))*(1.0 - smoothstep(0.4, 4.0, depth))*(1.0 - inland);
    float swash = (1.0 - smoothstep(0.0, mix(0.5, 0.15, inland) + 0.4*surfN*(1.0 - inland), depth))*mix(1.0, 0.3, inland);
    float surf = max(lines*bandw(PLANET_R/6.0, maxF), swash) + 0.5*(1.0 - smoothstep(0.0, 3.0, depth))*(1.0 - bandw(PLANET_R/6.0, maxF))*(1.0 - inland);
    float foam = clamp(max(max(caps, surf), rfoam) + capsFar, 0.0, 1.0);
    col = mix(col, vec3(0.75)*(sunC*max(dot(Nw, L), 0.0)*0.8 + sunC*max(muS, 0.0)*0.2 + amb), foam*0.8);
    // sand bars on the inner bends of big rivers (shallows around them), gravel on steep and glacial reaches; boulders in rapids
    if (onRiver > 0.0){
      vec3 rb = riverBar(cv, p, foot);
      vec3 bar = mix(vec3(0.36, 0.32, 0.24), vec3(0.29, 0.28, 0.26), max(rough, glacial))*E;
      col = mix(col, mix(col, bar*0.6, 0.5), rb.y*(1.0 - 0.5*silt));
      col = mix(col, bar*rb.z, rb.x*(1.0 - 0.5*rough));
      col = mix(col, vec3(0.19, 0.17, 0.15)*(sunC*(0.3*max(muS, 0.0) + 0.7*max(dot(Nw, L), 0.0)) + amb), rock);
    }
    // ice by today's temperature (seasonT); lakes freeze sooner, rivers a little later. Near freezing it breaks into
    // floes with dark young ice and open leads (not a see-through blend over the waves' glitter). Snow-covered and matte,
    // with drifts and wind-swept grey-blue patches; lit as a flat sheet, not by the bed below
    float frz = smoothstep(mix(-1.0, 1.5 - onRiver, inland), mix(-4.0, -1.5 - onRiver, inland), T + seasonT(p, info.x).x + 2.0*wn[4]);
    if (frz > 0.0){
      float fl = 0.0;                                // floes: two octaves, one call site
      for (int i = 0; i < loopN(2); i++){
        float k = i == 0 ? PLANET_R/2500.0 : PLANET_R/600.0;
        fl += (i == 0 ? 0.65 : 0.35)*snoise(p*k + (i == 0 ? vec3(3.1, 7.7, 1.9) : vec3(8.4, 2.6, 5.3)))*bandw(k, maxF);
      }
      float ice = smoothstep(0.35, 0.5, frz + 0.35*fl);
      float young = 1.0 - smoothstep(0.5, 0.9, frz + 0.35*fl), bare = smoothstep(0.4, 0.55, fl);
      vec3 snowC = vec3(0.80, 0.83, 0.88)*(1.0 + 0.08*surfN*bandw(PLANET_R/45.0, maxF));   // drifts
      vec3 iceA = mix(mix(snowC, vec3(0.48, 0.55, 0.62), bare), vec3(0.16, 0.19, 0.21), young);
      vec3 iceC = iceA*(sunC*max(muS, 0.0) + amb) + sunC*specGGX(p, V, L, 0.3)*mix(0.3, 1.0, max(young, bare));
      col = mix(col, iceC, ice);
    }
  } else {
    float slope = 1.0 - dot(N, p);
    vec3 mat;
#ifdef NO_BIOME
    vec3 alb = vec3(0.1); mat = vec3(0.0);
#else
    vec3 alb = landAlbedo(h, T, M, slope, info, p, maxF, mat);
#endif
#ifndef NO_FOREST
    alb = forestPaint(alb, p, foot, T, M, slope, h, cv);        // forest canopy (forest.glsl.js)
#endif
#ifndef NO_FARM
    alb = farmPaint(alb, p, foot, farmD, T, M, slope, h, cv);   // fields around settlements (farm.glsl.js)
#endif
    alb = mix(waterBank(alb, p, cv, h, foot, T, M), alb, mat.y);  // wet soil and reeds along lakes and rivers (water.glsl.js), not over snow
    // towns, roads and railways: exact painting up to a few km footprint (the terrain's own lookup up close,
    // civLand() past it), the coarse urban map from orbit
    float night = 1.0 - smoothstep(-0.1, 0.04, muS);
    float nearW = 1.0 - smoothstep(CIV_FAB0, CIV_FAB1, foot);
    vec3 emit = vec3(0.0);
    float gloss = 0.0, flatN = 0.0;
#ifndef NO_CIVPAINT
    if (nearW > 0.0){
      Civ fc = cv;
      if (maxF < CIV_FLAT_F) fc = civLand(p);
      CivOut co = civPaint(alb, fc, foot, night, smoothstep(8.0, 22.0, T));
      alb = mix(alb, co.alb, nearW);
      emit += co.emit*nearW*civOrbit();
      gloss = co.gloss*nearW; flatN = co.flatN*nearW;
      mat = mix(mat, vec3(0.0, 0.0, 0.0), flatN);
      // farms, hamlets and the lights thinning out around towns
      if (night > 0.0 && fc.urban <= 0.0) emit += civScatter(p, foot, max(um.r, 0.1*farmD))*night*nearW;
    }
    civFar(alb, emit, um, farmD, p, foot, 1.0 - nearW, night);
#endif
    // up close: fine grain and bumps in rock, sand, grass and snow
#ifndef NO_DETAIL
    if (foot < 4.0){
      float res;
      vec4 dt = detail(foot, res);
      float bump = mix(mix(0.35, 0.12, mat.z), 1.3, mat.x)*mix(1.0, 0.18, mat.y);
      vec3 gd = dt.yzw*bump*(1.0 - 0.85*flatN);
      N = normalize(N - (gd - N*dot(gd, N)));
      float grain = mix(0.45, 0.6, mat.x)*mix(1.0, 0.25, mat.y)*(1.0 - 0.6*flatN);
      alb *= 1.0 + grain*dt.x;
      alb.r *= 1.0 + 0.25*dt.x*(1.0 - mat.x - mat.y);   // dry vs green grass
    }
#endif
    float ndl = max(dot(N, L), 0.0);
    col = alb*(sunC*ndl + amb*(0.6 + 0.4*dot(N, p)));
    col += sunC*specGGX(N, V, L, 0.35)*gloss*2.0;       // sheen on asphalt and rails
    col += emit;                                        // street lights, windows, traffic
    // rivers narrower than a pixel, and every river from afar: coverage-weighted water lines
#ifndef NO_RIVERS
    // (waterFar is linear in silt, so one mix with the combined coverage and silt equals drawing the two in turn)
    vec2 rn = vec2(0.0);
    if (foot < 20000.0){
      Civ rc = cv;
      if (maxF < CIV_FLAT_F) rc = civRivers(p);
      rn = vec2(riverCover(rc, foot), smoothstep(20.0, 400.0, rc.riverHalf)*(0.3 + 0.6*smoothstep(12.0, 26.0, T)));
    }
    vec2 rf = riverFar(p, foot, T);   // big rivers from orbit: coarse river map (water.glsl.js)
    float cov = 1.0 - (1.0 - rn.x)*(1.0 - rf.x);
    if (cov > 0.0) col = mix(col, waterFar(p, V, L, sunC, amb, muS, (rn.x*(1.0 - rf.x)*rn.y + rf.x*rf.y)/cov), cov);
#endif
  }
  outColor = vec4(col, 1.0);
}`

const mod = (x, m) => ((x % m) + m) % m

export class Terrain {
  constructor(shared) {
    this.roots = FACES.map((_, f) => new Node(f, 0, 0, 2, 0))
    this.cubeArr = new Float32Array(MAX_INSTANCES * 4)
    this.relArr = new Float32Array(MAX_INSTANCES * 4)
    const geo = patchGeometry()
    this.iCube = new THREE.InstancedBufferAttribute(this.cubeArr, 4).setUsage(THREE.DynamicDrawUsage)
    this.iRel = new THREE.InstancedBufferAttribute(this.relArr, 4).setUsage(THREE.DynamicDrawUsage)
    geo.setAttribute('iCube', this.iCube)
    geo.setAttribute('iRel', this.iRel)
    geo.instanceCount = 0
    this.shared = shared
    this.uniforms = {
      uSun: shared.uSun, uTime: shared.uTime, uPixAngle: shared.uPixAngle,
      uTransLUT: shared.uTransLUT, uWeather: shared.uWeather, ...shared.civ,
      uDetailOff: { value: new THREE.Vector3() },
      uWave: { value: WAVES.map(() => new THREE.Vector4()) },
      uWavePh: { value: WAVES.map(() => 0) },
      uGRel: { value: null }, uGDir: { value: null },
    }
    // all three terrain programs share these uniforms: the civ modules add theirs to the material whose source has
    // forestPaint / farmPaint (the shading pass), which sits on the mesh until main.js swaps in the preview
    this.material = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: shared.quadVert, fragmentShader: FRAG, uniforms: this.uniforms, depthTest: false, depthWrite: false })
    this.quad = shared.makeQuad(this.material, false)
    this.gMaterial = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: VERT, fragmentShader: GBUF_FRAG, uniforms: this.uniforms })
    this.gMesh = new THREE.Mesh(geo, this.gMaterial)
    this.gMesh.frustumCulled = false
    this.gScene = new THREE.Scene().add(this.gMesh)
    this.mesh = new THREE.Mesh(geo, this.material)
    this.mesh.frustumCulled = false
    this.mesh.renderOrder = 1     // the preview: after buildings, trees etc., so early-Z skips ground they cover
    this.scene = new THREE.Scene()
    this.scene.add(this.mesh)
    this.deferred = false
    this.geo = geo
    this.frustum = new THREE.Frustum()
    this.sphere = new THREE.Sphere()
    this.m4 = new THREE.Matrix4()
    this.count = 0
    this.anchor = null
  }

  // Objects first, then the ground in two steps: the G-buffer pass (depth-tested against the objects in the depth
  // buffer it shares with target), then the full shading once per visible pixel; smoke and lamp beams last, over
  // the shaded ground. Until the full shader links (deferred = false) the preview surface draws directly.
  render(renderer, camera, target, gbuf) {
    if (!this.deferred) return renderer.render(this.scene, camera)
    const objs = this.scene.children.filter(m => m.visible && m !== this.mesh)
    const glass = objs.filter(m => m.material.transparent), solid = objs.filter(m => !m.material.transparent)
    this.mesh.visible = false
    for (const m of glass) m.visible = false
    renderer.render(this.scene, camera)
    renderer.setRenderTarget(gbuf)
    renderer.clear(true, false, false)
    renderer.render(this.gScene, camera)
    this.uniforms.uGRel.value = gbuf.textures[0]
    this.uniforms.uGDir.value = gbuf.textures[1]
    renderer.setRenderTarget(target)
    renderer.render(this.quad.scene, this.quad.camera)
    if (glass.length) {
      for (const m of solid) m.visible = false
      for (const m of glass) m.visible = true
      renderer.render(this.scene, camera)
      for (const m of solid) m.visible = true
    }
    this.mesh.visible = true
  }

  // camPos: planet-centred camera position (doubles); ground: terrain height under the camera
  update(camPos, camera, ground) {
    this.frustum.setFromProjectionMatrix(this.m4.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse), camera.coordinateSystem, camera.reversedDepth)
    const d = camPos.length()
    this.cam = camPos
    this.camDir = [camPos.x / d, camPos.y / d, camPos.z / d]
    this.horizon = Math.acos(Math.min(1, R / d)) + Math.acos(R / (R + MOUNTAIN))
    this.gr = R + Math.max(ground, 0)
    this.count = 0
    for (const root of this.roots) this.visit(root)
    this.geo.instanceCount = this.count
    this.iCube.needsUpdate = true
    this.iRel.needsUpdate = true
    this.updateSurface(camPos, d)
    this.shared.civ.uCamDir.value.set(camPos.x / d, camPos.y / d, camPos.z / d)
    this.shared.civ.uCamR.value = d
  }

  // detail-noise offset and wave phases, reduced in doubles so shaders only see small numbers
  updateSurface(c, d) {
    this.uniforms.uDetailOff.value.set(mod(c.x * DETAIL_F, 256), mod(c.y * DETAIL_F, 256), mod(c.z * DETAIL_F, 256))
    const up = this.camDir, a = this.anchor
    const ang = a ? Math.acos(Math.min(1, a.up[0] * up[0] + a.up[1] * up[1] + a.up[2] * up[2])) : 9
    // waves keep a fixed world frame; re-anchor only when far away or high up (where waves are invisible)
    if (ang > 0.3 || (d - R > 8e4 && ang > 0.005)) this.anchor = waveAnchor(up)
    const t = this.shared.uTime.value
    WAVES.forEach((w, i) => {
      const [x, y, z] = this.anchor.dirs[i]
      this.uniforms.uWave.value[i].set(x, y, z, w.k)
      this.uniforms.uWavePh.value[i] = mod(w.k * (x * c.x + y * c.y + z * c.z) - w.w * t + w.phi, TAU)
    })
  }

  // patch bounds as a vertical stack of spheres: terrain can sit up to v above or below the camera's ground
  inFrustum(n, rx, ry, rz, v) {
    const [dx, dy, dz] = n.dir, s = this.sphere
    s.radius = n.edge * 0.8 + v * 0.5
    for (const k of [0, -1, 1]) {
      s.center.set(rx + dx * v * k, ry + dy * v * k, rz + dz * v * k)
      if (this.frustum.intersectsSphere(s)) return true
    }
    return false
  }

  dist2(n) {
    const [dx, dy, dz] = n.dir, g = this.gr, c = this.cam
    return (dx * g - c.x) ** 2 + (dy * g - c.y) ** 2 + (dz * g - c.z) ** 2
  }

  visit(n) {
    const [dx, dy, dz] = n.dir, [cx, cy, cz] = this.camDir
    const theta = Math.acos(Math.max(-1, Math.min(1, dx * cx + dy * cy + dz * cz)))
    if (theta - n.edge * 0.8 / R > this.horizon) return
    const rx = dx * this.gr - this.cam.x, ry = dy * this.gr - this.cam.y, rz = dz * this.gr - this.cam.z
    const centre = Math.hypot(rx, ry, rz)
    if (!this.inFrustum(n, rx, ry, rz, Math.min(MOUNTAIN, 0.7 * centre + 30))) return
    const dist = centre - n.edge * 0.71
    if (n.level < MAX_LEVEL && dist < n.edge * SPLIT) {
      // nearest child first: patches then draw roughly front to back, so early-Z rejects ground hidden behind hills
      for (const c of n.split().sort((a, b) => this.dist2(a) - this.dist2(b))) this.visit(c)
      return
    }
    if (this.count >= MAX_INSTANCES) return
    const i = this.count++ * 4
    this.cubeArr.set([n.cube[0], n.cube[1], n.cube[2], n.size], i)
    // sea-level centre relative to the camera, computed in doubles
    this.relArr.set([dx * R - this.cam.x, dy * R - this.cam.y, dz * R - this.cam.z, n.face], i)
  }
}

// local east/north at 'up' and wave directions around the prevailing wind there
function waveAnchor(up) {
  const [x, y, z] = up
  const el = Math.hypot(z, x) || 1
  const east = [z / el, 0, -x / el]
  const north = [y * east[2] - z * east[1], z * east[0] - x * east[2], x * east[1] - y * east[0]]
  const lat = Math.asin(y), lon = Math.atan2(x, z), al = Math.abs(lat) * 180 / Math.PI
  // trades blow toward the west-equator, westerlies toward the east, polar easterlies west again
  let wind = al < 30 ? Math.PI + 0.35 * Math.sign(lat) : al < 60 ? 0.2 * Math.sign(lat) : Math.PI
  wind += 0.5 * Math.sin(lon * 3 + lat * 5)
  const dirs = WAVES.map((w, i) => {
    const th = wind + w.spread + (i < 4 ? 0.5 : 0)          // the longest waves are swell from elsewhere
    return east.map((e, j) => Math.cos(th) * e + Math.sin(th) * north[j])
  })
  return { up: [...up], dirs }
}
