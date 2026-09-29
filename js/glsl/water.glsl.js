// Inland water helpers for the terrain fragment shader (after skyReflect/specGGX/vnoised in terrain.js).
// Rivers wider than a pixel are real water (terrainCiv carves them below their level); narrower ones and
// all rivers seen from afar are drawn as coverage-weighted lines, like civRoadGlow() draws roads.
import { R } from '../config.js'

// coarse global river map (world.js riverMap, sampled by riverFar): size, presence tent half width and
// signed distance range (m; texels farther than that from every river are left 'far')
const TX = Math.PI * R / 2048
export const RMAP = { W: 4096, H: 2048, tent: 1.2 * TX, range: 2.8 * TX }
// opacity of a river line seen from orbit, by log10 of its discharge (m3/s): ~600 m3/s none .. 30000 m3/s 0.8
export const rmapAlpha = lq => { const t = Math.min(1, Math.max(0, (lq - 2.8) / 1.7)); return 0.8 * t * t * (3 - 2 * t) }

export const WATER = /* glsl */`
// share of the pixel covered by the river channel of c (c from civAt / civRivers). Big rivers keep a floor of a
// few % of a pixel; the whole line fades out once the pixel outgrows the river's lookup reach (it would break
// into dots), where riverFar() takes over from the river map
float riverCover(Civ c, float foot){
  if (c.riverHalf <= 0.0) return 0.0;
  float w = 2.0*c.riverHalf, reach = 1.9*c.riverHalf + riverPlain(c.riverHalf) + 360.0;
  float floorC = 0.16*smoothstep(15.0, 400.0, w);
  return min(1.0, max(w/foot, floorC))*(1.0 - smoothstep(0.0, min(foot, reach), c.dRiver - c.riverHalf))
    *(1.0 - smoothstep(1.2*reach, 3.0*reach, foot));
}

// water seen as a flat mirror: dark body, sky reflection, sun glint (rivers and lakes from afar). Once pixels are
// ~1 km and more a river line also averages in shallows, sand bars and wet banks: a lighter blue-grey
vec3 waterFar(vec3 N, vec3 V, vec3 L, vec3 sunC, vec3 amb, float muS, float silt){
  float nv = max(dot(N, V), 0.0);
  float fres = 0.02 + 0.98*pow(1.0 - nv, 5.0);
  vec3 body = mix(vec3(0.004, 0.011, 0.016), vec3(0.060, 0.047, 0.027), silt)
    + vec3(0.014, 0.045, 0.100)*smoothstep(300.0, 3000.0, length(vRel)*uPixAngle);
  return body*(sunC*max(muS, 0.0) + amb)*(1.0 - fres) + skyReflect(reflect(-V, N), N, muS)*fres + sunC*specGGX(N, V, L, 0.12);
}

// world.js riverMap(): r = presence tent x orbit opacity (its mips give line density), g = signed distance to the
// nearest centreline (0.5 = on it), b = log2(Q/100)/10 of that river; equirect, repeats in longitude
uniform sampler2D uRiverMap;
#define RMAP_TX ${TX.toFixed(1)}
#define RMAP_TENT ${RMAP.tent.toFixed(1)}
#define RMAP_RANGE ${RMAP.range.toFixed(1)}
float rmapAlpha(float lq){ return 0.8*smoothstep(2.8, 4.5, lq); }

// big rivers from afar, where the exact lookup (riverCover) runs out of reach. Pixels finer than a map texel
// get an anti-aliased line from the signed distance (sub-texel sharp, no staircase); coarser ones the
// mip-mapped line density (a pixel wide line of opacity a). Gradients come from foot, so any control flow is fine.
// Returns (coverage, silt) for waterFar(); the caller draws it (one waterFar call site: D3D inlines each one)
vec2 riverFar(vec3 p, float foot, float T){
  vec2 uv = vec2(atan(p.x, p.z)*(0.5/PI) + 0.5, asin(clamp(p.y, -1.0, 1.0))/PI + 0.5);
  vec4 m = textureLod(uRiverMap, uv, 0.0);
  float lq = 2.0 + 3.0103*m.b, hw = clamp(pow(10.0, 0.62*lq), 1.5, 750.0), a = rmapAlpha(lq);
  float reach = 1.9*hw + riverPlain(hw) + 360.0;
  float wm = smoothstep(1.2*reach, 3.0*reach, foot);
  if (wm <= 0.0) return vec2(0.0);
  float d = abs(m.g*2.0 - 1.0)*RMAP_RANGE - hw;
  float line = max(2.0*hw/foot, a)*(1.0 - smoothstep(0.0, foot, d))*smoothstep(0.15, 0.45, m.r/max(a, 0.004));
  float cl = max(sqrt(1.0 - p.y*p.y), 0.02);
  float dens = textureGrad(uRiverMap, uv, vec2(foot/(2.0*PI*PLANET_R*cl), 0.0), vec2(0.0, foot/(PI*PLANET_R))).r*foot/RMAP_TENT;
  float cov = min(1.0, mix(line, dens, smoothstep(0.6, 1.5, foot/RMAP_TX)))*wm;
  return vec2(cov, smoothstep(20.0, 400.0, hw)*(0.3 + 0.6*smoothstep(12.0, 26.0, T)));
}

// river surface: wavelets and foam drifting downstream; on steep reaches (rough 0..1) broken white water in
// stationary patches (standing waves over steps) with pools between, and boulders. The noise is world-fixed (detail
// frame, 8 m cells) and drifts along the channel with a two-phase flow map, so where the nearest polyline segment
// switches only the drift direction changes a little: no seams along the joints. Returns the slope gradient;
// foam = white share, rock = boulder share, both averaged once finer than a pixel.
vec3 riverFlow(Civ c, vec3 p, float foot, float time, float rough, out float foam, out float rock){
  vec3 tn = normalize(c.riverTan - p*dot(c.riverTan, p));
  vec3 sd = cross(p, tn);
  float speed = 0.6 + 1.2*smoothstep(5.0, 300.0, c.riverHalf) + 2.5*rough;   // m/s
  vec3 x = uDetailOff + vRel*DETAIL_F;
  vec3 g = vec3(0.0);
  float fn = 0.0, rn = 0.0;
  // one vnoised() call site (D3D inlines each): for flow phase k = j/4 three wavelet octaves (i = j%4 < 3) and the
  // foam noise (i = 3), then (j = 8) the boulder noise
  for (int j = 0; j < loopN(9); j++){
    int k = j >> 2, i = j & 3;
    float ph = fract(time*0.1 + 0.5*float(k)), wk = 1.0 - abs(2.0*ph - 1.0);   // 10 s cycles, weights sum to 1
    vec3 xk = x - tn*(speed*10.0*ph*DETAIL_F) + float(k)*vec3(37.0, 11.0, 23.0);
    float f = float(1 << i), bw = bandw(DETAIL_F*f, 0.5/foot);
    if (j < 8 && i < 3 && bw <= 0.0) continue;     // octaves past the pixel (the phase's finer ones too)
    vec4 n = vnoised(j == 8 ? x + 90.0 : i < 3 ? xk*f + float(i)*7.0 : xk*2.0 + 50.0);
    if (j == 8) rn = n.x*0.5 + 0.5;
    else if (i < 3) g += (tn*dot(n.yzw, tn)*0.6 + sd*dot(n.yzw, sd)*1.5)*(0.02 + 0.08*rough)*bw*wk;   // crests drawn out along the flow
    else fn += (n.x*0.5 + 0.5)*wk;
  }
  float bwF = bandw(2.0*DETAIL_F, 0.5/foot), bwR = bandw(DETAIL_F, 0.5/foot);
  float edge = clamp(1.0 - c.dRiver/max(c.riverHalf, 1.0), 0.0, 1.0);
  // white water covers a share cov of the surface: stationary reaches ~80 m apart, less near the banks
  float cov = 0.75*rough*smoothstep(-0.55, 0.45, snoise(p*(PLANET_R/80.0)))*(0.35 + 0.65*smoothstep(0.0, 0.5, edge));
  float white = mix(0.7*cov, smoothstep(1.0 - cov - 0.12, 1.0 - cov + 0.12, fn), bwF);
  float fleck = mix(0.03, smoothstep(0.78, 0.95, fn), bwF)*0.08*(1.0 - smoothstep(0.0, 0.2, rough));   // calm: faint foam flecks
  // boulders (8 m cells, fixed): more in the shallows by the banks, each with a collar of white water
  float thr = 0.9 - 0.14*(1.0 - edge);
  rock = rough*mix(0.05, smoothstep(thr, thr + 0.04, rn), bwR);
  foam = max(max(white, fleck), rough*mix(0.0, smoothstep(thr - 0.1, thr, rn), bwR))*smoothstep(0.0, 0.08, edge);
  return g;
}

// point bars: sand / gravel on the inner bank where the meander (riverMeander in spec.js) bends hardest. The bar
// grows with the bend, its waterline wanders (world-fixed noise, 70..220 m) and it stays inside the channel, so
// estuaries get no slabs out in the sea. Returns (bar share, shallow water around it, albedo factor: wet sand, ripples).
vec3 riverBar(Civ c, vec3 p, float foot){
  float hw = c.riverHalf;
  if (hw < 30.0) return vec3(0.0, 0.0, 1.0);
  vec3 tn = normalize(c.riverTan - p*dot(c.riverTan, p));
  float x = dot(civDelta(p, c.riverQ), cross(p, tn))*PLANET_R/hw;
  float k = 2.0*PI*c.riverS/(28.0*hw/riverNarrow(p) + 300.0);     // the meander's own (stored) width
  float curv = -(sin(k) + 3.65*sin(k/0.37 + 1.7))/4.65;           // offset'' normalised: > 0 bends towards +x
  float xs = x*sign(curv), grow = smoothstep(0.2, 1.0, abs(curv))*smoothstep(30.0, 200.0, hw);
  float nz = 0.0;   // two octaves through one snoise() call site
  for (int i = 0; i < loopN(2); i++) nz += (i == 0 ? 1.0 : 0.5)*snoise(i == 0 ? p*(PLANET_R/220.0) : p*(PLANET_R/70.0) + 5.0);
  float e = (xs - 1.0 + 0.6*grow)*hw + (0.12 + 0.25*grow)*hw*nz, aa = foot + 0.02*hw;   // metres past the bar waterline
  // inside the channel only, and only where it has banks (not out in the estuary at sea level)
  float inCh = (1.0 - smoothstep(-aa, aa, (xs - 1.0)*hw))*smoothstep(0.3, 1.5, c.riverLevel)*step(0.0, grow - 0.01);
  float bar = smoothstep(-aa, aa, e)*inCh;
  float shallow = smoothstep(-0.25*hw - 20.0, 0.0, e)*inCh;
  float shade = mix(0.72, 1.0, smoothstep(0.0, 25.0, e))*(1.0 + 0.12*nz);
  return vec3(bar, shallow, shade);
}

// banks: a band of dark wet soil along lake and river water, reeds on it in warm, wet climates. Lower-course
// floodplains sit 1..4 m above the level, where the biome paints beach sand: alluvial meadow or bare silt there
// instead, in patches, out to a wandering edge
vec3 waterBank(vec3 alb, vec3 p, Civ c, float h, float foot, float T, float M){
  if (c.waterKind == 0) return alb;
  if (c.waterKind == 2){
    float u = (c.dRiver - c.riverHalf)/riverPlain(c.riverHalf) + 0.3*riverWobble(p, 260.0);
    float fl = (1.0 - smoothstep(1.5, 4.5, h))*smoothstep(-0.3, 0.2, u)*(1.0 - smoothstep(0.5, 0.95, u));
    vec3 meadow = mix(vec3(0.24, 0.21, 0.15), vec3(0.075, 0.095, 0.04), smoothstep(0.2, 0.45, M)*smoothstep(0.0, 8.0, T));
    alb = mix(alb, meadow*(0.85 + 0.3*riverWobble(p, 35.0)), fl*0.9);
  }
  if (foot > 30.0) return alb;
  float nz = vnoised(uDetailOff + vRel*DETAIL_F + 17.0).x*bandw(DETAIL_F, 0.5/foot);   // 8 m clumps
  // lakes: height above the water; rivers: distance from the channel edge (their floodplain is flat)
  float u = c.waterKind == 1 ? (h - c.water)*4.0 : (c.dRiver - c.riverHalf)/(1.0 + 0.01*c.riverHalf);
  float fade = 1.0 - smoothstep(8.0, 30.0, foot);
  float wet = (1.0 - smoothstep(0.5, 3.0 + 1.5*nz, u))*fade;
  alb = mix(alb, alb*vec3(0.5, 0.47, 0.43), wet);
  float reeds = smoothstep(10.0, 18.0, T)*smoothstep(0.35, 0.6, M)*(1.0 - smoothstep(3.0, 7.0, u + 3.0*nz))*smoothstep(-0.2, 0.3, nz)*fade;
  return mix(alb, mix(vec3(0.07, 0.10, 0.035), vec3(0.16, 0.15, 0.07), nz*0.5 + 0.5), reeds*0.85);
}
`
