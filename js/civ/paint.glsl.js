import { INDUSTRY_PAINT } from './industry.glsl.js'
// Terrain-surface painting for civilisation: roads with markings, railways, city streets (district layout,
// notes/layout.md), lots, parks, plazas and night lighting. civPaint() runs up to a few km footprint (Civ from terrainCiv up to 150 m, then
// civLand()); civFar() uses the coarse global urban map from orbit. Every feature is box-filtered by the
// pixel footprint and fades to its area average once it is smaller than a pixel, so nothing shimmers.
import { LANES } from './spec.js'

const f = v => v.toFixed(2)
export const CIV_PAINT = /* glsl */`
// screen derivatives of the map uv; the terrain's deferred pass (terrain.js) supplies them from G-buffer neighbours
#ifndef UV_DX
#define UV_DX(uv) dFdx(uv)
#define UV_DY(uv) dFdy(uv)
#endif
uniform sampler2D uUrbanMap;
#define CIV_HW_LANES vec2(${f(LANES[0][0])}, ${f(LANES[0][1])})
#define CIV_LAMP 30.0            // street lamp spacing (m)
#define CIV_POOL 0.3             // lamp pool irradiance (x albedo) under a lamp: a lit street stays well below lit windows
#define CIV_E_STREET 0.018       // the pools averaged along a street (peak of the smeared profile, sigma 7 m)
#define CIV_E_WIN 0.006          // mean night radiance of lit windows seen from afar (matches the 3D buildings)
#define CIV_FAB0 2500.0          // footprint (m) where the exact city/road painting hands over to the coarse urban map
#define CIV_FAB1 5000.0

// coverage of the band |d| < w for a pixel of size aa
float civBand(float d, float w, float aa){ return clamp((w - d)/aa + 0.5, 0.0, 1.0)*min(1.0, 2.0*w/aa + 0.02); }
// four civBands at once (the same arithmetic per component: one set of vector instructions instead of four)
vec4 civBand4(vec4 d, vec4 w, float aa){ return clamp((w - d)/aa + 0.5, 0.0, 1.0)*min(vec4(1.0), 2.0*w/aa + 0.02); }
// dashed pattern along s: 'on' metres painted every 'period'; averages out when the pixel is big
float civDash(float s, float on, float period, float aa){
  if (aa > period*0.35) return on/period;
  float m = mod(s, period);
  return smoothstep(0.0, aa, m)*(1.0 - smoothstep(on - aa, on, m));
}
vec4 civDash4(vec4 s, vec4 on, vec4 period, float aa){   // four civDash at once (same arithmetic per component)
  vec4 m = mod(s, period);
  return mix(smoothstep(vec4(0.0), vec4(aa), m)*(1.0 - smoothstep(on - aa, on, m)), on/period, greaterThan(vec4(aa), period*0.35));
}
float civNoise(vec2 x){ return hash12(floor(x)); }
// loop bound n that D3D can't see is constant (uCamR > 0), so it keeps the loop instead of unrolling it: repeated
// calls go through one call site in such a loop (D3D inlines every call site; compile time grows with the copies)
int civN(int n){ return n + int(min(uCamR, 0.0)); }
// smooth value noise, flattened to its mean 0.5 once a cell is below the pixel size
float civVN(vec2 x, float cellPx){
  vec2 i = floor(x), u = fract(x);
  u = u*u*(3.0 - 2.0*u);
  float n = 0.0;
  for (int k = 0; k < civN(4); k++){               // the 4 corners, bilinear
    vec2 o = vec2(float(k & 1), float(k >> 1));
    n += hash12(i + o)*(o.x > 0.5 ? u.x : 1.0 - u.x)*(o.y > 0.5 ? u.y : 1.0 - u.y);
  }
  return mix(n, 0.5, 1.0 - smoothstep(0.5, 1.5, cellPx));
}
// the city's smooth noises civVN(x/S + O, S/aa), x = c.local (c.dl for the plaza's), all through the one civVN call
// site below; bit k of want picks the ones needed (the others stay 0.5). Slots (column, component):
// 0 asphalt patches, 1 sprawl fringe, 2 nb2, 3 nb3 (paintCity), 4 neighbourhood green (civGreen), 5 lawn, 6 trees,
// 7 canopy (civLots), 8 clump, 9 clump detail and industry lanes, 10 grass (park), 11 concrete, 12 yard asphalt,
// 15 heaps (paintIndustry, O = id.x), 13 stone, 14 crowns (plaza)
const float CIV_VN_S[16] = float[16](5.0, 170.0, 260.0, 2200.0, 700.0, 7.0, 6.0, 1.5, 14.0, 4.0, 3.0, 5.0, 3.0, 2.5, 1.5, 11.0);
const float CIV_VN_O[16] = float[16](7.0, 9.0, 5.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0);
mat4 civVNCity(vec2 local, vec2 dl, float aa, int want, float heapO){
  vec4 v0 = vec4(0.5), v1 = v0, v2 = v0, v3 = v0;
  for (int k = 0; k < civN(16); k++){
    float n = 0.5;
    if (((want >> k) & 1) != 0){
      float S = CIV_VN_S[k];
      n = civVN((k == 13 || k == 14 ? dl : local)/S + (k == 15 ? heapO : CIV_VN_O[k]), S/aa);
    }
    v0 = vec4(v0.yzw, v1.x); v1 = vec4(v1.yzw, v2.x); v2 = vec4(v2.yzw, v3.x); v3 = vec4(v3.yzw, n);
  }
  return mat4(v0, v1, v2, v3);
}
// night light gain from high up: the lit day crescent sets the exposure from orbit, so lights are lifted to stay visible
float civOrbit(){ float alt = uCamR - PLANET_R; return 1.0 + smoothstep(40e3, 400e3, alt) + 45.0*smoothstep(400e3, 6e6, alt); }

struct CivOut { vec3 alb; vec3 emit; float gloss; float flatN; };

// lamp colour of a city: older places burn sodium orange, others white LED
vec3 civLampCol(int seed, int tier){
  return hash2i(seed, 5, 31) < 0.25 + 0.15*float(tier) ? vec3(1.0, 0.86, 0.70) : vec3(1.0, 0.58, 0.24);
}
// share of gardens and trees: falls towards the core, leafier suburbs; neighbourhoods drift over ~700 m (smooth noise
// in grid space, continuous across districts), shared by the lots and their far average
float civGreen(Civ c, float nb){                  // nb = civVN(c.local/700.0, 700.0/aa)
  float kind = c.dkind == DK_SUBURB ? 0.15 : c.dkind == DK_CORE ? -0.12 : c.dkind == DK_SUPER ? 0.08 : 0.0;
  return clamp(mix(0.9, 0.12, smoothstep(0.15, 0.75, c.urban)) + kind + 0.7*(nb - 0.5), 0.0, 1.0);
}
// share of the ground under the local streets of this district (blocks ~c.block apart)
float civStreetFracD(Civ c){
  float l = min(0.9, 2.0*(PROF_HALF[c.sprof] + 1.0)/c.block);
  return 1.0 - (1.0 - l)*(1.0 - l);
}
// buildable rect of the block under p (district coords): the street-centre rect inset by the street half width + 1
// (spec.js blockInfo().lot)
vec4 civLotRect(Civ c){ float hw = PROF_HALF[c.sprof] + 1.0; return c.brect + vec4(hw, hw, -hw, -hw); }
${INDUSTRY_PAINT}

// ---- city blocks: lots with roofs, yards, gardens, parks, plazas ----
// returns albedo; sets roof coverage (for window glow) and the lot hash
vec3 civLots(Civ c, float aa, float warm, float green, mat4 vn, out float roofCov, out float lh){
  float dens = c.urban;
  vec4 rc = civLotRect(c);
  ivec2 nl = c.nlots;
  vec2 lotSize = max(rc.zw - rc.xy, vec2(4.0))/vec2(nl);
  ivec2 li = clamp(ivec2(floor((c.dl - rc.xy)/lotSize)), ivec2(0), nl - 1);
  vec2 lq = c.dl - rc.xy - (vec2(li) + 0.5)*lotSize;                 // from the lot centre (district units ~ m)
  int ix = c.kx*8 + li.x, iy = c.ky*8 + li.y;                         // lot hashes: hash2i(ix, iy, D.hseed + n)
  vec4 hl = vec4(0.0);
  for (int k = 0; k < civN(4); k++) hl = vec4(hl.yzw, hash2i(ix, iy, c.seed + (k == 0 ? 3 : 4 + k)));   // seeds 3, 5, 6, 7
  lh = hl.x;
  float lotMin = min(lotSize.x, lotSize.y);
  float r2 = hl.y, r3 = hl.z, r4 = hl.w;
  bool suburb = c.dkind == DK_SUBURB || dens < (c.tier == 3 ? 0.12 : 0.2);
  bool house = false, tower = false;
  if (c.tier <= 1 || suburb) house = !(c.tier == 1 && dens > 0.45 && !suburb && r2 < 0.7);
  else if (c.tier == 3){
    float cl = hash2i(c.dkey, 5, c.cseed + 21);                       // towers cluster by district
    float t = clamp((dens - 0.55)/0.4, 0.0, 1.0);
    tower = r2 < 0.65*t*t*(3.0 - 2.0*t)*(0.25 + 0.75*cl);
  }
  vec2 sz = lotSize - (c.dkind == DK_CORE ? 1.5 : 5.0);               // the old core builds out to the street
  if (house) sz = min(sz, vec2(8.0 + 8.0*r3, 7.0 + 5.0*r4));
  else if (tower) sz = min(sz*(0.65 + 0.3*r4), vec2(58.0));
  else if (r3 < 0.25) sz *= 0.8;
  vec2 hb = max(sz, vec2(4.0))*0.5;
  vec2 hv = fract(c.dl/9.0) - 0.5;
  vec4 bl = civBand4(vec4(abs(lq), max(abs(hv.x), abs(hv.y))*9.0, 0.0), vec4(hb, 1.3, 0.0), aa);   // roof box, rooftop plant
  roofCov = bl.x*bl.y;
  vec3 flatR = mix(vec3(0.12), vec3(0.28, 0.27, 0.25), lh);
  vec3 pitched = lh > 0.45 ? vec3(0.30, 0.12, 0.07) : vec3(0.13, 0.13, 0.14);
  float ridge = hb.x > hb.y ? lq.y : lq.x;                           // two slopes either side of the ridge
  pitched *= mix(1.0, ridge > 0.0 ? 1.15 : 0.82, 1.0 - smoothstep(0.5, 2.0, aa));
  bool pitch = (house && r2 < 0.8) || (c.dkind == DK_CORE && !tower && r2 < 0.65);
  vec3 roof = pitch ? pitched : tower ? mix(vec3(0.09, 0.095, 0.1), vec3(0.2, 0.2, 0.19), lh) : flatR;
  // rooftop plant on big flat roofs
  roof = mix(roof, roof*1.35 + 0.03, (pitch ? 0.0 : 1.0)*step(0.8, civNoise(c.dl/9.0 + 0.5))*bl.z*roofCov);
  // ground: gardens with trees in the suburbs, paving and parking downtown
  vec3 lawn = mix(vec3(0.07, 0.11, 0.04), vec3(0.10, 0.12, 0.05), vn[1].y);
  lawn = mix(lawn, vec3(0.19, 0.16, 0.10), warm*0.6);                // dry yards in hot places
  vec3 pave = mix(vec3(0.16, 0.155, 0.15), vec3(0.24, 0.23, 0.21), r2);
  vec3 ground = mix(pave, lawn, step(fract(lh*3.1), green));
  float trees = smoothstep(0.62, 0.72, vn[1].z)*green;
  vec3 canopy = mix(vec3(0.025, 0.05, 0.02), vec3(0.05, 0.085, 0.03), vn[1].w);
  ground = mix(ground, canopy, trees);
  vec3 lot = mix(ground, roof, roofCov);
  // average of this lot, used once a pixel spans a lot
  float roofFrac = min(1.0, 4.0*hb.x*hb.y/(lotSize.x*lotSize.y));
  vec3 groundAvg = mix(mix(pave, lawn, green), canopy, 0.25*green);
  lot = mix(lot, mix(groundAvg, roof, roofFrac), smoothstep(lotMin*0.08, lotMin*0.3, aa));
  if (c.park){
    vec2 pc = 0.5*(rc.xy + rc.zw), ph = 0.5*(rc.zw - rc.xy), q = c.dl - pc;
    float ed = max(abs(q.x) - ph.x, abs(q.y) - ph.y), along = abs(q.x) - ph.x > abs(q.y) - ph.y ? q.y : q.x;
    float r = length(q), fr = 3.0 + 3.0*lh;
    // plaza: slab joints, edge band, fountain ring; fountain; park: paths (edge loop, diagonals)
    vec4 bp = civBand4(vec4(abs(fract(q/3.0) - 0.5)*3.0, abs(ed + 5.0), abs(r - fr)), vec4(0.03, 0.03, 0.6, 0.5), aa);
    vec4 bq = civBand4(vec4(r, abs(ed + 8.0), abs(abs(q.x/max(ph.x, 1.0)) - abs(q.y/max(ph.y, 1.0)))*min(ph.x, ph.y), 0.0), vec4(fr - 0.5, 1.2, 1.2, 0.0), aa);
    if (c.plaza){
      // old-town square: stone slabs framed by darker bands, a fountain or plinth in the middle, trees round the edge
      float joint = max(bp.x, bp.y);
      vec3 stone = mix(vec3(0.33, 0.31, 0.28), vec3(0.41, 0.39, 0.35), vn[3].y)*(1.0 - 0.2*joint);
      stone = mix(stone, vec3(0.22, 0.21, 0.2), bp.z);
      stone = mix(stone, vec3(0.24, 0.23, 0.21), bp.w);
      stone = mix(stone, lh > 0.4 ? vec3(0.03, 0.06, 0.07) : vec3(0.2, 0.19, 0.18), bq.x);
      float crown = 1.0 - smoothstep(1.8, 2.6, length(vec2(ed + 2.5, mod(along + 3.5, 7.0) - 3.5)));
      lot = mix(stone, mix(vec3(0.025, 0.05, 0.02), vec3(0.05, 0.085, 0.03), vn[3].z), crown);
      lot = mix(lot, vec3(0.26, 0.25, 0.23), smoothstep(6.0, 25.0, aa));
    } else {
      // meadow, clumps of trees, gravel paths (a loop inside the edge and diagonals)
      float clump = smoothstep(0.45, 0.62, vn[2].x + 0.25*vn[2].y - 0.12);
      vec3 grass = mix(vec3(0.08, 0.13, 0.045), vec3(0.11, 0.14, 0.05), vn[2].z);
      float path = max(bq.y, bq.z);
      lot = mix(mix(grass, canopy, clump), vec3(0.30, 0.27, 0.22), path*(1.0 - clump));
      lot = mix(lot, vec3(0.05, 0.085, 0.03), smoothstep(8.0, 30.0, aa));
    }
    roofCov = 0.0;
  }
  return lot;
}

// ---- city streets (districts, notes/layout.md) ----
// a street near p: |d| m from the centreline, signed u, half width incl. sidewalks, s m along, index k,
// cross-section (PROF_*), role 0 row, 1 connector, 2 trunk, 3 runner-up trunk
struct CivSt { float d, u, hw, s, k; int prof, role; };
float civStKey(CivSt a){ return a.d - a.hw; }
void paintCity(inout CivOut o, Civ c, float aa, float night, float warm){
  float dens = c.urban, B = c.block;
  // industrial districts: the plot under p (its id seeds the heaps' noise); then the smooth noises this pixel needs
  ivec2 pid = ivec2(0);
  vec4 prect = vec4(0.0);
  if (c.zone == 1) prect = indPlot(c.dl, civLotRect(c), c.kx, c.ky, c.seed, pid);
  bool park = c.park && !c.plaza, plaza = c.park && c.plaza, ind = c.zone == 1;
  int want = 0xFF | (park ? 0x700 : 0) | (ind ? 0x9A00 : 0) | (plaza ? 0x6000 : 0);
  mat4 vn = civVNCity(c.local, c.dl, aa, want, float(pid.x));
  int lp = c.sprof, tk = int(c.trunk.w + 0.5);
  CivSt R = CivSt(abs(c.sRow.x), c.sRow.x, PROF_HALF[lp], c.dl.x*c.metric.x, c.sRow.y, lp, 0);
  CivSt C = CivSt(abs(c.sCon.x), c.sCon.x, PROF_HALF[lp], c.dl.y*c.metric.y, c.sCon.y, lp, 1);
  CivSt T = CivSt(abs(c.trunk.x), c.trunk.x, c.trunk.y, c.trunk.z, float(tk), profileOf(tk), 2);
  CivSt T2 = CivSt(c.trunk2.x, c.trunk2.x, c.trunk2.y, 0.0, 0.0, c.trunk2.y > 12.0 ? 3 : 2, 3);
  // the street p is on (deepest inside its half width: trunks win in their carriageway, so local streets end on
  // them in T-junctions) and the one crossing it
  CivSt me = R, ot = C;
  if (civStKey(C) < civStKey(R)){ me = C; ot = R; }
  if (civStKey(T) < civStKey(me)){ ot = me; me = T; } else if (civStKey(T) < civStKey(ot)) ot = T;
  if (civStKey(T2) < civStKey(ot)) ot = T2;
  // coverage of carriageways, walks and whole streets, the kerb line's signed distance, and (trunks) the far lines:
  // one loop over the four streets (one set of civBand call sites)
  CivSt S4[4] = CivSt[4](R, C, T, T2);
  float car = 0.0, walk = 0.0, street = 0.0, kerbSd = 1e30, trunkF = 0.0, trunkE = 0.0;
  for (int k = 0; k < civN(4); k++){
    CivSt q = S4[k];
    vec2 w = PROF_WALK[q.prof];
    vec4 b = civBand4(vec4(q.d), vec4(w, q.hw + 1.0, q.hw), aa);
    car = max(car, b.x); walk = max(walk, b.y); street = max(street, b.z);
    kerbSd = min(kerbSd, q.d - w.x);
    if (k >= 2){ trunkF = max(trunkF, b.w); trunkE += civBand(q.d, q.hw + 3.0, aa)*(q.prof == 3 ? 2.5 : 1.4); }
  }
  vec2 wm = PROF_WALK[me.prof], wo = PROF_WALK[ot.prof], ln = PROF_LANES[me.prof];
  float x = me.d, s = me.s;
  int sseed = me.role >= 2 ? c.cseed : c.seed, sid = int(me.k)*5 + me.role;
  float lane = min(abs(x - ln.x), abs(x - ln.y));
  bool stopJ = me.role < 2 && ot.role < 2;
  // the bands across this street, four at a time (civBand4): junction, open, wheel tracks, kerb; zebra, slabs;
  // setts, stop line; markings; markings and the median
  vec4 bs0 = civBand4(vec4(ot.d, ot.d, abs(lane - 0.85), abs(kerbSd - 0.15)), vec4(wo.x, wo.y + 2.5, 0.3, 0.15), aa);
  vec4 bs1 = civBand4(vec4(abs(ot.d - (wo.x + 1.6)), x, abs(fract(s/1.8) - 0.5)*1.8, abs(x - 0.5*(wm.x + wm.y))), vec4(1.3, wm.x - 0.5, 0.015, 0.015), aa);
  vec4 bs2 = civBand4(vec4(abs(fract(s/0.6) - 0.5)*0.6, abs(fract(me.u/0.3) - 0.5)*0.3, abs(ot.d - (wo.y + (stopJ ? 3.4 : 1.2))), x), vec4(0.02, 0.02, 0.2, wm.x - 0.3), aa);
  vec4 bs3 = civBand4(vec4(abs(x - 0.5*(ln.x + ln.y)), abs(x - (wm.x - 0.6)), abs(x - (PROF_MEDIAN[3] + 0.5)), abs(x - 0.15)), vec4(0.08, 0.1, 0.1, 0.06), aa);
  // dashes: lane dividers (9 m, 6 m), zebra, give-way
  vec4 ds = civDash4(vec4(s, s, me.u + 0.25, me.u + 20.0), vec4(3.0, 3.0, 0.5, 0.6), vec4(9.0, 6.0, 1.0, 1.2), aa);
  vec4 bs4 = civBand4(vec4(abs(x - 0.12), x, x, x), vec4(0.06, 0.07, PROF_MEDIAN[3], PROF_MEDIAN[3] - 0.3), aa);
  float inJunction = bs0.x;
  float open = 1.0 - bs0.y;                                         // away from the junction
  // carriageway: aged asphalt per stretch, repair patches, darker wheel tracks; setts in the old core's lanes
  float age = hash2i(sid, int(floor(s/80.0)), sseed + 19);
  vec3 asph = mix(vec3(0.075, 0.077, 0.082), vec3(0.13, 0.128, 0.125), age);
  if (me.prof == 0){
    float sett = max(bs2.x, bs2.y);
    asph = mix(vec3(0.15, 0.14, 0.13), vec3(0.21, 0.19, 0.17), age)*(1.0 - 0.3*sett);
  }
  asph *= 1.0 - 0.18*smoothstep(0.72, 0.78, vn[0].x);
  asph *= 1.0 - 0.14*bs0.z*(1.0 - inJunction);
  // markings: centre lines, lane dividers, edge lines; zebras and stop lines at junctions
  vec3 paintW = vec3(0.60, 0.60, 0.57), paintY = vec3(0.55, 0.42, 0.12);
  float mk = 0.0, mkY = 0.0;
  bool yellowCentre = hash2i(c.cseed, 7, 3) < 0.4;
  if (me.prof == 3){                                                // boulevard / ring (planted median below)
    mk = bs3.x*ds.x;
    mk = max(mk, bs3.y);
    mk = max(mk, bs3.z);
  } else if (me.prof == 2){                                         // edge streets: two lanes a side, double centre line
    mk = bs3.x*ds.x;
    float cl = bs3.w;
    if (yellowCentre) mkY = cl; else mk = max(mk, cl);
  } else if (me.prof == 1){
    if (yellowCentre) mkY = bs4.x;
    else mk = bs4.y*ds.y;
  }
  mk *= open; mkY *= open;
  float zebraOn = step(0.3, hash2i(sid, int(ot.k)*5 + ot.role, sseed + 23))*step(0.12, dens)*(me.prof == 0 ? 0.0 : 1.0);
  float zebra = bs1.x*bs1.y*ds.z*zebraOn;
  float stopL = 0.0;
  if (stopJ){                                                       // right-hand traffic heading into the junction
    bool approach = me.role == 1 ? me.u*ot.u < 0.0 : me.u*ot.u > 0.0;
    stopL = bs2.z*bs2.w*(approach ? 1.0 : 0.0);
  } else if (me.role < 2)                                           // give-way dashes where a local street meets a trunk
    stopL = bs2.z*bs2.w*ds.w;
  mk = max(max(mk, zebra*0.9), stopL*(me.prof == 0 ? 0.0 : 1.0));
  vec3 road = mix(mix(asph, paintW, mk), paintY, mkY);
  float medTree = 0.0, treeFar = smoothstep(0.1, 0.4, aa)*step(0.1, dens)*(1.0 - inJunction);
  if (me.prof == 3){
    // planted median with kerbs and a row of trees, broken at junctions
    road = mix(road, vec3(0.22, 0.21, 0.2), bs4.z*open);
    road = mix(road, mix(vec3(0.06, 0.1, 0.035), vec3(0.03, 0.055, 0.02), step(0.5, fract(s/8.0))), bs4.w*open);
    medTree = (1.0 - smoothstep(2.0, 2.8, length(vec2(x, mod(s, 8.0) - 4.0))))*treeFar*open;
  }
  // sidewalk: concrete slabs; kerb stones; verge with grass or paving and street trees (none in the core's lanes)
  float slab = max(bs1.z, bs1.w);
  vec3 walkC = mix(vec3(0.27, 0.265, 0.25), vec3(0.34, 0.33, 0.31), hash2i(sid, int(floor(s/80.0)), sseed + 29))*(1.0 - 0.25*slab);
  walkC = mix(walkC, vec3(0.40, 0.39, 0.37), bs0.w);
  vec3 verge = mix(vec3(0.075, 0.11, 0.04), vec3(0.2, 0.195, 0.185), smoothstep(0.35, 0.7, dens));
  vec2 tq = vec2(x - 0.5*(wm.y + me.hw + 1.0), mod(s + 5.0, 10.0) - 5.0);   // street tree every 10 m
  float crown = (1.0 - smoothstep(2.4, 3.4, length(tq)))*treeFar*(me.prof == 0 ? 0.0 : 1.0);
  vec3 st = mix(road, walkC, clamp(walk - car, 0.0, 1.0));
  st = mix(st, verge, clamp(street - walk, 0.0, 1.0));
  st = mix(st, vec3(0.03, 0.055, 0.02), max(crown*(1.0 - car*0.6), medTree));
  // lots, then the far-field average when whole blocks shrink below a few pixels
  float roofCov, lh, green = civGreen(c, vn[1].x);
  vec3 lot = civLots(c, aa, warm, green, vn, roofCov, lh);
  // the sprawl fringe frays into the countryside: low-density lots give way to the ground by a noise mask
  float built = smoothstep(0.0, 0.3, dens + 0.35*(vn[0].y - 0.5));
  vec3 col = mix(mix(o.alb, lot, built), st, street);
  // far off: a mosaic of districts (tiled old core, grey grid and superblocks, leafy suburbs, each its own tone),
  // a finer drift, trunk streets as lines (civBand area-averages them), parks and squares until they are sub-pixel
  float sf = civStreetFracD(c);
  float nb2 = vn[0].z, nb3 = vn[0].w;
  float dh = hash2i(c.dkey, 1, c.cseed + 43), dh2 = hash2i(c.dkey, 2, c.cseed + 43);
  vec3 roofs = c.dkind == DK_CORE ? mix(vec3(0.15, 0.115, 0.095), vec3(0.135, 0.125, 0.12), dh)
             : c.dkind == DK_SUBURB ? mix(vec3(0.145, 0.12, 0.10), vec3(0.14, 0.135, 0.13), dh)
             : mix(vec3(0.14, 0.135, 0.13), vec3(0.12, 0.12, 0.122), dh);
  roofs = mix(roofs, vec3(0.10, 0.10, 0.105), smoothstep(0.6, 0.9, dens)*(c.tier == 3 ? 1.0 : 0.0));
  roofs = mix(roofs, vec3(0.24, 0.16, 0.10), warm*0.4);
  vec3 lotFar = mix(roofs, mix(vec3(0.05, 0.085, 0.03), vec3(0.14, 0.13, 0.09), warm*0.6), green*0.8);
  vec3 far = mix(lotFar*(0.8 + 0.4*nb2)*(0.88 + 0.24*nb3)*(0.94 + 0.12*dh2), vec3(0.11, 0.11, 0.105), sf);
  far = mix(far, vec3(0.085, 0.085, 0.085), 0.6*trunkF);
  far = mix(o.alb, far, built);
  vec3 parkC = mix(c.plaza ? vec3(0.24, 0.23, 0.21) : vec3(0.06, 0.1, 0.035), far, smoothstep(B*0.4, B*1.2, aa));
  float farW = smoothstep(B*0.06, B*0.25, aa);
  col = mix(col, c.park ? parkC : far, farW);
  o.alb = mix(o.alb, col, c.urban > 0.0 ? 1.0 : 0.0);
  o.gloss = max(o.gloss, car*0.3*(1.0 - farW));
  o.flatN = max(o.flatN, max(street, c.park ? 0.3 : 0.8));
  // night: lamps every sp m (staggered on local streets, paired on trunks, wall lamps close together in the core's
  // lanes every 24 m), each a tight pool with dark gaps. Each district has its own share of lamps lit and some their own colour,
  // so the city reads as neighbourhoods with brighter trunk streets, not one even mesh
  vec3 lampCity = civLampCol(c.cseed, c.tier), lampD = lampCity;
  if (hash2i(c.dkey, 4, c.cseed + 31) < 0.3) lampD = c.dkind == DK_CORE ? vec3(1.0, 0.62, 0.3) : vec3(1.0, 0.9, 0.78);
  float lamp0 = clamp(0.25 + 0.25*float(c.tier) + 0.6*dens, 0.3, 1.0), distF = 0.65 + 0.7*hash2i(c.dkey, 3, c.cseed + 41);
  float lampF = clamp(lamp0*(me.role >= 2 ? 1.0 : distF), 0.25, 1.0);   // share of lamp posts present
  float sp = me.prof == 0 ? 24.0 : me.prof == 3 ? 36.0 : me.role >= 2 ? 32.0 : c.dkind == DK_SUBURB ? 40.0 : 30.0;
  float side = me.u < 0.0 ? 1.0 : 0.0, lampS = s + (me.role >= 2 ? 0.0 : side*sp*0.5);
  float lampOn = step(hash2i(sid*2 + int(side), int(floor(lampS/sp)), sseed + 37), lampF);
  float lx = me.prof == 0 ? wm.y - 0.3 : wm.x - 1.0;                  // wall lamps in lanes, kerbside poles elsewhere
  vec2 lq = vec2(x - lx, mod(lampS, sp) - sp*0.5);
  float pool = exp(-lq.x*lq.x/(2.0*12.0) - lq.y*lq.y/(2.0*20.0));
  float hs = max(0.3, aa*0.6), pk = me.prof == 0 ? 0.45 : 1.0;   // dim wall lamps: the core's dense lanes stay below the streets
  float head = (0.09/(hs*hs))*exp(-dot(lq, lq)/(2.0*hs*hs))*smoothstep(0.03, 0.12, aa);
  float lampNear = pk*(col.g*CIV_POOL*(pool*lampOn + 0.04*street) + 0.35*head*lampOn);
  float lampAvg = CIV_E_STREET*lampF*pk*(30.0/sp)*exp(-(x - lx)*(x - lx)/(2.0*49.0))*street;   // pools smeared along the street
  float lamps = mix(lampNear, lampAvg, smoothstep(6.0, 14.0, aa));
  float lit = step(0.3, fract(lh*9.1))*(0.35 + 0.65*dens);
  float windows = CIV_E_WIN*roofCov*lit*smoothstep(1.5, 8.0, aa)*(1.0 - street);
  // far off: local streets by length per area and the district's lamp share, trunks as brighter lines, windows
  float farE = CIV_E_STREET*lamp0*(18.0*distF/B + trunkE) + CIV_E_WIN*0.5*(0.35 + 0.65*dens)*(1.0 - sf);
  farE *= (c.park ? 0.3 : 1.0)*built*(0.6 + 0.8*nb2)*(0.5 + 1.0*nb3)*(0.3 + 1.1*dens);
  float e = mix(lamps*(0.6 + 0.4*dens) + windows, farE, farW);
  vec3 lampC = mix(me.role >= 2 ? lampCity : lampD, mix(lampD, lampCity, 0.5), farW);
  o.emit += mix(lampC, vec3(1.0, 0.78, 0.5), clamp(windows/max(e, 1e-5), 0.0, 1.0))*night*e;
  if (c.zone == 1) paintIndustry(o, c, aa, night, street, farW, prect, pid, vn);   // industrial districts (industry.glsl.js)
}

void paintRoad(inout CivOut o, Civ c, float aa, float night){
  float d = c.dRoad, hw = c.halfW;
  float cov = civBand(d, hw, aa);
  if (cov <= 0.0) return;
  bool hwy = c.roadType == CIV_HIGHWAY;
  vec3 asphalt = vec3(0.085, 0.086, 0.09)*(0.85 + 0.3*civVN(vec2(c.roadS/6.0, d/1.5), 1.5/aa));
  float track = hwy ? min(abs(d - CIV_HW_LANES.x), abs(d - CIV_HW_LANES.y)) : abs(d - ${f(LANES[1][0])});
  // wheel tracks, shoulder edge, road centre and edge lines; highway median, lines (civBand4)
  vec4 br = civBand4(vec4(abs(track - 0.85), d, d, abs(d - 3.4)), vec4(0.35, 10.0, 0.07, 0.08), aa);
  asphalt *= 1.0 - 0.15*br.x;
  vec3 paint = vec3(0.62, 0.62, 0.58), yellow = vec3(0.58, 0.44, 0.12);
  vec3 r = asphalt;
  float m = 0.0, dash = civDash(c.roadS, 3.0, hwy ? 12.0 : 8.0, aa);   // centre dashes
  if (hwy){
    vec4 bh = civBand4(vec4(d, abs(d - 2.4), abs(d - 0.5*(CIV_HW_LANES.x + CIV_HW_LANES.y)), abs(d - 9.8)), vec4(0.4, 0.1, 0.08, 0.12), aa);
    r = mix(r, vec3(0.26, 0.255, 0.245), bh.x);                              // concrete median barrier
    m = max(m, bh.y);                                                        // inner edge lines
    m = max(m, bh.z*dash);
    m = max(m, bh.w);                                                        // outer edge line
    r = mix(r, asphalt*1.3, cov - br.y);                                     // shoulder
  } else {
    m = max(m, br.z*dash);                                                   // centre dashes
    m = max(m, br.w);                                                        // edge lines
  }
  r = mix(r, hwy ? paint : mix(paint, yellow, 0.3), m);
  o.alb = mix(o.alb, r, cov);
  o.gloss = max(o.gloss, cov*0.35);
  o.flatN = max(o.flatN, cov);
  // night: distant highways read as streams of head- and tail-lights; lamps along the median near towns
  float farCars = smoothstep(1.0, 4.0, aa);
  vec3 e = vec3(0.0);
  if (hwy) e += mix(vec3(1.0, 0.9, 0.75), vec3(0.9, 0.08, 0.04), 0.5)*0.015*farCars*cov;   // sparse traffic: faint threads
  // lamps near towns: on the median of highways, on the verge of roads
  float gap = hwy ? 45.0 : 35.0, lampD = hwy ? 0.0 : hw + 0.5, poolD = hwy ? 5.5 : hw - 2.0;
  float q = mod(c.roadS, gap) - 0.5*gap;
  float lampNear = exp(-q*q/(2.0*25.0) - (d - poolD)*(d - poolD)/(2.0*16.0))*r.g*3.0*cov
                 + exp(-(q*q + (d - lampD)*(d - lampD))/(2.0*max(0.09, aa*aa*0.36)))*0.3*smoothstep(0.03, 0.12, aa);
  float lamp = mix(lampNear, 0.025*cov, smoothstep(8.0, 20.0, aa));
  e += vec3(1.0, 0.8, 0.55)*lamp*smoothstep(0.0, 0.3, c.flatW);
  o.emit = mix(o.emit, e*night, cov);
}

void paintRail(inout CivOut o, Civ c, float aa){
  float d = c.dRail;
  vec4 b = civBand4(vec4(d, d, d, abs(d - RAIL_GAUGE*0.5 - 0.036)), vec4(2.1, 1.0, 1.3, 0.036), aa);   // bed, between the rails, ties, rails
  float bed = b.x;
  if (bed <= 0.0) return;
  vec3 ballast = vec3(0.17, 0.155, 0.14)*(0.75 + 0.5*civNoise(vec2(c.railS, d)*9.0)*min(1.0, 0.1/aa + 0.3));
  ballast = mix(ballast, vec3(0.11, 0.09, 0.075), b.y*0.5);   // oil and rust between the rails
  float ties = b.z*civDash(c.railS, 0.26, 0.65, aa);
  vec3 r = mix(ballast, vec3(0.12, 0.10, 0.08), ties);
  float rails = b.w;
  r = mix(r, vec3(0.45, 0.44, 0.43), rails);
  o.alb = mix(o.alb, r, bed);
  o.gloss = max(o.gloss, rails*0.8);
  o.flatN = max(o.flatN, bed*0.7);
}

// warm = 0 (cold climate) .. 1 (hot): picks roof styles
CivOut civPaint(vec3 alb, Civ c, float foot, float night, float warm){
  CivOut o; o.alb = alb; o.emit = vec3(0.0); o.gloss = 0.0; o.flatN = 0.0;
  float aa = max(foot, 0.02);
  if (c.urban > 0.0) paintCity(o, c, aa, night, warm);
  if (c.roadType >= 0) paintRoad(o, c, aa, night);
  if (c.dRail < 3.0) paintRail(o, c, aa);
  return o;
}

// distant look from the urban map (sampled with explicit gradients to survive the longitude seam)
vec2 civFarUV(vec3 p){ return vec2(atan(p.x, p.z)*(0.5/PI) + 0.5, asin(clamp(p.y, -1.0, 1.0))/PI + 0.5); }
vec4 civFarSample(vec3 p){
  vec2 uv = civFarUV(p);
  vec2 dx = UV_DX(uv), dy = UV_DY(uv);
  dx.x -= floor(dx.x + 0.5); dy.x -= floor(dy.x + 0.5);
  return textureGrad(uUrbanMap, uv, dx, dy);
}
// how close to a town: 80 km blur of the urban channel (towns and their halos)
float civNearCity(vec3 p){ return smoothstep(0.003, 0.06, textureLod(uUrbanMap, civFarUV(p), 3.0).r); }
// r = urban cover (dense at big city cores), g = highway, b = rail, a = road; foot = pixel size (m)
void civFar(inout vec3 alb, inout vec3 emit, vec4 um, float farmD, vec3 p, float foot, float w, float night){
  if (w <= 0.0) return;                             // everything below is weighted by w (0 below CIV_FAB0)
  float core = smoothstep(0.3, 0.9, um.r);
  vec3 sn = vec3(0.0);                               // snoise at 3e3, 1.2e4 and 600 per unit radius, one call site
  for (int k = 0; k < civN(3); k++) sn = vec3(sn.yz, snoise(p*(k == 0 ? 3.0e3 : k == 1 ? 1.2e4 : 600.0)));
  // neighbourhood-scale patchwork, visible from tens of km, averaged out from orbit
  float patchN = mix(sn.x*0.5 + 0.5, 0.5, smoothstep(700.0, 2500.0, foot));
  vec3 city = mix(vec3(0.12, 0.125, 0.10), vec3(0.16, 0.155, 0.15), core)*(0.85 + 0.3*patchN);
  alb = mix(alb, city, min(1.0, um.r*1.1)*w);
  alb = mix(alb, vec3(0.1), (um.g*0.35 + um.a*0.2 + um.b*0.15)*w);
  // night: clustered lights, brightest in the cores, highways as faint threads between them. At the handover
  // (CIV_FAB0..1) this matches the painted blocks' average; higher up it is lifted further (with civOrbit), since
  // from orbit the lit day crescent sets the exposure
  // towns cover a sliver of a texel but should still read as points from orbit;
  // a finer mottle breaks the cores into districts when seen from a few hundred km
  float mottle = mix(sn.y*0.5 + 0.5, 0.5, smoothstep(300.0, 900.0, foot));
  // sub-linear only from orbit (else texels show); further from high orbit, where towns are a sliver of a pixel
  float cover = mix(mix(um.r, pow(um.r, 0.55), smoothstep(1500.0, 5000.0, foot)), pow(um.r, 0.42), smoothstep(15e3, 60e3, foot));
  // roads are lit near the towns they join and fade to faint car lights out in open country, so the threads
  // visibly run into the city glows instead of floating on their own (80 km blur of the urban channel)
  float roads = (um.g*0.006 + um.a*0.0012 + um.b*0.0006)*(0.3 + 0.7*civNearCity(p));
  float high = smoothstep(4000.0, 20000.0, foot);
  float lights = (cover*(0.5 + 0.5*core)*(0.5 + patchN)*(0.4 + 1.2*mottle)*0.005 + roads*0.1)*(1.0 + 48.0*high);
  // settled country: hamlets and farms too small to be places (civScatter's average), clustered by a coarse speckle
  lights += farmD*3.5e-4*(0.3 + 1.4*max(sn.z, 0.0))*(1.0 + 12.0*high);
  emit += vec3(1.0, 0.7, 0.4)*night*w*lights*civOrbit();
}
// cities and roads only (no rivers) for the painting past the terrain's own lookup (footprint > 150 m): terrainCiv's
// walk found them (civ.glsl.js civFarMode; main sets it before terrainCiv), so the shader holds one copy of civCity.
// p must be the p main passed to terrainCiv
Civ civLand(vec3 p){ return civFarLand(p); }

// scattered lights: farms, hamlets and outer suburbs, thinning out around towns (halo = urban map value, or the
// regional settlement density out in the country). Up close a lit yard with a bright lamp, then a point about a
// pixel wide that never fades below faint, and the average once the 150 m cells shrink below a pixel.
// None inside woods: civCanopy is the painted canopy share of the pixel (set by forestPaint(), 0 without forest)
float civCanopy = 0.0;
vec3 civScatter(vec3 p, float foot, float halo){
  float dens = clamp(halo*0.45, 0.0, 0.35);                  // chance of a light per ~150 m cell
  if (dens < 0.003 || civCanopy >= 1.0) return vec3(0.0);
  vec3 q = p*(PLANET_R/150.0), id = floor(q);
  vec3 c = id + 0.5 + (vec3(hash13(id + 1.7), hash13(id + 3.1), hash13(id + 5.3)) - 0.5)*0.7;
  float d = length(q - c)*150.0, r = max(foot*0.7, 1.0);
  float lamp = step(hash13(id), dens)*(0.004*exp(-d*d/(2.0*64.0)) + clamp(0.8/(r*r), 0.003, 0.3)*exp(-d*d/(2.0*r*r)));
  lamp = mix(lamp, dens*0.0015, smoothstep(100.0, 220.0, foot));
  // hamlets: a small glow (~120 m) in some 2.4 km cells, a faint point about a pixel wide from high up
  float hd = clamp(halo*2.0, 0.0, 0.6);
  vec3 q2 = p*(PLANET_R/2400.0), id2 = floor(q2);
  vec3 c2 = id2 + 0.5 + (vec3(hash13(id2 + 2.3), hash13(id2 + 4.1), hash13(id2 + 6.7)) - 0.5)*0.6;
  float d2 = length(q2 - c2)*2400.0, r2 = max(foot*0.7, 120.0);
  float ham = step(hash13(id2 + 8.9), hd)*max(0.001*14400.0/(r2*r2), 0.002)*exp(-d2*d2/(2.0*r2*r2));
  lamp += mix(ham, hd*0.001, smoothstep(900.0, 1800.0, foot));
  return vec3(1.0, 0.72, 0.42)*lamp*(1.0 - civCanopy);
}
`
