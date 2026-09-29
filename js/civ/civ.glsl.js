// Civilisation lookup for shaders. A lat/lon cell grid (8 per degree, busy cells split 4x4) lists the
// road/rail segments and cities near each point; civAt(p) returns the nearest road, rail and the city
// block under p. terrain() wraps terrainBase(): city ground is smoothed and road/rail beds are levelled
// across their width with embankments and cuttings, so every shader and the CPU probe agree.
// Rivers (RIVER_BIT items, hydro.js) are carved as channels below their water level with a floodplain;
// lakes come from a coarse level map, their shoreline is wherever the terrain is below that level.
import { SPEC_GLSL, HYDRO_GLSL, RIVER_MEANDER_GLSL, BRIDGE_GLSL } from './spec.js'

export const CIV = /* glsl */`
${SPEC_GLSL}
${HYDRO_GLSL}
${RIVER_MEANDER_GLSL}
${BRIDGE_GLSL}
uniform highp usampler2D uCivGrid;    // cell header: bit31 subgrid, bits 8..30 offset, bits 0..7 count
uniform highp usampler2D uCivItems;   // item ids; CITY_BIT marks cities
uniform highp sampler2D uCivSeg;      // 2 texels per segment: (A, type), (B, distance along route at A)
uniform highp sampler2D uCivCity;     // 2 texels per city: (centre, radius), (grid angle, block, seed, tier)
uniform highp sampler2D uCivRiver;    // 3 texels per river segment: (A, half width), (B, half width), (level at A, level at B, distance at A, river length)
uniform highp sampler2D uLakeMap;     // lake level per flood-grid cell (-1e5 = dry); 1x1 = no lakes yet
uniform vec3 uCamDir;                 // camera direction (unit), for precise offsets near the camera
uniform float uCamR;                  // camera distance from the planet centre (m)

// Fragment shaders call civSetRel(vRel): offsets p - A are then (camDir - A) + civRel with no float
// cancellation, so street edges and markings stay crisp at walking height. civRel is the exact
// p - camDir: for p = (camPos + vRel)/L, p - camDir = vRel/L - camDir*(L - |camPos|)/L, and L - |camPos|
// is formed from small quantities, so it also holds when the camera is high above the ground.
vec3 civRel = vec3(0.0);
bool civHasRel = false;
void civSetRel(vec3 vRel){
  float L = length(uCamDir*uCamR + vRel);
  float dL = (2.0*uCamR*dot(uCamDir, vRel) + dot(vRel, vRel))/(L + uCamR);
  civRel = vRel/L - uCamDir*(dL/L);
  civHasRel = true;
}
vec3 civDelta(vec3 p, vec3 A){ return civHasRel ? (uCamDir - A) + civRel : p - A; }

struct Civ {
  float dRoad, halfW, roadS; int roadType; vec3 roadQ, roadTan;   // nearest road (edge-distance wise)
  float dRail, railS; vec3 railQ, railTan;                          // nearest railway
  float flatW;                                                      // city ground smoothing 0..1
  // city block under p (spec.js cityCell; notes/layout.md): local = grid coords, dl = district coords, seed = district hash
  // seed (cseed = the city's), block (kx, ky) of district dkey, brect its street-centre rect (district coords), zone 1 = industrial
  float urban; vec2 local; float block; int seed, kx, ky, tier, zone; bool park;
  vec2 metric;                                                      // metres per district unit across connectors (x) / rows (y)
  int cseed, dkey, dkind, sprof; bool plaza; vec2 dl; vec4 brect; ivec2 nlots;
  vec2 sRow, sCon;             // local streets: (signed m to the nearest row line, its index), (signed m to the nearest connector segment, index)
  vec4 trunk; vec2 trunk2;     // nearest trunk street (signed m, half width, m along, SK_ kind); runner-up (m, half width)
  float dRiver, riverHalf, riverLevel, riverS, riverSlope; vec3 riverQ, riverTan;   // nearest river (edge-distance wise); slope = level drop per metre
  float lakeLevel, lakeM;                                           // lake around p: level (-1e9 outside its shore mask), share of its cells
  float water; int waterKind;                                       // local water level (-1e9 none; the sea is at 0), 1 lake 2 river
};

ivec2 civTexel(uint i){ return ivec2(int(i % CIV_ROW), int(i / CIV_ROW)); }

void civSegment(inout Civ c, vec3 p, uint id){
  vec4 a = texelFetch(uCivSeg, civTexel(id*2u), 0);
  vec4 b = texelFetch(uCivSeg, civTexel(id*2u + 1u), 0);
  vec3 ab = b.xyz - a.xyz;
  float L2 = max(dot(ab, ab), 1e-20);
  vec3 pa = civDelta(p, a.xyz);
  float t = clamp(dot(pa, ab)/L2, 0.0, 1.0);
  vec3 q = a.xyz + ab*t;
  vec3 v = pa - ab*t;
  v -= p*dot(v, p);                          // tangential distance only (the chord sags below the sphere)
  float d = length(v)*PLANET_R;
  int type = int(a.w + 0.5);
  float s = b.w + t*sqrt(L2)*PLANET_R;
  vec3 tn = ab*inversesqrt(L2);
  if (type == CIV_RAIL){
    if (d < c.dRail){ c.dRail = d; c.railS = s; c.railQ = q; c.railTan = tn; }
  } else if (d - CIV_HALF[type] < c.dRoad - c.halfW){
    c.dRoad = d; c.halfW = CIV_HALF[type]; c.roadType = type; c.roadS = s; c.roadQ = q; c.roadTan = tn;
  }
}

// slow world-fixed wobble -1..1 (three sines of p: the same in every shader, no noise call): irregular channel
// and floodplain edges. Wavelengths ~3.3, 6.3 and 10 x scale metres
float riverWobble(vec3 p, float scale){
  vec3 q = p*(PLANET_R/scale);
  return (sin(dot(q, vec3(0.83, 0.42, -0.37))) + sin(dot(q, vec3(-0.29, 0.71, 0.64))*0.61 + 1.3) + 0.6*sin(dot(q, vec3(0.45, -0.62, 0.64))*1.9 + 0.4))/2.6;
}
// c.riverHalf = stored half width x this. Anything phased along riverS by the width (the meander) must divide it back
// out: s is ~1e6 m, so a few % of width change would scramble the phase
float riverNarrow(vec3 p){ return 0.87 + 0.13*riverWobble(p, 150.0); }

void civRiver(inout Civ c, vec3 p, uint id){
  vec4 a = texelFetch(uCivRiver, civTexel(id*3u), 0);
  vec4 b = texelFetch(uCivRiver, civTexel(id*3u + 1u), 0);
  vec3 ab = b.xyz - a.xyz;
  float L2 = max(dot(ab, ab), 1e-20);
  vec3 pa = civDelta(p, a.xyz);
  float t = clamp(dot(pa, ab)/L2, 0.0, 1.0);
  vec3 v = pa - ab*t;
  v -= p*dot(v, p);
  float hw = mix(a.w, b.w, t);
  if (length(v)*PLANET_R - 0.9*hw - hw > c.dRiver - c.riverHalf) return;   // can't beat the current river even meandering
  vec4 e = texelFetch(uCivRiver, civTexel(id*3u + 2u), 0);
  vec3 tn = ab*inversesqrt(L2), sd = normalize(cross(p, tn));
  float s = e.z + t*sqrt(L2)*PLANET_R, off = riverMeander(s, hw, e.w - s);
  float d = length(v - sd*(off/PLANET_R))*PLANET_R;                       // distance to the meandering channel centre
  if (d - hw < c.dRiver - c.riverHalf){
    // banks wander in and out (narrower only: world.nearestWater's edge stays outside the water)
    c.dRiver = d; c.riverHalf = hw*riverNarrow(p); c.riverLevel = mix(e.x, e.y, t); c.riverS = s; c.riverSlope = (e.x - e.y)/(sqrt(L2)*PLANET_R);
    c.riverQ = a.xyz + ab*t + sd*(off/PLANET_R); c.riverTan = tn;
  }
}

// lake around p: the lake with the most weight among the 4 nearest map cells (cells of one lake share its level).
// Straight-line vector code: loops over arrays here cost seconds of D3D compile time in every shader.
void civLake(inout Civ c, vec3 p){
  ivec2 sz = textureSize(uLakeMap, 0);
  if (sz.x < 2) return;
  vec2 cc = vec2((atan(p.x, p.z)/(2.0*PI) + 0.5)*float(sz.x), (asin(clamp(p.y, -1.0, 1.0))/PI + 0.5)*float(sz.y)) - 0.5;
  vec2 f = cc - floor(cc);
  ivec2 i0 = ivec2(floor(cc));
  int x0 = (i0.x + sz.x) % sz.x, x1 = (i0.x + 1) % sz.x, y0 = clamp(i0.y, 0, sz.y - 1), y1 = clamp(i0.y + 1, 0, sz.y - 1);
  vec4 lv = vec4(texelFetch(uLakeMap, ivec2(x0, y0), 0).r, texelFetch(uLakeMap, ivec2(x1, y0), 0).r,
                 texelFetch(uLakeMap, ivec2(x0, y1), 0).r, texelFetch(uLakeMap, ivec2(x1, y1), 0).r);
  vec4 w = vec4((1.0 - f.x)*(1.0 - f.y), f.x*(1.0 - f.y), (1.0 - f.x)*f.y, f.x*f.y);
  vec4 m = vec4(dot(w, vec4(equal(lv, lv.xxxx))), dot(w, vec4(equal(lv, lv.yyyy))), dot(w, vec4(equal(lv, lv.zzzz))), dot(w, vec4(equal(lv, lv.wwww))))
         *vec4(greaterThan(lv, vec4(-1e4)));
  float best = max(max(m.x, m.y), max(m.z, m.w));
  if (best <= 0.0) return;
  c.lakeM = best;
  c.lakeLevel = best == m.x ? lv.x : best == m.y ? lv.y : best == m.z ? lv.z : lv.w;
}

void civTrunk(inout vec4 t1, inout vec4 t2, vec4 t){
  if (abs(t.x) - t.y < abs(t1.x) - t1.y){ t2 = t1; t1 = t; } else if (abs(t.x) - t.y < abs(t2.x) - t2.y) t2 = t;
}
void civCity(inout Civ c, vec3 p, uint id){
  vec4 a = texelFetch(uCivCity, civTexel(id*2u), 0);
  vec4 b = texelFetch(uCivCity, civTexel(id*2u + 1u), 0);
  vec3 C = a.xyz;
  float rad = a.w;
  // street frame, identical to cityFrame() in spec.js
  vec3 e = normalize(vec3(C.z, 0.0, -C.x));
  vec3 n = cross(C, e);
  float cs = cos(b.x), sn = sin(b.x);
  vec3 ex = e*cs + n*sn, ny = -e*sn + n*cs;
  vec3 v = civDelta(p, C);
  vec2 local = vec2(dot(v, ex), dot(v, ny))*PLANET_R;
  int seed = int(b.z + 0.5), tier = int(b.w + 0.5);
  // organic outline and warped grid space (spec.js): local metres -> grid space
  CityOutline o = cityOutline(seed, rad);
  float shape = cityShape(o, local);
  c.flatW = max(c.flatW, smoothstep(-0.2, 0.02, shape));
  if (shape < -0.16) return;
  CityWarp w = cityWarp(seed, rad, o);
  mat2 J;
  vec2 g = localToGrid(w, local, J);
  // district (Voronoi), district coords, row and block
  float edge; vec2 en, em; ivec2 nb;
  District D = district(districtAt(g, rad, seed, edge, en, em, nb), w, seed, tier, b.y);
  vec2 dl = toDistrict(D, g);
  float y0, y1;
  int ky = rowAt(D, dl.y, y0, y1);
  vec2 cn = rowConnectors(D, ky);
  int kx = int(floor((dl.x - cn.y)/cn.x));
  vec4 rect = vec4(float(kx)*cn.x + cn.y, y0, float(kx + 1)*cn.x + cn.y, y1);
  float dens = blockDensityD(w, fromDistrict(D, 0.5*(rect.xy + rect.zw)), kx, ky, D.hseed, seed);
  if (dens <= c.urban) return;
  // metres across a line of district x / y: 1 / |grad_local|
  mat2 Jt = transpose(inverse(J));
  float wk = D.A*D.k*cos(D.k*dl.x);
  float mx = 1.0/length(Jt*vec2(D.c, D.s)), my = 1.0/length(Jt*vec2(-D.s - wk*D.c, D.c - wk*D.s));
  c.urban = dens; c.local = g; c.block = sqrt(D.bx*D.by); c.seed = D.hseed; c.cseed = seed; c.kx = kx; c.ky = ky; c.tier = tier;
  c.dkey = D.key; c.dkind = D.kind; c.sprof = profileOf(D.prof); c.dl = dl; c.brect = rect; c.metric = vec2(mx, my);
  c.zone = D.kind == DK_INDUSTRY ? 1 : 0;
  c.park = c.zone == 0 && hash2i(kx, ky, D.hseed + 7) < (D.kind == DK_SUPER ? 0.12 : D.kind == DK_CORE ? 0.03 : 0.07);
  c.plaza = c.park && D.kind == DK_CORE;
  float hw = PROF_HALF[c.sprof] + 1.0;
  c.nlots = blockLotsD(D.kind, kx, ky, D.hseed, rect + vec4(hw, hw, -hw, -hw));
  // local streets: the nearest row line; connectors of this row and of the row across that line (they end on it)
  bool lower = dl.y - y0 < y1 - dl.y;
  c.sRow = vec2((dl.y - (lower ? y0 : y1))*my, float(lower ? ky : ky + 1));
  float k = floor((dl.x - cn.y)/cn.x + 0.5);
  c.sCon = vec2((dl.x - (k*cn.x + cn.y))*mx, k);
  vec2 ca = rowConnectors(D, lower ? ky - 1 : ky + 1);
  float ka = floor((dl.x - ca.y)/ca.x + 0.5), ya0 = lower ? rowLine(D, ky - 1) : y1, ya1 = lower ? y0 : rowLine(D, ky + 2);
  float dxa = (dl.x - (ka*ca.x + ca.y))*mx, da = length(vec2(dxa, max(0.0, max(ya0 - dl.y, dl.y - ya1))*my));
  if (da < abs(c.sCon.x)) c.sCon = vec2(dxa < 0.0 ? -da : da, ka);
  // trunk streets: the district edge (lattice distance pulled back through the log-radial map, signed from the lower
  // district key to the higher), radial boulevards, the ring road; the nearest two by edge distance
  float r = length(g), R1 = LAT_R1*rad, s0 = latScale(r, R1, false), s1 = 1.0/(1.0 + r/R1);
  vec2 gh = r > 1e-3 ? g/r : vec2(1.0, 0.0);
  vec2 eg = s0*en + (s1 - s0)*dot(gh, en)*gh, mg = em*latScale(length(em), R1, true);
  float sg = D.key < nb.x*1024 + nb.y ? 1.0 : -1.0;
  vec4 t1 = vec4(-sg*edge/length(Jt*eg), PROF_HALF[2], sg*dot(g - mg, vec2(-eg.y, eg.x))/length(eg), float(SK_EDGE)), t2 = vec4(1e9, 0.0, 0.0, float(SK_EDGE));
  int nbv = tier == 0 ? 0 : 2*tier;
  float a0 = hash2i(seed, 1, 81)*2.0*PI;
  for (int i = 0; i < nbv; i++){
    float ang = a0 + 2.0*PI*(float(i) + 0.35*(hash2i(i, 2, seed + 81) - 0.5))/float(nbv);
    vec2 ed = vec2(cos(ang), sin(ang)), nd = vec2(-ed.y, ed.x);
    float t = dot(g, ed);
    if (t > 0.0 && t < BOULEVARD_END*rad) civTrunk(t1, t2, vec4(dot(g, nd)/length(Jt*nd), PROF_HALF[3], t, float(SK_BOULEVARD)));
  }
  if (tier >= 2){
    float th = atan(g.y, g.x), h1 = hash2i(seed, 1, 83)*2.0*PI, h2 = hash2i(seed, 2, 83)*2.0*PI, R0 = rad*(0.4 + 0.1*hash2i(seed, 0, 84));
    float Rr = R0*(1.0 + 0.1*sin(2.0*th + h1) + 0.05*sin(3.0*th + h2)), dR = R0*(0.2*cos(2.0*th + h1) + 0.15*cos(3.0*th + h2));
    vec2 gr = gh + dR/max(r, 1.0)*vec2(gh.y, -gh.x);
    civTrunk(t1, t2, vec4((r - Rr)/length(Jt*gr), PROF_HALF[3], th*R0, float(SK_RING)));
  }
  c.trunk = t1; c.trunk2 = vec2(abs(t2.x), t2.y);
}

Civ civNone(vec3 p){
  Civ c;
  c.dRoad = 1e9; c.halfW = 0.0; c.roadS = 0.0; c.roadType = -1; c.roadQ = p; c.roadTan = vec3(1.0, 0.0, 0.0);
  c.dRail = 1e9; c.railS = 0.0; c.railQ = p; c.railTan = vec3(1.0, 0.0, 0.0);
  c.flatW = 0.0; c.urban = 0.0; c.local = vec2(0.0); c.block = 100.0; c.seed = 0; c.kx = 0; c.ky = 0; c.tier = -1; c.zone = 0; c.park = false; c.metric = vec2(1.0);
  c.cseed = 0; c.dkey = 0; c.dkind = 0; c.sprof = 1; c.plaza = false; c.dl = vec2(0.0); c.brect = vec4(0.0, 0.0, 100.0, 100.0); c.nlots = ivec2(1);
  c.sRow = vec2(1e9, 0.0); c.sCon = vec2(1e9, 0.0); c.trunk = vec4(1e9, 0.0, 0.0, 2.0); c.trunk2 = vec2(1e9, 0.0);
  c.dRiver = 1e9; c.riverHalf = 0.0; c.riverLevel = -1e9; c.riverS = 0.0; c.riverSlope = 0.0; c.riverQ = p; c.riverTan = vec3(1.0, 0.0, 0.0);
  c.lakeLevel = -1e9; c.lakeM = 0.0; c.water = -1e9; c.waterKind = 0;
  return c;
}

// cell header of the lookup cell under p (offset << 8 | count)
uint civCell(vec3 p){
  vec2 cc = vec2((atan(p.x, p.z)/(2.0*PI) + 0.5)*float(GRID_W), (asin(clamp(p.y, -1.0, 1.0))/PI + 0.5)*float(GRID_H));
  ivec2 ic = clamp(ivec2(floor(cc)), ivec2(0), ivec2(GRID_W - 1, GRID_H - 1));
  uint hdr = texelFetch(uCivGrid, ic, 0).r;
  if ((hdr & 0x80000000u) != 0u){
    ivec2 sc = clamp(ivec2(fract(cc)*4.0), ivec2(0), ivec2(3));
    hdr = texelFetch(uCivItems, civTexel(((hdr >> 8u) & 0x7FFFFFu) + uint(sc.y*4 + sc.x)), 0).r;
  }
  return hdr;
}

// walk the cell list under p: everything, or (all = false) only the rivers (they come first in every cell list)
Civ civWalk(vec3 p, bool all){
  Civ c = civNone(p);
  uint hdr = civCell(p);
  uint off = (hdr >> 8u) & 0x7FFFFFu, cnt = min(hdr & 255u, 160u);
  for (uint i = 0u; i < cnt; i++){
    uint id = texelFetch(uCivItems, civTexel(off + i), 0).r;
    if ((id & CITY_BIT) != 0u){ if (!all) break; civCity(c, p, id & 0x7FFFFFFFu); }
    else if ((id & RIVER_BIT) != 0u) civRiver(c, p, id & 0x3FFFFFFFu);
    else { if (!all) break; civSegment(c, p, id); }
  }
  return c;
}
Civ civAt(vec3 p){ return civWalk(p, true); }

// The painting past the terrain's own lookup (footprint > 150 m: civLand for cities / roads, civRivers for the far water
// lines) rides on terrainCiv's walk: the fragment shader sets civFarMode before terrainCiv (0 none, 1 rivers, 2 all) and
// terrainCiv's one civWalk call site leaves the result in civFarC. D3D inlines every call site, so this keeps one copy
// of civCity / civSegment / civRiver instead of two. Cities, roads and rivers write disjoint Civ fields (and read only
// their own), so one walk splits exactly into the land-only and rivers-only lookups.
int civFarMode = 0;
Civ civFarC;
void civRiverFrom(inout Civ c, Civ f){
  c.dRiver = f.dRiver; c.riverHalf = f.riverHalf; c.riverLevel = f.riverLevel; c.riverS = f.riverS; c.riverSlope = f.riverSlope;
  c.riverQ = f.riverQ; c.riverTan = f.riverTan;
}
// cities and roads only (civFarMode 2)
Civ civFarLand(vec3 p){ Civ c = civFarC; civRiverFrom(c, civNone(p)); return c; }
// rivers only (civFarMode >= 1): the far water lines
Civ civRivers(vec3 p){ Civ c = civNone(p); civRiverFrom(c, civFarC); return c; }

// river cross-section from the stored water level: parabolic channel below it; beside it a floodplain, higher
// ground cut down to 1..4 m above the level and lower ground filled only to 0.4 m above it (no levee walls), then
// back to natural ground. Its width wanders (riverWobble), so its edge doesn't run parallel to the channel. Where
// the ground is already under water (sea, lake) the bed only cuts, so river mouths don't build dikes out into the water.
vec4 civRiverBed(vec4 t, vec3 p, inout Civ c){
  float hw = c.riverHalf, d = c.dRiver, L = c.riverLevel;
  float plain = riverPlain(hw)*(0.8 + 0.2*riverWobble(p, 700.0)), D = riverDepth(hw), x = d/hw, bank = 1.0 + 0.004*hw;
  vec3 v = civDelta(p, c.riverQ); v -= p*dot(v, p);
  vec3 gd = v/max(length(v), 1e-12)*PLANET_R;            // gradient of d
  vec2 pb = sstepd(hw, hw + plain, d);
  float lo = L + 0.4*pb.x, hi = L + bank*pb.x;
  float f = d < hw ? L - D*(1.0 - x*x) : clamp(t.x, lo, hi);
  vec3 gf = d < hw ? gd*(2.0*D*x/hw) : t.x > hi ? gd*(bank*pb.y) : t.x < lo ? gd*(0.4*pb.y) : t.yzw;
  float dh = t.x - f;
  vec2 s = sstepd(hw + plain, hw + plain + 15.0 + min(abs(dh), 200.0)*1.5, d);
  vec4 r = vec4(f + dh*s.x, gf*(1.0 - s.x) + t.yzw*s.x + dh*s.y*gd);
  // test before the river claims the water: a river entering a lake keeps no floodplain (no levee strips) and the lake's level
  bool lakeWet = c.waterKind == 1 && t.x < c.water - 0.5;
  if (d < hw + plain && L > c.water && !lakeWet){ c.water = L; c.waterKind = 2; }
  return (t.x < 0.0 || lakeWet) && r.x > t.x ? t : r;
}

// level a road or rail bed: constant height across the width (hc = smoothed terrain at the centreline);
// over lakes the bed rises to a causeway above the water. Approaching a river it crosses (spec.js BRIDGE) the bed
// descends at grade towards bridgeClear() over the floodplain (not in towns), then gives way (open 0..1) to the
// carved ground over the channel and, out of towns, wherever that lies > BRIDGE_E0 below it: bridges.js spans those
vec4 civBed(vec4 t, vec4 hc, vec3 p, vec3 q, vec3 tn, float d, float halfW, float grade, Civ c, out float open){
  float bed = max(hc.x, 2.0), gt = dot(hc.yzw, tn);                // bed and its gradient along the road
  vec3 v = civDelta(p, q); v -= p*dot(v, p);
  float hw = c.riverHalf, fp = hw + riverPlain(hw), reach = fp + BRIDGE_REACH, gate = 0.0, town = 0.0;
  vec3 sd = vec3(0.0);
  if (c.dRiver < reach){
    // descent: dq = the centreline's distance from the channel centre; the ramp reaches the bed BRIDGE_RAMP m beyond the floodplain
    sd = normalize(cross(p, c.riverTan));
    float sn = dot(tn, sd), Gq = dot(civDelta(p, c.riverQ) - v, sd)*PLANET_R, dq = abs(Gq), up = grade*sign(Gq)*sn*PLANET_R;
    gate = smoothstep(BRIDGE_SIN, BRIDGE_SIN + 0.05, abs(sn));
    town = 1.0 - smoothstep(0.0, BRIDGE_TOWN, c.flatW);
    float ramp = c.riverLevel + bridgeClear(hw) + grade*max(0.0, dq - fp), lim = bed - grade*max(0.0, fp + BRIDGE_RAMP - max(dq, fp));   // level over the floodplain
    float low = min(bed, max(ramp, lim)), gLow = low == bed ? gt : (low == ramp ? 0.0 : gt) + up*step(fp, dq);
    bed = mix(bed, low, gate*town); gt = mix(gt, gLow, gate*town);
  }
  if (c.lakeM > 0.0) bed = mix(bed, max(bed, c.lakeLevel + 1.5), smoothstep(0.05, 0.3, c.lakeM));
  vec3 gb = tn*gt;
  float dh = t.x - bed;
  vec2 s = sstepd(halfW + 1.5, halfW + 4.0 + abs(dh)*1.5, d);   // 0 on the bed .. 1 on natural ground
  vec3 gd = v/max(length(v), 1e-12)*PLANET_R;
  vec4 r = vec4(bed + dh*s.x, gb + (t.yzw - gb)*s.x + dh*s.y*gd);
  open = 0.0;
  if (c.dRiver >= reach) return r;
  // ground and river distance at the centreline point level with p (first order on the road, so bridge ends are square)
  float k = 1.0 - smoothstep(halfW, halfW + 10.0, d);
  float dq = abs(dot(civDelta(p, c.riverQ) - v*k, sd))*PLANET_R, e = bed - (t.x - dot(t.yzw, v)*k);
  open = gate*(1.0 - smoothstep(reach - 100.0, reach, c.dRiver))*max(1.0 - smoothstep(hw, hw + 6.0, dq), smoothstep(BRIDGE_E0, BRIDGE_E0 + 1.5, e)*town);
  return mix(r, t, open);
}

#define CIV_FLAT_F (PLANET_R/300.0)
// terrain with civilisation applied; c is valid only when maxF >= CIV_FLAT_F.
// All terrainBase() evaluations go through one call site in a loop: D3D inlines every call site,
// and several copies of the height field made shader compiles take many seconds.
vec4 terrainCiv(vec3 p, float maxF, out vec3 info, out Civ c){
  c = civNone(p);
#ifdef NO_CIV
  bool civ = false;
#else
  bool civ = maxF >= CIV_FLAT_F;
#endif
  int walk = civ ? 2 : civFarMode;
  civFarC = c;
  if (walk > 0){ Civ w = civWalk(p, walk == 2); if (civ) c = w; else civFarC = w; }
  float cityF = min(maxF, PLANET_R/2500.0);
  civLake(c, p);
  // requests: 0 natural (or city-smoothed) ground, 1 city-smoothed ground, 2 road bed, 3 rail bed
  vec3 P[4]; float F[4]; bool use[4];
  P[0] = p; F[0] = c.flatW >= 0.999 ? cityF : maxF; use[0] = true;
  P[1] = p; F[1] = cityF; use[1] = civ && c.flatW > 0.0 && c.flatW < 0.999;
  P[2] = normalize(c.roadQ); F[2] = min(maxF, mix(PLANET_R/1500.0, PLANET_R/2500.0, c.flatW)); use[2] = civ && c.dRoad < c.halfW + 90.0;
  P[3] = normalize(c.railQ); F[3] = min(maxF, mix(PLANET_R/4000.0, PLANET_R/2500.0, c.flatW)); use[3] = civ && c.dRail < 95.0;
  vec4 R4[4];
  int last = use[3] ? 4 : use[2] ? 3 : use[1] ? 2 : 1;
  for (int k = 0; k < last; k++){
    vec3 i2;
    R4[k] = use[k] ? terrainBase(P[k], F[k], i2) : vec4(0.0);
    if (k == 0){ info = i2; lfBase = lfLast; }
  }
  vec4 t = R4[0];
  if (use[1]) t = mix(t, R4[1], c.flatW);
  // lakes: water wherever the natural ground is below the level inside the lake cells (islands and a ragged
  // shore come from the fine relief), and a rim at the edge of the cells so water never shows outside its basin
  if (c.lakeM > 0.0){
    // the rim follows the lake cells bent by the relief (low ground pulls it in), so where it shapes the shore
    // the shore isn't a smooth grid curve (and no extra noise call: every call site costs D3D compile time).
    // The sine of the ground height bends it along the terrain's contours (every 25 m), and two sine wobbles (no noise
    // call) where the ground outside lies flat below the level: no ruler-straight or cell-stepped shores
    float mN = c.lakeM + clamp((t.x - c.lakeLevel)*0.002, -0.15, 0.15) + 0.06*sin((t.x - c.lakeLevel)*0.25)
             + 0.1*riverWobble(p, 2000.0) + 0.05*riverWobble(p, 350.0);
    float rim = c.lakeLevel + 40.0*(0.45 - mN);
    if (mN < 0.45 && t.x < rim){ float wr = smoothstep(0.0, 0.2, mN); t = vec4(mix(t.x, rim, wr), t.yzw*(1.0 - wr)); }
    if (mN > 0.3){ c.water = c.lakeLevel; c.waterKind = 1; } else c.lakeLevel = -1e9;   // lakeLevel: only where lake water may be
  }
  if (civ && c.dRiver < c.riverHalf + riverPlain(c.riverHalf) + 360.0) t = civRiverBed(t, p, c);   // dRiver already includes the meander
  // road and rail beds after rivers: causeways where roads run along rivers, open under bridges
  // (no road / rail paint under bridges; trees stay cleared)
  float open;
  if (use[2]){ t = civBed(t, R4[2], p, c.roadQ, c.roadTan, c.dRoad, c.halfW, BRIDGE_GRADE[clamp(c.roadType, 0, 2)], c, open); if (open > 0.5) c.dRoad = max(c.dRoad, c.halfW + 3.0); }
  if (use[3]){ t = civBed(t, R4[3], p, c.railQ, c.railTan, c.dRail, CIV_HALF[CIV_RAIL], BRIDGE_GRADE[CIV_RAIL], c, open); if (open > 0.5) c.dRail = max(c.dRail, 6.0); }
  return t;
}
vec4 terrain(vec3 p, float maxF, out vec3 info){ Civ c; return terrainCiv(p, maxF, info, c); }
`
