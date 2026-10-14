// Farmland painting: field patchwork around settlements. farmMapSample() must run in uniform control
// flow (it uses derivatives); farmPaint() is called on land before the town/road painting.
// Layout (mirrored in farms.js): Voronoi districts over a lat/lon grid of ~4.5 km cells, each with its own
// orientation, parcel size and style (0 fields, 1 centre pivots, 2 rice paddies). Parcels sit in staggered
// rows and split into strips = fields; every field gets a crop whose colour follows the season.
// FARM_VERT/FARM_FRAG draw the farmstead buildings (gabled boxes and silos) placed by farms.js.
import { OBJ_COMMON, OBJ_LIGHT } from './objects.glsl.js'

export const FARM_CELL = 4500           // district grid cell (m)
export const FARM_RANGE = 6000          // farmsteads within this distance; they dither out from 0.6 to 0.9 of it
export const FARM_REG0 = 0.56           // farming-country threshold on the regional noise (sets the global farmed share)
export const FARM_PAINT = /* glsl */`
// screen derivatives of the map uv; the terrain's deferred pass (terrain.js) supplies them from G-buffer neighbours
#ifndef UV_DX
#define UV_DX(uv) dFdx(uv)
#define UV_DY(uv) dFdy(uv)
#endif
uniform sampler2D uFarmMap;     // r = regional settlement density (0..1), ~40 km texels, mipmapped
uniform sampler2D uFarmTerr;    // r = mountain mask, g = roughness/1.5 (farms.js, from the sampler; ~40 km texels)
uniform float uFarmDay, uFarmOn; // day of the year, 1 = paint fields (farms.js)
uniform vec3 uFarmA, uFarmCamA; // fp32 anchor direction at the camera, (camera direction - anchor) from doubles
#define FARM_CELL ${FARM_CELL.toFixed(1)}
#define FARM_REG0 ${FARM_REG0.toFixed(3)}

// 1 on dry ground, 0 under a lake (h below its level + 0.3..1.2 m) and within 'bank' m of a river channel's edge.
// Shared with the forests (forest.glsl.js). c from terrainCiv: the river part only works close up (civAt ran).
float waterClear(Civ c, float h, float bank){
  return smoothstep(0.3, 1.2, h - c.lakeLevel)*smoothstep(0.4*bank, bank, c.dRiver - c.riverHalf);
}
// fields and farmyards keep off a grass / reed strip along rivers
float farmBank(Civ c){ return 15.0 + 0.3*c.riverHalf; }

vec2 farmTerr;                  // (mountain mask, roughness) at p, set by farmMapSample()
float farmMapSample(vec3 p){
  vec2 uv = vec2(atan(p.x, p.z)*(0.5/PI) + 0.5, asin(clamp(p.y, -1.0, 1.0))/PI + 0.5);
  vec2 dx = UV_DX(uv), dy = UV_DY(uv);
  dx.x -= floor(dx.x + 0.5); dy.x -= floor(dy.x + 0.5);
  farmTerr = textureGrad(uFarmTerr, uv, dx, dy).rg*vec2(1.0, 1.5);
  return textureGrad(uFarmMap, uv, dx, dy).r;
}

// Compile time: fxc (ANGLE's D3D11 backend) inlines every call, so the repeated hashes and noises below go through
// one call site in a small loop, which fxc keeps rolled. (A uniform-dependent bound made these loops slower to
// compile, not faster: fxc does not unroll them anyway. notes/compile-farm.md.) No loop sits inside another one:
// fxc compiles nested loops much more slowly (notes/compile-farm3.md).
// one-hot vec4 (component j is 1 where i == base + j): gathers loop results into vectors exactly
vec4 farmSlot(int i, int base){ return vec4(equal(ivec4(i), ivec4(base, base + 1, base + 2, base + 3))); }
int farmPick(ivec4 v, int i){ return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
// hash2i(x[i], y[i], s[i]) for i < n (the rest 0)
vec4 farmHash4(ivec4 x, ivec4 y, ivec4 s, int n){
  vec4 r = vec4(0.0);
  for (int i = 0; i < n; i++) r += hash2i(farmPick(x, i), farmPick(y, i), farmPick(s, i))*farmSlot(i, 0);
  return r;
}

// smooth value noise on the integer lattice from hash2i (mirrored in farms.js): 2D for local field frames,
// 3D for positions on the sphere (z folds into the seed); the lattice corners share one hash2i.
// farmVNMix2/3 blend the corner hashes hv (2D) or h0, h1 (3D: the layers seed + c*1013 and seed + (c + 1)*1013).
float farmVNMix2(vec2 x, vec4 hv){
  vec2 u = fract(x);
  u = u*u*(3.0 - 2.0*u);
  return mix(mix(hv.x, hv.y, u.x), mix(hv.z, hv.w, u.x), u.y);
}
float farmVNMix3(vec3 x, vec4 h0, vec4 h1){
  vec2 u = fract(x.xy);
  u = u*u*(3.0 - 2.0*u);
  float w = fract(x.z);
  return mix(mix(mix(h0.x, h0.y, u.x), mix(h0.z, h0.w, u.x), u.y), mix(mix(h1.x, h1.y, u.x), mix(h1.z, h1.w, u.x), u.y), w*w*(3.0 - 2.0*w));
}
// corner k of 8 of the 3D lattice cell at x
float farmVNCorner(vec3 x, int seed, int k){
  return hash2i(int(floor(x.x)) + (k & 1), int(floor(x.y)) + ((k >> 1) & 1), seed + int(floor(x.z))*1013 + (k >> 2)*1013);
}
float farmVN(vec2 x, int seed){
  int a = int(floor(x.x)), b = int(floor(x.y));
  vec4 hv = vec4(0.0);
  for (int k = 0; k < 4; k++) hv += hash2i(a + (k & 1), b + (k >> 1), seed)*farmSlot(k, 0);
  return farmVNMix2(x, hv);
}
float farmVN(vec3 x, int seed){
  vec4 h0 = vec4(0.0), h1 = vec4(0.0);
  for (int k = 0; k < 8; k++){
    float hk = farmVNCorner(x, seed, k);
    h0 += hk*farmSlot(k, 0); h1 += hk*farmSlot(k, 4);
  }
  return farmVNMix3(x, h0, h1);
}
// four 3D noises (n of them, the rest 0): one flat loop over their 8n corners (fxc compiles a loop nested in a
// loop much more slowly)
vec4 farmVN4(vec3 x0, vec3 x1, vec3 x2, vec3 x3, ivec4 seed, int n){
  vec4 r = vec4(0.0), h0 = vec4(0.0), h1 = vec4(0.0);
  for (int k = 0; k < 8*n; k++){
    int i = k >> 3, j = k & 7;
    vec3 x = i == 0 ? x0 : i == 1 ? x1 : i == 2 ? x2 : x3;
    float hk = farmVNCorner(x, farmPick(seed, i), j);
    h0 += hk*farmSlot(j, 0); h1 += hk*farmSlot(j, 4);
    if (j == 7){ r += farmVNMix3(x, h0, h1)*farmSlot(i, 0); h0 = vec4(0.0); h1 = vec4(0.0); }
  }
  return r;
}
// metres on the sphere: the frame for every regional noise below. (A lat/lon chart shears by lon*sin(lat) per
// metre north, which smeared every pattern into diagonal streaks away from the prime meridian.)
vec3 farmLL(vec3 p){ return p*PLANET_R; }
vec3 farmRot(vec3 x){ return vec3(0.8*x.y + 0.6*x.z, -0.8*x.x + 0.36*x.y - 0.48*x.z, -0.6*x.x - 0.48*x.y + 0.64*x.z); }   // so octaves don't share lattice planes

// share of the land farmed around here (0..1); farmLocal() then picks which fields. Mirrored in farms.js cover().
// Terrain comes from the regional mountain mask and roughness (terr) and the height: the per-pixel slope grows
// with detail (the hills are fractal), so it only rules out cliffs.
float farmCover(vec3 ll, float farmD, vec2 terr, float T, float M, float slope, float h){
  // mountain belts, hill country, cliffs, high, cold, swampy and hot-wet ground stay wild
  float hill = terr.y - 1.1*terr.x;
  float ok = smoothstep(0.3, 0.08, terr.x)*(1.0 - 0.7*smoothstep(0.35, 0.55, hill))*smoothstep(0.4, 0.25, slope)
           *smoothstep(1500.0, 600.0, h)*smoothstep(2.0, 7.0, T)
           *(1.0 - 0.8*smoothstep(0.72, 0.88, M)*smoothstep(60.0, 10.0, h))*(1.0 - 0.68*smoothstep(0.7, 0.85, M)*smoothstep(20.0, 24.0, T));
  if (ok*farmD <= 0.0) return 0.0;
  // farming regions and wild tracts between them: domain-warped 25-250 km noise; the edges retreat from poor
  // ground (ok) and settled country tips the balance a little (farmD is regional, so no town sits in the middle)
  vec3 x = ll/150e3;
  x += 1.6*farmVN4(x*0.35, x*0.35 + 5.2, x*0.35 + 9.7, x, ivec4(41, 42, 47, 0), 3).xyz - 0.8;
  vec4 r = farmVN4(x, farmRot(x)*2.7, x*6.3 + 3.3, farmRot(x)*1.4 + 9.1, ivec4(43, 44, 46, 45), 4);
  float reg = 0.5*r.x + 0.3*r.y + 0.2*r.z;
  // meandering valleys are farmed even in dry or thinly settled country
  float val = 1.0 - smoothstep(0.012, 0.035, abs(r.w - 0.5));
  float dry = max(smoothstep(0.1, 0.28, M), val*smoothstep(0.02, 0.07, M));
  // suitability moves the region edges and cuts off sharply on poor ground (a thin spread there reads as specks)
  float suit = ok*dry;
  float lvl = reg + 0.12*smoothstep(0.05, 0.7, farmD) + 0.3*(suit - 0.6) + 0.2*val;
  return 0.53*smoothstep(0.25, 0.55, suit)*smoothstep(0.0, 0.12, farmD)*smoothstep(FARM_REG0, FARM_REG0 + 0.08, lvl);
}
// field-cluster noise (2 and 6.5 km: woods, commons, field blocks); a field is farmed when
// 0.2*hash + 0.8*nb < cover. Octaves below the pixel size (aa, m) flatten; .y = the spread they leave behind.
// n = (farmVN(ll/2000.0, 17), farmVN(farmRot(ll)/6500.0, 18)) at the point ll (callers batch them with farmVN4)
vec2 farmLocal(vec2 n, float aa){
  float f1 = smoothstep(0.5, 1.5, 2000.0/aa), f2 = smoothstep(0.5, 1.5, 6500.0/aa);
  float s = 0.55*f1*(n.x - 0.5) + 0.45*f2*(n.y - 0.5);
  return vec2(clamp(0.5 + 1.6*s, 0.0, 1.0), 0.9*(0.55*(1.0 - f1) + 0.45*(1.0 - f2)));
}
// the farm-belt edge follows the field-cluster noise at the pixel, not whole fields (a staircase): fields pass a
// looser test at their centre (FARM_LOOSE), then woods and heath cut curved bites out of those on the rim
// (n: farmLocal's noises at the pixel)
#define FARM_LOOSE 0.08
float farmTrim(vec2 n, float hf0, float prob){ return smoothstep(-0.004, 0.004, prob - hf0 - 0.8*farmLocal(n, 1e-3).x); }
// expected farmed share of the fields here (the far view of the per-field test)
float farmShare(float cover, vec2 nb){ return clamp((cover - 0.8*nb.x + 0.4*nb.y)/(0.2 + 0.8*nb.y), 0.0, 1.0); }

// p - D, precise near the camera: (camDir - A) + (A - D) + civRel has no float cancellation
vec3 farmDelta(vec3 p, vec3 D){ return civHasRel ? uFarmCamA + (uFarmA - D) + civRel : p - D; }
// cos of the view zenith angle at p (ground height h): the view ray is civRel*R + p*h - camDir*(camera altitude)
float farmGraze(vec3 p, float h){ vec3 v = civRel*PLANET_R + p*h - uCamDir*(uCamR - PLANET_R); return civHasRel ? max(-dot(normalize(v), p), 0.08) : 1.0; }
vec3 farmDir(vec2 ll){ return vec3(cos(ll.x)*sin(ll.y), sin(ll.x), cos(ll.x)*cos(ll.y)); }
// district centre (lat, lon) of grid cell (i, j) in a row of n cells; r = (hash2i(i, j, 71), hash2i(i, j, 72))
vec2 farmCentre(int i, int j, float n, vec2 r){
  return vec2((float(j) + 0.15 + 0.7*r.y)*(FARM_CELL/PLANET_R), ((float(i) + 0.15 + 0.7*r.x)/n - 0.5)*2.0*PI);
}
float farmRowCells(int j){ return max(floor(2.0*PI*cos((float(j) + 0.5)*(FARM_CELL/PLANET_R))*PLANET_R/FARM_CELL), 1.0); }

// nearest district: centre D, cell id, distance to the district boundary (m)
struct FarmDistrict { vec3 D; ivec2 id; float edge; };
FarmDistrict farmDistrict(vec3 p){
  float lat = asin(clamp(p.y, -1.0, 1.0)), lon = atan(p.x, p.z), cl = cos(lat);
  float fj = lat/(FARM_CELL/PLANET_R);
  float b1 = 1e30, b2 = 1e30;
  ivec3 c1 = ivec3(0), c2 = ivec3(0);
  vec2 r1 = vec2(0.0), r2 = vec2(0.0);      // their centre hashes (71, 72)
  // approximate metric distances pick the two nearest centres (3 x 3 cells, rows outer)
  for (int t = 0; t < 9; t++){
    int j = int(floor(fj)) + t/3 - 1;
    float n = farmRowCells(j), fx = (lon/(2.0*PI) + 0.5)*n;
    int i = int(floor(fx)) + t%3 - 1, iw = int(mod(float(i), n));
    vec2 r = vec2(hash2i(iw, j, 71), hash2i(iw, j, 72));   // (not farmHash4: no loop in this loop)
    vec2 dv = vec2((float(i) + 0.15 + 0.7*r.x - fx)*2.0*PI*cl/n, float(j) + 0.15 + 0.7*r.y - fj);
    dv.y *= FARM_CELL/PLANET_R;
    float d = dot(dv, dv);
    if (d < b1){ b2 = b1; c2 = c1; r2 = r1; b1 = d; c1 = ivec3(iw, j, int(n)); r1 = r; }
    else if (d < b2){ b2 = d; c2 = ivec3(iw, j, int(n)); r2 = r; }
  }
  // exact distances for those two
  vec3 D1 = farmDir(farmCentre(c1.x, c1.y, float(c1.z), r1)), D2 = farmDir(farmCentre(c2.x, c2.y, float(c2.z), r2));
  vec3 v1 = farmDelta(p, D1), v2 = farmDelta(p, D2);
  float d1 = length(v1 - D1*dot(v1, D1))*PLANET_R, d2 = length(v2 - D2*dot(v2, D2))*PLANET_R;
  FarmDistrict o;
  if (d2 < d1){ o.D = D2; o.id = c2.xy; } else { o.D = D1; o.id = c1.xy; }
  o.edge = abs(d2*d2 - d1*d1)/(2.0*length(D2 - D1)*PLANET_R);
  return o;
}

// box-filtered periodic pulse: coverage of pulses of width w (period 1) over a footprint fw (period units)
float farmBoxI(float x, float w){ return floor(x)*w + clamp(fract(x) - 0.5 + 0.5*w, 0.0, w); }
float farmPulse(float x, float w, float fw){
  if (fw > 1.5) return w;
  fw = max(fw, 1e-4);
  return (farmBoxI(x + 0.5*fw, w) - farmBoxI(x - 0.5*fw, w))/fw;
}

// texture noise (value v = farmVN(x, 5)) with a cell of cellPx pixels, flattened to its mean 0.5 once the cell is
// below the pixel size
float farmTex(float v, float cellPx){ return cellPx < 0.5 ? 0.5 : mix(0.5, v, smoothstep(0.5, 1.5, cellPx)); }
// four 2D ones (n of them, the rest 0), farmVN(x, 5) in one flat loop over their 4n corners; the noise only runs
// where it shows
vec4 farmTex4(vec2 x0, vec2 x1, vec2 x2, vec2 x3, vec4 cellPx, int n){
  vec4 r = vec4(0.0), hv = vec4(0.0);
  for (int k = 0; k < 4*n; k++){
    int i = k >> 2, j = k & 3;
    float cp = i == 0 ? cellPx.x : i == 1 ? cellPx.y : i == 2 ? cellPx.z : cellPx.w;
    vec2 x = i == 0 ? x0 : i == 1 ? x1 : i == 2 ? x2 : x3;
    if (cp >= 0.5) hv += hash2i(int(floor(x.x)) + (j & 1), int(floor(x.y)) + (j >> 1), 5)*farmSlot(j, 0);
    if (j == 3){ r += (cp >= 0.5 ? farmTex(farmVNMix2(x, hv), cp) : 0.5)*farmSlot(i, 0); hv = vec4(0.0); }
  }
  return r;
}

// coverage of the region sd < 0 (sd = signed distance, m) for a pixel of size aa
float farmIn(float sd, float aa){ return clamp(0.5 - sd/aa, 0.0, 1.0); }

// crops: 0 cereal, 1 maize, 2 rapeseed/sunflower, 3 grass/hay, 4 roots/vegetables, 5 fallow, 6 orchard/vineyard
#define FARM_CROPS 7
// crop shares by climate, shifted by a 16-22 km crop-mix noise (arable vs grassland country):
// n = (farmVN(ll/22e3, 56), farmVN(farmRot(ll)/16e3, 57))
void farmWeights(float T, float M, vec2 n, out float w[FARM_CROPS]){
  float warm = smoothstep(14.0, 20.0, T), wet = smoothstep(0.45, 0.7, M), dry = smoothstep(0.35, 0.18, M);
  float a = n.x, b = n.y;
  w[0] = (0.34 + 0.1*dry)*(0.4 + 1.2*a); w[1] = (0.16 + 0.06*warm)*(0.5 + b); w[2] = 0.12*(1.5 - b);
  w[3] = (0.14 + 0.3*wet - 0.08*dry)*(1.6 - 1.2*a); w[4] = 0.1; w[5] = 0.07 + 0.12*dry; w[6] = 0.14*warm + 0.6*warm*wet*smoothstep(21.0, 25.0, T);
}
int farmCrop(float r, float w[FARM_CROPS]){
  float tot = 0.0;
  for (int k = 0; k < FARM_CROPS; k++) tot += w[k];
  r *= tot;
  for (int k = 0; k < FARM_CROPS - 1; k++){ if (r < w[k]) return k; r -= w[k]; }
  return FARM_CROPS - 1;
}

// mean field colour of a crop at season t (0..1 from January in the north); soil = bare earth colour
// .w = how much the rows show (0 closed canopy .. 1 bare drills)
vec4 farmCropColour(int crop, float t, vec3 soil, float warm){
  vec3 young = vec3(0.075, 0.12, 0.045), green = vec3(0.055, 0.13, 0.035), gold = vec3(0.36, 0.28, 0.12);
  vec3 stubble = vec3(0.30, 0.26, 0.16), ploughed = soil*0.75;
  vec3 c = soil; float rows = 1.0;
  if (crop == 0){                       // winter cereal: sown Oct, green all spring, gold in July, stubble, ploughed
    float s = fract(t - 0.78 + 0.05*warm);
    c = mix(c, young, smoothstep(0.03, 0.14, s)*0.75);
    c = mix(c, green, smoothstep(0.33, 0.52, s));
    c = mix(c, gold, smoothstep(0.64, 0.74, s));
    c = mix(c, stubble, smoothstep(0.79, 0.81, s));
    c = mix(c, ploughed, smoothstep(0.9, 0.93, s));
    rows = 1.0 - smoothstep(0.1, 0.45, s)*0.8 + smoothstep(0.79, 0.81, s)*0.6;
  } else if (crop == 1){                // maize: sown May, rows close by July, dry tan in autumn
    float s = fract(t - 0.33 + 0.04*warm);
    c = mix(c, young, smoothstep(0.08, 0.2, s)*0.6);
    c = mix(c, vec3(0.04, 0.095, 0.028), smoothstep(0.17, 0.32, s));
    c = mix(c, vec3(0.24, 0.20, 0.11), smoothstep(0.4, 0.5, s));
    c = mix(c, vec3(0.17, 0.13, 0.085), smoothstep(0.52, 0.54, s));
    c = mix(c, ploughed, smoothstep(0.65, 0.7, s));
    rows = 1.0 - 0.85*smoothstep(0.15, 0.32, s)*(1.0 - smoothstep(0.52, 0.54, s));
  } else if (crop == 2){                // rapeseed (sunflower where warm): yellow in bloom
    float s = fract(t - 0.65 - 0.25*warm);
    c = mix(c, green*0.9, smoothstep(0.04, 0.3, s));
    c = mix(c, mix(vec3(0.50, 0.42, 0.04), vec3(0.30, 0.24, 0.04), warm), smoothstep(0.58, 0.63, s)*(1.0 - smoothstep(0.7, 0.75, s)));
    c = mix(c, vec3(0.13, 0.12, 0.055), smoothstep(0.72, 0.78, s));
    c = mix(c, stubble*0.7, smoothstep(0.85, 0.87, s));
    c = mix(c, ploughed, smoothstep(0.93, 0.96, s));
    rows = 1.0 - 0.9*smoothstep(0.05, 0.3, s)*(1.0 - smoothstep(0.85, 0.87, s));
  } else if (crop == 3){                // pasture and hay: greener in spring, parched in warm summers
    float grow = 0.5 + 0.5*sin((t - 0.2)*2.0*PI);
    c = mix(vec3(0.07, 0.095, 0.045), vec3(0.065, 0.14, 0.04), grow);
    c = mix(c, vec3(0.20, 0.18, 0.09), warm*smoothstep(0.45, 0.6, t)*(1.0 - smoothstep(0.75, 0.85, t)));
    rows = 0.0;
  } else if (crop == 4){                // beet, potatoes, vegetables: green rows May to October
    float s = fract(t - 0.28);
    c = mix(c, young, smoothstep(0.1, 0.22, s)*0.7);
    c = mix(c, vec3(0.06, 0.14, 0.04), smoothstep(0.2, 0.34, s));
    c = mix(c, ploughed, smoothstep(0.55, 0.58, s));
    rows = 1.0 - 0.7*smoothstep(0.2, 0.34, s)*(1.0 - smoothstep(0.55, 0.58, s));
  } else if (crop == 5){                // fallow: weeds over dry soil
    c = mix(soil*1.1, vec3(0.11, 0.12, 0.06), 0.5 + 0.3*sin((t - 0.2)*2.0*PI));
    rows = 0.2;
  } else {                              // orchard/vineyard: grass alleys, dark trees, bare in winter
    float leaf = smoothstep(0.25, 0.35, t)*(1.0 - smoothstep(0.8, 0.9, t));
    c = mix(vec3(0.10, 0.10, 0.06), vec3(0.06, 0.11, 0.04), leaf);
    rows = 0.0;
  }
  return vec4(c, rows);
}

// rice paddy at stage s (0..1): flooded, planted, green, ripe, cut; water is faked with a sky tint
vec3 farmRice(float s, float graze){
  vec3 col = mix(vec3(0.10, 0.11, 0.12), vec3(0.06, 0.063, 0.048), smoothstep(0.05, 0.5, graze));
  col = mix(col, vec3(0.06, 0.12, 0.04), smoothstep(0.12, 0.3, s));
  col = mix(col, vec3(0.07, 0.18, 0.035), smoothstep(0.3, 0.45, s));
  col = mix(col, vec3(0.25, 0.23, 0.08), smoothstep(0.7, 0.8, s));
  return mix(col, vec3(0.16, 0.13, 0.08), smoothstep(0.88, 0.9, s));
}

float farmRiceStage(float n, float season){ return fract(season*2.0 + 1.6*n); }   // n = farmVN(ll/20e3, 58)

// district layout styles: 1 centre pivots (dry warm), 2 rice paddies (hot wet lowland), else 0 fields; a district is
// style 1 when hS < farmPivotP, else 2 when hS < farmRiceP, with hS = 0.75*farmVN(centre/30 km, 51) + 0.25*hash
float farmPivotP(float T, float M){ return 0.8*smoothstep(0.35, 0.15, M)*smoothstep(10.0, 16.0, T); }
float farmRiceP(float T, float M, float h){ return smoothstep(0.55, 0.7, M)*smoothstep(20.0, 24.0, T)*smoothstep(400.0, 150.0, h); }
vec3 farmSoil(float n, float M){ return mix(vec3(0.13, 0.095, 0.065), vec3(0.24, 0.19, 0.13), smoothstep(0.35, 0.15, M))*(0.85 + 0.3*n); }   // n = farmVN(ll/12e3, 55)

// the pixel's noises used near and far: farmLocal's two, soil (55), district style share (51)
vec4 farmNoiseP(vec3 ll){ return farmVN4(ll/2000.0, farmRot(ll)/6500.0, ll/12e3, ll/30e3, ivec4(17, 18, 55, 51), 4); }
// the district's: style (51), orientation (54), parcel size (52, 53)
vec4 farmNoiseD(vec3 llD){ return farmVN4(llD/30e3, llD/40e3, llD/60e3, farmRot(llD)/45e3, ivec4(51, 54, 52, 53), 4); }

// far away: the average of many fields, the crop mix mottled at 0.5-6 km (no district shows); .w = farmed share
// (nP = farmNoiseP(ll), soil = farmSoil(nP.z, M))
vec4 farmFar(vec3 ll, vec4 nP, vec3 soil, float cover, float aa, float T, float M, float h, float season, float graze){
  // district styles as shares of the ground
  float sv = 0.75*nP.w;
  float piv = clamp((farmPivotP(T, M) - sv)/0.25, 0.0, 1.0), rice = max(clamp((farmRiceP(T, M, h) - sv)/0.25, 0.0, 1.0) - piv, 0.0);
  vec4 nw = farmVN4(ll/22e3, farmRot(ll)/16e3, ll/20e3, ll, ivec4(56, 57, 58, 0), rice > 0.0 ? 3 : 2);
  float warm = smoothstep(14.0, 20.0, T), wts[FARM_CROPS];
  vec3 mean = vec3(0.0), c0 = vec3(0.0), c3 = vec3(0.0);
  farmWeights(T, M, nw.xy, wts);
  float tot = 0.0;
  for (int i = 0; i < FARM_CROPS; i++){
    vec3 ci = farmCropColour(i, season, soil, warm).rgb;
    mean += ci*wts[i]; tot += wts[i];
    if (i == 0) c0 = ci; else if (i == 3) c3 = ci;
  }
  mean *= 0.9/tot;
  if (rice > 0.0){
    mean = mix(mean, farmRice(farmRiceStage(nw.z, season), graze), rice);
  }
  // the mottling octaves coarse to fine, as many as show
  vec4 nt = farmVN4(ll/6000.0 + 31.0, farmRot(ll)/1700.0 + 17.0, ll/500.0, ll, ivec4(5),
                    int(6000.0/aa >= 0.5) + int(1700.0/aa >= 0.5) + int(500.0/aa >= 0.5));
  float mott = (farmTex(nt.z, 500.0/aa) - 0.5) + 0.8*(farmTex(nt.y, 1700.0/aa) - 0.5)
             + 0.6*(farmTex(nt.x, 6000.0/aa) - 0.5);
  mean = max(mean*(1.0 + 0.35*mott) + (c0 - c3)*0.4*mott, 0.0);
  return vec4(mean, farmShare(cover, farmLocal(nP.xy, aa))*(1.0 - 0.25*piv));
}

// for the 3D trees (trees.js): x = 1 on a farmed field or farmyard, y = 1 on a hedgerow; farmPaint's layout, no colour
vec2 farmFieldWeight(vec3 p, float farmD, float T, float M, float slope, float h, Civ c){
  float pix = (1.0 - c.flatW)*smoothstep(c.halfW + 1.5, c.halfW + 4.0, c.dRoad)*smoothstep(2.5, 5.0, c.dRail)*waterClear(c, h, farmBank(c));
  vec3 ll = farmLL(p);
  float prob = farmCover(ll, farmD, farmTerr, T, M, slope, h);
  if (pix*prob <= 0.0 || uFarmOn < 0.5) return vec2(0.0);
  FarmDistrict fd = farmDistrict(p);
  int ds = fd.id.x*1013 + fd.id.y*7919;
  vec3 llD = farmLL(fd.D);
  vec4 nD = farmNoiseD(llD), hD = farmHash4(ivec4(ds), ivec4(1, 2, 3, 4), ivec4(74, 73, 75, 76), 4);
  float hS = 0.75*nD.x + 0.25*hD.x, hA = hD.y;
  int style = hS < farmPivotP(T, M) ? 1 : hS < farmRiceP(T, M, h) ? 2 : 0;
  float ang = style == 1 ? (hA - 0.5)*0.06 : PI*(1.6*nD.y + 0.25*hA);
  vec3 e = normalize(vec3(fd.D.z, 0.0, -fd.D.x)), nn = cross(fd.D, e);
  vec3 ex = e*cos(ang) + nn*sin(ang), ny = -e*sin(ang) + nn*cos(ang), dv = farmDelta(p, fd.D);
  vec2 q = vec2(dot(dv, ex), dot(dv, ny))*PLANET_R;
  float h1 = clamp(0.5 + 1.5*(0.7*nD.z + 0.3*hD.z - 0.5), 0.0, 1.0);
  float h2 = clamp(0.5 + 1.5*(0.6*nD.w + 0.4*hD.w - 0.5), 0.0, 1.0);
  vec2 S = style == 1 ? vec2(h1 < 0.7 ? 805.0 : 402.0) : style == 2 ? vec2(50.0 + 70.0*h1, 30.0 + 40.0*h2) : vec2(260.0 + 560.0*h1);
  if (style == 0) S.y *= 0.35 + 0.55*h2;
  int ry = int(floor(q.y/S.y));
  float xs = q.x + (style == 1 ? 0.0 : hash2i(ry, ds, 78)*S.x);
  int rx = int(floor(xs/S.x));
  vec2 f = vec2(xs - float(rx)*S.x, q.y - float(ry)*S.y);
  int k = style == 1 ? 1 : 1 + int(hash2i(rx, ry, ds + 7)*(style == 2 ? 2.0 : 4.0));
  float w = S.x/float(k);
  int strip = min(int(f.x/w), k - 1);
  vec2 g = vec2(f.x - float(strip)*w, f.y), fs = vec2(w, S.y), fc = q - g + 0.5*fs;
  vec3 llC = farmLL(normalize(fd.D + (ex*fc.x + ny*fc.y)/PLANET_R));
  float h5 = 0.2*hash2i(rx*8 + strip, ry, ds + 5);
  vec4 nL = farmVN4(llC/2000.0, farmRot(llC)/6500.0, ll/2000.0, farmRot(ll)/6500.0, ivec4(17, 18, 17, 18), 4);   // farmLocal's at llC, ll
  float hf = h5 + 0.8*farmLocal(nL.xy, 1e-3).x, farmed = step(hf, prob + FARM_LOOSE), trim = step(0.5, farmTrim(nL.zw, h5, prob));
  float inside = style == 1 ? step(length(f - 0.5*S), 0.49*S.x) : 1.0;
  if (style != 2 && hash2i(rx, ry, ds + 14) < min(0.35, S.x*S.y/6e5)){
    vec2 yh = 0.5*vec2(45.0 + 30.0*hash2i(rx, ry, ds + 15), 35.0 + 25.0*hash2i(rx, ry, ds + 16)), yd = abs(f - 10.0 - yh) - yh;
    float yard = step(max(yd.x, yd.y), 0.0);
    inside = max(inside, yard); trim = max(trim, yard*step(hf, prob));   // farms.js builds on strictly farmed fields
  }
  farmed *= trim;
  float hedge = 0.0;
  if (style == 0){
    float hedgeP = 0.55*smoothstep(0.35, 0.55, M)*smoothstep(3.0, 8.0, T)*(1.0 - smoothstep(14.0, 20.0, T));
    int exi = g.x < fs.x*0.5 ? rx*8 + strip : strip == k - 1 ? (rx + 1)*8 : rx*8 + strip + 1;
    int eyi = g.y < fs.y*0.5 ? ry : ry + 1;
    float edgeX = min(g.x, fs.x - g.x), edgeY = min(g.y, fs.y - g.y);
    hedge = max(step(edgeX, 3.0)*step(hash2i(exi, ry, ds + 12), hedgeP), step(edgeY, 3.0)*step(hash2i(eyi, int(floor(q.x/700.0)), ds + 13), hedgeP));
  }
  return pix*farmed*vec2(inside, hedge);
}

vec3 farmPaint(vec3 alb, vec3 p, float foot, float farmD, float T, float M, float slope, float h, Civ c){
  // off town ground, road beds, lakes and river banks; farmCover() keeps steep, high, cold, dry, wet and unsettled land wild
  float pix = (1.0 - c.flatW)*smoothstep(c.halfW + 1.5, c.halfW + 4.0, c.dRoad)*smoothstep(2.5, 5.0, c.dRail)*waterClear(c, h, farmBank(c));
  if (pix <= 0.0 || uFarmOn < 0.5) return alb;
  vec3 ll = farmLL(p);
  float prob = farmCover(ll, farmD, farmTerr, T, M, slope, h);
  if (prob <= 0.0) return alb;
  float aa = max(foot, 0.01);
  float graze = farmGraze(p, h);
  float season = fract(uFarmDay/365.0 + (p.y < 0.0 ? 0.5 : 0.0));
  vec4 nP = farmNoiseP(ll);
  vec3 soil = farmSoil(nP.z, M);
  // every field is under half a pixel (the largest are ~800 m): skip the layout (wFar is 1 there anyway)
  vec3 col = vec3(0.0);
  float wt = 0.0, wFar = 1.0;
  if (aa <= 410.0){
    float aaS = aa/graze;                           // footprint along the view, for fine stripes
    float warm = smoothstep(14.0, 20.0, T);

    // district frame and style; style, orientation and parcel size drift smoothly from district to district
    FarmDistrict fd = farmDistrict(p);
    int ds = fd.id.x*1013 + fd.id.y*7919;
    vec3 llD = farmLL(fd.D);
    vec4 nD = farmNoiseD(llD), hD = farmHash4(ivec4(ds), ivec4(1, 2, 3, 4), ivec4(74, 73, 75, 76), 4);
    float hS = 0.75*nD.x + 0.25*hD.x, hA = hD.y;
    int style = hS < farmPivotP(T, M) ? 1 : hS < farmRiceP(T, M, h) ? 2 : 0;
    float ang = style == 1 ? (hA - 0.5)*0.06 : PI*(1.6*nD.y + 0.25*hA);
    vec3 e = normalize(vec3(fd.D.z, 0.0, -fd.D.x)), nn = cross(fd.D, e);
    vec3 ex = e*cos(ang) + nn*sin(ang), ny = -e*sin(ang) + nn*cos(ang);
    vec3 dv = farmDelta(p, fd.D);
    vec2 q = vec2(dot(dv, ex), dot(dv, ny))*PLANET_R;

    // parcels in staggered rows, split into 1-4 strips (fields)
    float h1 = clamp(0.5 + 1.5*(0.7*nD.z + 0.3*hD.z - 0.5), 0.0, 1.0);
    float h2 = clamp(0.5 + 1.5*(0.6*nD.w + 0.4*hD.w - 0.5), 0.0, 1.0);
    vec2 S = style == 1 ? vec2(h1 < 0.7 ? 805.0 : 402.0) : style == 2 ? vec2(50.0 + 70.0*h1, 30.0 + 40.0*h2) : vec2(260.0 + 560.0*h1);
    if (style == 0) S.y *= 0.35 + 0.55*h2;
    int ry = int(floor(q.y/S.y));
    float xs = q.x + (style == 1 ? 0.0 : hash2i(ry, ds, 78)*S.x);
    int rx = int(floor(xs/S.x));
    vec2 f = vec2(xs - float(rx)*S.x, q.y - float(ry)*S.y);        // metres within the parcel
    int k = style == 1 ? 1 : 1 + int(hash2i(rx, ry, ds + 7)*(style == 2 ? 2.0 : 4.0));
    float w = S.x/float(k);
    int strip = min(int(f.x/w), k - 1);
    vec2 g = vec2(f.x - float(strip)*w, f.y);                        // metres within the field
    vec2 fs = vec2(w, S.y);
    // the field's hashes: 5 wild, 6 crop, 8 sowing date, 9 tone; 10 row direction, 11 vines, 12/13 hedges
    // (edge ids exi/eyi: the field edge nearest along x/y); 14-16 farmyard
    int fid = rx*8 + strip;
    int exi = g.x < fs.x*0.5 ? fid : strip == k - 1 ? (rx + 1)*8 : fid + 1;
    int eyi = g.y < fs.y*0.5 ? ry : ry + 1;
    vec4 hF = farmHash4(ivec4(fid), ivec4(ry), ds + ivec4(5, 6, 8, 9), 4);
    vec4 hG = farmHash4(ivec4(fid, fid, exi, eyi), ivec4(ry, ry, ry, int(floor(q.x/700.0))), ds + ivec4(10, 11, 12, 13), 4);
    vec4 hY = farmHash4(ivec4(rx), ivec4(ry), ds + ivec4(14, 15, 16, 0), 3);
    float hc = hF.y;
    // farmed or left wild, per field: the field-cluster noise at the field centre (lat/lon, so it runs across districts)
    vec2 fc = q - g + 0.5*fs;
    vec3 llC = farmLL(normalize(fd.D + (ex*fc.x + ny*fc.y)/PLANET_R));
    vec4 nC = farmVN4(llC/2000.0, farmRot(llC)/6500.0, llC/22e3, farmRot(llC)/16e3, ivec4(17, 18, 56, 57), 4);   // farmLocal's, farmWeights'
    float h5 = 0.2*hF.x, hf = h5 + 0.8*farmLocal(nC.xy, 1e-3).x;
    float farmed = smoothstep(hf - 0.03, hf + 0.03, prob + FARM_LOOSE), trim = farmTrim(nP.xy, h5, prob);

    // crop and its colour this season (dates shift a little field by field)
    float wts[FARM_CROPS];
    farmWeights(T, M, nC.zw, wts);
    int crop = farmCrop(hc, wts);
    vec4 cc = farmCropColour(crop, fract(season + (hF.z - 0.5)*0.07), soil, warm);
    col = cc.rgb*(0.9 + 0.2*hF.w);
    float edgeX = min(g.x, fs.x - g.x), edgeY = min(g.y, fs.y - g.y);

    if (style == 2){                         // rice: flooded, planted, green, ripe, cut; twice a year
      float s = fract(farmRiceStage(farmVN(llC/20e3, 58), season) + 0.05*(hc - 0.5));
      col = farmRice(s, graze);
      cc.w = 1.0 - smoothstep(0.1, 0.35, s);
      col = mix(col, vec3(0.20, 0.17, 0.11), civBand(min(edgeX, edgeY), 0.5, aa));   // bunds
    }

    // rows along one axis (turned round on the headlands), drills and tramlines
    bool alongX = hG.x < 0.75;
    bool head = (alongX ? edgeX : edgeY) < 11.0;
    float u = (alongX != head) ? g.y : g.x;
    float P = crop == 0 ? 0.17 : crop == 1 ? 0.76 : crop == 4 ? 0.5 : 0.3;
    float rowsN = farmPulse(u/P, 0.5, aaS/P);
    col *= 1.0 + (rowsN - 0.5)*0.7*cc.w*(style == 1 ? 0.3 : 1.0);
    if (crop == 0 || crop == 2 || crop == 4){
      float v = u/24.0;
      float tram = farmPulse(v - 0.9/24.0, 0.45/24.0, aaS/24.0) + farmPulse(v + 0.9/24.0, 0.45/24.0, aaS/24.0);
      col = mix(col, soil*0.9, tram*(1.0 - cc.w)*0.8);
    }
    if (crop == 3) col *= 1.0 + 0.12*(farmPulse(u/12.0, 0.5, aaS/12.0) - 0.5);       // mowing stripes
    if (crop == 6){                                                                     // tree rows / vines
      bool vine = hG.y < 0.5;
      float cov = vine ? farmPulse(u/2.6, 0.35, aaS/2.6) : farmPulse(u/6.0, 0.55, aaS/6.0)*farmPulse((alongX ? g.x : g.y)/5.0, 0.6, aaS/5.0);
      col = mix(col, vec3(0.03, 0.065, 0.025), cov*0.9);
    }
    // texture noises: soil moisture blotches, clods, leaves, hedge texture; farmyard ground, hedge widths
    vec2 qm = mod(q, 700.0);
    vec4 tA = farmTex4(q/70.0 + float(ds & 255), qm/0.4, qm/0.07, q/2.5, vec4(70.0/aa, 0.4/aa, 0.07/aa, 2.5/aa), 4);
    vec4 tB = farmTex4(q/1.3, vec2(q.y/9.0, float(exi)), vec2(q.x/9.0, float(eyi)), q, vec4(1.3/aa, 9.0/aa, 9.0/aa, 0.0), style == 0 ? 3 : 1);
    col *= 0.9 + 0.2*tA.x;
    col *= 0.8 + 0.4*tA.y;
    col *= 0.8 + 0.4*tA.z;

    // centre pivots: the crop circle with wheel-track rings; corners stay dry ground
    float inside = 1.0;
    if (style == 1){
      float r = length(f - 0.5*S);
      col *= 1.0 - 0.3*farmPulse(r/48.0, 0.8/48.0, aaS/48.0);
      inside = farmIn(r - 0.49*S.x, aa);
    }

    // field edges: hedgerows in temperate country, farm tracks, grass margins; lanes on district boundaries
    float hedgeP = 0.55*smoothstep(0.35, 0.55, M)*smoothstep(3.0, 8.0, T)*(1.0 - warm);
    float hx = hG.z, hy = hG.w;
    vec3 hedge = vec3(0.03, 0.06, 0.022)*(0.7 + 0.6*tA.w);
    vec3 margin = vec3(0.075, 0.11, 0.045), track = vec3(0.22, 0.19, 0.14);
    float noEdge = style == 1 ? 0.0 : 1.0;
    col = mix(col, margin, civBand(edgeX, 1.3, aa)*noEdge);
    col = mix(col, margin, civBand(edgeY, 1.3, aa)*noEdge);
    if (style == 0){
      float hwX = 2.0 + 2.5*tB.y, hwY = 2.0 + 2.5*tB.z;
      col = mix(col, hedge, civBand(edgeX, hwX, aa)*step(hx, hedgeP));
      col = mix(col, hedge, civBand(edgeY, hwY, aa)*step(hy, hedgeP));
      col = mix(col, track, civBand(edgeY, 1.8, aa)*step(1.0 - hy, 0.12));
    }
    col = mix(col, track, civBand(fd.edge, 2.5, aa));

    // farmyards in some parcel corners (farms.js puts the buildings there)
    if (style != 2 && hY.x < min(0.35, S.x*S.y/6e5)){
      vec2 yh = 0.5*vec2(45.0 + 30.0*hY.y, 35.0 + 25.0*hY.z);
      vec2 yd = abs(f - 10.0 - yh) - yh;
      float yc = farmIn(max(yd.x, yd.y), aa)*step(hf, prob);             // farms.js builds on strictly farmed fields
      col = mix(col, vec3(0.19, 0.18, 0.16)*(0.85 + 0.3*tB.x), yc);
      inside = max(inside, yc); trim = max(trim, yc);
    }

    // each field fades to the far average once it is under a couple of pixels (small fields first)
    wt = farmed*inside*trim; wFar = smoothstep(0.12, 0.5, aa/min(fs.x, fs.y));
  }
  if (wFar > 0.0){
    vec4 fr = farmFar(ll, nP, soil, prob, aa, T, M, h, season, graze);
    col = mix(col, fr.rgb, wFar);
    wt = mix(wt, fr.w, wFar);
  }
  return mix(alb, col, pix*wt);
}
`

// ---- farmstead buildings: kinds 0 house, 1 barn, 2 machine shed (gabled boxes), 3 silo, 4 grain bin ----
export const FARM_VERT = /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
uniform mat4 projectionMatrix, viewMatrix;
uniform vec3 uAnchor;          // instance anchor minus camera (m)
uniform int uShape;            // 0 gabled box, 1 cylinder with a dome
in vec3 position, normal;      // box: x, y in [-0.5, 0.5], z 0 ground, 1 eaves, 2 ridge; silo: unit circle, z 0, 1, 1..2 dome
in float around;               // silo: 0..1 round the wall
in vec3 iPos, iUp, iEx;        // base centre relative to the anchor (m), local vertical, long axis
in vec4 iSize;                 // box: length, depth, wall, roof rise; silo: radius, dome/radius, wall, 0
in vec4 iMisc;                 // kind, seed, foundation depth, load time (s)
out vec3 vRel, vN;
out vec4 vFace;                // u along the face (m), height (m), face width (m), 1 roof / -1 gable end / 0 side
flat out vec3 vUp, vSunC, vAmbC;
flat out vec4 vSize, vMisc;
void main(){
  vec3 U = iUp, E = normalize(iEx - U*dot(iEx, U)), Nn = cross(U, E);
  float z = position.z;
  vec3 lp, nl;
  if (uShape == 0){
    float zm = z < 0.5 ? -iMisc.z : min(z, 1.0)*iSize.z + max(z - 1.0, 0.0)*iSize.w;
    lp = vec3(position.x*iSize.x, position.y*iSize.y, zm);
    nl = normal.z > 0.5 ? normalize(vec3(0.0, normal.y*iSize.w, iSize.y*0.5)) : normal;
    bool end = abs(normal.x) > 0.5;
    vFace = normal.z > 0.5 ? vec4(position.x*iSize.x, position.y*iSize.y, iSize.x, 1.0)
          : vec4(end ? (position.y + 0.5)*iSize.y : (position.x + 0.5)*iSize.x, zm, end ? iSize.y : iSize.x, end ? -1.0 : 0.0);
  } else {
    float r = iSize.x, dome = iSize.y;
    float zm = z < 0.01 ? -iMisc.z : z <= 1.0 ? z*iSize.z : iSize.z + (z - 1.0)*r*dome;
    lp = vec3(position.xy*r, zm);
    nl = normalize(vec3(normal.xy, normal.z/max(dome, 0.05)));
    vFace = vec4(around*2.0*PI*r, zm, 2.0*PI*r, z > 1.001 ? 1.0 : 0.0);
  }
  vec3 rel = uAnchor + iPos + E*lp.x + Nn*lp.y + U*lp.z;
  vN = E*nl.x + Nn*nl.y + U*nl.z;
  vRel = rel;
  vUp = U;
  vSunC = objSun(U, 200.0);
  vAmbC = objAmbient(U);
  vSize = iSize; vMisc = iMisc;
  gl_Position = projectionMatrix*viewMatrix*vec4(rel, 1.0);
}`

export const FARM_FRAG = /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
uniform float uPixAngle;
in vec3 vRel, vN;
in vec4 vFace;
flat in vec3 vUp, vSunC, vAmbC;
flat in vec4 vSize, vMisc;
out vec4 outColor;

float boxI(float x, float w){ return floor(x)*w + clamp(fract(x) - 0.5 + 0.5*w, 0.0, w); }
float pulse(float x, float w, float fw){ if (fw > 1.5) return w; fw = max(fw, 1e-4); return (boxI(x + 0.5*fw, w) - boxI(x - 0.5*fw, w))/fw; }
float rect(vec2 q, vec2 lo, vec2 hi, float aa){ vec2 a = clamp((q - lo)/aa + 0.5, 0.0, 1.0)*clamp((hi - q)/aa + 0.5, 0.0, 1.0); return a.x*a.y; }

void main(){
  // dither in over the last third of the range and for 0.8 s after the farmstead loads (vMisc.w = load time)
  float fade = smoothstep(${(0.9 * FARM_RANGE).toFixed(1)}, ${(0.6 * FARM_RANGE).toFixed(1)}, length(vRel))*clamp((uTime - vMisc.w)/0.8, 0.0, 1.0);
  if (hash12(gl_FragCoord.xy) >= fade) discard;
  int kind = int(vMisc.x + 0.5);
  float r1 = hash12(vec2(vMisc.y, 1.7)), r2 = hash12(vec2(vMisc.y, 5.3)), r3 = hash12(vec2(vMisc.y, 9.1));
  vec3 N = normalize(vN), V = -normalize(vRel);
  float aa = max(length(vRel)*uPixAngle, 0.01);
  bool roof = vFace.w > 0.5, end = vFace.w < -0.5;
  vec2 q = vFace.xy;
  float wall = vSize.z;
  vec3 alb; float metal = 0.0, rough = 0.6;
  vec3 emit = vec3(0.0);
  float night = objNight(vUp);
  if (kind == 0){                                      // farmhouse: plaster, brick or boards; tile or slate roof
    alb = r1 < 0.55 ? vec3(0.56, 0.53, 0.47)*(0.85 + 0.3*r2) : r1 < 0.8 ? vec3(0.28, 0.13, 0.08) : vec3(0.45, 0.42, 0.36);
    if (r1 >= 0.8) alb *= 0.9 + 0.2*pulse(q.y/0.22, 0.8, aa/0.22);
    if (roof) alb = r2 < 0.5 ? vec3(0.22, 0.08, 0.05) : vec3(0.08, 0.085, 0.095);
    else {
      float win = 0.0;
      for (int f = 0; f < 2; f++){
        float z0 = 1.0 + 2.7*float(f);
        if (z0 + 1.2 < wall) win += rect(vec2(mod(q.x, 3.2), q.y), vec2(1.0, z0), vec2(2.2, z0 + 1.2), aa);
      }
      win *= step(1.2, q.x)*step(q.x, vFace.z - 1.2);
      alb = mix(alb, vec3(0.03, 0.035, 0.04), win);
      emit += vec3(1.0, 0.72, 0.42)*0.02*win*step(0.4, r3)*night;
    }
  } else if (kind == 1){                               // barn: red, weathered or grey boards; metal roof; big doors
    alb = r1 < 0.45 ? vec3(0.32, 0.06, 0.04) : r1 < 0.75 ? vec3(0.20, 0.17, 0.13) : vec3(0.30, 0.30, 0.29);
    alb *= 0.85 + 0.3*pulse(q.x/0.3, 0.85, aa/0.3);
    if (roof){ alb = r2 < 0.6 ? vec3(0.22, 0.23, 0.24) : vec3(0.18, 0.09, 0.05); metal = 0.5; }
    else if (end){
      alb = mix(alb, vec3(0.05, 0.045, 0.04), rect(q, vec2(0.5*vFace.z - 2.4, -1.0), vec2(0.5*vFace.z + 2.4, 4.4), aa));
      vec2 l = q - vec2(0.5*vFace.z, 5.0);          // yard lamp over the door
      float s = max(0.2, aa*0.7);
      emit += vec3(1.0, 0.75, 0.45)*night*(0.05/(s*s))*exp(-dot(l, l)/(2.0*s*s));
    }
  } else if (kind == 2){                               // machine shed: corrugated steel, open front
    alb = r1 < 0.5 ? vec3(0.40, 0.41, 0.42) : r1 < 0.75 ? vec3(0.10, 0.17, 0.12) : vec3(0.40, 0.34, 0.25);
    alb *= 0.9 + 0.2*pulse(q.x/0.2, 0.5, aa/0.2);
    metal = 0.35;
    if (roof) alb = vec3(0.30, 0.31, 0.32);
    else if (!end) alb = mix(alb, vec3(0.04), rect(q, vec2(2.0, -1.0), vec2(vFace.z - 2.0, wall - 0.8), aa)*step(0.5, r2));
  } else if (kind == 3){                               // silo: concrete staves or blue glass-lined steel
    alb = r1 < 0.65 ? vec3(0.46, 0.45, 0.42)*(0.92 + 0.16*pulse(q.y/0.76, 0.1, aa/0.76)) : vec3(0.04, 0.07, 0.13);
    if (roof) alb = r1 < 0.65 ? vec3(0.35, 0.36, 0.37) : alb;
    metal = r1 < 0.65 ? 0.1 : 0.4; rough = 0.45;
  } else {                                             // grain bin: galvanised, ribbed
    alb = vec3(0.42, 0.43, 0.44)*(0.9 + 0.2*pulse(q.y/0.15, 0.5, aa/0.15));
    metal = 0.6; rough = 0.4;
  }
  vec3 col = alb*(vSunC*max(dot(N, uSun), 0.0) + vAmbC*(0.6 + 0.4*dot(N, vUp)));
  col += vSunC*objSpec(N, V, rough)*metal;
  outColor = vec4(col + emit, 1.0);
}`
