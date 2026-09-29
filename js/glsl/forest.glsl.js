// Forests: canopy painting on land (called after landAlbedo, before farm and town painting, so fields and towns
// override it). forestDensity() is shared with the 3D trees (js/civ/trees.js) so both agree. Notes: notes/forests.md.
export const FOREST_CELL = 8            // crown lattice cell of the painted canopy (m)
export const FOREST = /* glsl */`
uniform vec3 uForestOff;                // camDir*R/FOREST_CELL mod 1024, from doubles (trees.js): precise crown lattice
uniform float uTreeR0, uTreeMax;        // 3D trees: all drawn within r0, thinned as (r0/d)^2 out to max (0 = none)
#define FOREST_CELL ${FOREST_CELL.toFixed(1)}

// climate: x = share of the land under stands, y = trees per lattice cell inside a stand,
// z = conifer share, w = tropical share. Moisture need rises with heat; the treeline follows T (lapse rate) and height.
vec4 forestClimate(float T, float M, float h){
  float need = 0.30 + 0.12*smoothstep(0.0, 26.0, T);
  float wet = smoothstep(need, need + 0.2, M);                              // closed forest
  float dry = smoothstep(need - 0.17, need - 0.03, M)*(1.0 - wet);          // open woodland: savanna, mediterranean
  float cold = smoothstep(-4.5, -0.5, T)*smoothstep(4200.0, 3200.0, h);
  float open = dry/(wet + dry + 1e-4);
  float share = sqrt(cold)*(0.86*wet + 0.7*dry);
  float dens = mix(0.95, 0.2, open)*(0.3 + 0.7*cold);
  float conifer = clamp(smoothstep(11.0, 3.0, T) + 0.4*open*smoothstep(12.0, 17.0, T)*smoothstep(25.0, 21.0, T), 0.0, 1.0);
  return vec4(share, dens, conifer, smoothstep(18.0, 23.0, T));
}
// loop count the compiler can't fold (D3D inlines and unrolls constant loops; uForestOff.x >= 0)
int forestN(int n){ return n + int(min(uForestOff.x, 0.0)); }
// no trees on rock, beaches or under water
float forestGround(float slope, float h){ return smoothstep(0.42, 0.24, slope)*smoothstep(1.0, 4.0, h); }
// clearings for towns, and cleared verges along roads and railways (c from terrainCiv; empty far away)
float forestClear(Civ c){ return (1.0 - c.flatW)*smoothstep(c.halfW + 5.0, c.halfW + 16.0, c.dRoad)*smoothstep(12.0, 28.0, c.dRail); }
// trunks stand at least this far (m) from a river channel's edge (crowns may overhang); lakes: waterClear (farm.glsl.js)
float forestBank(Civ c){ return 4.0 + 0.03*c.riverHalf; }
// gallery trees lining river banks, also in open country: extra stand share in a strip beyond the bank (aa = pixel m)
float forestRiparian(Civ c, float T, float h, float aa){
  float d = c.dRiver - c.riverHalf, b = forestBank(c);
  return 0.6*civBand(abs(d - b - 9.0), 9.0, aa)*smoothstep(-4.5, -0.5, T)*smoothstep(4200.0, 3200.0, h)*smoothstep(0.3, 1.2, h - c.lakeLevel);
}

// share of the ground under stands for a pixel of aa metres: multi-scale noise (150 km .. 160 m) thresholded so that
// 'share' of it is forest. Octaves under the pixel average out and widen the threshold instead (no aliasing).
// forestStandN = octave count, forestStandX = octave i's noise input, forestStandEnd = the share from the octave sum n
// and the variance vu left under the pixel (forestPaint runs the octaves in its own noise loop)
const float FOREST_AMP[7] = float[7](0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4);   // 150 km .. 160 m: mostly 1-20 km patches
const float FOREST_VT = 0.2832;            // total variance: 0.12*sum(FOREST_AMP^2)
int forestStandN(float aa){ return forestN(clamp(int(ceil(log(75e3/aa)/1.1394)), 0, 7)); }   // octaves above 2 pixels
vec3 forestStandX(vec3 p, int i, float lam){ return p*(PLANET_R/lam) + vec3(float(i)*5.1, 3.7, float(i)*1.3); }
float forestStandEnd(float n, float vu, float share){
  float s = clamp(share, 0.005, 0.995), t = sqrt(FOREST_VT)*0.6*log(s/(1.0 - s));
  return clamp(0.5 + 0.4*(t - n)/sqrt(max(vu, 0.0) + 4e-4), 0.0, 1.0)*step(0.0, share - 0.005);
}
float forestStand(vec3 p, float share, float aa){
  float n = 0.0, vu = FOREST_VT, lam = 150e3;
  int nO = forestStandN(aa);
  for (int i = 0; i < nO; i++){
    float a = FOREST_AMP[i], bw = bandw(1.0/lam, 0.5/aa);
    n += a*bw*snoise(forestStandX(p, i, lam));
    vu -= a*a*0.12*bw;
    lam *= 0.32;
  }
  return forestStandEnd(n, vu, share);
}

// expected trees per lattice cell at p (full detail; the 3D trees apply forestClear and the farm fields themselves)
float forestDensity(vec3 p, float T, float M, float slope, float h, Civ c){
  vec4 cl = forestClimate(T, M, h);
  float rip = forestRiparian(c, T, h, 1.0);
  if (cl.x*cl.y <= 0.0 && rip <= 0.0 || h < 0.0) return 0.0;
  return cl.y*max(forestStand(p, cl.x, 1.0)*forestGround(slope, h)*waterClear(c, h, forestBank(c)), rip);
}

// canopy colour by forest type; autumn tints temperate broadleaf (uFarmDay = day of the year)
// (boreal: dark spruce/pine green, red above blue so the blue haze can't turn it into sea)
vec3 forestColour(vec4 cl, vec3 p, float T){
  vec3 col = mix(mix(vec3(0.034, 0.066, 0.024), vec3(0.026, 0.058, 0.020), cl.w), vec3(0.033, 0.060, 0.020), cl.z);
  col = mix(col, vec3(0.062, 0.064, 0.034), smoothstep(0.7, 0.3, cl.y)*(1.0 - cl.z));      // dry woodland: olive
  float season = fract(uFarmDay/365.0 + (p.y < 0.0 ? 0.5 : 0.0));
  float autumn = smoothstep(0.72, 0.79, season)*(1.0 - smoothstep(0.84, 0.88, season))*(1.0 - cl.z)*(1.0 - cl.w)*smoothstep(17.0, 9.0, T);
  return mix(col, vec3(0.12, 0.07, 0.02), 0.7*autumn);
}

vec3 forestPaint(vec3 alb, vec3 p, float foot, float T, float M, float slope, float h, Civ c){
  vec4 cl = forestClimate(T, M, h);
  float aa = max(foot, 0.01), rip = forestRiparian(c, T, h, aa);
  float clear = forestClear(c);
  if (cl.x*cl.y <= 0.0 && rip <= 0.0 || clear <= 0.0) return alb;
  // where the 3D trees stand the ground is forest floor (share of trees drawn at this distance)
  float d = aa/uPixAngle;
  float keep = uTreeMax > 0.0 ? min(1.0, pow(uTreeR0/d, 2.0))*clamp((uTreeMax - d)/(0.5*uTreeMax), 0.0, 1.0) : 0.0;
  // (precise lattice coordinates near the camera; the pattern jumps once per 8 km where uForestOff wraps)
  vec3 x = civHasRel ? uForestOff + civRel*(PLANET_R/FOREST_CELL) : p*(PLANET_R/FOREST_CELL);
  // one snoisedR() call site (D3D inlines each; 105.0*snoisedR = snoised): first forestStand's nS octaves (stand share
  // -> cover; none of the rest without cover), then jobs 0-2 the stands' tone (species, age), band-limited mottle at
  // 3 km, 700 m, 150 m; then, under the 3D trees, five forest floor noises fn of x (x*FS + FO)
  const float FS[5] = float[5](1.4, 8.0, 20.0, 1.3, 24.0), FO[5] = float[5](11.0, 0.0, 5.0, 7.0, 3.0);
  float sn = 0.0, vu = FOREST_VT, sl = 150e3, cover = 0.0;
  float mt = 0.0, lam = 3000.0, fn[5] = float[5](0.0, 0.0, 0.0, 0.0, 0.0);
  int nS = forestStandN(aa);
  for (int j = 0; j < nS + forestN(keep > 0.0 ? 8 : 3); j++){
    int i = j - nS, k = max(i - 3, 0);
    if (i == 0){
      cover = max(forestStandEnd(sn, vu, cl.x)*forestGround(slope, h)*waterClear(c, h, forestBank(c)), rip)*clear;
      if (cover <= 0.001) break;
    }
    float n = snoisedR(i < 0 ? forestStandX(p, j, sl) : i < 3 ? p*(PLANET_R/lam) + vec3(9.0 + float(i)) : x*FS[k] + FO[k]).x;
    if (i < 0){
      float a = FOREST_AMP[j], bw = bandw(1.0/sl, 0.5/aa);
      sn += a*bw*(105.0*n);
      vu -= a*a*0.12*bw;
      sl *= 0.32;
      continue;
    }
    n = 105.0*n;
    if (i < 3){ mt += bandw(1.0/lam, 0.5/aa)*n; lam *= 0.23; }
    else fn[k] = n;
  }
  // openings in wet boreal forest are bog and heath (sphagnum, sedge): yellow-olive moss, green above red and little
  // blue, so the aerial blue can't turn it mauve
  alb = mix(alb, vec3(0.072, 0.084, 0.030), 0.75*cl.z*smoothstep(0.35, 0.55, M)*smoothstep(0.12, 0.04, slope)*smoothstep(3.0, 15.0, h)*clear);
  if (cover <= 0.001) return alb;
  // a canopy is a volume: its sunlit crowns fall off slower than flat ground as the sun gets low
  float mu = dot(p, uSun), vol = 1.0 + 0.6*(1.0 - clamp(mu, 0.0, 1.0))*smoothstep(-0.05, 0.15, mu);
  // (no view-angle term: a x0.7 grazing darkening made the canopy change brightness as the view tilted)
  vec3 col = forestColour(cl, p, T);
  col *= 1.0 + 0.16*mt;
  // from afar the sunlit crown tops fill the view and sub-pixel clearings, bog and heath lighten it: a brighter olive,
  // so the canopy stays green through the aerial blue instead of turning sea-grey (up close the crowns show it)
  col *= mix(vec3(1.0), vec3(1.6, 1.5, 0.9), smoothstep(5.0, 50.0, aa));
  float fc = cover*cl.y;                                        // canopy fraction
  civCanopy = min(fc*1.15, 1.0);                                // woods stay dark at night (civScatter, paint.glsl.js)
  vec3 res = mix(alb, col*vol, civCanopy);
  // crowns on a 3D jittered lattice sliced by the ground: lit and shaded sides, dark gaps (fade out below ~2 px)
  float wc = civHasRel ? 1.0 - smoothstep(0.25, 0.6, aa/FOREST_CELL) : 0.0;
  if (wc > 0.0){
    vec3 x = uForestOff + civRel*(PLANET_R/FOREST_CELL), b = floor(x - 0.5);
    float best = 1e9; vec3 bd = vec3(0.0);
    for (int k = 0; k < forestN(8); k++){
      vec3 cc = b + vec3(float(k & 1), float((k >> 1) & 1), float(k >> 2)), cm = mod(cc, 1024.0);
      vec3 j = vec3(hash13(cm + 0.5), hash13(cm + 17.3), hash13(cm + 41.9));
      if (j.x*0.999 >= fc) continue;
      vec3 d = x - (cc + 0.2 + 0.6*j);
      float rad = mix(0.42 + 0.2*j.y, 0.3 + 0.12*j.y, cl.z);          // conifers: narrow crowns, more floor between
      float r = dot(d, d)/(rad*rad);
      if (r < best){ best = r; bd = d/rad; }
    }
    vec3 gap = mix(alb, mix(col, vec3(0.05, 0.04, 0.028), 0.5)*0.6, smoothstep(0.3, 0.8, cl.y));   // shaded floor
    vec3 near = gap;
    if (best < 1.0){
      vec3 t = bd - p*dot(bd, p);
      vec3 dn = normalize(t + p*sqrt(1.0 - min(dot(t, t), 1.0)));   // dome normal of the crown
      float lit = max(dot(dn, uSun), 0.0)/max(dot(p, uSun), 0.2);
      near = col*(0.72 + 0.5*clamp(lit, 0.0, 1.5))*(0.85 + 0.3*fract(best*7.0 + dot(bd, vec3(3.0))));
      near = mix(near, gap, smoothstep(0.8, 1.0, best)*0.5);
    }
    res = mix(res, near, wc);
  }
  if (keep > 0.0){
    // litter and moss, mostly in the canopy's shade: small sun flecks up close, ~0.3 on average further out
    float dense = cover*smoothstep(0.3, 0.8, cl.y);
    // (irregular spots of ~1 m and ~0.4 m, clustered under ~6 m canopy gaps; litter grain at ~0.3 m)
    float gapN = smoothstep(-0.2, 0.5, fn[0]), spot = fn[1] + 0.5*bandw(2.5, 0.5/aa)*fn[2];
    float fleck = 0.4 + bandw(1.0, 0.5/aa)*(0.6*gapN*smoothstep(0.5, 0.9, spot) - 0.02);
    vec3 floorC = mix(vec3(0.045, 0.036, 0.024), vec3(0.025, 0.038, 0.017), 0.5 + 0.5*fn[3])*(0.8 + 0.2*cl.w)
                *(1.0 + 0.2*bandw(3.0, 0.5/aa)*fn[4]);
    res = mix(res, mix(alb, floorC*mix(1.0, fleck, dense), dense), keep);
  }
  return res;
}
`
