// Land surface colour by biome (climate, height, slope, rock/snow/beach) and landform (dune seas, mesas,
// badlands, volcanoes, wetlands, salt flats). Called from the terrain fragment shader before forest, farm
// and town painting. mat = (rock, snow, sand/dry) weights for close-up detail. Landforms, the hilliness noise and the
// volcano distance come from terrainCiv's evaluation at p (lfBase), so call it after terrainCiv(p, ...).
export const BIOME = /* glsl */`
// horizontal rock layers: 0..1 band value from height; faded out where the bands would be finer than a pixel
float strataBand(float h, float slope, float spacing, float maxF){
  float fade = bandw(PLANET_R*max(slope, 0.02)/spacing, maxF);
  return 0.5 + 0.5*fade*sin(h*6.2832/spacing + 1.7*sin(h*6.2832/(2.7*spacing)));
}

// coverage of a line |d| < w with a pixel-wide edge aa; lines thinner than a pixel fade to their mean coverage
float washLine(float d, float w, float aa){ return smoothstep(w + aa, max(w - aa, 0.0), d)*min(1.0, w/aa); }

vec3 landAlbedo(float h, float T, float M, float slope, vec3 info, vec3 p, float maxF, out vec3 mat){
  Landform L = lfBase.L;
  float n8 = lfBase.n8, vx = lfBase.vx;
  // One snoise() call site (D3D inlines each): job 0 the slow tint drift (hue), then the fine colour variation that
  // appears as you get closer (stops before float precision runs out). The raw octaves o (cells ~7 km, 3 km, 1.3 km,
  // 580, 250, 110, 48, 21 m) and their band weights ob also draw the desert.
  float hue = 0.0, v = 0.0, amp = 0.5, f = 900.0;
  float o[8] = float[8](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), ob[8] = float[8](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (int j = 0; j < loopN(9); j++){
    float bw = j == 0 ? 1.0 : bandw(f, maxF);
    if (bw <= 0.0) break;
    float n = snoise(j == 0 ? p*23.0 + vec3(4.0, 1.0, 7.0) : p*f + 17.0);
    if (j == 0){ hue = n; continue; }
    o[j - 1] = n; ob[j - 1] = bw;
    v += n*amp*bw;
    f *= 2.3; amp *= 0.8;
  }

  // dry end: gravel plains (red-brown hamada, pale reg), orange dune sand, grey cold desert
  vec3 reg = mix(vec3(0.36, 0.28, 0.19), vec3(0.42, 0.25, 0.14), smoothstep(0.1, 0.6, hue));
  reg = mix(reg, vec3(0.48, 0.42, 0.32), smoothstep(-0.2, -0.7, hue));
  vec3 sand = mix(vec3(0.50, 0.35, 0.19), vec3(0.54, 0.32, 0.15), smoothstep(-0.2, 0.5, hue));
  vec3 dry = mix(vec3(0.29, 0.27, 0.23), reg, smoothstep(9.0, 17.0, T));
  dry = mix(dry, sand, smoothstep(0.1, 0.5, L.erg));
  // semi-arid: cold steppe, mediterranean scrub, savanna
  vec3 semi = mix(vec3(0.20, 0.18, 0.12), vec3(0.14, 0.13, 0.08), smoothstep(8.0, 13.0, T));
  semi = mix(semi, vec3(0.24, 0.19, 0.10), smoothstep(19.0, 23.0, T));
  // grassland (prairie -> lush), forest floors (the forest pass paints canopy on top), rainforest, taiga, tundra
  vec3 grass = mix(vec3(0.13, 0.14, 0.065), vec3(0.085, 0.125, 0.05), smoothstep(0.36, 0.5, M));
  vec3 forest = vec3(0.04, 0.075, 0.03), jungle = vec3(0.025, 0.06, 0.02), taiga = vec3(0.03, 0.05, 0.034);
  vec3 tundra = mix(vec3(0.13, 0.12, 0.09), vec3(0.10, 0.11, 0.08), smoothstep(-0.3, 0.4, hue));
  vec3 snow = vec3(0.80, 0.83, 0.88);

  vec3 c = mix(dry, semi, smoothstep(0.08, 0.24, M));
  c = mix(c, grass, smoothstep(0.28, 0.44, M));
  c = mix(c, forest, smoothstep(0.48, 0.66, M));
  c = mix(c, jungle, smoothstep(0.66, 0.82, M)*smoothstep(18.0, 23.0, T));
  c = mix(c, taiga, smoothstep(6.0, 1.0, T)*smoothstep(0.3, 0.5, M));
  c = mix(c, tundra, smoothstep(-2.0, -6.0, T));
  // alpine meadows above the tree line, below the snow
  float alp = smoothstep(0.3, 0.6, info.y)*smoothstep(1400.0, 2200.0, h)*smoothstep(-7.0, -3.0, T)*(1.0 - smoothstep(2.0, 6.0, T));
  c = mix(c, vec3(0.10, 0.12, 0.055), alp*smoothstep(0.15, 0.3, M));
  // slow tint drift so wide biomes are not flat colour
  c *= 1.0 + 0.1*hue;

  // desert ground off the dune seas: dark varnished gravel pavement and paler sand sheets, rock outcrops, dry
  // washes with pale beds and shrubby banks. Each fades in with its octave, so it resolves instead of shimmering.
  float des = (1.0 - smoothstep(0.12, 0.3, M))*(1.0 - L.erg), gravel = 0.0, outcrop = 0.0;
  if (des > 0.0){
    float pv = smoothstep(-0.3, 0.35, 0.7*o[3] + 0.6*o[4] + 0.35*o[5] + 0.2*o[6]);
    c = mix(c, mix(c*vec3(1.08, 1.05, 1.0) + vec3(0.025, 0.02, 0.012), c*vec3(0.78, 0.69, 0.63), pv), des*ob[3]);
    gravel = 0.4*pv*des*ob[5];
    outcrop = smoothstep(0.5, 0.6, 0.7*o[3] + 0.6*o[4] + 0.35*o[5] + 0.15*o[7])*des*ob[4];
    c = mix(c, mix(vec3(0.13, 0.095, 0.075), vec3(0.34, 0.29, 0.23), smoothstep(-0.3, 0.6, o[6] + hue)), outcrop);
    float footm = PLANET_R/(2.0*maxF);
    for (int i = 1; i <= 3; i += 2){
      // washes on the zero lines of octaves 1 and 3 (meandering with the next one), coming and going with the
      // coarser one; w = half width in noise units (~50 m and ~12 m), aa = one pixel in noise units
      float d = abs(o[i] + 0.35*o[i + 1]), w = i == 1 ? 0.04 : 0.05, aa = 2.5*footm*900.0*pow(2.3, float(i))/PLANET_R;
      float on = des*ob[i]*smoothstep(-0.35, 0.05, o[i - 1]);
      float bed = washLine(d, w, aa), bank = max(washLine(d, 2.6*w, aa) - bed, 0.0);
      c = mix(c, c*0.5 + vec3(0.25, 0.22, 0.17), bed*on*0.8);
      c = mix(c, vec3(0.09, 0.085, 0.05), bank*on*mix(0.3, smoothstep(0.0, 0.5, o[7]), ob[7])*mix(0.3, 0.9, smoothstep(0.04, 0.2, M)));
      outcrop *= 1.0 - bed*on;
      gravel *= 1.0 - bed*on;
    }
  }

  // wetlands on wet coastal flats: dark sedge with open pools
  float wet = L.plain*smoothstep(0.45, 0.6, M)*(1.0 - smoothstep(4.0, 18.0, h))*smoothstep(1.0, 6.0, T);
  c = mix(c, mix(vec3(0.06, 0.075, 0.035), vec3(0.025, 0.035, 0.03), smoothstep(0.0, 0.4, v)), wet);
  // salt flats in the driest flat basins
  float salt = (1.0 - smoothstep(0.03, 0.09, M))*(1.0 - smoothstep(0.008, 0.03, slope))*smoothstep(0.35, 0.6, -n8)*smoothstep(0.3, 0.55, -hue)*(1.0 - L.erg);
  c = mix(c, vec3(0.70, 0.68, 0.64)*(1.0 + 0.1*v), salt);
  // mesa country: red and cream sandstone layers (strong on canyon walls)
  vec3 redRock = mix(vec3(0.40, 0.17, 0.09), vec3(0.45, 0.28, 0.17), strataBand(h, slope, 70.0, maxF));
  redRock = mix(redRock, vec3(0.26, 0.18, 0.15), 0.4*smoothstep(0.7, 1.0, strataBand(h + 23.0, slope, 190.0, maxF)));
  c = mix(c, redRock, L.mesa*mix(0.45, 1.0, smoothstep(0.05, 0.25, slope))*(1.0 - smoothstep(0.35, 0.6, M)));
  // badlands: bare banded clay, grey, buff and rust
  vec3 clay = mix(vec3(0.33, 0.30, 0.26), vec3(0.42, 0.26, 0.16), strataBand(h, slope, 18.0, maxF));
  c = mix(c, clay, L.bad*smoothstep(0.08, 0.3, slope + 0.1));
  // volcanoes: dark basalt and ash on the cone, greening on the lower flanks where it is wet
  float volc = (1.0 - smoothstep(0.35, 1.0, vx))*(1.0 - 0.6*smoothstep(0.45, 0.75, M)*smoothstep(0.4, 0.9, vx));
  c = mix(c, mix(vec3(0.075, 0.068, 0.062), vec3(0.15, 0.135, 0.12), smoothstep(-0.3, 0.5, v + hue)), volc);

  c *= 1.0 + 0.35*v*(1.0 - 0.6*L.erg);
  // bare rock on steep ground and high peaks, with strata on the cliffs
  vec3 rock = mix(vec3(0.15, 0.14, 0.13), vec3(0.20, 0.15, 0.11), smoothstep(-0.4, 0.4, hue));
  rock = mix(rock, vec3(0.09, 0.085, 0.078), volc);
  rock = mix(rock, redRock, L.mesa);
  float strata = 0.85 + 0.3*sin(h*0.02 + 4.0*v)*bandw(PLANET_R/40.0, maxF);
  // bare rock on steep ground; high in the ranges also on the ridges and moderate slopes, not on the flats
  float rockAmt = clamp(smoothstep(0.18, 0.42, slope + 0.08*v) + smoothstep(1800.0, 3200.0, h)*smoothstep(0.3, 0.8, info.y)*mix(0.2, 0.8, smoothstep(0.06, 0.22, slope)), 0.0, 1.0);
  c = mix(c, rock*(1.0 + 0.3*v)*mix(1.0, strata, smoothstep(0.3, 0.6, slope)), rockAmt);
  // snow: seasonal cover from today's temperature (thinner where it is dry), perennial snow and glacier ice
  // (bluish where it flows steeply) where even the warmest month stays below freezing
  vec2 sn = seasonT(p, info.x);
  float Tnow = T + sn.x, Tsum = T + sn.y;
  float snowAmt = max(smoothstep(-1.5, -6.0, Tnow + 3.0*v)*mix(0.45, 1.0, smoothstep(0.06, 0.3, M)), smoothstep(-1.5, -5.0, Tsum + 4.0*v));
  snowAmt *= 1.0 - smoothstep(0.35, 0.6, slope);
  c = mix(c, mix(snow, vec3(0.60, 0.70, 0.80), smoothstep(0.12, 0.3, slope)*smoothstep(-2.0, -8.0, Tsum)), snowAmt);
  float beach = (1.0 - smoothstep(1.5, 4.0, h))*smoothstep(-2.0, 4.0, T)*(1.0 - smoothstep(0.1, 0.3, slope))*(1.0 - snowAmt)*(1.0 - wet);
  c = mix(c, mix(vec3(0.55, 0.47, 0.34), vec3(0.09, 0.085, 0.08), volc), beach*0.9);
  mat = vec3(max(max(rockAmt, outcrop), gravel), snowAmt,max(max(beach, 1.0 - smoothstep(0.1, 0.35, M)), max(L.erg, salt)));
  return c;
}
`
