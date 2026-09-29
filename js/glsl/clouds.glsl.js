// Volumetric cloud layer between CLOUD_BASE and CLOUD_TOP plus a thin cirrus sheet above it. Needs
// uniforms declared by the including shader: uWeather, uNoise, uTransLUT, uSun, uWind, uTime, uPixAngle, uCamAlt, uFlash.
export const CLOUDS = /* glsl */`
const float SIGMA = 0.03;     // extinction per metre at density 1
const float CIR_H = 10800.0;  // cirrus sheet height

float gSoft = 0.0;            // 0 near .. 1 from orbit: rim thinning in cloudShape, set per sample by marchCumulus

float remap(float x, float a, float b, float c, float d){ return c + (x - a)/(b - a)*(d - c); }

// stable ray/sphere roots for a shell at height hs above sea level (camera height known precisely)
vec2 shell(float b, float hs){
  float c = (uCamAlt - hs)*(2.0*PLANET_R + uCamAlt + hs);
  float disc = b*b - c;
  if (disc < 0.0) return vec2(-1.0);
  float q = -(b + (b >= 0.0 ? 1.0 : -1.0)*sqrt(disc));
  float t0 = q, t1 = c/(abs(q) > 1e-6 ? q : 1e-6);
  return vec2(min(t0, t1), max(t0, t1));
}

// noise lookup with period S km; lod 0 = one 390 m pixel, so this mip matches the texel to the pixel
vec4 noiseS(vec3 q, float S, float lod){ return textureLod(uNoise, q*(1.0/S), max(lod + log2(25.0/S), 0.0)); }

// weather looked up through a warped direction and a cubic filter (no map texels, even round a sub-texel eye),
// then broken up by mesoscale noise: smooth patches for stratiform cloud, cell clusters where convective;
// solid decks stay solid. r = coverage, g = convective type.
vec4 coverAt(vec3 P, vec3 q, float lod){
  vec3 up = P/length(P);
  // skip clear air on a coarse mip (~80-160 km support, wider than the warp): a bilinear test at the unwarped
  // point cut the warped cloud along a texel staircase
  if (textureLod(uWeather, weatherUV(up), 2.5).r < 0.002) return vec4(0.0);
  vec3 wn = noiseS(q, 900.0, lod).rga - 0.5;
  vec4 w = weatherSmooth(uWeather, normalize(up + wn*(40000.0/PLANET_R)));
  vec4 mm = noiseS(q + 0.37, 240.0, lod);
  float m = mix(clamp(2.2*mm.a - 0.6, 0.0, 1.0), mm.r, smoothstep(0.3, 0.7, w.g));
  w.r = clamp(mix(w.r*(0.25 + 1.5*m), w.r, smoothstep(0.7, 1.0, w.r)), 0.0, 1.0);
  return w;
}

// Stratiform cloud is a soft-edged sheet of smooth value noise whose thickness follows its density;
// convective cloud is billowy Perlin-Worley cells that climb with w.g. Dense convective cores (hurricane
// CDO, fronts) flatten back into a sheet with a high top.
float cloudShape(vec3 q, float hN, vec4 w, float lod, bool detail){
  float cov = w.r;
  if (cov < 0.03) return 0.0;
  float cv = smoothstep(0.2, 0.7, w.g)*(1.0 - 0.7*smoothstep(0.85, 1.0, cov));
  // tops sink towards the edge of a cloud field (scattered cells are shallow), so decks end in a slope, not a cliff
  float top = mix(0.22, 1.0, w.g)*mix(0.45, 1.0, smoothstep(0.15, 0.85, cov));
  vec4 a = noiseS(q, 91.0, lod + 1.2);
  // far away the small cells are unresolved: a homogenised medium with the same mean opacity (never an opaque
  // blob), its edge frayed and its top domed by the 91 km octave the pixel still resolves, so decks seen from
  // orbit get lit tops and shaded flanks instead of a flat cut-out
  float n0 = a.a - 0.5 + 0.6*(a.r - 0.25);
  float covF = smoothstep(0.05, 0.95, cov + 0.9*n0*(1.0 - smoothstep(0.8, 1.0, cov)));
  float topF = top*(0.55 + 0.45*covF)*max(1.0 + 1.5*n0, 0.4);
  float dFar = smoothstep(0.0, 0.07, hN)*(1.0 - smoothstep(topF*0.35, topF, hN))*-log(1.0 - 0.985*covF)/(SIGMA*5500.0*topF);
  float far = smoothstep(1.5, 4.5, lod);
  if (far > 0.99) return dFar;
  vec4 b = noiseS(q + 0.5, 29.0, lod + 1.0), c = noiseS(q + 0.2, 9.0, lod + 0.5);
  // convective tops domed by the Perlin-Worley billows (dense cores would otherwise be flat-topped)
  float topC = top*(0.75 + 0.5*b.r + 0.3*(a.a - 0.5));
  float prof = smoothstep(0.0, 0.07, hN)*(1.0 - smoothstep(topC*0.45, topC, hN));
  float ns = a.a*0.55 + b.a*0.3 + c.g*0.15;
  float sheet = max(smoothstep(0.1, 0.32, ns - mix(0.66, 0.34, cov) + 0.1), smoothstep(0.85, 1.0, cov));
  float topS = top*(0.4 + 0.6*sheet)*(0.3 + 1.4*ns);   // relief on the deck top
  float dS = sheet*sheet*smoothstep(0.0, 0.06, hN)*(1.0 - smoothstep(topS*0.5, topS, hN));
  // the threshold falls faster than linearly so dense cover closes into solid decks
  float dC = remap((a.r*0.3 + b.r*0.45 + c.g*0.25)*prof, pow(1.0 - cov, 1.5), 1.0, 0.0, 1.0);
  float d = mix(dS, dC, cv);
  if (detail && d > 0.0){
    float e = noiseS(q + vec3(0.0, uTime*0.002, 0.0), 4.3, lod - 0.5).g;
    d = remap(d, e*mix(0.25, 0.45, cv)*(1.0 - 0.5*hN), 1.0, 0.0, 1.0);
  }
  float dn = clamp(d*mix(0.6, 1.4, cov), 0.0, 1.4);
  // seen from far off the rim is thinned: sigma*path saturates by d ~ 0.03, so opacity used to jump in the first
  // few percent of the soft edge (a paper cut-out); cubed, it ramps over the whole edge (feathered, see-through)
  dn *= mix(1.0, min(dn*dn, 1.0), gSoft);
  return mix(dn, dFar, far);
}

float heightFrac(vec3 P){ return (length(P) - PLANET_R - CLOUD_BASE)/(CLOUD_TOP - CLOUD_BASE); }

vec4 marchCumulus(vec3 ro, vec3 rd, float b, float tMax, float jitter, out float tRep){
  tRep = tMax;
  vec2 top = shell(b, CLOUD_TOP), bot = shell(b, CLOUD_BASE);
  if (top.y <= 0.0) return vec4(0.0, 0.0, 0.0, 1.0);
  float t0, t1;
  if (uCamAlt > CLOUD_TOP){ t0 = top.x; t1 = bot.x > 0.0 ? bot.x : top.y; }
  else if (uCamAlt > CLOUD_BASE){ t0 = 0.0; t1 = bot.x > 0.0 ? bot.x : top.y; }
  else { t0 = bot.y; t1 = top.y; }
  t0 = max(t0, 0.0);
  t1 = min(min(t1, tMax), t0 + 260000.0);
  if (t1 <= t0) return vec4(0.0, 0.0, 0.0, 1.0);

  int N = int(clamp((t1 - t0)/190.0, 24.0, 96.0));
  // from below or inside, bunch the samples up close to the camera where detail is visible
  float ex = uCamAlt < CLOUD_TOP ? 1.6 : 1.0;
  float mu = dot(rd, uSun);
  float phF = mix(phaseHG(mu, 0.8), phaseHG(mu, -0.25), 0.3);
  vec3 L = vec3(0.0);
  float T = 1.0, tw = 0.0, ts = 0.0, tPrev = t0;
  for (int i = 0; i < 96; i++){
    if (i >= N) break;
    float t = t0 + (t1 - t0)*pow((float(i) + jitter)/float(N), ex);
    float dt = t - tPrev; tPrev = t;
    vec3 P = ro + rd*t;
    float hN = heightFrac(P);
    if (hN < 0.0 || hN > 1.0) continue;
    float lod = max(log2(t*uPixAngle/390.0), 0.0);
    gSoft = smoothstep(40.0, 300.0, t*uPixAngle);
    vec3 q = P*0.001 + uWind;
    vec4 w = coverAt(P, q, lod);
    float d = cloudShape(q, hN, w, lod, lod < 4.0);
    if (d > 0.0005){
      vec3 up = normalize(P);
      float muS = dot(up, uSun);
      // optical depth towards the sun (weather reused: the march is short next to a map texel)
      // far away fewer, longer steps cover the same distance
      int nL = lod < 2.5 ? 5 : 3;
      float od = 0.0, ls = nL == 5 ? 90.0 : 300.0, lt = 0.0;
      for (int j = 0; j < 5; j++){
        if (j >= nL) break;
        lt += ls*0.5;
        vec3 P2 = P + uSun*lt;
        od += cloudShape(P2*0.001 + uWind, heightFrac(P2), w, lod + 1.0, false)*ls;
        lt += ls*0.5; ls *= 2.0;
      }
      vec3 sunC = SUN_I*sunTrans(uTransLUT, max(length(P) - PLANET_R, 0.0), muS);
      // multiple-scattering approximation (octaves of weakening extinction and flattening phase)
      vec3 S = vec3(0.0);
      float ea = 1.0, sb = 1.0, pc = 1.0;
      for (int k = 0; k < 4; k++){
        float ph = mix(phF, 1.0/(4.0*PI), 1.0 - pc);
        S += sb*sunC*exp(-od*SIGMA*0.6*ea)*ph;   // 0.6: coarse view steps land deeper than the lit skin
        ea *= 0.18; sb *= 0.8; pc *= 0.5;   // 0.8: thick decks reflect ~75% from above (were grey at 0.62)
      }
      // dark edges on the sunlit side ("powder"), only where the shapes are resolved
      S *= PI*mix(1.0, 1.0 - exp(-d*SIGMA*600.0), 0.35*(1.0 - smoothstep(1.5, 4.5, lod))*(0.5 - 0.5*mu));
      float day = smoothstep(-0.2, 0.3, muS);
      // skylight from above, weaker deep in the layer, plus warm light bounced off the ground
      vec3 amb = SUN_I*day*(vec3(0.05, 0.062, 0.085)*mix(0.4, 1.0, hN) + vec3(0.03, 0.028, 0.025)*max(muS, 0.0)*(1.0 - hN));
      // lightning inside storm cells, and a local flash when the camera is near a storm
      // (a soft glow round the 10 km cell's centre: a flat-lit cell drew a white box from orbit)
      vec3 lc = P*0.0001, lf = fract(lc) - 0.5;
      float cell = hash13(floor(lc) + floor(uTime*3.0)*17.0);
      vec3 flash = vec3(0.7, 0.75, 1.0)*(step(0.9993, cell)*w.a*900.0*exp(-14.0*dot(lf, lf)) + uFlash*400.0*exp(-t/6000.0))*(1.0 - hN);
      float Ti = exp(-SIGMA*d*dt);
      L += T*(S + amb + flash)*(1.0 - Ti);
      tw += T*(1.0 - Ti); ts += t*T*(1.0 - Ti);
      T *= Ti;
      if (T < 0.01) break;
    }
  }
  if (tw > 0.0) tRep = ts/tw;
  return vec4(L, T);
}

// thin ice sheet: streaks stretched along the zonal flow, thickest over the jet latitudes and storm anvils
vec4 cirrus(vec3 ro, vec3 rd, float b, float tMax, out float tC){
  vec2 s = shell(b, CIR_H);
  tC = uCamAlt < CIR_H ? s.y : s.x;
  if (tC <= 0.0 || tC > tMax) return vec4(0.0, 0.0, 0.0, 1.0);
  vec3 P = ro + rd*tC;
  vec3 up = normalize(P);
  float lod = max(log2(tC*uPixAngle/390.0), 0.0);
  vec3 q = P*0.001 + uWind*2.0;
  vec4 w = weatherAt(uWeather, up);
  float amt = smoothstep(0.5, 0.72, noiseS(q, 2600.0, lod).a)*mix(0.35, 1.0, smoothstep(0.35, 0.75, abs(up.y)))
            + w.g*smoothstep(0.5, 0.9, w.r)*0.9;
  if (amt < 0.01) return vec4(0.0, 0.0, 0.0, 1.0);
  vec3 qs = q*vec3(1.0, 4.0, 1.0) + (noiseS(q, 700.0, lod).rgb - 0.5)*60.0;
  float f = noiseS(qs, 180.0, lod + 2.5).a*0.65 + noiseS(qs, 41.0, lod + 2.5).g*0.35;
  float c = smoothstep(0.35, 0.8, f*0.8 + amt*0.35)*min(amt, 1.0);
  float tau = c*0.6/max(abs(dot(rd, up)), 0.04);
  float muS = dot(up, uSun), mu = dot(rd, uSun);
  vec3 sunC = SUN_I*sunTrans(uTransLUT, CIR_H, muS);
  vec3 Lc = sunC*mix(phaseHG(mu, 0.75), phaseHG(mu, 0.0), 0.5)*PI + SUN_I*vec3(0.06, 0.08, 0.12)*smoothstep(-0.2, 0.3, muS);
  float T = exp(-tau);
  return vec4(Lc*(1.0 - T), T);
}

// returns (in-scattered radiance, transmittance); tRep = transmittance-weighted cloud distance
vec4 marchClouds(vec3 ro, vec3 rd, float b, float tMax, float jitter, out float tRep){
  vec4 cu = marchCumulus(ro, rd, b, tMax, jitter, tRep);
  float tC;
  vec4 ci = cirrus(ro, rd, b, tMax, tC);
  if (ci.a > 0.999) return cu;
  if (cu.a > 0.99) tRep = tC;
  return uCamAlt < CIR_H ? vec4(cu.rgb + cu.a*ci.rgb, cu.a*ci.a) : vec4(ci.rgb + ci.a*cu.rgb, cu.a*ci.a);
}
`
