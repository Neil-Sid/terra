// Planet height field and climate. Everything is a function of the unit direction p,
// so the vertex shader, fragment shader and CPU probe all agree.
// maxF band-limits the noise: octaves whose features are smaller than the sample spacing fade out.
import { CIV } from '../civ/civ.glsl.js'

// natural terrain only (terrainBase, climate); TERRAIN adds roads and cities on top (terrain, terrainCiv)
export const TERRAIN_BASE = /* glsl */`
uniform vec2 uSeason;   // see seasonT()
float bandw(float f, float maxF){ return clamp(2.0 - 2.0*f/maxF, 0.0, 1.0); }
// smoothstep and its derivative
// how many octaves (f0 * lac^i) lie below maxF; a data-dependent loop bound keeps D3D from unrolling
int octaves(float f0, float lac, float maxF, int cap){ return clamp(int(ceil(log(maxF/f0)/log(lac))), 0, cap); }
vec2 sstepd(float a, float b, float x){ float t = clamp((x - a)/(b - a), 0.0, 1.0); return vec2(t*t*(3.0 - 2.0*t), 6.0*t*(1.0 - t)/(b - a)); }
float gauss(float x){ return exp(-x*x); }

// large-scale moisture (0..1-ish) at sea level: climate() adds detail, terrainBase places landforms with it.
// mn = snoise(p*2.6 + vec3(7.0, 3.0, 1.0)), from the caller's batched noises
float moist0(float mn, float al, float land){
  return 0.5 + 0.32*mn + 0.30*gauss(al/11.0) - 0.42*gauss((al - 23.0)/9.0)
       + 0.12*gauss((al - 52.0)/12.0) + 0.22*(1.0 - smoothstep(0.0, 0.1, land));
}

// Where each landform applies (0..1), from climate and geology. Masks change over hundreds of km,
// so their gradients are left out. n8 = hilliness noise, pn = plateau noise, mn = moisture noise (moist0), all three
// from terrainBase's batch.
struct Landform { float M0, erg, mesa, bad, karst, cliff, plain, glac; };
Landform landforms(vec3 p, float land, float mask, float n8, float pn, float mn){
  Landform L = Landform(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  float al = degrees(asin(min(abs(p.y), 1.0)));
  L.glac = smoothstep(48.0, 62.0, al);
  if (land < -0.05) return L;
  float M0 = moist0(mn, al, land);
  float inl = smoothstep(0.02, 0.08, land), low = 1.0 - smoothstep(0.15, 0.4, mask), coast = 1.0 - smoothstep(0.04, 0.12, land);
  float semi = smoothstep(0.06, 0.12, M0)*(1.0 - smoothstep(0.28, 0.36, M0));
  L.M0 = M0;
  // sand seas in the basins of hot deserts (not on the plateaus)
  L.erg = (1.0 - smoothstep(0.05, 0.14, M0))*(1.0 - smoothstep(30.0, 36.0, al))*(1.0 - smoothstep(0.2, 0.45, n8))*(1.0 - smoothstep(0.12, 0.22, pn))*inl*low;
  // mesas, buttes and canyons on dry plateaus; badlands on dry hilly lowlands
  L.mesa = semi*smoothstep(0.02, 0.16, pn)*smoothstep(0.05, 0.12, land)*low*(1.0 - smoothstep(25.0, 34.0, abs(al - 25.0)));
  L.bad = semi*smoothstep(0.3, 0.55, n8)*inl*low*(1.0 - L.mesa)*(1.0 - L.erg);
  // tower karst in some wet tropics
  L.karst = smoothstep(0.6, 0.72, M0)*(1.0 - smoothstep(20.0, 27.0, al))*smoothstep(-0.28, -0.42, pn)*inl*low;
  // rocky cliffed coasts where the land is hilly; flat alluvial plains on wet, quiet coasts
  L.cliff = coast*smoothstep(0.2, 0.45, n8)*smoothstep(0.1, 0.2, M0)*(1.0 - L.glac);
  L.plain = coast*smoothstep(0.3, 0.45, M0)*smoothstep(0.05, -0.25, n8)*low;
  return L;
}

// Volcanoes on a 2-degree lat/lon lattice, at most one per cell and never crossing its edge: arcs along the
// mountain belts, rare hotspot shields anywhere (their tops make islands at sea). Returns (height, gradient);
// hb, dhb = the ground it stands on; xr = distance from the summit in radii (>= 1 outside).
vec4 volcano(vec3 p, float mask, float hb, vec3 dhb, out float xr){
  xr = 2.0;
  const float CELL = 0.03490659;
  float lat = asin(clamp(p.y, -1.0, 1.0)), lon = atan(p.x, p.z);
  vec2 id = floor(vec2(lat, lon)/CELL);
  float h1 = hash12(id + 17.0), h2 = hash12(id + 91.0), h3 = hash12(id + 43.0);
  bool shield = h1 < 0.012;   // hotspot (per cell, so a cone never changes type across its flank)
  float live = shield ? 1.0 : clamp((0.012 + 0.45*smoothstep(0.3, 0.7, mask) - h1)*25.0, 0.0, 1.0);
  if (live <= 0.0 || abs(lat) > 1.25) return vec4(0.0);
  vec2 cl = (id + 0.5 + (vec2(h2, h3) - 0.5)*0.3)*CELL;
  vec3 c = vec3(cos(cl.x)*sin(cl.y), sin(cl.x), cos(cl.x)*cos(cl.y));
  vec3 dv = p - c;
  float r = length(dv)*PLANET_R;
  float rad = min(mix(8e3, 16e3, h2), 0.33*CELL*PLANET_R*cos(cl.x));
  // shields: broad and low on land; at sea they rise from the floor to a summit 1-2.5 km above the waves
  float H = mix(1800.0, 3000.0, h3);
  vec3 dH = vec3(0.0);
  if (shield){
    rad = min(mix(25e3, 50e3, h3), 0.33*CELL*PLANET_R*cos(cl.x));
    H = max(1600.0, 1000.0 + 1500.0*h2 - hb);
    dH = H > 1600.0 ? -dhb : vec3(0.0);
  }
  xr = r/rad;
  if (xr >= 1.0) return vec4(0.0);
  vec3 dx = dv/max(length(dv), 1e-9)/rad*PLANET_R;
  // strato cones are concave with a summit crater; a third collapse into calderas; shields are domes
  float caldera = step(0.65, h1*17.0 - floor(h1*17.0));
  float xc = shield ? 0.07 : mix(0.06, 0.28, caldera), dc = (shield ? 0.04 : mix(0.13, 0.2, caldera))*H;
  float y, yd;
  if (shield){ float u = 1.0 - xr*xr; y = u*u; yd = -4.0*xr*u; }
  else { float u = 1.0 - max(xr, xc); y = u*u; yd = xr > xc ? -2.0*u : 0.0; }
  vec2 pit = sstepd(caldera > 0.5 ? 0.7*xc : 0.0, xc, xr);
  float hv = H*y - dc*(1.0 - pit.x);
  vec3 g = (H*yd + dc*pit.y)*dx + y*dH;
  return vec4(hv, g)*live;
}

// terrainBase's landform inputs, kept for landAlbedo (biome.glsl.js) so the fragment shader doesn't evaluate them
// twice: lfLast = the last call, lfBase = terrainCiv's natural ground at p (its later calls are road and rail bed points)
struct BaseLF { Landform L; float n8, vx; };
BaseLF lfLast, lfBase;

// Returns (height m, gradient of height w.r.t. p). info = (land, mountain mask, roughness).
vec4 terrainBase(vec3 p, float maxF, out vec3 info){
  // The noises before the ridges share one snoisedR() call site (D3D inlines every call site; 105.0*snoisedR = snoised,
  // written at each use so fxc rounds it as in separate calls), in this order: jobs 0-2 the continent warp w, then nC
  // domain-warped continent octaves of q (c), then 6 single noises ns of p (plateaus, hilliness, two mountain belt ones,
  // belt edge, moisture), then nH hill octaves of p (coarse dc, fine df, used further down). An octave past maxF ends its sum.
  const vec3 WO[3] = vec3[3](vec3(1.3, 7.1, 2.9), vec3(8.2, 1.4, 5.5), vec3(3.7, 4.6, 9.1));
  const float NS[6] = float[6](3.2, 8.0, 2.2, 5.3, 1.5, 2.6);
  const vec3 NO[6] = vec3[6](vec3(5.2, 1.7, 3.3), vec3(3.0), vec3(4.1, 2.3, 8.7), vec3(1.0, 9.0, 3.0), vec3(9.3, 2.2, 6.1), vec3(7.0, 3.0, 1.0));
  vec3 w = vec3(0.0), q = p;
  vec4 c = vec4(0.0), dc = vec4(0.0), df = vec4(0.0), ns[6];
  float a = 0.6, f = 1.1, cC = 1e9, ah = 1.0, fh = 60.0;
  int nC = octaves(1.1, 2.03, maxF, 9), nH = octaves(60.0, 2.1, maxF, 15);
  for (int j = 0; j < 9 + nC + nH; j++){
    int i = j - 3, k = i - nC, ih = k - 6;       // continent octave, single noise, hill octave
    float bw = 1.0;
    vec3 x;
    if (j < 3) x = p*1.8 + WO[j];
    else if (i < nC){ bw = bandw(f, maxF); x = q*f + float(i)*7.31; }
    else if (k < 6) x = p*NS[k] + NO[k];
    else { bw = bandw(fh, maxF); x = p*fh + vec3(float(ih)*1.91, 0.0, float(ih)*0.73); }
    if (bw <= 0.0){ if (k < 0) continue; break; }
    vec4 n = snoisedR(x);
    if (j < 3){ w[j] = n.x; q = p + 0.21*(105.0*w); }   // domain-warped continents
    else if (i < nC){
      n = 105.0*n;
      c += vec4(n.x, n.yzw*f)*a*bw;
      if (i == 4) cC = c.x;         // continents without the < ~300 km octaves (the inland floor below; 1e9 = not reached)
      f *= 2.03; a *= 0.52;
    }
    else if (k < 6) ns[k] = n;
    else {
      n = 105.0*n*vec4(1.0, fh, fh, fh)*ah*bw;
      if (ih < 5) dc += n; else df += n;
      fh *= 2.1; ah *= 0.6;
    }
  }
  // a polar continent in the south, like Antarctica
  vec2 ant = sstepd(0.88, 0.97, -p.y);
  float land = c.x - 0.05 + 0.5*ant.x;
  vec3 dl = c.yzw - vec3(0.0, 0.5*ant.y, 0.0);
  vec4 pn = 105.0*ns[0];   // plateaus (and where mesas and karst go)
  vec4 n8 = 105.0*ns[1];   // hilly vs quiet country
  // inland: land well above 0, or the coarse continent (cC) well inland, where pockets that only the fine octaves
  // take below 0 are land, not sea. No sea there (the floor at the end), so no quiet coast and no sea cliffs either
  float inland = max(smoothstep(0.02, 0.06, land), smoothstep(0.015, 0.04, (cC < 1e8 ? cC : c.x) - 0.05 + 0.5*ant.x));

  // base elevation: coastal plains inland, continental shelf then abyss offshore
  float h; vec3 dh;
  if (land > 0.0){
    h = 700.0*land + 2600.0*land*land;
    dh = dl*(700.0 + 5200.0*land);
    // high plateaus with escarpments, away from the coast
    vec2 pl = sstepd(0.22, 0.34, pn.x), pin = sstepd(0.05, 0.16, land);
    h += 800.0*pl.x*pin.x;
    dh += 800.0*(pl.y*pin.x*pn.yzw*3.2 + pl.x*pin.y*dl);
    // ice dome over the pole
    vec2 dome = sstepd(0.9, 1.0, -p.y);
    h += 2400.0*dome.x;
    dh.y -= 2400.0*dome.y;
  } else {
    float s = -land;
    float t1 = clamp(s/0.025, 0.0, 1.0), t2 = clamp((s - 0.03)/0.19, 0.0, 1.0);
    // shelf drops off right at the shore (ease-out), so shallows are a fringe, not a flat
    float s1 = t1*(2.0 - t1), s2 = t2*t2*(3.0 - 2.0*t2);
    h = -(140.0*s1 + 4300.0*s2);
    dh = dl*(140.0*2.0*(1.0 - t1)/0.025 + 4300.0*6.0*t2*(1.0 - t2)/0.19);
  }

  // mountain belts: inland orogens from a slow noise, plus cordilleras hugging some coasts
  float mm = (105.0*ns[2].x)*0.65 + (105.0*ns[3].x)*0.35;
  float belt = smoothstep(-0.03, 0.02, land)*(1.0 - smoothstep(0.06, 0.18, land))*smoothstep(0.15, 0.5, 105.0*ns[4].x);
  float mask = max(smoothstep(0.08, 0.5, mm)*smoothstep(-0.04, 0.05, land), belt)*(1.0 - ant.x);
  Landform L = landforms(p, land, mask, n8.x, pn.x, 105.0*ns[5].x);

  // The landform noises and the ridge octaves share a second snoisedR() call site: first the landform slots lf
  // (canyons 0-2, karst 3, dune meander and crest height 4-5, badland gullies 6-7; gradients pre-scaled), then nO ridge
  // octaves of q
  const float LFF[8] = float[8](80.0, 230.0, 160.0, 9000.0, 700.0, 1500.0, 9000.0, 21600.0);
  bool lfOn[4] = bool[4](L.mesa > 0.0 && land > 0.0, L.karst > 0.0, L.erg > 0.0, L.bad > 0.0);
  vec4 lf[8];
  for (int i = 0; i < 8; i++) lf[i] = vec4(0.0);
  int nLF = lfOn[0] || lfOn[1] || lfOn[2] || lfOn[3] ? 8 : 0;
  // ridged multifractal; octaves fade on steep ground so flanks stay smooth like eroded valleys
  vec4 r = vec4(0.0);
  vec3 acc = vec3(0.0);
  float amp = 1.0, fr = 36.0, wgt = 1.0;
  int nO = mask > 0.0 ? octaves(36.0, 1.98, maxF, 10) : 0;
  for (int j = 0; j < nLF + nO; j++){
    int i = j - nLF;                // ridge octave
    float bw = 1.0;
    vec3 x;
    if (j < nLF){
      if (!lfOn[j < 3 ? 0 : j == 3 ? 1 : j < 6 ? 2 : 3]) continue;
      x = p*LFF[j] + vec3(1.1 + 2.37*float(j), 4.2 + 0.71*float(j), 3.3);
    } else {
      bw = bandw(fr, maxF);
      if (bw <= 0.0) break;
      x = q*fr + vec3(float(i)*3.17);
    }
    vec4 n = snoisedR(x);
    // (bracketed like this the result is bit-identical to a separate loop of snoised(x)*vec4(1.0, vec3(LFF[j])); (105.0*n)*... is not)
    if (j < nLF){ lf[j] = 105.0*(n*vec4(1.0, vec3(LFF[j]))); continue; }
    n = 105.0*n;
    float s = 1.0 - abs(n.x);
    vec3 ds = -sign(n.x)*n.yzw*fr;
    float ero = 1.0/(1.0 + 1.5*dot(acc, acc));
    // the ~5-45 km octaves are boosted up to 2.7x so ranges have real valley-to-peak relief at that scale
    float mb = 1.0 + 1.7*gauss((float(i) - 3.5)/1.8);
    vec4 o = vec4(s*s*wgt, 2.0*s*ds*wgt)*amp*bw*ero*mb;
    r += o;
    acc += o.yzw*(2400.0*mask/PLANET_R);
    wgt = clamp(s*s*1.7, 0.0, 1.0);
    fr *= 1.98; amp *= 0.52;
  }
  if (L.glac > 0.0 && mask > 0.0){
    // glacial troughs: valleys get broad flat floors under steep walls (u = (1+k) x^2/(x+k) keeps the peaks)
    const float k = 0.35;
    float x = r.x, u = (1.0 + k)*x*x/(x + k), ud = (1.0 + k)*x*(x + 2.0*k)/((x + k)*(x + k));
    // near the coast the troughs are drowned into fjords
    vec2 fv = sstepd(0.04, 0.45, x);
    float fj = L.glac*(1.0 - smoothstep(0.02, 0.1, land))*smoothstep(0.25, 0.55, mask)*650.0;
    h -= fj*(1.0 - fv.x);
    dh += fj*fv.y*r.yzw;
    r = vec4(mix(x, u, L.glac), r.yzw*mix(1.0, ud, L.glac));
  }
  h += r.x*2400.0*mask;
  dh += r.yzw*2400.0*mask;
  float vx;
  vec4 vol = volcano(p, mask, h, dh, vx);
  h += vol.x; dh += vol.yzw;

  // hills from ~100 km down to ~10 m features; quieter right at the coast so shorelines stay clean,
  // near-flat in sand seas and alluvial plains, more relief where mesas and badlands get carved.
  // Coarse octaves (> ~5 km, dc) and fine ones (df), from the first batch, are kept apart so mesas can terrace the coarse relief.
  float rough = 0.22 + 1.1*mask + 0.35*smoothstep(0.1, 0.7, n8.x);
  float coastQuiet = max(mix(0.2, 1.0, smoothstep(0.0, 0.05, abs(land))), inland);
  float amp0 = 480.0*rough*coastQuiet*(land > 0.0 ? 1.0 : mix(0.35, 1.0, inland))/(1.0 + dot(acc, acc));
  amp0 *= (1.0 - 0.85*L.erg)*(1.0 - 0.75*L.plain)*(1.0 - 0.6*L.karst)*(1.0 + 0.7*L.mesa + 0.4*L.bad);
  amp0 *= 1.0 + mix(12.0, 3.0, smoothstep(-0.05, 0.02, land))*smoothstep(1.0, 0.2, vx);   // eroded, ragged volcano flanks and island coasts
  h += dc.x*amp0;
  dh += dc.yzw*amp0;

  if (lfOn[0]){
    // canyons: flat-floored gorges along the zero lines of a two-octave noise; a second noise makes them
    // start and end (box canyons) so they never close into loops
    vec4 c1 = lf[0] + 0.35*lf[1];
    vec2 k1 = sstepd(0.03, 0.22, abs(c1.x)), sg = sstepd(-0.35, 0.5, lf[2].x);
    float cut = L.mesa*(1.0 - k1.x)*sg.x;
    vec3 dcut = L.mesa*(-k1.y*sg.x*sign(c1.x)*c1.yzw + (1.0 - k1.x)*sg.y*lf[2].yzw);
    vec2 dep = sstepd(80.0, 1000.0, h);   // never cut below sea level
    float D = 450.0*dep.x;
    dh = dh*(1.0 - cut*450.0*dep.y) - D*dcut;
    h -= cut*D;
    // stepped plateaus: caprock treads and steep risers; risers soften where they would alias
    const float S = 200.0;
    float wr = clamp(2.0*length(dh - p*dot(dh, p))/(maxF*S), 0.12, 1.0);
    float x = h/S, fl = floor(x), t = x - fl;
    vec2 g = sstepd(1.0 - wr, 1.0, t);
    float tw = L.mesa*bandw(PLANET_R/5000.0, maxF);
    h = mix(h, S*(fl + 0.15*t + 0.85*g.x), tw);
    dh *= mix(1.0, 0.15 + 0.85*g.y, tw);
  }
  float fineA = amp0*(1.0 - 0.6*L.mesa);
  h += df.x*fineA;
  dh += df.yzw*fineA;

  if (lfOn[1]){
    // tower karst: steep round-topped limestone towers ~700 m apart over a flat floor
    vec2 kt = sstepd(0.12, 0.55, lf[3].x);
    float kh = 190.0*L.karst*bandw(12000.0, maxF);
    h += kh*kt.x; dh += kh*kt.y*lf[3].yzw;
  }
  if (lfOn[3]){
    // badlands: dense sharp gullies, two ridged octaves (~700 m, ~300 m)
    for (int i = 6; i < 8; i++){
      float s = 1.0 - abs(lf[i].x), bw = 32.0*L.bad*(i == 6 ? 1.0 : 0.45)*bandw(LFF[i]*1.5, maxF);
      h += s*s*bw; dh += 2.0*s*(-sign(lf[i].x))*lf[i].yzw*bw;
    }
  }
  if (lfOn[2]){
    // dune sea: asymmetric transverse ridges (gentle stoss, steep lee facing west, downwind of the trades)
    // with flat interdune corridors; crests meander and break into barchan-like crescents toward the
    // edges; small dunes ride on top. Longitude wave numbers are whole cycles per turn: they wrap at 180 deg.
    float lat = asin(clamp(p.y, -1.0, 1.0)), lon = atan(p.x, p.z);
    vec3 gLat = vec3(0.0, inversesqrt(max(1.0 - p.y*p.y, 1e-6)), 0.0), gLon = vec3(p.z, 0.0, -p.x)/max(p.x*p.x + p.z*p.z, 1e-6);
    vec4 wn = lf[4], bn = lf[5];
    float edge = 1.0 - smoothstep(0.25, 0.85, L.erg);
    vec2 bc = sstepd(-0.15, 0.45, bn.x);
    float m = (0.75 + 0.25*bn.x)*mix(1.0, bc.x, edge), md = 0.25*mix(1.0, bc.x, edge) + (0.75 + 0.25*bn.x)*edge*bc.y;
    for (int i = 0; i < 2; i++){
      // (lon cycles/rad, lat cycles/rad, height m): 1.8 km ridges, then 160 m dunes at another angle
      vec3 k = i == 0 ? vec3(3014.37, 1203.4, 100.0) : vec3(18038.06, 34483.0, 5.0);
      float bw = bandw(1.5*(i == 0 ? 3539.4 : 39819.0), maxF);
      if (bw <= 0.0) break;
      float ph = -(k.x*lon + k.y*lat + 0.8*wn.x);
      vec3 gph = -(k.x*gLon + k.y*gLat + 0.8*wn.yzw);
      float t = fract(ph), up = t < 0.76 ? t/0.76 : (1.0 - t)/0.24, dup = t < 0.76 ? 1.0/0.76 : -1.0/0.24;
      float u = max(up - 0.2, 0.0)/0.8, ud = up > 0.2 ? dup/0.8 : 0.0;
      float pr = u*u*(3.0 - 2.0*u), prd = 6.0*u*(1.0 - u)*ud;
      float A = k.z*L.erg*bw, mi = i == 0 ? m : 1.0;
      h += A*mi*pr;
      dh += A*(mi*prd*gph + (i == 0 ? pr*md*bn.yzw : vec3(0.0)));
    }
  }
  if (L.cliff > 0.0 && h > 0.0){
    // cliffed coast: the land steps up right at the waterline (as sharp as the sample spacing allows)
    float e = max(0.6, 2.0*length(dh - p*dot(dh, p))/maxF);
    vec2 cs = sstepd(0.0, e, h);
    float C = 55.0*L.cliff*(1.0 - inland);
    h += C*cs.x; dh *= 1.0 + C*cs.y;
  }
  if (L.plain > 0.0 && h > 0.0){
    // alluvial plain: low ground compressed toward sea level (h^2/(h + H0))
    const float H0 = 40.0;
    float P = h*h/(h + H0), Pd = h*(h + 2.0*H0)/((h + H0)*(h + H0));
    h = mix(h, P, L.plain); dh *= mix(1.0, Pd, L.plain);
  }

  // inland ground never dips below sea level (every shader draws h < 0 as sea, so inland hollows became
  // false seas); a soft floor near +5 m (above the beach band) keeps the gradient continuous. Coastal fjords and
  // inlets (small land) still can.
  if (inland > 0.0 && h < 40.0){
    float x = (h - 5.0)/3.0, sp = 5.0 + 3.0*log(1.0 + exp(x));
    h = mix(h, sp, inland);
    dh *= mix(1.0, 1.0/(1.0 + exp(-x)), inland);
  }

  info = vec3(land, mask, rough);
  lfLast = BaseLF(L, n8.x, vx);
  return vec4(h, dh);
}

// x = today's departure from the annual mean temperature (C), y = half the annual range. The range grows toward
// the poles and inland; the peak lags the solstice by ~4 weeks inland, ~7 at sea; the south is flipped.
// uSeason = (cos, sin) of the year's phase from the June solstice; (0, 0) = annual mean.
vec2 seasonT(vec3 p, float land){
  float s2 = p.y*p.y, cont = smoothstep(-0.05, 0.2, land);
  float A = mix(0.5 + 8.0*s2, 1.0 + 22.0*s2, cont);
  vec2 lag = mix(vec2(0.70, 0.72), vec2(0.90, 0.44), cont);   // (cos, sin) of 46 and 26 days
  return vec2(A*clamp(p.y*12.0, -1.0, 1.0)*dot(uSeason, lag), A);
}

// x = annual mean temperature (C), y = moisture 0..1. Stays the annual mean: forest, farm and tree placement
// and world generation read it; add seasonT().x for today's temperature (snow, ice, precipitation type).
vec2 climate(vec3 p, float h, float land){
  // one snoisedR() call site: temperature, large-scale moisture (moist0), moisture detail
  const float S[3] = float[3](6.0, 2.6, 9.0);
  const vec3 O[3] = vec3[3](vec3(2.0), vec3(7.0, 3.0, 1.0), vec3(1.0, 5.0, 2.0));
  vec3 n;
  for (int i = 0; i < loopN(3); i++) n[i] = snoisedR(p*S[i] + O[i]).x;
  float al = degrees(asin(min(abs(p.y), 1.0)));
  float T = 28.0 - 0.0078*al*al - 6.5*max(h, 0.0)/1000.0 + 3.0*(105.0*n.x);
  float M = moist0(105.0*n.y, al, land) + 0.14*(105.0*n.z);
  return vec2(T, clamp(M, 0.0, 1.0));
}
`
export const TERRAIN = TERRAIN_BASE + CIV
