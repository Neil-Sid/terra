// Industrial blocks (zone 1). INDUSTRY_LAYOUT mirrors plots()/plan()/grid() in factories.js, so the painted
// roofs, yards and tanks sit exactly under the 3D ones; INDUSTRY_COLOURS is shared with factories.glsl.js.
// paintIndustry() runs at the end of paintCity() for INDUSTRY districts: street = street coverage, farW = far blend
// (0 close, 1 far). Plots live in district coords (c.dl) inside the block's lot rect (civLotRect).

// roof albedo per style (0 hall, 1 shed, 2 tank, 3 silo, 4 works, 5 office, 6 sawtooth), tone = plot hash;
// lp = roof position (m) from the l start of the building, pitch = sawtooth spacing (0 = flat), aa = pixel size
export const INDUSTRY_COLOURS = /* glsl */`
vec3 indRoofCol(int style, float tone){
  if (style == 0 || style == 6) return mix(vec3(0.29, 0.30, 0.31), vec3(0.40, 0.40, 0.39), tone);
  if (style == 1) return mix(vec3(0.50, 0.51, 0.51), vec3(0.64, 0.64, 0.62), tone);
  if (style == 2) return mix(vec3(0.52, 0.52, 0.50), vec3(0.66, 0.65, 0.62), tone);
  if (style == 3) return vec3(0.42, 0.41, 0.39);
  if (style == 4) return mix(vec3(0.19, 0.165, 0.15), vec3(0.29, 0.23, 0.19), tone);
  return vec3(0.17, 0.17, 0.165);
}
float indPulse(float x, float w, float aa){        // periodic band |fract(x) - 0.5| < w/2, averaged below a pixel
  float d = abs(fract(x) - 0.5);
  return mix(clamp((0.5*w - d)/max(aa, 1e-4) + 0.5, 0.0, 1.0), w, smoothstep(0.25, 0.5, aa));
}
vec3 indRoof(int style, float tone, float lp, float pitch, float aa){
  vec3 c = indRoofCol(style, tone);
  if (style == 0 && pitch > 0.0){                  // sawtooth seen from above: slopes brighten away from the glazing
    float t = fract(lp/pitch), f = smoothstep(pitch*0.15, pitch*0.5, aa);
    c *= mix(0.8 + 0.3*t - 0.45*(1.0 - smoothstep(0.0, max(0.06, aa/pitch), t)), 0.93, f);
  }
  if (style == 1) c = mix(c, vec3(0.34, 0.37, 0.40), indPulse(lp/8.0, 0.12, aa/8.0));   // skylight strips
  return c;
}
`

export const INDUSTRY_LAYOUT = /* glsl */`
float indH(ivec2 id, int seed, int n){ return hash2i(id.x, id.y, seed + 45 + n); }
// plot of an industrial block containing q (district coords): r = the block's lot rect (x0, y0, x1, y1; spec.js
// blockInfo().lot), seed = the district's hseed; returns the plot rect and its id
vec4 indPlot(vec2 q, vec4 r, int kx, int ky, int seed, out ivec2 id){
  vec3 h = vec3(0.0);                               // seeds 41, 42, 43 through one call site (civN: paint.glsl.js)
  for (int k = 0; k < civN(3); k++) h = vec3(h.yz, hash2i(kx, ky, seed + 41 + k));
  float hb = h.x, f = 0.35 + 0.3*h.z;
  int nx = hb < 0.4 ? 1 : hb < 0.8 ? (h.y < 0.5 ? 2 : 1) : 2;
  int ny = hb < 0.4 ? 1 : hb < 0.8 ? 3 - nx : 2;
  int i = 0, j = 0;
  if (nx == 2){ float m = mix(r.x, r.z, f); if (q.x > m){ i = 1; r.x = m + 2.0; } else r.z = m - 2.0; }
  if (ny == 2){ float m = mix(r.y, r.w, f); if (q.y > m){ j = 1; r.y = m + 2.0; } else r.w = m - 2.0; }
  id = ivec2(kx*4 + i, ky*4 + j);
  return r;
}
struct IndPlan { int kind; bool ax; vec2 c; float L, S, side, yard, lmin, lmax, smin, smax; vec4 ha, hb; };
IndPlan indPlan(vec4 r, ivec2 id, int seed){
  IndPlan P;
  vec4 h0 = vec4(0.0), h1 = vec4(0.0);             // indH 0..8 through one call site: h0 = 0..3, h1 = 4..7, h8 = 8
  float h8 = 0.0;
  for (int k = 0; k < civN(9); k++){ h0 = vec4(h0.yzw, h1.x); h1 = vec4(h1.yzw, h8); h8 = indH(id, seed, k); }
  float w = r.z - r.x, d = r.w - r.y, ph = h0.x;
  P.ax = w > d - 0.5; P.L = max(w, d); P.S = min(w, d); P.c = 0.5*(r.xy + r.zw);
  P.kind = P.S < 34.0 ? (ph < 0.6 ? 0 : 1) : ph < 0.3 ? 0 : ph < 0.55 ? 1 : ph < 0.68 ? 2 : ph < 0.77 ? 3 : 4;
  P.side = h0.y < 0.5 ? -1.0 : 1.0;
  P.yard = P.S*(0.24 + 0.14*h0.z);
  P.lmin = 4.0 - 0.5*P.L; P.lmax = 0.5*P.L - 4.0;
  P.smin = P.side > 0.0 ? 4.0 - 0.5*P.S : P.yard - 0.5*P.S;
  P.smax = P.side > 0.0 ? 0.5*P.S - P.yard : 0.5*P.S - 4.0;
  P.ha = vec4(h0.w, h1.xyz);
  P.hb = vec4(h1.w, h8, 0.0, 0.0);
  return P;
}
// tank / silo grid: g = (l0, s0, pitch l, pitch s), n = (cols, rows, diameter, sphere)
void indGrid(IndPlan P, out vec4 g, out vec4 n){
  float Lm = P.lmax - P.lmin, Sm = P.smax - P.smin;
  if (P.kind == 2){
    float D = min(12.0 + 14.0*P.ha.x, Sm - 2.0), pitch = D + 5.0;
    vec2 k = vec2(clamp(floor((Lm + 5.0)/pitch), 1.0, 6.0), clamp(floor((Sm + 5.0)/pitch), 1.0, 2.0));
    bool sph = P.ha.y < 0.35;
    g = vec4(P.lmin, P.smin, Lm/k.x, Sm/k.y); n = vec4(k, sph ? min(D, 18.0) : D, sph ? 1.0 : 0.0);
  } else {
    float D = 7.0 + 3.0*P.ha.x;
    vec2 k = vec2(clamp(floor(Lm*0.7/D), 2.0, 10.0), clamp(floor(Sm/D), 1.0, 3.0));
    g = vec4(P.lmin, 0.5*(P.smin + P.smax) - 0.5*k.y*D, D, D); n = vec4(k, D - 0.4, 0.0);
  }
}
`

export const INDUSTRY_PAINT = /* glsl */`
${INDUSTRY_COLOURS}
${INDUSTRY_LAYOUT}
float indBox(vec2 q, vec4 r, float aa){   // coverage of the rect (x0, y0, x1, y1)
  vec2 c = 0.5*(r.xy + r.zw), h = 0.5*(r.zw - r.xy);
  vec4 b = civBand4(vec4(abs(q - c), 0.0, 0.0), vec4(h, 0.0, 0.0), aa);
  return b.x*b.y;
}
// rect, id = indPlot() of p (paintCity finds it), vn = paintCity's civVNCity() noises
void paintIndustry(inout CivOut o, Civ c, float aa, float night, float street, float farW, vec4 rect, ivec2 id, mat4 vn){
  IndPlan P = indPlan(rect, id, c.seed);
  vec2 q = c.dl - P.c, ls = P.ax ? q : q.yx;
  float l = ls.x, s = ls.y, Lm = P.lmax - P.lmin, Sm = P.smax - P.smin, sm = 0.5*(P.smin + P.smax);
  float edge = P.side > 0.0 ? P.smax : P.smin, ys = (s - edge)*P.side;   // metres out into the yard from the main face
  // plot box and slab joints; yard, fence line and stock-yard box (civBand4: paint.glsl.js)
  vec4 b0 = civBand4(vec4(abs(l), abs(s), abs(fract(l/6.0 + 0.5) - 0.5)*6.0, abs(fract(s/6.0 + 0.5) - 0.5)*6.0),
                     vec4(0.5*P.L, 0.5*P.S, 0.04, 0.04), aa);
  vec4 b1 = civBand4(vec4(abs(ys - 0.5*P.yard), abs(max(abs(l) - 0.5*P.L, abs(s) - 0.5*P.S)), abs(l - 0.5*(P.lmin + 0.58*Lm + P.lmax + 8.0)), abs(s - sm)),
                     vec4(0.5*P.yard, 0.08, 0.5*(0.42*Lm - 12.0), 0.5*Sm), aa);
  float lot = b0.x*b0.y;
  // corridors kept clear for rail sidings and roads (factories.js drops 3D parts there too)
  float keep = smoothstep(12.0, 14.0, c.dRail)*smoothstep(c.halfW + 2.0, c.halfW + 4.0, c.dRoad);

  // ground: concrete apron with slab joints and stains, asphalt yard, parking stalls, gravel lanes between plots
  float jn = max(b0.z, b0.w);
  vec3 conc = mix(vec3(0.25, 0.245, 0.23), vec3(0.31, 0.30, 0.28), P.ha.w)*(0.85 + 0.3*vn[2].w)*(1.0 - 0.3*jn);
  vec3 asph = vec3(0.095, 0.095, 0.10)*(0.85 + 0.3*vn[3].x);
  float yardCov = b1.x;
  vec3 g = mix(conc, asph, yardCov);
  float lotF = abs(l) < 0.5*P.L - 8.0 ? 1.0 : 0.0;                       // drive-ins at both ends stay clear
  vec3 carAvg = vec3(0.16, 0.15, 0.15);
  if (P.kind == 1 && P.yard > 20.0){
    // truck yard: trailers backed onto the dock doors every 4 m, yellow bay lines
    float k = floor(l/4.0), lf = l - (k + 0.5)*4.0;
    vec4 b = civBand4(vec4(abs(lf - 2.0), abs(ys - 9.0), abs(ys - 7.6), abs(lf)), vec4(0.06, 9.0, 6.8, 1.25), aa);
    float bay = b.x*b.y*lotF;
    g = mix(g, vec3(0.5, 0.4, 0.1), bay);
    float here = step(hash2i(int(k), id.x*7 + id.y, c.seed + 51), 0.55)*lotF;
    vec3 tc = hash2i(int(k), id.y, c.seed + 52) < 0.7 ? vec3(0.62, 0.62, 0.6) : vec3(0.35, 0.12, 0.08);
    float area = b.z*lotF;
    g = mix(g, tc, b.w*area*here);
    g = mix(g, mix(asph, vec3(0.55), 0.35), smoothstep(0.6, 1.6, aa)*area);
  } else {
    // car park along the plot edge
    float k = floor(l/2.6), lf = l - (k + 0.5)*2.6;
    vec4 b = civBand4(vec4(abs(ys - (P.yard - 3.5)), abs(abs(lf) - 1.3), abs(lf), abs(ys - (P.yard - 3.5))), vec4(2.5, 0.06, 0.9, 2.2), aa);
    float row = b.x*lotF;
    float line = b.y;
    float here = step(hash2i(int(k), id.x*7 + id.y, c.seed + 53), 0.6);
    float hc = hash2i(int(k), id.y, c.seed + 54);
    vec3 cc = hc < 0.3 ? vec3(0.55) : hc < 0.5 ? vec3(0.05) : hc < 0.65 ? vec3(0.3, 0.05, 0.04) : hc < 0.8 ? vec3(0.08, 0.12, 0.22) : vec3(0.3);
    float car = b.z*b.w*here;
    vec3 park = mix(mix(asph, vec3(0.55, 0.55, 0.52), line), cc, car);
    park = mix(park, mix(asph, carAvg, 0.5), smoothstep(0.5, 1.5, aa));
    g = mix(g, park, row);
  }
  if (P.kind == 4){
    // stock yard beyond the works hall: coal, ore and gravel heaps
    float heap = smoothstep(0.52, 0.7, vn[3].w);
    vec3 hc = P.ha.z < 0.4 ? vec3(0.045, 0.043, 0.04) : P.ha.z < 0.7 ? vec3(0.27, 0.13, 0.07) : vec3(0.34, 0.32, 0.28);
    g = mix(g, hc, heap*b1.z*b1.w);
  }
  vec3 lane = mix(vec3(0.2, 0.19, 0.17), vec3(0.08, 0.11, 0.045), 0.5 + 0.5*vn[2].y);
  g = mix(lane, g, lot);
  g *= 1.0 - 0.45*b1.y;                                                  // fence line

  // roofs (the 3D models' footprints)
  float roof = 0.0, roofFrac = 0.0;
  vec3 rc = vec3(0.0);
  float tone = P.ha.w;
  float od = 0.0, ow = 0.0, of = 0.0;
  if (P.kind <= 1){
    bool hall = P.kind == 0;
    float teeth = hall && P.ha.y < 0.65 ? Lm/max(2.0, floor(Lm/(8.0 + 4.0*P.ha.z) + 0.5)) : 0.0;
    od = min(P.yard - (hall ? 8.0 : 10.0), hall ? 12.0 : 8.0); ow = min(hall ? 14.0 + 12.0*P.ha.z : 12.0 + 8.0*P.ha.z, Lm*0.45);
    float lo = hall ? P.lmax - ow : P.lmin;           // office block (used below when od > 5)
    vec4 b = civBand4(vec4(abs(l - 0.5*(P.lmin + P.lmax)), abs(s - sm), abs(l - lo - 0.5*ow), abs(ys - 0.5*od)), vec4(0.5*Lm, 0.5*Sm, 0.5*ow, 0.5*od), aa);
    roof = b.x*b.y; of = b.z*b.w;
    rc = indRoof(P.kind, tone, l - P.lmin, teeth, aa);
    roofFrac = Lm*Sm;
  } else if (P.kind == 4){
    float lw = P.lmin + 0.58*Lm;
    vec4 b = civBand4(vec4(abs(l - 0.5*(P.lmin + lw)), abs(s - sm), 0.0, 0.0), vec4(0.5*(lw - P.lmin), 0.5*Sm, 0.0, 0.0), aa);
    roof = b.x*b.y;
    rc = indRoof(4, tone, 0.0, 0.0, aa);
    roofFrac = 0.58*Lm*Sm;
  } else {
    vec4 gr, n;
    indGrid(P, gr, n);
    vec2 k = clamp(floor((vec2(l, s) - gr.xy)/gr.zw), vec2(0.0), n.xy - 1.0);
    float dc = length(vec2(l, s) - gr.xy - (k + 0.5)*gr.zw);
    vec4 b = civBand4(vec4(dc, dc, abs(max(abs(l - 0.5*(P.lmin + P.lmax)) - 0.5*Lm - 2.0, abs(s - sm) - 0.5*Sm - 2.0)), 0.0), vec4(0.5*n.z, 0.5*n.z - 1.0, 0.3, 0.0), aa);
    roof = b.x;
    rc = P.kind == 2 ? indRoofCol(2, P.hb.x)*(1.0 - 0.25*b.y*(1.0 - n.w)) : indRoofCol(3, 0.0);
    roofFrac = 0.785*n.z*n.z*n.x*n.y;
    if (P.kind == 2){                                 // bund wall round the tank farm
      g = mix(g, vec3(0.38, 0.37, 0.35), b.z);
    } else {                                          // head house beside the silo battery
      float lh = gr.x + n.x*gr.z + 1.0, wh = min(10.0, P.lmax - lh), sh = min(Sm, 12.0);
      float hh = wh > 4.0 ? indBox(vec2(l, s), vec4(lh, sm - 0.5*sh, lh + wh, sm + 0.5*sh), aa) : 0.0;
      rc = mix(rc, indRoofCol(3, 0.0)*0.9, hh);
      roof = max(roof, hh);
      roofFrac += wh*sh;
    }
  }
  if (od > 5.0){                                      // office block on the yard side
    rc = mix(rc, indRoofCol(5, 0.0), of);
    roof = max(roof, of);
    roofFrac += ow*od;
  }
  roof *= keep;
  vec3 col = mix(g, rc, roof);

  // averages: per plot once a plot is a few pixels, then one district tone once blocks shrink to pixels
  roofFrac = min(roofFrac/(P.L*P.S), 0.9);
  vec3 gAvg = mix(mix(vec3(0.27, 0.265, 0.25), vec3(0.11), P.yard/P.S), lane, 0.08);
  vec3 plotAvg = mix(gAvg, indRoofCol(P.kind == 3 ? 3 : P.kind, P.kind == 2 ? P.hb.x : tone), roofFrac);
  col = mix(col, plotAvg, smoothstep(P.S*0.08, P.S*0.3, aa));
  float sf = civStreetFracD(c);
  vec3 district = vec3(0.16, 0.16, 0.15);    // close to the housing around it (metal roofs, concrete, verges), not glaring
  vec3 far = mix(mix(plotAvg*0.72, district, smoothstep(c.block*0.25, c.block*0.8, aa)), vec3(0.14, 0.14, 0.135), sf);
  o.alb = mix(mix(o.alb, col, 1.0 - street), far, farW);
  o.flatN = max(o.flatN, 0.85);

  // night: sodium floodlight masts along the yards (pools and heads), a little glow from night-shift roofs
  vec3 sodium = vec3(1.0, 0.56, 0.22);
  float mk = floor(l/30.0), ml = l - (mk + 0.5)*30.0, my = ys - 0.5*P.yard;
  float pool = exp(-(ml*ml + my*my)/(2.0*70.0))*yardCov*lot;
  float hs = max(0.35, aa*0.6);
  float head = (0.12/(hs*hs))*exp(-(ml*ml + my*my)/(2.0*hs*hs))*smoothstep(0.03, 0.12, aa)*lot;
  float flood = mix(col.g*0.25*pool + 0.2*head, 0.012*P.yard/P.S, smoothstep(5.0, 14.0, aa));
  // far off about as bright as housing, and whiter as the yards blur together, so districts don't stand out as
  // flat orange polygons (and block-sized differences don't alias once blocks are below a pixel)
  float farE = 0.0015*(1.0 - sf);
  vec3 farC = mix(sodium, vec3(1.0, 0.75, 0.48), smoothstep(c.block*0.5, c.block*2.0, aa));
  o.emit = o.emit*mix(mix(1.0, 0.35, 1.0 - street), 0.6, farW) + night*mix(sodium*flood*(1.0 - street), farC*farE, farW);
}
`
