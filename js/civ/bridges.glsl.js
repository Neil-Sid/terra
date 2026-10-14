// Bridge shaders. Each instance is a unit box stretched over a chord A->B of the deck centreline (sea level,
// relative to an anchor near the camera) with lateral extent y0..y1 and absolute heights z0..z1 at A; the box is
// sheared by dz from A to B so decks, chords and diagonals follow the deck profile. No terrain() call: the deck
// heights come from the CPU (sampled road beds). Surfaces are painted per kind; railings and stone arches discard.
import { OBJ_COMMON, OBJ_LIGHT } from './objects.glsl.js'

export const KIND = { HW: 0, ROAD: 1, RAIL: 2, STREET: 3, CONCRETE: 4, STEEL: 5, STONE: 6, RAILING: 7, LAMP: 8, STONE_ARCH: 9, POST: 10 }

export const BRIDGE_VERT = /* glsl */`${OBJ_COMMON}
uniform mat4 projectionMatrix, viewMatrix;
uniform vec3 uAnchor, uAnchorRel;  // anchor (planet frame) and anchor minus camera
in vec3 position, normal;          // unit box: x along 0..1, y across 0..1, z up 0..1
in vec3 iA, iB;                    // chord ends on the deck centreline at sea level minus the anchor
in vec4 iBox;                      // y0, y1 (m, +y = left of A->B), z0, z1 (m above sea level at A)
in vec4 iMisc;                     // dz (height change A->B), kind, seed (+4096 when lit), first drawn (s)
in vec4 iArch;                     // x of A along the bridge, arch opening half span (0 none), spring height, rise
out vec3 vN, vLN, vUp, vRel, vLocal, vSize, vUnit;
flat out vec4 vMisc, vArch;
void main(){
  vec3 c = mix(iA, iB, position.x);
  vec3 up = normalize(uAnchor + c);
  vec3 ax = iB - iA;
  float len = max(length(ax), 1e-3), dz = iMisc.x;
  ax = normalize(ax - up*dot(ax, up));
  vec3 left = cross(up, ax);
  vec3 size = max(vec3(len, iBox.y - iBox.x, iBox.w - iBox.z), vec3(0.01));
  vec3 local = vec3(position.x*len, mix(iBox.x, iBox.y, position.y), mix(iBox.z, iBox.w, position.z) + dz*position.x);
  vec3 rel = uAnchorRel + c + left*local.y + up*local.z;
  vec3 n = vec3(normal.x/len - normal.z*dz/(len*size.z), normal.y/size.y, normal.z/size.z);   // shear: inverse transpose
  vN = ax*n.x + left*n.y + up*n.z;
  vLN = normal; vUp = up; vRel = rel; vLocal = local; vSize = size; vUnit = position; vMisc = iMisc; vArch = iArch;
  gl_Position = projectionMatrix*viewMatrix*vec4(rel, 1.0);
}`

export const bridgeFrag = range => /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
in vec3 vN, vLN, vUp, vRel, vLocal, vSize, vUnit;
flat in vec4 vMisc, vArch;
out vec4 outColor;
float band(float x, float a, float b){ return step(a, x)*step(x, b); }
// antialiased line of half width w around 0 (aa = metres per pixel)
float line(float x, float w, float aa){ return clamp((w - abs(x))/aa + 0.5, 0.0, 1.0)*min(1.0, 2.0*w/aa); }
float dash(float s, float on, float period, float aa){ return mix(step(fract(s/period), on/period), on/period, smoothstep(0.3, 1.0, aa*4.0/period)); }
float ign(vec2 p){ return fract(52.9829189*fract(dot(p, vec2(0.06711056, 0.00583715)))); }
void main(){
  float dist = length(vRel);
  // dithered in over 0.8 s once built and out towards the range
  float fade = smoothstep(${range.toFixed(1)}, ${(0.8 * range).toFixed(1)}, dist)*clamp((uTime - vMisc.w)/0.8, 0.0, 1.0);
  float dth = ign(gl_FragCoord.xy);
  if (dth >= fade) discard;
  int kind = int(vMisc.y + 0.5), seed = int(vMisc.z + 0.5) & 4095;
  bool lit = int(vMisc.z + 0.5) >= 4096;
  vec3 N = normalize(vN), V = -vRel/dist, n = vLN;
  float x = vLocal.x + vArch.x, y = vLocal.y, h = vLocal.z;     // x along the whole bridge, h above sea level
  float aa = dist*0.0012 + 0.002;                                 // ~ metres per pixel
  float night = objNight(vUp), top = step(0.5, n.z), wall = 1.0 - step(0.5, abs(n.z));
  float h1 = hash2i(seed, 1, 9), h2 = hash2i(seed, 2, 9);
  vec3 steel = h1 < 0.25 ? vec3(0.16, 0.26, 0.2) : h1 < 0.5 ? vec3(0.3, 0.1, 0.07) : h1 < 0.75 ? vec3(0.34, 0.36, 0.37) : vec3(0.12, 0.17, 0.27);
  vec3 concrete = vec3(0.47, 0.46, 0.43)*(0.95 + 0.05*step(0.1, fract(h/1.2)))*(1.0 - 0.12*wall*hash2i(int(floor(x/1.7)), 3, seed));
  vec3 stoneC = h2 < 0.5 ? vec3(0.5, 0.43, 0.33) : vec3(0.42, 0.41, 0.39);
  vec3 alb = concrete, emit = vec3(0.0);
  float rough = 0.8, metal = 0.0;
  if (kind <= 3) {
    // deck slab: road surface on top with the terrain paint's markings, concrete fascia and soffit
    float d = abs(y);
    vec3 asphalt = vec3(0.085, 0.086, 0.09)*(0.9 + 0.1*hash2i(int(floor(x/7.0)), int(floor(y/2.0)), seed));
    vec3 paint = vec3(0.62, 0.62, 0.58), r = asphalt;
    float m = 0.0;
    if (kind == 0) {
      r = mix(r, vec3(0.26, 0.255, 0.245), line(d, 0.4, aa));
      m = max(max(line(d - 2.4, 0.1, aa), line(d - 9.8, 0.12, aa)), line(d - 6.1, 0.08, aa)*dash(x, 3.0, 12.0, aa));
      r = mix(r, vec3(0.42, 0.41, 0.39), step(12.0, d));                                  // service walkways
    } else if (kind == 1) {
      m = max(line(d, 0.07, aa)*dash(x, 3.0, 8.0, aa), line(d - 3.4, 0.08, aa));
      r = mix(r, vec3(0.42, 0.41, 0.39), step(4.5, d));
    } else if (kind == 2) {
      // ballast, sleepers every 0.65 m, two rails; walkways beside the track
      r = vec3(0.3, 0.28, 0.25)*(0.85 + 0.15*hash2i(int(floor(x/0.3)), int(floor(y/0.3)), seed));
      r = mix(r, vec3(0.2, 0.15, 0.11), band(d, 0.0, 1.3)*mix(step(0.55, fract(x/0.65)), 0.45, smoothstep(0.1, 0.3, aa)));
      metal = line(d - 0.7175, 0.04, aa);
      r = mix(r, vec3(0.5, 0.5, 0.52), metal);
      r = mix(r, vec3(0.4, 0.39, 0.37), step(2.9, d));
    } else {
      // city trunk street, as paint.glsl.js draws its profile (vArch.y: 2 edge street, 3 boulevard / ring; spec.js
      // PROFILES): dashed lane dividers between the lane centres, a double centre line (white or, per city, yellow:
      // vArch.z) or a raised median with edge lines, kerbs and sidewalks
      bool blv = vArch.y > 2.5;
      float kerb = blv ? 10.6 : 7.2, lane = blv ? 6.8 : 3.5;
      m = line(d - lane, 0.08, aa)*dash(x, 3.0, 9.0, aa);
      if (blv) {
        m = max(m, max(line(d - (kerb - 0.6), 0.1, aa), line(d - 3.7, 0.1, aa)));
        r = mix(r, vec3(0.4, 0.39, 0.37), step(d, 3.2));                                // median (paved on the deck)
        r = mix(r, vec3(0.62, 0.61, 0.58), band(d, 2.95, 3.2));
      } else {
        float cl = line(d - 0.15, 0.06, aa);
        if (vArch.z > 0.5) r = mix(r, vec3(0.55, 0.42, 0.12), cl); else m = max(m, cl);
      }
      r = mix(r, vec3(0.34, 0.33, 0.31)*(1.0 - 0.25*line(fract(x/1.8) - 0.5, 0.01, aa)), step(kerb + 0.3, d));   // sidewalk slabs
      r = mix(r, vec3(0.4, 0.39, 0.37), band(d, kerb, kerb + 0.3));                           // kerb
    }
    r = mix(r, paint, m);
    alb = mix(concrete, r, top);
    rough = mix(0.8, 0.55, top);
    float lx = fract(x/32.0) - 0.5;                                  // lamp pools (lamps every 32 m)
    if (lit) emit += top*vec3(1.0, 0.78, 0.5)*0.03*night*exp(-lx*lx*40.0);
  } else if (kind == 5 || kind == 10) {
    alb = kind == 10 ? vec3(0.1, 0.1, 0.11) : steel*(0.85 + 0.15*step(0.06, fract(x/0.9 + 0.5*n.z)));
    rough = 0.5; metal = 0.4;
  } else if (kind == 6 || kind == 9) {
    // masonry: courses 0.55 m high, blocks 1.1 m long and staggered; voussoirs around an arch opening
    float c = floor(h/0.55), u = (abs(n.x) > 0.5 ? y : x)/1.1 + 0.5*mod(c, 2.0);
    float joint = max(step(0.92, fract(h/0.55)), step(0.95, fract(u)));
    vec3 blk = stoneC*(0.85 + 0.25*hash2i(int(floor(u)), int(c), seed));
    if (kind == 9) {
      float xa = vLocal.x - 0.5*vSize.x, e = length(vec2(xa/vArch.y, (h - vArch.z)/vArch.w));
      if (abs(xa) < vArch.y && (h < vArch.z || e < 1.0)) discard;                           // the opening
      if (h > vArch.z && e < 1.0 + 1.1/vArch.w) {
        float a = atan(h - vArch.z, xa)*vArch.y*1.1;
        joint = step(0.9, fract(a));
        blk = stoneC*(1.0 + 0.2*hash2i(int(floor(a)), 7, seed));
      }
    }
    alb = mix(blk, vec3(0.2, 0.19, 0.17), joint*(1.0 - smoothstep(0.03, 0.09, aa)));
    rough = 0.9;
  } else if (kind == 7) {
    // railing: top rail, posts every 2 m, balusters every 0.16 m (dithered to their coverage once sub-pixel)
    float zf = vUnit.z*vSize.z, solid = max(step(vSize.z - 0.09, zf), step(fract(x/2.0), 0.05));
    float bar = mix(step(fract(x/0.16), 0.25), step(dth, 0.25*fade), smoothstep(0.02, 0.06, aa));
    if (max(solid, bar*step(0.1, zf)) < 0.5) discard;
    alb = steel; rough = 0.5; metal = 0.4;
  } else if (kind == 8) {
    alb = vec3(0.1);
    emit += (step(n.z, -0.5) + 0.3*wall)*vec3(1.0, 0.82, 0.55)*(0.3 + 3.0*night);   // glowing lens, lit sides
  }
  vec3 col = objLight(alb, N, vUp, h) + emit + alb*vec3(0.010, 0.012, 0.018)*night;
  col += objSun(vUp, h)*objSpec(N, V, mix(rough, 0.3, metal))*(0.3 + metal);
  outColor = vec4(col, 1.0);
}`
