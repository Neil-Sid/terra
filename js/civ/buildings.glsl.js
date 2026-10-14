// Building shaders: one instanced unit box (optionally with a gable roof) per building. Facades get
// a procedural window grid that is box-filtered by the pixel footprint, so distant facades average
// out instead of shimmering; at night a hashed subset of windows glows.
import { OBJ_COMMON, OBJ_LIGHT } from './objects.glsl.js'

// styles: 0 house, 1 brick block, 2 concrete block, 3 glass tower, 4 concrete tower with ribbon windows
export const VERT = /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
uniform mat4 projectionMatrix, viewMatrix;
uniform vec3 uAnchor;          // instance anchor minus camera (m)
uniform float uLod;            // range scale (< 1 while over the instance cap)
in vec3 position;              // x, y in [-0.5, 0.5]; z: 0 ground, 1 eaves, 2 ridge
in vec3 normal;                // local face normal; z = 1 marks the roof (slope side in y)
in vec3 iPos;                  // base centre relative to the anchor (m)
in vec3 iUp, iEx, iEy;         // local vertical, long axis, short axis (not always square: sheared lots)
in vec4 iSize;                 // length, depth, wall height, roof rise (m)
in vec4 iMisc;                 // foundation depth, seed, style, lot hash
in vec4 iFade;                 // range (m), time first drawn (s), sink depth (m: lot top into the ground, rooftop parts into the roof),
                               // inner range (far-tier block masses: they rise over [0.8, 1] of it while the buildings sink)
out vec3 vRel, vN;
out vec4 vFace;                // u (m along the face), v (m above base), face width, 1 on roofs
flat out vec3 vUp, vSunC, vAmbC;
flat out vec4 vSize, vMisc;
flat out float vFade;          // dithered opacity (new blocks, range edge)
flat out vec2 vHt;             // part base above ground, building top (main parts)
void main(){
  float dist = length(uAnchor + iPos), range = iFade.x*uLod;
  // over the last 20% of the range the part slides down by up to iFade.z (position only: the facade keeps its
  // coordinates) and dithers out over the last 5%; new blocks dither in over 0.8 s. Hidden ones are dropped here.
  float grow = smoothstep(0.0, 1.0, (range - dist)/(0.2*range)), r0 = iFade.w*uLod;
  if (r0 > 0.0) grow = min(grow, smoothstep(0.8*r0, r0, dist));
  vFade = clamp((uTime - iFade.y)/0.8, 0.0, 1.0)*min(grow*4.0, 1.0);
  if (vFade <= 0.0){ gl_Position = vec4(0.0, 0.0, -2.0, 1.0); return; }
  vec3 U = iUp, E = normalize(iEx - U*dot(iEx, U)), Nn = normalize(iEy - U*dot(iEy, U));
  float z = position.z;
  float found = iMisc.x < 0.0 ? 0.3 : iMisc.x + dist*0.002;   // far terrain is coarser: dig deeper
  int style = int(iMisc.z + 0.5);
  float par = iSize.w < 0.1 && style >= 1 && style <= 4 && normal.z < 0.5 ? 1.1 : 0.0;   // walls rise above flat roofs
  float zm = z < 0.5 ? -found : min(z, 1.0)*iSize.z + max(z - 1.0, 0.0)*iSize.w + par;
  vec3 rel = uAnchor + iPos + E*(position.x*iSize.x) + Nn*(position.y*iSize.y) + U*(zm - (1.0 - grow)*iFade.z);
  vec3 nl = normal;
  if (nl.z > 0.5) nl = normalize(vec3(0.0, nl.y*iSize.w, iSize.y*0.5));
  vN = cross(Nn, U)*nl.x + cross(U, E)*nl.y + cross(E, Nn)*nl.z;   // normals of the sheared frame (dual basis)
  bool sideX = abs(normal.x) > 0.5;
  vFace = normal.z > 0.5 ? vec4(position.x*iSize.x, position.y*iSize.y, 0.0, 1.0)
        : vec4(sideX ? (position.y + 0.5)*iSize.y : (position.x + 0.5)*iSize.x, zm, sideX ? iSize.y : iSize.x, 0.0);
  vRel = rel;
  vUp = U;
  vSunC = objSun(U, 100.0);                            // transmittance barely changes over city heights
  vAmbC = objAmbient(U);
  vSize = iSize; vMisc = iMisc;
  vHt = vec2(max(-iMisc.x - 1.0, 0.0), iFade.z - 2.0);
  gl_Position = projectionMatrix*viewMatrix*vec4(rel, 1.0);
}`

export const FRAG = /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
in vec3 vRel, vN;
in vec4 vFace;
flat in vec3 vUp, vSunC, vAmbC;
flat in vec4 vSize, vMisc;
flat in float vFade;
flat in vec2 vHt;
uniform vec4 uAct;             // share of windows lit now: homes, offices, shop fronts, office towers
out vec4 outColor;
#define GLOW vec3(1.0, 0.62, 0.36)   // sodium / LED city light: the night sky glow and lamp-lit streets

// fraction of [x - fw/2, x + fw/2] covered by a periodic pulse of width w centred in each cell
float boxI(float x, float w){ return floor(x)*w + clamp(fract(x) - 0.5 + 0.5*w, 0.0, w); }
float pulse(float x, float w, float fw){ fw = max(fw, 1e-3); return (boxI(x + 0.5*fw, w) - boxI(x - 0.5*fw, w))/fw; }

// the sky a window mirrors, as bright as post.js draws it (sky radiance x PI, like the terrain lighting): haze at
// the horizon, blue overhead, brighter towards the sun and warm while it is low; at night the city's glow
vec3 skyRefl(vec3 r, vec3 up, float night){
  float el = clamp(dot(r, up), 0.0, 1.0), muS = dot(up, uSun);
  vec3 c = mix(vec3(0.36, 0.40, 0.46), vec3(0.08, 0.16, 0.34), sqrt(el));
  c = mix(c, vec3(0.55, 0.32, 0.16), exp(-muS*muS/0.02)*(1.0 - el)*0.7);
  c *= 1.0 + 2.0*pow(max(dot(r, uSun), 0.0), 6.0);
  return SUN_I*PI*0.1*c*smoothstep(-0.15, 0.2, muS) + GLOW*night*0.012*(1.0 - 0.8*el);
}
// what a window mirrors: sky above, the facades across the street and the street itself near and below the
// horizon (sunlit and sky-lit grey-blue by day, lamp-lit at night)
vec3 envRefl(vec3 r, vec3 up, float night){
  vec3 city = 0.1*vSunC*max(dot(up, uSun), 0.0) + 0.6*vAmbC + GLOW*night*0.007;
  return mix(skyRefl(r, up, night), city, 1.0 - smoothstep(-0.1, 0.25, dot(r, up)));
}

// curtain-wall glass tint (multiplies the reflection): blue, silver grey, teal, bronze
vec3 glassTint(float r2){
  return r2 < 0.3 ? vec3(0.55, 0.76, 1.0) : r2 < 0.55 ? vec3(0.80, 0.86, 0.92) : r2 < 0.75 ? vec3(0.50, 0.88, 0.84) : vec3(1.0, 0.76, 0.52);
}
// facade albedo per style from two hashes
vec3 wallColour(int style, float r, float r2){
  if (style == 0){
    vec3 plaster = r2 < 0.33 ? vec3(0.60, 0.57, 0.50) : r2 < 0.66 ? vec3(0.58, 0.48, 0.32) : vec3(0.52, 0.53, 0.52);
    return r < 0.6 ? plaster*(0.8 + 0.3*r) : mix(vec3(0.34, 0.17, 0.11), vec3(0.46, 0.36, 0.26), r2);
  }
  if (style == 1) return mix(vec3(0.26, 0.12, 0.08), vec3(0.40, 0.25, 0.17), r)*(0.85 + 0.3*r2);
  if (style == 2) return r2 < 0.5 ? mix(vec3(0.32, 0.31, 0.30), vec3(0.48, 0.47, 0.45), r)
                                  : mix(vec3(0.46, 0.40, 0.31), vec3(0.56, 0.50, 0.40), r);
  if (style == 3) return glassTint(r2)*0.1;                                                        // spandrel panels
  if (style == 4) return mix(vec3(0.42, 0.42, 0.40), vec3(0.55, 0.52, 0.47), r);
  return vec3(0.30, 0.30, 0.29);                                                                   // plant room
}

void main(){
  // screen-door fade: interleaved gradient noise, fixed on screen so it doesn't crawl
  if (fract(52.9829189*fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) >= vFade) discard;
  int style = int(vMisc.z + 0.5), seed = int(vMisc.y + 0.5);
  vec3 N = normalize(vN), up = vUp;
  float dist = length(vRel);
  vec3 V = -vRel/dist;
  float r1 = hash2i(seed, 1, 5), r2 = hash2i(seed, 2, 5);
  float night = objNight(up);
  float muS = max(dot(up, uSun), 0.0);
  vec3 light = vSunC*(max(dot(N, uSun), 0.0) + 0.05*muS*(1.0 - dot(N, up))) + vAmbC*(0.6 + 0.4*dot(N, up))   // + ground bounce
             + GLOW*night*0.001;                                                                               // city glow
  vec3 col, emit = vec3(0.0);

  if (vFace.w > 0.5){
    // roofs: gravel or tar on flat roofs, tiles or slate on pitched ones
    float lh = vMisc.w;
    vec3 alb = vSize.w > 0.1 ? (lh > 0.45 ? vec3(0.30, 0.12, 0.07) : vec3(0.13, 0.13, 0.14))
                             : mix(vec3(0.12, 0.12, 0.12), vec3(0.28, 0.27, 0.25), lh);
    vec2 q = vFace.xy, e = 0.5*vSize.xy - abs(q);
    float rim = 1.0 - smoothstep(0.35, 0.6, min(e.x, e.y));
    if (vSize.w < 0.1) alb = mix(alb, wallColour(style, r1, r2)*0.8, rim);
    if (vSize.w < 0.1 && style >= 1 && style <= 4) alb *= 0.65 + 0.35*smoothstep(0.6, 2.0, min(e.x, e.y));   // parapet shade
    else alb *= 0.85 + 0.3*pulse(q.y/0.35, 0.5, fwidth(q.y/0.35));
    col = alb*light;
    col += vSunC*objSpec(N, V, style == 3 ? 0.3 : 0.6)*0.5;
  } else {
    float u = vFace.x, v = vFace.y, width = vFace.z, H = vSize.z;
    bool upper = vMisc.x < 0.0;
    float fh = style == 0 ? 2.9 : style >= 3 ? 3.6 : 3.2;
    float colW = style == 0 ? 3.4 : style == 3 ? 1.6 : style == 4 ? 4.5 : 3.2;
    float floors = style == 5 ? 0.0 : max(1.0, floor((H - (style == 0 ? 0.3 : 0.9))/fh + 0.05));
    float nCol = max(1.0, floor(width/colW + 0.5));
    vec2 c = vec2(u/width*nCol, v/fh), fw = fwidth(c);
    bool ground = style != 0 && !upper && c.y < 1.0;
    float wx = style == 0 ? 0.42 : style == 1 ? 0.48 : style == 2 ? 0.55 : style == 3 ? 0.92 : 0.9;
    float wy = style == 0 ? 0.45 : style == 3 ? 0.78 : 0.5;
    float oy = 0.04;                                       // opening centre above mid-floor (cells)
    if (ground){ wx = 0.8; wy = 0.62; oy = -0.1; }         // shop windows with a fascia above
    float inside = step(0.0, v)*smoothstep(floors + 0.02, floors - 0.02, c.y);
    float win = pulse(c.x, wx, fw.x)*pulse(c.y - oy, wy, fw.y)*inside;
    // frame: a thin band inside the opening (about 8 cm)
    vec2 fr = vec2(0.16*nCol/width, 0.16/fh);
    float pane = pulse(c.x, wx - fr.x, fw.x)*pulse(c.y - oy, wy - fr.y, fw.y)*inside;
    // close-up detail, faded out as a window cell nears a pixel: recessed openings (the reveal shades the top
    // and sides), sills, mullions in wide windows
    float sub = smoothstep(0.3, 1.0, max(fw.x, fw.y)), near = 1.0 - sub;
    vec2 cw = vec2(width/nCol, fh), mq = (fract(vec2(c.x, c.y - oy)) - 0.5)*cw, hw = 0.5*vec2(wx, wy)*cw;
    float reveal = mix(1.0, 0.45 + 0.55*smoothstep(0.02, 0.2, min(hw.x - abs(mq.x), hw.y - mq.y)), near);
    float sill = style <= 2 && !ground ? near*inside*step(abs(mq.x), hw.x + 0.08)*step(-hw.y - 0.1, mq.y)*step(mq.y, -hw.y) : 0.0;
    if (style != 3) pane *= 1.0 - near*step(1.1, 2.0*hw.x)*step(abs(mq.x), 0.035);   // (curtain-wall panes are one sheet)
    float fascia = ground ? clamp(min(c.y - 0.76, 0.95 - c.y)/max(fw.y, 1e-3) + 0.5, 0.0, 1.0) : 0.0;
    float sh = hash2i(int(floor(c.x/2.0)), 0, seed + 17);  // one shop per two bays
    vec3 signC = sh < 0.25 ? vec3(0.30, 0.05, 0.04) : sh < 0.45 ? vec3(0.05, 0.16, 0.10) : sh < 0.65 ? vec3(0.04, 0.06, 0.16)
               : sh < 0.85 ? vec3(0.55, 0.50, 0.40) : vec3(0.03);
    vec3 wall = wallColour(style, r1, r2);
    if (ground) wall = mix(wall*0.45, signC, fascia);
    wall *= 1.0 + 0.3*sill;
    if (style >= 1 && style <= 4)                          // shadow line under the parapet coping
      wall *= 1.0 - 0.3*clamp(min(v - H + 0.5, H - 0.25 - v)/max(fwidth(v), 1e-3) + 0.5, 0.0, 1.0);
    if (v < 0.0) wall = vec3(0.30, 0.29, 0.27);            // exposed foundation
    if (style == 5) wall *= 0.8 + 0.2*pulse(v/0.4, 0.5, fwidth(v/0.4));   // louvres
    if (!upper) wall *= mix(0.6, 1.0, smoothstep(0.0, 3.0, v));           // contact shadow
    // glass: dim interior (hashed blinds and rooms) plus reflections and sun glint
    ivec2 cell = ivec2(floor(c));
    float rh = hash2i(cell.x + seed*7, cell.y, seed + 4);
    float room = mix(rh, 0.5, sub);
    // reflective curtain glass mirrors about a third of its surroundings head-on, all of them at grazing angles
    float nv = max(dot(N, V), 0.0);
    float f0 = style == 3 ? 0.32 : 0.14;
    float fres = f0 + (1.0 - f0)*pow(1.0 - nv, 5.0);
    vec3 tint = style == 3 ? glassTint(r2) : vec3(1.0);
    vec3 refl = (envRefl(reflect(-V, N), up, night)*fres + vSunC*objSpec(N, V, 0.12))*tint;
    vec3 glass = (vec3(0.03, 0.032, 0.035)*(0.4 + 1.6*room*room)*light*tint + refl)*reveal;
    vec3 frameC = style == 0 ? vec3(0.7, 0.68, 0.64) : style == 3 ? vec3(0.30, 0.31, 0.33) : style == 4 ? vec3(0.12, 0.13, 0.14) : vec3(0.35, 0.34, 0.32);
    col = wall*light;
    if (style == 3) col += refl*0.8;                       // curtain wall: the spandrels are glass too
    col = mix(col, frameC*light*reveal, win - pane);
    col = mix(col, glass, pane);
    // night: the lit share follows the local hour (uAct: homes, offices, shops) and varies per building
    // and floor; lights switch on per apartment or office bay. Each level averages once below a pixel.
    // office towers keep more floors lit into the night (uAct.w) and whole floors switch together; their lobbies stay lit
    bool office = style >= 3 || (style == 2 && r1 > 0.7), tower = office && vHt.y > 60.0;
    float r3 = hash2i(seed, 5, 5);                         // towers: many nearly dark, a few mostly lit
    float share = ground ? (tower ? 0.9 : uAct.z) : tower ? uAct.w*(0.1 + 1.8*r3*r3) : (office ? uAct.y : uAct.x)*(0.4 + 1.2*r2);
    int face = int(dot(N, vec3(1.0, 2.0, 3.0))*3.0);
    float gw = office ? 4.0 : 2.0, zw = tower ? 1e3 : gw*3.0;   // apartment / bay width; zones of 3 bays (towers: whole floors) x 4 floors
    float zs = share*(0.3 + 1.4*hash2i(int(floor(c.x/zw)) + face*97, int(floor(c.y/4.0)), seed + 13));
    float fs = min(zs*(0.2 + 1.6*hash2i(cell.y, face, seed + 11)), 1.0);
    float grp = step(hash2i(int(floor(c.x/gw)) + seed*7, cell.y, seed + face), fs);
    float on = mix(grp*(rh > 0.2 ? 1.0 : 0.3), grp*0.86, sub);
    on = mix(on, fs*0.86, smoothstep(0.3, 1.0, max(fw.x/gw, fw.y)));
    on = mix(on, zs*0.86, smoothstep(0.3, 1.0, fw.y));
    on = mix(on, share*0.86, smoothstep(0.3, 1.0, max(fw.x/zw, fw.y/4.0)));
    on *= 1.0 - 0.4*smoothstep(0.3, 1.0, fw.y);             // sub-pixel windows read as sparse points, not a glow
    float tone = hash2i(cell.x, cell.y, seed + 9);
    vec3 homeC = tone < 0.65 ? vec3(1.0, 0.62, 0.32) : tone < 0.9 ? vec3(1.0, 0.8, 0.58) : vec3(0.5, 0.62, 1.0);
    vec3 officeC = tone < 0.8 ? vec3(0.82, 0.9, 1.0) : vec3(1.0, 0.88, 0.7);
    vec3 lampC = mix(office ? officeC : homeC, office ? vec3(0.86, 0.9, 0.97) : vec3(0.95, 0.66, 0.4), sub);
    float boost = mix(3.5, 1.0, smoothstep(0.03, 0.3, max(fw.x, fw.y)));   // a lit window reads bright up close
    emit = lampC*night*on*pane*(ground ? 0.12 : office ? 0.045 : 0.035)*(0.6 + 0.8*room)*boost;
    if (tower) emit += officeC*night*pane*(1.0 - on)*0.004*near*room;   // corridor and exit lights behind dark office floors (close up only)
    emit += signC*fascia*night*uAct.z*0.1*step(0.3, sh);    // lit shop signs
    if (!upper) emit += wall*vec3(1.0, 0.72, 0.45)*night*0.1*exp(-max(v, 0.0)/5.0)*(1.0 - pane);   // street lamps on the walls
    // the tallest towers: floodlit crowns (most of them) and red aviation lights on the roof corners of the very
    // tallest, kept at least ~1.5 px wide so they read from afar, flashing
    float zt = vHt.x + v, top = vHt.y, ch = hash2i(seed, 3, 5);
    if (style >= 2 && style <= 4 && top > 180.0 && vHt.x + H > top - 5.0){
      vec3 crownC = ch < 0.45 ? vec3(1.0, 0.86, 0.62) : ch < 0.75 ? vec3(0.85, 0.92, 1.0) : vec3(0.55, 0.75, 1.0);
      emit += crownC*night*step(0.55, ch)*0.012*smoothstep(top*0.95 - 3.0, top*0.95, zt)*mix(1.0, 0.4, pane)*(style == 3 ? 1.0 : 2.0*dot(wall, vec3(0.33)) + 0.3);
      float s = max(0.7, 1.5*max(fwidth(u), fwidth(v)));
      float lamp = step(300.0, top)*step(min(u, width - u), s)*step(H + 1.1 - s, v);
      emit += vec3(1.0, 0.04, 0.02)*night*lamp*0.12*step(0.45, fract(uTime*0.6 + r1));
    }
  }
  outColor = vec4(col + emit, 1.0);
}`
