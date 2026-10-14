// Factory shaders. One program draws every shape (uShape: 0 box, 1 sawtooth tooth, 2 cylinder, 3 sphere),
// each shape its own instanced mesh. Facades are procedural (cladding ribs, roller doors, dock doors,
// clerestories, office windows, chimney bands), box-filtered by the pixel footprint so nothing shimmers.
// SMOKE_* draws chimney plumes (soft lit puffs, premultiplied alpha) and red aviation lights (additive).
import { OBJ_COMMON, OBJ_LIGHT } from './objects.glsl.js'
import { INDUSTRY_COLOURS } from './industry.glsl.js'

// styles: 0 hall, 1 shed, 2 tank, 3 silo, 4 works hall, 5 office, 6 tooth, 7 chimney, 8 banded chimney, 9 steel, 10 plant
export const VERT = /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
uniform mat4 projectionMatrix, viewMatrix;
uniform vec3 uAnchor;
uniform int uShape;
uniform float uLod;            // range scale (< 1 while a shape is over its instance cap)
in vec3 position;              // x, y in [-0.5, 0.5], z in [0, 1]
in vec3 normal;
in float aU;                   // around the circumference (cylinder, sphere) 0..1
in vec3 iPos, iUp, iEx, iSize; // base centre rel. anchor, vertical, local x axis, size (x, y, height)
in vec4 iMisc;                 // foundation depth (-1 none), seed, style, tone
in vec2 iExt;                  // box: dock side (local y sign), sawtooth pitch / shear rise; cylinder: taper
in vec3 iEy;                   // local y axis (not always square to iEx: sheared plots)
in vec3 iFade;                 // range (m), time first drawn (s), sink depth (m: into the ground, roof parts into the roof)
out vec3 vRel, vN;
out vec4 vFace;                // u (m along the face), v (m above base), face width, kind (0 wall, 1 roof, 2 glazing)
flat out vec3 vUp, vSunC, vAmbC, vSize;
flat out vec4 vMisc;
flat out vec3 vExt;            // iExt, local y sign of the face
flat out float vFade;          // dithered opacity (new blocks, range edge)
void main(){
  float z = position.z, dist = length(uAnchor + iPos);
  // over the last 20% of the range the part slides down by up to iFade.z (position only: the facade keeps its
  // coordinates) and dithers out over the last 5%; new blocks dither in over 0.8 s. Hidden ones are dropped here.
  float range = iFade.x*uLod, grow = smoothstep(0.0, 1.0, (range - dist)/(0.2*range));
  vFade = clamp((uTime - iFade.y)/0.8, 0.0, 1.0)*min(grow*4.0, 1.0);
  if (vFade <= 0.0){ gl_Position = vec4(0.0, 0.0, -2.0, 1.0); return; }
  vec3 U = iUp, E = normalize(iEx - U*dot(iEx, U)), Nn = normalize(iEy - U*dot(iEy, U));
  float found = iMisc.x < 0.0 ? 0.0 : iMisc.x + dist*0.002;
  float taper = uShape == 2 ? 1.0 - iExt.x*z : 1.0;
  vec3 loc = vec3(position.xy*iSize.xy*taper, uShape != 3 && z < 0.01 ? -found : z*iSize.z);
  float shear = uShape == 0 && iMisc.z > 8.5 && iMisc.z < 9.5 ? iExt.y : 0.0;   // sloped conveyors (steel boxes)
  loc.z += (position.x + 0.5)*shear;
  vec3 rel = uAnchor + iPos + E*loc.x + Nn*loc.y + U*(loc.z - (1.0 - grow)*iFade.z);
  vec3 nl = normal;
  float kind = 0.0;
  if (uShape == 1 && nl.z > 0.5){ nl = normalize(vec3(iSize.z, 0.0, iSize.x)); kind = 1.0; }
  else if (uShape == 1 && nl.x < -0.5) kind = 2.0;
  else if (uShape == 2 && nl.z < 0.5) nl = normalize(vec3(nl.xy, 0.5*iSize.x*iExt.x/iSize.z));
  else if (uShape == 3) nl = normalize(nl/iSize);
  else if (nl.z > 0.5){ kind = 1.0; nl = normalize(vec3(-shear/iSize.x, 0.0, 1.0)); }
  vN = cross(Nn, U)*nl.x + cross(U, E)*nl.y + cross(E, Nn)*nl.z;   // normals of the sheared frame (dual basis)
  bool sideX = abs(normal.x) > 0.5;
  float circ = 3.14159*iSize.x;
  vFace = kind == 1.0 ? vec4(position.x*iSize.x, position.y*iSize.y, 0.0, 1.0)
        : uShape >= 2 ? vec4(aU*circ, loc.z, circ, 0.0)
        : vec4(sideX ? (position.y + 0.5)*iSize.y : (position.x + 0.5)*iSize.x, loc.z, sideX ? iSize.y : iSize.x, kind);
  vRel = rel;
  vUp = U; vSize = iSize; vMisc = iMisc;
  vExt = vec3(iExt, sideX || uShape >= 2 ? 0.0 : sign(normal.y));
  vSunC = objSun(U, 100.0);
  vAmbC = objAmbient(U);
  gl_Position = projectionMatrix*viewMatrix*vec4(rel, 1.0);
}`

export const FRAG = /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
${INDUSTRY_COLOURS}
in vec3 vRel, vN;
in vec4 vFace;
flat in vec3 vUp, vSunC, vAmbC, vSize;
flat in vec4 vMisc;
flat in vec3 vExt;
flat in float vFade;
uniform float uShift;          // share of plants working through the night
out vec4 outColor;

// fraction of [x - fw/2, x + fw/2] covered by a periodic pulse of width w centred in each cell
float boxI(float x, float w){ return floor(x)*w + clamp(fract(x) - 0.5 + 0.5*w, 0.0, w); }
float pulse(float x, float w, float fw){ fw = max(fw, 1e-3); return (boxI(x + 0.5*fw, w) - boxI(x - 0.5*fw, w))/fw; }
float band(float x, float a, float b, float fw){ fw = max(fw, 1e-3); return clamp((min(x - a, b - x))/fw + 0.5, 0.0, 1.0); }

vec3 cladding(int style, float t){
  if (style == 4) return t < 0.4 ? vec3(0.30, 0.15, 0.10) : t < 0.7 ? vec3(0.21, 0.21, 0.22) : vec3(0.28, 0.21, 0.16);
  if (style == 1) return t < 0.6 ? vec3(0.62, 0.62, 0.60) : t < 0.85 ? vec3(0.50, 0.53, 0.56) : vec3(0.60, 0.57, 0.50);
  return t < 0.3 ? vec3(0.30, 0.34, 0.38) : t < 0.55 ? vec3(0.55, 0.52, 0.45) : t < 0.75 ? vec3(0.36, 0.15, 0.10) : vec3(0.28, 0.33, 0.30);
}

void main(){
  // screen-door fade: interleaved gradient noise, fixed on screen so it doesn't crawl
  if (fract(52.9829189*fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) >= vFade) discard;
  int style = int(vMisc.z + 0.5), seed = int(vMisc.y + 0.5);
  float tone = vMisc.w;
  vec3 N = normalize(vN), up = vUp;
  float dist = length(vRel);
  vec3 V = -vRel/dist;
  float night = objNight(up), muS = max(dot(up, uSun), 0.0);
  vec3 light = vSunC*(max(dot(N, uSun), 0.0) + 0.05*muS*(1.0 - dot(N, up))) + vAmbC*(0.6 + 0.4*dot(N, up));
  float u = vFace.x, v = vFace.y, width = vFace.z, H = vSize.z;
  int kind = int(vFace.w + 0.5);
  vec2 fw = vec2(fwidth(u), fwidth(v));
  bool shift = hash2i(seed >> 4, 3, 17) < uShift;
  vec3 sodium = vec3(1.0, 0.56, 0.22), alb, emit = vec3(0.0);
  float lampFar = 1.0 - 0.8*smoothstep(0.3, 1.5, max(fw.x, fw.y));   // wall lamps thin out to a glow far off
  float rough = 0.75, spec = 0.0;

  if (kind == 1){
    // roofs, matching the terrain paint
    alb = indRoof(style == 6 ? 0 : style, tone, u + 0.5*vSize.x, style == 0 ? vExt.y : 0.0, max(fw.x, fw.y));
    if (style == 0 || style == 1 || style == 6) { alb *= 0.92 + 0.08*pulse(vFace.y/0.8, 0.5, fwidth(vFace.y/0.8)); spec = 0.6; rough = 0.45; }
    if (style == 7 || style == 8) alb = vec3(0.03);                       // soot inside the chimney mouth
    if (style == 9) alb = tone < 0.5 ? vec3(0.55, 0.40, 0.05) : vec3(0.20, 0.27, 0.34);
    if (style == 10) alb = vec3(0.24, 0.24, 0.23);
  } else if (kind == 2){
    // sawtooth north lights: glazing bars over dim hall interior, sky reflection
    float bars = pulse(u/1.5, 0.9, fw.x/1.5)*band(v, 0.25, vSize.z - 0.2, fw.y);
    alb = mix(indRoofCol(0, tone), vec3(0.03, 0.035, 0.04), bars);
    float fres = 0.08 + 0.92*pow(1.0 - max(dot(N, V), 0.0), 5.0);
    emit += bars*fres*0.25*vAmbC*4.0;
    emit += bars*vec3(0.85, 0.92, 1.0)*night*(shift ? 0.05 : 0.004);
    spec = 1.0; rough = 0.2;
  } else if (style == 6){
    alb = cladding(0, tone);
  } else if (style <= 1 || style == 4){
    // clad halls and sheds: ribs, concrete plinth, clerestory band, doors
    alb = cladding(style, tone);
    alb *= style == 1 ? 0.9 + 0.2*pulse(v/0.6, 0.5, fw.y/0.6) : 0.88 + 0.24*pulse(u/0.3, 0.5, fw.x/0.3);
    if (style == 1) alb = mix(alb, tone < 0.5 ? vec3(0.08, 0.2, 0.42) : vec3(0.45, 0.08, 0.06), band(v, H - 2.6, H - 1.2, fw.y));   // livery stripe
    float cl = band(v, H - 3.4, H - 1.8, fw.y)*(style == 1 ? 0.0 : 1.0);
    float mull = pulse(u/1.5, 0.85, fw.x/1.5);
    alb = mix(alb, vec3(0.04, 0.045, 0.05), cl*mull);
    emit += cl*mull*vec3(1.0, 0.85, 0.6)*night*(shift ? 0.04 : 0.003);
    bool longFace = vExt.z != 0.0;
    float door = 0.0;
    if (longFace && style == 1 && vExt.z == vExt.x){
      // dock doors every 4 m with dark shelters, a lamp over each
      float k = u/4.0;
      door = pulse(k, 0.72, fw.x/4.0)*band(v, 1.2, 4.4, fw.y);
      float seal = pulse(k, 0.85, fw.x/4.0)*band(v, 1.0, 4.7, fw.y) - door;
      alb = mix(alb, vec3(0.05), seal);
      alb = mix(alb, vec3(0.36, 0.37, 0.38)*(0.85 + 0.3*pulse(v/0.5, 0.5, fw.y/0.5)), door);
      alb = mix(alb, vec3(0.3, 0.29, 0.28), band(v, -9.0, 1.2, fw.y));    // dock wall
      vec2 lq = vec2((fract(k) - 0.5)*4.0, v - 5.2);
      float hs = max(0.12, 0.7*max(fw.x, fw.y));
      emit += sodium*night*lampFar*min(0.02/(hs*hs), 3.0)*exp(-dot(lq, lq)/(2.0*hs*hs));
    } else if (longFace){
      // roller doors every ~24 m
      float dw = style == 4 ? 8.0 : 5.0, dh = style == 4 ? 9.0 : 5.5, per = 24.0;
      float k = (u - 0.5*mod(width, per))/per;
      door = pulse(k, dw/per, fw.x/per)*band(v, 0.0, dh, fw.y)*step(0.0, u - 6.0)*step(u, width - 6.0);
      alb = mix(alb, vec3(0.34, 0.33, 0.31)*(0.85 + 0.3*pulse(v/0.25, 0.5, fw.y/0.25)), door);
      vec2 lq = vec2((fract(k + 0.5) - 0.5)*per, v - dh - 1.0);
      float hs = max(0.15, 0.7*max(fw.x, fw.y));
      emit += sodium*night*lampFar*min(0.03/(hs*hs), 3.0)*exp(-dot(lq, lq)/(2.0*hs*hs));
    }
    alb = mix(alb, vec3(0.33, 0.32, 0.30), band(v, -99.0, 1.2, fw.y)*(1.0 - door));   // concrete plinth
    spec = 0.3;
  } else if (style == 5){
    // offices: ribbon windows per floor
    alb = tone < 0.5 ? vec3(0.44, 0.43, 0.41) : vec3(0.33, 0.18, 0.12);
    float fl = v/3.6, win = band(fract(fl), 0.35, 0.85, fw.y/3.6)*step(0.0, v)*step(v, H - 1.0)*pulse(u/1.8, 0.9, fw.x/1.8);
    ivec2 cell = ivec2(floor(u/3.6), floor(fl));
    float on = step(hash2i(cell.x + seed, cell.y, 7), shift ? 0.45 : 0.15);
    on = mix(on, shift ? 0.45 : 0.15, smoothstep(0.3, 1.0, fw.x/3.6));
    vec3 glass = vec3(0.03, 0.04, 0.05) + 0.2*vAmbC*(0.1 + 0.9*pow(1.0 - max(dot(N, V), 0.0), 5.0));
    alb = mix(alb, vec3(0.03), win);
    emit += win*(glass*0.5 + vec3(0.85, 0.9, 1.0)*night*on*0.04);
  } else if (style == 2){
    // tanks: white or silver steel, weld seams, a darker base ring
    alb = indRoofCol(2, tone)*(1.0 - 0.1*pulse(v/2.4, 0.06, fw.y/2.4))*(1.0 - 0.3*band(v, -99.0, 0.6, fw.y));
    spec = 0.8; rough = 0.35;
  } else if (style == 3){
    // silos: slipformed concrete with rain streaks under the top
    alb = vec3(0.47, 0.46, 0.43)*(0.94 + 0.06*pulse(u/1.2, 0.5, fw.x/1.2))*(1.0 - 0.2*smoothstep(H - 8.0, H, v)*(0.5 + 0.5*pulse(u/0.7, 0.3, fw.x/0.7)));
  } else if (style == 7 || style == 8){
    // chimneys: concrete, red and white bands on the tallest, soot at the top
    alb = vec3(0.42, 0.41, 0.39);
    if (style == 8) alb = mix(alb, vec3(0.45, 0.06, 0.04), band(v, 0.5*H, H, fw.y)*pulse(v/(H/14.0), 0.5, fw.y/(H/14.0)));
    alb = mix(alb, vec3(0.05), smoothstep(H - 4.0, H - 0.5, v));
  } else if (style == 9){
    alb = tone < 0.5 ? vec3(0.55, 0.40, 0.05) : vec3(0.20, 0.27, 0.34);
    alb *= 0.8 + 0.2*pulse(u/2.0, 0.15, fw.x/2.0);
  } else {
    alb = vec3(0.26, 0.26, 0.25)*(0.8 + 0.2*pulse(v/0.3, 0.5, fw.y/0.3));
  }
  if (style != 9 && style != 6) alb *= mix(0.65, 1.0, smoothstep(0.0, 3.0, v + (kind == 1 ? 99.0 : 0.0)));   // contact shadow
  vec3 col = alb*light + vSunC*objSpec(N, V, rough)*spec*0.5;
  // floodlights wash the lower walls at night
  if (kind != 1) emit += alb*sodium*night*0.05*exp(-max(v, 0.0)/6.0);
  outColor = vec4(col + emit, 1.0);
}`

export const SMOKE_VERT = /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
uniform mat4 projectionMatrix, viewMatrix;
uniform vec3 uAnchor, uWind;
uniform float uSmoke;          // plume reach (m): SMOKE, less while the puff cap is hit
in vec3 position;              // quad corner (-1..1)
in vec3 iTop, iUp;             // chimney mouth rel. anchor, vertical
in vec4 iP;                    // seed, mouth diameter, type (0 smoke, 1 steam, 2 aviation light), phase
out vec2 vUV;
flat out float vA, vType, vSeed;
flat out vec3 vSunC, vAmbC, vUp;
void main(){
  vec3 top = uAnchor + iTop, c;
  vec3 camR = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]), camU = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
  float size;
  vType = iP.z; vSeed = iP.x;
  if (iP.z > 1.5){
    c = top + iUp*0.8;
    size = max(0.8, length(c)*0.004);
    vA = 1.0/(size*size);
  } else {
    // puffs leave the mouth, rise, bend downwind and spread; each recycles once it has faded
    float life = iP.z > 0.5 ? 26.0 : 50.0, ph = fract(uTime/life + iP.w), t = ph*life;
    int k = int(iP.w*64.0);
    vec3 j = vec3(hash2i(int(iP.x), k, 1), hash2i(int(iP.x), k, 2), hash2i(int(iP.x), k, 3)) - 0.5;
    c = top + iUp*(5.0*pow(t, 0.7) + 0.5*iP.y) + uWind*t*(0.5 + 0.5*ph) + j*(1.0 + 0.8*t);
    size = 0.7*iP.y + 1.0*pow(t, 0.85);
    vA = smoothstep(0.0, 0.06, ph)*pow(1.0 - ph, 1.5)*(iP.z > 0.5 ? 0.6 : 0.5);
  }
  vA *= clamp((uSmoke - length(top))/(0.2*uSmoke), 0.0, 1.0);   // fades out over the last 20% of the reach
  vUp = iUp;
  vSunC = objSun(iUp, 300.0);
  vAmbC = objAmbient(iUp);
  vUV = position.xy;
  gl_Position = projectionMatrix*viewMatrix*vec4(c + (camR*position.x + camU*position.y)*size, 1.0);
}`

export const SMOKE_FRAG = /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
uniform mat4 viewMatrix;
in vec2 vUV;
flat in float vA, vType, vSeed;
flat in vec3 vSunC, vAmbC, vUp;
out vec4 outColor;
void main(){
  float r2 = dot(vUV, vUV);
  if (r2 > 1.0) discard;
  if (vType > 1.5){
    // red obstruction light, blinking; additive (alpha 0)
    float on = step(0.5, fract(uTime*0.7 + vSeed*0.37));
    outColor = vec4(vec3(1.0, 0.04, 0.02)*exp(-r2*6.0)*on*min(vA*4.0, 30.0)*(0.15 + 0.85*objNight(vUp)), 0.0);
  } else {
    vec3 camR = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]), camU = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
    vec3 camB = vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]);     // towards the camera
    vec3 n = normalize(camR*vUV.x + camU*vUV.y + camB*sqrt(1.0 - r2));
    float lump = 0.7 + 0.3*sin(vUV.x*4.1 + vSeed)*sin(vUV.y*3.7 + vSeed*1.3);
    float a = vA*(1.0 - r2)*(1.0 - r2)*lump;
    float fwd = pow(max(dot(-camB, uSun), 0.0), 6.0);                          // silver lining against the sun
    vec3 alb = vType > 0.5 ? vec3(0.9) : vec3(0.42, 0.41, 0.40);
    vec3 col = alb*(vSunC*(0.35 + 0.65*max(dot(n, uSun), 0.0) + 0.8*fwd) + vAmbC*1.5);
    outColor = vec4(col*a, a);
  }
}`
