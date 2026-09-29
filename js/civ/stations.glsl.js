// Station shaders. Each instance is a unit box (or vault) stretched over a chord A->B of the rail centreline
// (sea level, camera-relative) with lateral extent y0..y1 and height z0..z1 above the rail top; one terrain()
// call lifts it onto the levelled rail bed. Surfaces (coping, windows, clock faces, glazing) are painted per piece.
import { OBJ_COMMON, OBJ_LIGHT } from './objects.glsl.js'

export const STATION_VERT = /* glsl */`${OBJ_COMMON}
uniform mat4 projectionMatrix, viewMatrix;
uniform vec3 uCamPos;
in vec3 position, normal;   // unit shape: x along 0..1, y across 0..1, z up 0..1
in vec3 iA, iB;             // chord ends on the rail centreline at sea level minus the camera
in vec4 iBox;               // y0, y1 (m, +y = left of A->B), z0, z1 (m above the rail top)
in vec3 iMisc;              // piece kind, station seed, building side
out vec3 vN, vLN, vUp, vRel, vLocal, vSize, vUnit;
flat out vec3 vMisc;        // flat: the seed must not wobble between pixels
out float vH, vFragDepth;
void main(){
  vec3 c = mix(iA, iB, position.x);
  vec3 up = normalize(uCamPos + c);
  vec3 ax = iB - iA;
  float len = length(ax);
  ax = normalize(ax - up*dot(ax, up));
  vec3 left = cross(up, ax);
  vec3 info;
  float h = max(terrain(normalize(uCamPos + c), CIV_FLAT_F*1.001, info).x, 0.0);
  vec3 size = vec3(len, iBox.y - iBox.x, iBox.w - iBox.z);
  vec3 local = vec3(position.x*len, mix(iBox.x, iBox.y, position.y), mix(iBox.z, iBox.w, position.z));
  vec3 rel = c + left*local.y + up*(h + local.z);
  vec3 n = normal/max(size, vec3(0.01));
  vN = ax*n.x + left*n.y + up*n.z;
  vLN = normal; vUp = up; vRel = rel; vLocal = local; vSize = size; vUnit = position; vMisc = iMisc; vH = h + local.z;
  gl_Position = projectionMatrix*viewMatrix*vec4(rel, 1.0);
  vFragDepth = 1.0 + gl_Position.w;
}`

export const STATION_FRAG = /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
uniform float uHour;        // local time (h) for the clock hands
in vec3 vN, vLN, vUp, vRel, vLocal, vSize, vUnit;
flat in vec3 vMisc;
in float vH, vFragDepth;
out vec4 outColor;
float band(float x, float a, float b){ return step(a, x)*step(x, b); }
// distance-based hand of a clock: angle a (clockwise from 12), length l, half width w
float hand(vec2 p, float a, float l, float w){
  vec2 d = vec2(sin(a), cos(a));
  float t = dot(p, d);
  return band(t, -0.1*l, l)*step(abs(p.x*d.y - p.y*d.x), w);
}
void main(){
  gl_FragDepth = log2(vFragDepth)*uLogDepthFC*0.5;
  vec3 N = normalize(vN), V = normalize(-vRel), L = vLocal, n = vLN;
  if (dot(N, V) < 0.0) { N = -N; n = -n; }          // vaults are seen from inside too
  float night = objNight(vUp), top = step(0.5, n.z), wall = 1.0 - step(0.5, abs(n.z));
  int kind = int(vMisc.x + 0.5), seed = int(vMisc.y);
  float hs = hash2i(seed, 7, 3);
  vec3 livery = hs < 0.33 ? vec3(0.05, 0.16, 0.09) : hs < 0.66 ? vec3(0.22, 0.04, 0.04) : vec3(0.04, 0.07, 0.18);
  vec3 alb = vec3(0.4), emit = vec3(0.0);
  float rough = 0.7, glass = 0.0;
  float y = abs(L.y);
  if (kind == 0) {
    // platform: concrete top with pale coping at the edge and a yellow line; brick-faced sides
    alb = mix(vec3(0.28, 0.2, 0.16), vec3(0.42, 0.41, 0.39)*(0.92 + 0.08*step(0.08, fract(L.x/2.5))), top);
    alb = mix(alb, vec3(0.66, 0.65, 0.6), top*band(y, 1.7, 2.15));
    alb = mix(alb, vec3(0.8, 0.62, 0.08), top*band(y, 2.4, 2.55));
    emit += vec3(1.0, 0.8, 0.55)*0.012*night*top*exp(-abs(fract(L.x/24.0 + 0.1) - 0.5)*6.0);   // lamp pools
  } else if (kind == 1) {
    // canopy / roof slab: livery fascia, lit underside
    alb = mix(livery*1.4, vec3(0.2, 0.2, 0.21), top);
    alb = mix(alb, vec3(0.55, 0.53, 0.5), step(n.z, -0.5));
    emit += step(n.z, -0.5)*vec3(1.0, 0.85, 0.6)*0.02*night;
  } else if (kind == 2) {
    alb = livery*1.2; rough = 0.5;
  } else if (kind == 3) {
    // lamp head: glowing underside at night
    alb = vec3(0.1);
    emit += step(n.z, -0.5)*vec3(1.0, 0.82, 0.55)*(0.3 + 3.0*night);
  } else if (kind == 4) {
    // station building: brick or render, tall arched-top windows, lit at night; door towards the track
    vec3 wallC = hs < 0.5 ? vec3(0.36, 0.17, 0.11) : vec3(0.62, 0.56, 0.45);
    float u = abs(n.x) > 0.5 ? L.y : L.x, fl = floor(L.z/4.5), roof = L.z + (1.0 - vUnit.z)*vSize.z;
    vec2 cell = vec2(fract(u/3.6), fract(L.z/4.5));
    float win = wall*step(0.5, L.z)*step(L.z, roof - 1.2)*band(cell.x, 0.28, 0.72)*band(cell.y, 0.2, 0.78);
    alb = wallC*(0.9 + 0.1*band(cell.y, 0.0, 0.06));
    alb = mix(alb, vec3(0.75, 0.72, 0.65), wall*band(L.z, -0.5, 0.5));   // plinth
    glass = win;
    float lit = step(0.35, hash2i(int(floor(u/3.6)), int(fl), seed));
    emit += vec3(1.0, 0.78, 0.5)*0.05*night*win*lit;
  } else if (kind == 5) {
    // clock: white dial with hour and minute hands on every vertical face
    float u = n.x*(vUnit.y - 0.5)*vSize.y - n.y*(vUnit.x - 0.5)*vSize.x;   // towards the viewer's right
    vec2 p = vec2(u, (vUnit.z - 0.5)*vSize.z);
    float r = 0.46*min(abs(n.x) > 0.5 ? vSize.y : vSize.x, vSize.z), d = length(p);
    float h = uHour*PI/6.0, m = fract(uHour)*2.0*PI;
    float hands = max(hand(p, h, 0.55*r, 0.045*r), hand(p, m, 0.85*r, 0.03*r));
    float dial = wall*step(d, r), rim = dial*step(0.9*r, d);
    alb = mix(vec3(0.12, 0.1, 0.09), vec3(0.85, 0.83, 0.76), dial);
    alb = mix(alb, vec3(0.05), max(rim, dial*hands));
    emit += dial*(1.0 - hands)*vec3(1.0, 0.92, 0.75)*0.04*night;
  } else if (kind == 6) {
    // name board: navy with pale lettering blocks
    vec2 q = vec2(abs(n.x) > 0.5 ? vUnit.y : vUnit.x, vUnit.z);
    float letter = band(q.y, 0.3, 0.7)*band(q.x, 0.08, 0.92)*step(0.25, fract(q.x*14.0))*step(0.3, hash2i(int(q.x*14.0), 1, seed));
    alb = mix(vec3(0.02, 0.04, 0.14), vec3(0.85), wall*letter);
    emit += wall*letter*vec3(0.9, 0.95, 1.0)*0.03*night;
  } else {
    // train shed: iron arches every 5 m and purlins, glazing between
    float a = atan(vUnit.z, 0.5 - vUnit.y)/PI;
    float iron = max(step(fract(L.x/5.0), 0.08), step(fract(a*12.0), 0.1));
    alb = mix(vec3(0.3, 0.34, 0.33), livery*0.8, iron);
    glass = 1.0 - iron;
    rough = 0.3;
    emit += step(dot(N, vUp), 0.0)*vec3(1.0, 0.85, 0.6)*0.004*night;
  }
  if (kind == 4) alb = mix(alb, vec3(0.03, 0.035, 0.04), glass);
  vec3 col = objLight(alb, N, vUp, vH) + emit + alb*vec3(0.010, 0.012, 0.018)*night;
  col += objSun(vUp, vH)*objSpec(N, V, mix(rough, 0.15, glass))*(0.3 + glass);
  outColor = vec4(col, 1.0);
}`
