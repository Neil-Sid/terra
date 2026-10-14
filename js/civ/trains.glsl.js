// Train car shaders. Each instance is one car given by its two bogie points on the rail centreline
// (sea level, camera-relative); vertices are placed along that chord and lifted onto the rail bed by
// one terrain() call, or onto a bridge deck (heights at the bogies from the CPU). Car details (windows, stripes, lamps, container colours) are painted per part.
import { OBJ_COMMON, OBJ_LIGHT } from './objects.glsl.js'

export const TRAIN_VERT = /* glsl */`${OBJ_COMMON}
uniform mat4 projectionMatrix, viewMatrix;
uniform vec3 uCamPos;       // camera position (float precision is fine: only used for directions)
in vec3 position;           // car-local metres: x forward, y left, z up from the rail top
in vec3 normal;
in float aPart;             // 0 plain livery, 1 running gear, 2 coach, 3 locomotive, 4 container, 5 wheels, 6 headlight beam
in vec3 iA, iB;             // rear and front bogie centres at sea level minus the camera
in vec3 iCol;               // livery (linear rgb)
in vec2 iMisc;              // lead unit (1 = headlights, 0 = tail lights), seed
in vec2 iH;                 // deck heights at the rear / front bogie (m; < -1e8 = on the rail bed)
out vec3 vN, vLN, vUp, vRel, vLocal, vCol;
flat out vec2 vMisc;        // flat: the seed must not wobble between pixels
out float vPart, vH;
void main(){
  if (aPart > 5.5 && iMisc.x < 0.5) { gl_Position = vec4(0.0, 0.0, -2.0, 1.0); return; }   // beams only ahead of lead units
  vec3 mid = 0.5*(iA + iB);
  vec3 up = normalize(uCamPos + mid);
  vec3 ax = iB - iA;
  ax = normalize(ax - up*dot(ax, up));
  vec3 left = cross(up, ax);
  vec3 c = mid + ax*position.x;
  vec3 info;
  float h = mix(iH.x, iH.y, position.x/max(length(iB - iA), 1.0) + 0.5);          // on a bridge
  if (iH.x < -1e8) h = max(terrain(normalize(uCamPos + c), CIV_FLAT_F*1.001, info).x, 0.0);  // the levelled rail bed
  vec3 rel = c + left*position.y + up*(h + position.z);
  vN = ax*normal.x + left*normal.y + up*normal.z;
  vLN = normal; vUp = up; vRel = rel; vLocal = position; vCol = iCol; vMisc = iMisc; vPart = aPart; vH = h + position.z;
  gl_Position = projectionMatrix*viewMatrix*vec4(rel, 1.0);
}`

export const TRAIN_FRAG = /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
in vec3 vN, vLN, vUp, vRel, vLocal, vCol;
flat in vec2 vMisc;
in float vPart, vH;
out vec4 outColor;
float band(float x, float a, float b){ return step(a, x)*step(x, b); }
float inside(vec2 p, vec2 lo, vec2 hi){ return step(lo.x, p.x)*step(p.x, hi.x)*step(lo.y, p.y)*step(p.y, hi.y); }
void main(){
  vec3 N = normalize(vN), V = normalize(-vRel), L = vLocal;
  float night = objNight(vUp), side = step(0.7, abs(vLN.y));
  int part = int(vPart + 0.5);
  if (part == 6) {
    // headlight beam on the track ahead (additive): narrow and bright near the nose, spreading and fading
    float x = L.x - 10.5, w = 0.8 + 0.04*x;
    float k = smoothstep(0.0, 4.0, x)*exp(-x/30.0)*(1.0 - smoothstep(0.3, 1.0, abs(L.y)/w));
    outColor = vec4(vec3(1.0, 0.86, 0.66)*0.016*k*night, 1.0);
    return;
  }
  vec3 alb = vCol, emit = vec3(0.0);
  float glass = 0.0, rough = 0.45;
  if (part == 1) { alb = vec3(0.035); rough = 0.7; }
  else if (part == 5) { alb = vec3(0.09, 0.085, 0.08); rough = 0.35; }
  else if (part == 2) {
    // coach: window band (lit at night), doors near the ends, a pale stripe below the windows
    float win = side*inside(vec2(fract(L.x/2.3), L.z), vec2(0.12, 2.25), vec2(0.88, 3.15))*step(abs(L.x), 10.6);
    float door = side*inside(vec2(abs(L.x), L.z), vec2(11.0, 1.35), vec2(12.1, 3.3));
    alb = mix(alb, vec3(0.8), side*inside(vec2(0.0, L.z), vec2(-1.0, 1.75), vec2(1.0, 1.95)));
    alb = mix(alb, alb*0.5, door);
    glass = win;
    float lit = step(0.18, hash2i(int(floor(L.x/2.3)), int(L.y > 0.0), int(vMisc.y)));
    emit += mix(vec3(1.0, 0.64, 0.32), vec3(0.72, 0.82, 0.86), step(0.7, hash2i(int(vMisc.y), 5, 1)))*0.022*night*win*lit;
    emit += alb*vec3(1.0, 0.8, 0.6)*0.05*night*side*band(L.z, 1.6, 2.2);   // window light on the body below
  } else if (part == 3) {
    // locomotive: raked windscreen and cab side windows up front, stripe, head/tail lamps on the nose
    float screen = step(0.3, vLN.x)*step(0.3, vLN.z)*step(abs(L.y), 1.15);
    float cab = side*inside(L.xz, vec2(6.9, 2.95), vec2(8.0, 3.7));
    alb = mix(alb, vec3(0.9, 0.75, 0.12), side*inside(vec2(0.0, L.z), vec2(-1.0, 1.55), vec2(1.0, 1.85)));
    glass = max(cab, screen);
    float front = step(0.9, vLN.x);
    // lamps grow with distance (dimmer per area) so they stay visible as points far away
    float r = max(0.18, length(vRel)*0.0012), k = 0.03/(r*r);
    float low = step(length(vec2(abs(L.y) - 0.85, L.z - 1.9)), r), top = step(length(vec2(L.y, L.z - 2.9)), r*0.8);
    vec3 lamp = vMisc.x > 0.5 ? vec3(1.0, 0.92, 0.75)*40.0*min(low + top, 1.0) : vec3(1.0, 0.05, 0.02)*12.0*low;
    alb = mix(alb, vec3(0.9), front*min(low + top, 1.0)*step(r, 0.18));
    emit += front*lamp*min(k, 1.0)*(0.06 + night) + vec3(1.0, 0.8, 0.55)*0.06*night*glass;
  } else if (part == 4) {
    // container: one colour per box, corrugated
    float hk = hash2i(int(floor((L.x + 9.3)/6.2)), 3, int(vMisc.y));
    vec3 cc[6] = vec3[6](vec3(0.45, 0.06, 0.04), vec3(0.05, 0.14, 0.35), vec3(0.08, 0.22, 0.08), vec3(0.55, 0.28, 0.04), vec3(0.5, 0.5, 0.48), vec3(0.3, 0.05, 0.18));
    alb = cc[int(hk*5.99)]*(0.85 + 0.15*step(0.5, fract(L.x*2.0)));
  } else {
    alb *= 1.0 - 0.1*side*step(0.5, fract(L.x));   // ribbed wagon sides
  }
  if (part != 1 && part != 5 && vLN.z > 0.9) alb *= 0.6;   // grimy roofs
  alb = mix(alb, vec3(0.015, 0.02, 0.025), glass);
  vec3 col = objLight(alb, N, vUp, vH) + emit + alb*vec3(0.022, 0.026, 0.036)*night;   // sky glow keeps bodies from going black
  col += alb*objSun(vUp, vH)*max(dot(vUp, uSun), 0.0)*0.09*(0.5 - 0.5*dot(N, vUp));   // light bounced off the ground
  col += objSun(vUp, vH)*objSpec(N, V, mix(rough, 0.12, glass))*(0.3 + glass);
  outColor = vec4(col, 1.0);
}`
