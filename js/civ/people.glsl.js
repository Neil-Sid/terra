// Shaders for pedestrians. A tiny pre-pass evaluates terrain() once per person (the only terrain()
// call site) into a 64x32 float target; the mesh shader reads that height by gl_InstanceID and
// poses a low-poly humanoid (walk cycle, standing, sitting). Contracts in notes/CIV.md.
import { GLSL_HEAD } from '../config.js'
import { NOISE } from '../glsl/noise.js'
import { TERRAIN } from '../glsl/terrain.glsl.js'
import { OBJ_COMMON, OBJ_LIGHT } from './objects.glsl.js'

export const TEX_W = 64, TEX_H = 32              // one texel per person

// ground height under each person: r = terrain height (m)
export const HEIGHT_FRAG = /* glsl */`${GLSL_HEAD}
uniform highp sampler2D uDirs;
uniform int uCount;
out vec4 outH;
${NOISE}
${TERRAIN}
void main(){
  ivec2 px = ivec2(gl_FragCoord.xy);
  if (px.y*${TEX_W} + px.x >= uCount){ outH = vec4(0.0); return; }
  vec3 info;
  outH = vec4(terrain(normalize(texelFetch(uDirs, px, 0).xyz), 1.2e6, info).x, 0.0, 0.0, 1.0);
}`

export const PEOPLE_VERT = /* glsl */`${OBJ_COMMON}
uniform mat4 projectionMatrix, viewMatrix;
uniform highp sampler2D uDirs, uHeights;
uniform float uHRef;
in vec3 position, normal;
in float pm;                // part*8 + material
in vec3 iRel, iFwd;         // anchor relative to the camera at height uHRef; heading (unit)
in vec4 iMisc;              // walk phase, height (m), colour seed, pose (0 walk, 1 stand, 2 sit)
out vec3 vN, vUp, vAlb;
out float vH;

// rotate about the lateral axis through height py; positive a moves points below the pivot forward (+z)
vec3 swing(vec3 q, float py, float a){
  float c = cos(a), s = sin(a);
  q.y -= py;
  return vec3(q.x, q.y*c + q.z*s + py, q.z*c - q.y*s);
}
vec3 pick(vec3 a, vec3 b, vec3 c, vec3 d, float u){ return u < 0.25 ? a : u < 0.5 ? b : u < 0.75 ? c : d; }
// materials: 0 skin, 1 hair, 2 top, 3 trousers, 4 shoes
vec3 albedo(int mat, float seed){
  float u = hash12(vec2(seed, float(mat)*7.1 + 1.3)), w = hash12(vec2(seed, float(mat)*3.7 + 9.1));
  if (mat == 0) return pick(vec3(0.58, 0.37, 0.26), vec3(0.42, 0.25, 0.15), vec3(0.16, 0.085, 0.05), vec3(0.50, 0.33, 0.21), u);
  if (mat == 1) return w < 0.15 ? vec3(0.30, 0.29, 0.28) : pick(vec3(0.018), vec3(0.07, 0.04, 0.02), vec3(0.30, 0.21, 0.10), vec3(0.03, 0.02, 0.015), u);
  if (mat == 2){
    vec3 hue = clamp(abs(fract(u + vec3(0.0, 2.0/3.0, 1.0/3.0))*6.0 - 3.0) - 1.0, 0.0, 1.0);
    return mix(vec3(1.0), hue, 0.15 + 0.7*fract(w*5.3))*mix(0.025, 0.55, w*w);
  }
  if (mat == 3) return pick(vec3(0.035, 0.05, 0.10), vec3(0.018), vec3(0.22, 0.17, 0.10), vec3(0.10, 0.10, 0.11), u);
  return u < 0.7 ? vec3(0.025) : vec3(0.5);
}

void main(){
  int part = int(pm + 0.5)/8, mat = int(pm + 0.5) - part*8;   // parts: 0 body, 1 head, 2 thigh, 3 shin, 4 arm
  float ph = iMisc.x, pose = iMisc.w, side = position.x > 0.0 ? 1.0 : -1.0;
  const float HIP = 0.95, KNEE = 0.5, SHOULDER = 1.44;
  float legA = 0.0, kneeA = 0.03, armA = 0.04*sin(ph + side), lean = 0.0, drop = 0.0;
  if (pose < 0.5){                               // walk: legs and arms swing in opposition
    float p = ph + (side > 0.0 ? 0.0 : PI);
    legA = 0.42*sin(p);
    kneeA = 0.08 + 0.8*pow(max(cos(p), 0.0), 1.5);
    armA = -0.55*legA;
    lean = -0.04;
    drop = HIP*(1.0 - cos(legA));                  // keeps the stance foot on the ground
  } else if (pose > 1.5){                          // sit on the ground, legs out, leaning back on the hands
    legA = 1.5; kneeA = 0.2; armA = -0.5; lean = 0.25; drop = HIP - 0.13;
  }
  vec3 q = position, n = normal;
  if (part == 3){ q = swing(q, KNEE, -kneeA); n = swing(n, 0.0, -kneeA); }
  if (part == 2 || part == 3){ q = swing(q, HIP, legA); n = swing(n, 0.0, legA); }
  if (part == 4){ q = swing(q, SHOULDER, armA); n = swing(n, 0.0, armA); }
  if (part == 0 || part == 1 || part == 4){ q = swing(q, HIP, lean); n = swing(n, 0.0, lean); }
  q.y -= drop;
  q *= iMisc.y/1.75;

  ivec2 tc = ivec2(gl_InstanceID % ${TEX_W}, gl_InstanceID / ${TEX_W});
  vec3 up = normalize(texelFetch(uDirs, tc, 0).xyz);
  float h = texelFetch(uHeights, tc, 0).x;
  vec3 fwd = normalize(iFwd - up*dot(iFwd, up)), X = cross(up, fwd);
  vec3 rel = iRel + up*(h - uHRef) + X*q.x + up*q.y + fwd*q.z;
  vN = X*n.x + up*n.y + fwd*n.z;
  vUp = up; vH = h + q.y;
  vAlb = albedo(mat, iMisc.z);
  gl_Position = projectionMatrix*viewMatrix*vec4(rel, 1.0);
}`

export const PEOPLE_FRAG = /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
in vec3 vN, vUp, vAlb;
in float vH;
out vec4 outColor;
void main(){
  vec3 N = normalize(vN);
  // at night the street lamps overhead light people about as brightly as the pavement around them: CIV_POOL (0.3 x albedo)
  // under a lamp, dark between (paint.glsl.js), so about a third of that on average, mostly on heads and shoulders
  vec3 col = objLight(vAlb, N, vUp, vH) + vAlb*vec3(1.0, 0.72, 0.45)*(0.07 + 0.06*dot(N, vUp))*objNight(vUp);
  outColor = vec4(col, 1.0);
}`
