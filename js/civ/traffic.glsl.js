// Vehicle shaders. Each instance is one vehicle: a sea-level anchor relative to the camera, its unit
// direction (for the ground height), heading and paint. The body sits on terrain() at the anchor (or on a bridge
// deck height from the CPU) and pitches with the road; lamp quads become camera-facing glows at night that stay a few pixels wide
// far away, so traffic reads as streams of light from altitude.
import { OBJ_COMMON, OBJ_LIGHT } from './objects.glsl.js'

// vertex parts (aInfo.x), shared with traffic.js
export const PART = { PAINT: 0, GLASS: 1, DARK: 2, BOX: 3, HEAD: 4, TAIL: 5 }

export const TRAFFIC_VERT = /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
uniform mat4 projectionMatrix, viewMatrix;
uniform float uPixAngle;
in vec3 position, normal;   // local metres: x right, y up, z forward
in vec3 aInfo;              // part, lamp corner
in vec4 iA, iB, iC, iD;     // (anchor - camera, length scale) (unit dir, height scale) (heading, seed) (paint, lit windows)
in vec2 iE;                 // bridge deck height (m; < -1e8 on the ground) and its grade along the heading
out vec3 vN, vRel, vUp, vAlb;
out vec2 vCorner;
out float vH, vGlow, vWin, vFragDepth;
flat out int vPart;
void main(){
  vec3 up = normalize(iB.xyz), info;
  vec4 t = vec4(iE.x, 0.0, 0.0, 0.0);
  float slope = iE.y;
  if (iE.x < -1e8){ t = terrain(up, 1.2e6, info); slope = clamp(dot(t.yzw, iC.xyz)/PLANET_R, -0.3, 0.3); }
  vec3 fw = normalize(iC.xyz + up*slope), rt = normalize(cross(fw, up)), u2 = cross(rt, fw);
  int part = int(aInfo.x + 0.5);
  vec3 lp = position*vec3(1.0, iB.w, iA.w);
  vec3 rel = iA.xyz + up*t.x + rt*lp.x + u2*lp.y + fw*lp.z;
  vec3 n = rt*normal.x + u2*normal.y + fw*normal.z;
  float night = objNight(up), brake = step(1.5, iD.w), win = iD.w - 2.0*brake;   // iD.w = lit windows + 2 * braking
  vGlow = 0.0;
  vec4 vp;
  if (part >= ${PART.HEAD}){
    // lamp: a disc facing the camera, at least ~3 px wide at night
    float dist = length(rel), size = max(0.13, uPixAngle*dist*1.6*night);
    vp = viewMatrix*vec4(rel + n*size, 1.0);
    vp.xy += aInfo.yz*size;
    float facing = smoothstep(-0.15, 0.6, dot(n, -rel/dist));
    // brake lights: brighter tails at night, a small red glow by day
    vGlow = facing*(part == ${PART.HEAD} ? 60.0*night : (14.0 + 22.0*brake)*night + 1.5*brake)*sqrt(0.13/size);
  } else vp = viewMatrix*vec4(rel, 1.0);
  float seed = iC.w;
  vAlb = part == ${PART.PAINT} ? iD.rgb : part == ${PART.GLASS} ? vec3(0.018, 0.022, 0.028)
       : part == ${PART.DARK} ? vec3(0.022) : vec3(0.62 - 0.35*fract(seed*7.31), 0.6 - 0.33*fract(seed*7.31), 0.58 - 0.3*fract(seed*3.7));
  vWin = part == ${PART.GLASS} ? win*night : 0.0;
  vN = n; vRel = rel; vUp = up; vH = t.x + lp.y; vCorner = aInfo.yz; vPart = part;
  gl_Position = projectionMatrix*vp;
  vFragDepth = 1.0 + gl_Position.w;
}`

export const TRAFFIC_FRAG = /* glsl */`${OBJ_COMMON}
${OBJ_LIGHT}
in vec3 vN, vRel, vUp, vAlb;
in vec2 vCorner;
in float vH, vGlow, vWin, vFragDepth;
flat in int vPart;
out vec4 outColor;
void main(){
  vec3 N = normalize(vN), V = normalize(-vRel), col;
  if (vPart >= ${PART.HEAD}){
    float r2 = dot(vCorner, vCorner);
    if (r2 > 1.0) discard;
    vec3 lamp = vPart == ${PART.HEAD} ? vec3(1.0, 0.93, 0.8) : vec3(1.0, 0.03, 0.015);
    float core = (1.0 - r2)*(1.0 - r2);
    col = objLight(lamp*0.5, N, vUp, vH) + lamp*vGlow*core;
  } else {
    float rough = vPart == ${PART.GLASS} ? 0.1 : vPart == ${PART.DARK} ? 0.8 : 0.3;
    float fr = 0.04 + 0.96*pow(1.0 - max(dot(N, V), 0.0), 5.0);
    col = objLight(vAlb, N, vUp, vH) + objSun(vUp, vH)*objSpec(N, V, rough)*(vPart == ${PART.DARK} ? 0.1 : 1.0)
        + objAmbient(vUp)*fr*(vPart == ${PART.DARK} ? 0.1 : 0.5)
        + vec3(1.0, 0.8, 0.55)*vWin*0.35;
  }
  outColor = vec4(col, 1.0);
  gl_FragDepth = log2(vFragDepth)*uLogDepthFC*0.5;
}`
