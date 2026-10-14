// Full-screen composite: sky, aerial perspective, volumetric clouds (marched at half resolution), sun, stars -> HDR,
// then bloom, auto-exposure and tone mapping to the screen.
// Reads the terrain colour + reversed float depth, so everything is resolved per pixel along the view ray.
import * as THREE from 'three'
import { GLSL_HEAD } from './config.js'
import { NOISE } from './glsl/noise.js'
import { ATMO } from './glsl/atmo.glsl.js'
import { CLOUDS } from './glsl/clouds.glsl.js'

const LUT_FRAG = /* glsl */`${GLSL_HEAD}
in vec2 vUv;
out vec4 outT;
${ATMO}
void main(){
  float s = vUv.x*2.0 - 1.0;
  float mu = sign(s)*s*s;
  float h = vUv.y*vUv.y*ATMO_H;
  float r = PLANET_R + h;
  vec3 rd = vec3(sqrt(max(1.0 - mu*mu, 0.0)), mu, 0.0);
  float b = r*mu;
  float cG = h*(2.0*PLANET_R + h);
  if (mu < 0.0 && b*b - cG > 0.0){ outT = vec4(0.0, 0.0, 0.0, 1.0); return; }
  float cT = (h - ATMO_H)*(2.0*PLANET_R + h + ATMO_H);
  float tTop = -b + sqrt(max(b*b - cT, 0.0));
  vec3 od = vec3(0.0);
  float dt = tTop/48.0;
  for (int i = 0; i < 48; i++){
    vec3 P = vec3(0.0, r, 0.0) + rd*(float(i) + 0.5)*dt;
    od += atmoExtinction(atmoDensity(length(P) - PLANET_R))*dt;
  }
  outT = vec4(exp(-od), 1.0);
}`

// Terrain and clouds are lit with pi*E (Lambert without the 1/pi), so sky radiance gets the same factor.
// what the cloud pass and the composite share: uniforms, the view ray through a pixel, the scene distance along it
const VIEW = /* glsl */`
uniform sampler2D uDepth, uTransLUT, uWeather, uBlue;
uniform sampler3D uNoise;
uniform mat4 uInvProj;
uniform mat3 uCamRot;
uniform vec3 uCamPos, uSun, uWind;
uniform float uCamAlt, uTime, uFlash, uPixAngle, uFog;
uniform vec2 uDepthCD;      // reversed-Z projection terms: view distance = d/(depth + c)
in vec2 vUv;
${NOISE}
${ATMO}
${CLOUDS}
// unproject on the near plane: the far plane is ill-conditioned when near is tiny (far/near ~ 1e9); rv in view space
vec3 viewRay(vec2 uv, out vec3 rv){
  vec4 vp = uInvProj*vec4(uv*2.0 - 1.0, 1.0, 1.0);
  rv = normalize(vp.xyz/vp.w);
  return normalize(uCamRot*rv);
}
float sceneDist(float dz, vec3 rv){ return dz > 0.0 ? uDepthCD.y/(dz + uDepthCD.x)/max(-rv.z, 1e-4) : 1e12; }   // 1e12: sky
// blue-noise ray offset: fine grain with no visible pattern (IGN left diagonal hatching in cloud edges)
float rayJitter(){ return (texelFetch(uBlue, ivec2(gl_FragCoord.xy) & 63, 0).r*255.0 + 0.5)/256.0; }
`

// Clouds at half resolution (the march is most of the composite's cost, and clouds are soft): in-scatter and
// transmittance, plus the scene distance the march stopped at and the cloud distance, for the depth-aware upsampling
const CLOUD_FRAG = /* glsl */`${GLSL_HEAD}
${VIEW}
layout(location = 0) out vec4 outCl;
layout(location = 1) out vec4 outT;
void main(){
  // the farthest of the 2x2 depths below: across an edge this sample takes the far side (sky, distant ground);
  // pixels on the near side find a neighbour like them in the composite, or march their own clouds
  ivec2 q = ivec2(gl_FragCoord.xy)*2, m = textureSize(uDepth, 0) - 1;
  float dz = min(min(texelFetch(uDepth, min(q, m), 0).r, texelFetch(uDepth, min(q + ivec2(1, 0), m), 0).r),
                 min(texelFetch(uDepth, min(q + ivec2(0, 1), m), 0).r, texelFetch(uDepth, min(q + 1, m), 0).r));
  vec3 rv, rd = viewRay(vUv, rv);
  float tScene = sceneDist(dz, rv), tCloud;
  outCl = marchClouds(uCamPos, rd, dot(uCamPos, rd), tScene, rayJitter(), tCloud);
  outT = vec4(tScene, tCloud, 0.0, 0.0);
}`

const FRAG = /* glsl */`${GLSL_HEAD}
#define SKY_E (SUN_I*PI)
#define SUN_R 0.00465
#define NIGHT_SKY 2.5e-4
uniform sampler2D uColor, uCloud, uCloudT;
${VIEW}
out vec4 outColor;

// the half-resolution clouds here: the 4 nearest samples, bilinear, among those marched to about this pixel's scene
// distance (10 %); with none such (the near side of an edge, a thin pole against the sky) the pixel marches its own
bool halfClouds(float tScene, out vec4 cl, out float tCloud){
  ivec2 n = textureSize(uCloud, 0);
  vec2 x = vUv*vec2(n) - 0.5, f = fract(x);
  ivec2 i0 = ivec2(floor(x));
  cl = vec4(0.0); tCloud = 0.0;
  float ws = 0.0;
  for (int k = 0; k < 4; k++){
    ivec2 o = ivec2(k & 1, k >> 1), q = clamp(i0 + o, ivec2(0), n - 1);
    vec2 tt = texelFetch(uCloudT, q, 0).rg;
    float w = max((o.x == 1 ? f.x : 1.0 - f.x)*(o.y == 1 ? f.y : 1.0 - f.y), 1e-3)*step(abs(log(tt.x/tScene)), 0.1);
    cl += w*texelFetch(uCloud, q, 0); tCloud += w*tt.y; ws += w;
  }
  if (ws <= 0.0) return false;
  cl /= ws; tCloud /= ws;
  return true;
}

// point stars on a jittered 3D grid, drawn as pixel-sized gaussians (checks the 8 nearest cells)
vec3 stars(vec3 rd){
  vec3 p = rd*170.0;
  vec3 base = floor(p - 0.5);
  float sig = max(uPixAngle*170.0*0.55, 0.05);
  float twinkle = uCamAlt < ATMO_H*0.5 ? 1.0 : 0.0;
  vec3 c = vec3(0.0);
  for (int k = 0; k < 8; k++){
    vec3 id = base + vec3(k & 1, (k >> 1) & 1, k >> 2);
    float hs = hash13(id);
    if (hs < 0.95) continue;
    vec3 sp = id + 0.5 + (vec3(hash13(id + 1.3), hash13(id + 2.7), hash13(id + 5.1)) - 0.5)*0.8;
    vec3 dp = p - sp;
    float d2 = dot(dp, dp) - pow(dot(dp, rd), 2.0);
    float mag = 1.2e-4 + 0.06*pow(hash13(id + 9.1), 16.0);
    float tw = 1.0 + twinkle*0.45*sin(uTime*(9.0 + 13.0*hs) + hs*80.0);
    vec3 tint = mix(vec3(1.0, 0.9, 0.8), vec3(0.85, 0.9, 1.0), smoothstep(0.2, 0.8, hash13(id + 4.4)));
    c += tint*mag*tw*exp(-d2/(2.0*sig*sig));
  }
  return c;
}

// faint galactic band with a bright bulge and dust lanes, plus airglow
vec3 milkyWay(vec3 rd){
  const vec3 G = vec3(0.46, 0.62, -0.64), C = vec3(0.77, -0.19, 0.61);
  float x = dot(rd, G);
  float n = snoise(rd*5.0)*0.5 + snoise(rd*13.0)*0.3 + snoise(rd*31.0)*0.2;
  float band = exp(-x*x/0.011)*smoothstep(-0.6, 0.8, n);
  float bulge = exp(-pow(1.0 - dot(rd, C), 2.0)*6.0);
  float dust = 1.0 - 0.75*exp(-pow(x + 0.02*snoise(rd*9.0), 2.0)/0.0012)*smoothstep(-0.3, 0.6, snoise(rd*7.0 + 3.1));
  vec3 mw = mix(vec3(0.75, 0.85, 1.0), vec3(1.0, 0.95, 0.85), bulge)*band*(1.0 + 2.5*bulge)*dust;
  return NIGHT_SKY*(vec3(0.3, 0.45, 0.9) + 2.0*mw);
}

vec3 sunDisk(vec3 rd){
  float a = length(cross(rd, uSun));
  if (dot(rd, uSun) < 0.0 || a > SUN_R*1.5) return vec3(0.0);
  float r = a/SUN_R;
  float edge = 1.0 - smoothstep(1.0 - uPixAngle/SUN_R, 1.0 + uPixAngle/SUN_R, r);
  float limb = 1.0 - 0.6*(1.0 - sqrt(max(1.0 - r*r, 0.0)));
  return vec3(30000.0)*edge*limb;
}

float warp(float f, int mode){ return mode == 1 ? f*f : mode == 2 ? 1.0 - (1.0 - f)*(1.0 - f) : f; }

void main(){
  vec3 rv, rd = viewRay(vUv, rv);
  vec3 ro = uCamPos;
  float b = dot(ro, rd);
  float dz = texture(uDepth, vUv).r;
  bool hit = dz > 0.0;
  float tScene = sceneDist(dz, rv);
  vec3 scene = texture(uColor, vUv).rgb;
  float jitter = rayJitter();

  float tCloud;
  vec4 cl;
  if (!halfClouds(tScene, cl, tCloud)) cl = marchClouds(ro, rd, b, tScene, jitter, tCloud);

  float mu = dot(rd, uSun);
  vec3 bg = scene;
  if (!hit){
    // the sky turns with the sun: rotate stars by the sun's hour angle
    float ha = atan(uSun.x, uSun.z);
    vec3 rs = vec3(cos(ha)*rd.x - sin(ha)*rd.z, rd.y, sin(ha)*rd.x + cos(ha)*rd.z);
    // stars only where they can show: at night, or from space
    bool dark = uCamAlt > ATMO_H || dot(normalize(ro), uSun) < 0.25;
    bg = (dark ? stars(rs) + milkyWay(rs) : vec3(0.0)) + sunDisk(rd);
  }

  vec3 col;
  vec2 at = shell(b, ATMO_H);
  if (at.y <= 0.0){
    col = cl.rgb + cl.a*bg;
  } else {
    float t0 = max(at.x, 0.0), t1 = min(at.y, tScene), len = t1 - t0;
    float pr = phaseR(mu), pm = phaseCS(mu, 0.8);
    bool inside = uCamAlt < ATMO_H;
    // dense samples near the camera from inside, near the ground when looking down from space
    int mode = inside ? 1 : (hit ? 2 : 0);
    const int N = 32;
    vec3 od = vec3(0.0), sF = vec3(0.0), sB = vec3(0.0), odFront = vec3(-1.0);
    for (int i = 0; i < N; i++){
      float ta = t0 + len*warp(float(i)/float(N), mode), tb = t0 + len*warp(float(i + 1)/float(N), mode);
      float t = mix(ta, tb, jitter), dt = tb - ta;
      vec3 P = ro + rd*t;
      float r = length(P), h = max(r - PLANET_R, 0.0);
      vec3 d = atmoDensity(h);
      d.y += uFog*exp(-max(h - uCamAlt, 0.0)/400.0);
      vec3 ext = max(atmoExtinction(d), vec3(1e-12));
      if (t > tCloud && odFront.x < 0.0) odFront = od;
      float muS = dot(P, uSun)/r;
      vec3 Ts = sunTrans(uTransLUT, h, muS);
      // multiple scattering: isotropic. Past the terminator it is skylight from the sunlit air above Earth's shadow
      // (height R e^2/2 at sun depression e), so it fades as that line climbs: blue hour, then night by ~-15 deg.
      // (A fixed muS + 0.1 lit high air 6 deg too deep through a long reddened path: a red band at sun -20.)
      float eS = max(-muS, 0.0), hSh = 0.5*PLANET_R*eS*eS;
      vec3 Tms = sunTrans(uTransLUT, max(h, hSh), min(max(muS, 0.0) + 0.1, 1.0))*exp(-max(hSh - h, 0.0)/9000.0);
      vec3 sc = BETA_R*d.x*pr + BETA_M*d.y*pm;
      vec3 ms = (BETA_R*d.x + BETA_M*d.y)*0.035*Tms;
      // energy-conserving segment integral
      vec3 Tseg = exp(-ext*dt);
      vec3 s = exp(-od)*(Ts*sc + ms)*(1.0 - Tseg)/ext;
      od += ext*dt;
      if (t < tCloud) sF += s; else sB += s;
    }
    if (odFront.x < 0.0) odFront = od;
    col = SKY_E*sF + exp(-odFront)*cl.rgb + cl.a*(SKY_E*sB + exp(-od)*bg);
  }
  col += uFlash*vec3(0.5, 0.55, 0.7)*0.4*(1.0 - cl.a);
  // keep half-float targets finite
  col = all(lessThan(abs(col), vec3(6e4))) ? col : vec3(0.0);
  // alpha marks ground (or planet) pixels for the exposure meter
  outColor = vec4(col, hit ? 1.0 : 0.0);
}`

// 2x downsample (4 bilinear taps = 4x4 box); the first level applies exposure and caps fireflies
const DOWN_FRAG = /* glsl */`${GLSL_HEAD}
uniform sampler2D uSrc, uExp;
uniform vec2 uTexel;
uniform float uFirst;
in vec2 vUv;
out vec4 outC;
void main(){
  vec4 c = 0.25*(texture(uSrc, vUv + uTexel*vec2(-1.0, -1.0)) + texture(uSrc, vUv + uTexel*vec2(1.0, -1.0))
               + texture(uSrc, vUv + uTexel*vec2(-1.0, 1.0)) + texture(uSrc, vUv + uTexel*vec2(1.0, 1.0)));
  float A = texelFetch(uExp, ivec2(0), 0).r;
  if (uFirst > 0.5) c.rgb = min(c.rgb*(A > 0.0 ? A : 1.0), vec3(1000.0));
  outC = c;
}`

// 3x3 tent upsample, added onto the next larger level
const UP_FRAG = /* glsl */`${GLSL_HEAD}
uniform sampler2D uSrc;
uniform vec2 uTexel;
in vec2 vUv;
out vec4 outC;
void main(){
  vec3 c = vec3(0.0);
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++)
    c += texture(uSrc, vUv + uTexel*vec2(x, y)).rgb*float((2 - abs(x))*(2 - abs(y)));
  outC = vec4(c/16.0, 1.0);
}`

// Auto-exposure. Inside the atmosphere it follows the light falling on the land (incident metering: sun through the
// cloud deck plus skylight on a card at the camera, as terrain.js lights the ground), not what the camera looks at, so
// tilting or turning never changes the brightness (a reflected meter swung x1.4 by day and ~x40 at sunset between the
// sky and the ground below). Overcast and rain read grey, night sits at the x60 cap. Towards space (50-100 km) it
// hands over to reflected metering: centre-weighted log-average, floored by the luminance-weighted mean (a small bright
// planet in black space still meters on the planet), with a sunlit grey-card ("sunny 16") floor keeping white cloud
// white. Adapted over time in a 1x1 ping-pong; .g/.b hold the incident and reflected targets (debug).
const ADAPT_FRAG = /* glsl */`${GLSL_HEAD}
uniform sampler2D uSrc, uPrev, uTransLUT, uWeather, uUrbanMap;
uniform vec3 uUp, uSun;
uniform float uDt, uCamAlt, uAgl;
in vec2 vUv;
out vec4 outC;
${ATMO}
void main(){
  float prev = texelFetch(uPrev, ivec2(0), 0).r;
  float A = prev > 0.0 ? prev : 1.0;
  float muS = dot(uUp, uSun);
  float air = 1.0 - smoothstep(0.5*ATMO_H, ATMO_H, uCamAlt), day = smoothstep(0.02, 0.2, muS)*air;
  float sw = 0.0, sl = 0.0, s1 = 0.0, s2 = 0.0, sg = 0.0;
  float lg[144], wg[144];
  for (int y = 0; y < 9; y++) for (int x = 0; x < 16; x++){
    vec2 uv = (vec2(x, y) + 0.5)/vec2(16.0, 9.0);
    vec2 q = (uv - 0.5)*vec2(1.6, 1.0);
    vec4 c = textureLod(uSrc, uv, 0.0);   // a = ground fraction
    float w = exp(-dot(q, q)*3.0), wf = w*mix(1.0, c.a, day);
    w *= mix(1.0, mix(0.3, 1.0, c.a), day);
    float L = max(dot(c.rgb, vec3(0.2126, 0.7152, 0.0722))/A, 1e-7);
    sw += w; sl += w*log(L); s1 += wf*L; s2 += wf*L*L;
    lg[y*16 + x] = log(L); wg[y*16 + x] = exp(-dot(q, q)*1.5); sg += wg[y*16 + x];
  }
  // highlight guard: the level the brightest 12 % of the view sits above (bisection; the sun disk is a single sample)
  float lo = -16.0, hi = 8.0;
  int n = 144 + int(min(uDt, 0.0));   // data-dependent bound keeps ANGLE from unrolling
  for (int k = 0; k < 12; k++){
    float m = 0.5*(lo + hi), f = 0.0;
    for (int i = 0; i < n; i++) f += wg[i]*step(m, lg[i]);
    if (f > 0.12*sg) lo = m; else hi = m;
  }
  float meter = 0.2/max(exp(sl/sw), 0.4*s2/max(s1, 1e-12));
  float Ts = sunTrans(uTransLUT, max(uCamAlt, 0.0), muS).g;
  // incident: sun through the cloud deck (same weather lookup as terrain cloudShadow) plus skylight, which carries on
  // into twilight like terrain.js skyAmbient (luminance 0.044, gone with the sun ~10 deg down)
  vec3 qc = normalize(uUp*(PLANET_R + 3000.0) + uSun*(3000.0/max(muS, 0.08)));
  float sh = mix(1.0 - 0.8*smoothstep(0.35, 0.95, weatherSmooth(uWeather, qc).r), 1.0, smoothstep(CLOUD_BASE, CLOUD_TOP, uCamAlt));
  // plus street light: towns light the ground around the camera (urban map at ~20 km blur), fading with height,
  // so a lit city at night is exposed for its lamps (x5-30 near the ground) instead of for a dark field (x60)
  vec2 uvU = vec2(atan(uUp.x, uUp.z)*(0.5/PI) + 0.5, asin(clamp(uUp.y, -1.0, 1.0))/PI + 0.5);
  float Ea = 0.0032*textureLod(uUrbanMap, uvU, 1.0).r/(1.0 + max(uAgl, 0.0)/100.0);
  float E = max(Ts*(sh*max(muS, 0.0) + 0.15), 0.044*smoothstep(-0.18, 0.35, muS)) + Ea;
  // key: by day the land (mostly dark vegetation) sits ~0.8 EV over a grey card (the old 40 % reflected share gave
  // that); at dusk the bright sky is protected instead, so the ground darkens and the afterglow keeps its colour
  float key = mix(0.2, 1.8, smoothstep(-0.02, 0.25, muS));
  float inc = key*0.2/(0.18*SUN_I*max(E, 1e-6));
  float card = 0.2/(0.18*SUN_I*max(Ts, 0.3))*smoothstep(0.0, 0.3, Ts);
  float target = clamp(max(exp(mix(log(meter), log(min(inc, 60.0)), air)), card), 0.03, 60.0);
  target = max(min(target, 2.0*exp(-lo)), min(card, target));   // only ever darkens, and only when that much would clip white
  float a = prev > 0.0 ? mix(log(prev), log(target), 1.0 - exp(-uDt*1.8)) : log(target);
  outC = vec4(exp(a), inc, meter, 2.0*exp(-lo));   // .gba: debug (incident, reflected, highlight guard)
}`

const FINAL_FRAG = /* glsl */`${GLSL_HEAD}
uniform sampler2D uHdr, uBloom, uExp;
uniform float uUser;
in vec2 vUv;
out vec4 outColor;
${NOISE}
vec3 aces(vec3 x){
  const mat3 I = mat3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
  const mat3 O = mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
  x = I*x;
  vec3 a = x*(x + 0.0245786) - 0.000090537;
  vec3 b = x*(0.983729*x + 0.4329510) + 0.238081;
  return clamp(O*(a/b), 0.0, 1.0);
}
void main(){
  float A = texelFetch(uExp, ivec2(0), 0).r;
  vec3 c = texture(uHdr, vUv).rgb*A + texture(uBloom, vUv).rgb*0.006;
  c = pow(aces(c*uUser), vec3(1.0/2.2));
  outColor = vec4(c + (hash12(gl_FragCoord.xy) - 0.5)/255.0, 1.0);
}`

const LEVELS = 6

// 64x64 blue-noise ranks (void filling: each new point goes into the emptiest spot of a toroidal gaussian energy field)
function blueNoise(N = 64) {
  const n = N * N, E = new Float32Array(n), rank = new Int32Array(n).fill(-1), K = []
  let seed = 7
  for (let i = 0; i < n; i++) { seed = Math.imul(seed, 1664525) + 1013904223 | 0; E[i] = (seed >>> 8) * 1e-12 }   // random tie-break
  for (let y = -7; y <= 7; y++) for (let x = -7; x <= 7; x++) K.push(x, y, Math.exp(-(x * x + y * y) / 4.5))
  let p = 0
  for (let r = 0; r < n; r++) {
    rank[p] = r
    const px = p % N, py = (p - px) / N
    for (let k = 0; k < K.length; k += 3) E[((py + K[k + 1] + N) % N) * N + (px + K[k] + N) % N] += K[k + 2]
    let best = Infinity
    for (let i = 0; i < n; i++) if (rank[i] < 0 && E[i] < best) { best = E[i]; p = i }
  }
  const tex = new THREE.DataTexture(Uint8Array.from(rank, r => r * 256 / n), N, N, THREE.RedFormat)
  tex.needsUpdate = true
  return tex
}
const DEFAULT_EXPOSURE = 0.55   // settings.exposure that means "no user adjustment"

export class Post {
  constructor(shared, noiseTex) {
    const raw = (fragmentShader, uniforms, extra) => shared.makeQuad(new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: shared.quadVert, fragmentShader, uniforms, depthTest: false, depthWrite: false, ...extra,
    }))
    this.lut = this.target(256, 128, THREE.FloatType)
    shared.uTransLUT.value = this.lut.texture
    this.lutQuad = raw(LUT_FRAG, {})
    this.u = {
      uColor: { value: null }, uDepth: { value: null }, uNoise: { value: noiseTex },
      uTransLUT: shared.uTransLUT, uWeather: shared.uWeather, uSun: shared.uSun, uTime: shared.uTime, uPixAngle: shared.uPixAngle,
      uInvProj: { value: new THREE.Matrix4() }, uCamRot: { value: new THREE.Matrix3() },
      uCamPos: { value: new THREE.Vector3() }, uCamAlt: { value: 0 }, uWind: { value: new THREE.Vector3() },
      uDepthCD: { value: new THREE.Vector2() }, uFlash: { value: 0 }, uFog: { value: 0 }, uBlue: { value: blueNoise() },
      uCloud: { value: null }, uCloudT: { value: null },
    }
    this.clouds = raw(CLOUD_FRAG, this.u)
    this.quad = raw(FRAG, this.u)
    this.down = raw(DOWN_FRAG, { uSrc: { value: null }, uExp: { value: null }, uTexel: { value: new THREE.Vector2() }, uFirst: { value: 0 } })
    this.up = raw(UP_FRAG, { uSrc: { value: null }, uTexel: { value: new THREE.Vector2() } }, { blending: THREE.AdditiveBlending, transparent: true })
    this.adapt = raw(ADAPT_FRAG, {
      uSrc: { value: null }, uPrev: { value: null }, uDt: { value: 0 }, uTransLUT: shared.uTransLUT, uWeather: shared.uWeather,
      uSun: shared.uSun, uUp: { value: new THREE.Vector3() }, uCamAlt: { value: 0 }, uAgl: { value: 0 }, uUrbanMap: shared.civ.uUrbanMap,
    })
    this.final = raw(FINAL_FRAG, { uHdr: { value: null }, uBloom: { value: null }, uExp: { value: null }, uUser: { value: 1 } })
    const exp = () => new THREE.WebGLRenderTarget(1, 1, { type: THREE.FloatType, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter })
    this.exp = [exp(), exp()]
    this.hdr = null
    this.cloud = null
    this.levels = []
    this.sizes = new Map()   // targets per scene size: main.js switches between full and reduced resolution
    this.lutReady = false
    this.last = performance.now()
    Post.last = this   // debug handle: (await import('/js/post.js')).Post.last
  }

  target(w, h, type = THREE.HalfFloatType) {
    return new THREE.WebGLRenderTarget(w, h, { type, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter })
  }

  resize(w, h) {
    if (this.hdr && this.hdr.width === w && this.hdr.height === h) return
    const key = `${w}x${h}`
    let s = this.sizes.get(key)
    if (!s) {
      s = { hdr: this.target(w, h), levels: [] }
      s.cloud = new THREE.WebGLRenderTarget(Math.ceil(w / 2), Math.ceil(h / 2), {
        count: 2, type: THREE.FloatType, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
      })
      for (let i = 1; i <= LEVELS; i++) s.levels.push(this.target(Math.max(1, w >> i), Math.max(1, h >> i)))
      this.sizes.set(key, s)
      // keep the two sizes in use (full and reduced); a window resize retires the older ones
      for (const [k, o] of this.sizes) {
        if (this.sizes.size <= 2) break
        if (k === key) continue
        for (const t of [o.hdr, o.cloud, ...o.levels]) t.dispose()
        this.sizes.delete(k)
      }
    }
    Object.assign(this, s)
    this.u.uCloud.value = s.cloud.textures[0]
    this.u.uCloudT.value = s.cloud.textures[1]
  }

  pass(renderer, quad, target) {
    renderer.setRenderTarget(target)
    renderer.render(quad.scene, quad.camera)
  }

  render(renderer, sceneRT, camera, cam, opts) {
    const now = performance.now(), dt = Math.min((now - this.last) / 1000, 0.25)
    this.last = now
    if (!this.lutReady) { this.pass(renderer, this.lutQuad, this.lut); this.lutReady = true }
    this.resize(sceneRT.width, sceneRT.height)

    const u = this.u
    u.uColor.value = sceneRT.texture
    u.uDepth.value = sceneRT.depthTexture
    u.uInvProj.value.copy(camera.projectionMatrixInverse)
    u.uDepthCD.value.set(camera.projectionMatrix.elements[10], camera.projectionMatrix.elements[14])
    u.uCamRot.value.setFromMatrix4(camera.matrixWorld)
    u.uCamPos.value.copy(cam.pos)
    u.uCamAlt.value = cam.altitude
    u.uFlash.value = opts.flash
    u.uFog.value = opts.fog
    u.uWind.value.copy(opts.wind)
    this.pass(renderer, this.clouds, this.cloud)
    this.pass(renderer, this.quad, this.hdr)

    // bloom pyramid; level 0 is exposed and capped
    const [expPrev, expNext] = this.exp
    const d = this.down.mesh.material.uniforms
    d.uExp.value = expPrev.texture
    let src = this.hdr
    this.levels.forEach((lvl, i) => {
      d.uSrc.value = src.texture
      d.uTexel.value.set(1 / src.width, 1 / src.height)
      d.uFirst.value = i === 0 ? 1 : 0
      this.pass(renderer, this.down, lvl)
      src = lvl
    })

    const a = this.adapt.mesh.material.uniforms
    a.uSrc.value = this.levels[3].texture
    a.uPrev.value = expPrev.texture
    a.uDt.value = dt
    a.uCamAlt.value = cam.altitude
    a.uAgl.value = cam.agl
    a.uUp.value.copy(cam.pos).normalize()
    this.pass(renderer, this.adapt, expNext)
    this.exp.reverse()

    const up = this.up.mesh.material.uniforms
    for (let i = LEVELS - 1; i > 0; i--) {
      up.uSrc.value = this.levels[i].texture
      up.uTexel.value.set(1 / this.levels[i].width, 1 / this.levels[i].height)
      this.pass(renderer, this.up, this.levels[i - 1])
    }

    const f = this.final.mesh.material.uniforms
    f.uHdr.value = this.hdr.texture
    f.uBloom.value = this.levels[0].texture
    f.uExp.value = expNext.texture
    f.uUser.value = opts.exposure / DEFAULT_EXPOSURE
    this.pass(renderer, this.final, null)
  }
}
